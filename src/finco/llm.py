"""Client gọi LLM qua OpenAI-compatible API (Groq), có Retry + Exponential Backoff + Jitter (W2).

Đọc file từ trên xuống:
  1. get_client / load_prompt   : khóa API từ .env, đọc prompt từ prompts/*.txt
  2. _create_with_retry         : NƠI DUY NHẤT quyết định lỗi nào retry
  3. call_llm                   : hỏi đáp thuần văn bản -> (nội dung, Usage)
  4. call_llm_tools             : cho model gọi tool (function calling) -> ChatTurn

Quy tắc retry: lỗi tạm thời (429, 5xx, timeout, mất kết nối) thì chờ rồi thử lại, ưu tiên header
Retry-After; lỗi cố định (401, 400, 403, 404) thì raise ngay. Chỉ log ra stderr: stdout dành cho MCP.
"""

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from finco.types import Usage

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))  # hạ xuống khi server có ngữ cảnh nhỏ (vLLM 8k)
DEFAULT_TIMEOUT = 60.0    # giây cho một lần gọi, để một lần treo không làm treo cả lượt
MAX_RETRY_AFTER = 60.0    # Retry-After lớn hơn mức này thì chỉ chờ tối đa bằng mức này
# Hết hạn mức TOKEN/NGÀY (TPD): thử lại sau vài giây vô ích. Server báo rõ "try again in 9m15s" nên chờ
# đúng thời gian đó (không đốt lần retry), nhưng không chờ quá MAX_DAILY_WAIT: lâu hơn thì dừng ngay.
MAX_DAILY_WAIT = float(os.getenv("LLM_MAX_DAILY_WAIT", "1800"))
MAX_DAILY_WAITS = 12
_TRY_AGAIN = re.compile(r"try again in\s+((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s(?!\w))?(?:[\d.]+ms)?)", re.IGNORECASE)

_RETRY = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_NO_RETRY = (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError)
_CLIENT: Optional[OpenAI] = None



class EmptyResponseError(RuntimeError):
    """LLM trả về phản hồi không dùng được (không có choices, hoặc nội dung rỗng)."""

def _extra_content(raw_tool_call: Any) -> Optional[dict]:
    """Phần `extra_content` (vd. thought_signature của Gemini) mà SDK giữ ở model_extra, nếu có."""
    value = getattr(raw_tool_call, "extra_content", None)
    if value is None:
        extra = getattr(raw_tool_call, "model_extra", None)
        value = extra.get("extra_content") if isinstance(extra, dict) else None
    return value if isinstance(value, dict) and value else None


def _parse_args(raw: str) -> tuple[dict, str]:
    """Parse chuỗi JSON tham số: trả (dict, '') hoặc ({}, 'Error: ...')."""
    try:
        args = json.loads((raw or "").strip() or "{}")
        if not isinstance(args, dict):
            return {}, 'Error: Tham số gọi tool phải là một đối tượng JSON dạng {"tên": giá_trị}.'
        return args, ""
    except json.JSONDecodeError as e:
        return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})."


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"                                         # Chuỗi thô (để tương thích test)
    args: dict = field(default_factory=dict, compare=False)      # Dict đã parse (cho agent)
    error: Optional[str] = field(default=None, compare=False)    # Báo lỗi nếu JSON hỏng
    # Gemini 3 gắn chữ ký suy luận vào tool_call (extra_content.google.thought_signature) và đòi gửi lại
    # nguyên vẹn ở lượt sau; thiếu thì HTTP 400. Các nhà cung cấp khác không có trường này (None).
    extra_content: Optional[dict] = field(default=None, compare=False)

    def __post_init__(self):
        """Tự động parse arguments sang args và error nếu chưa có."""
        if not self.args and not self.error and self.arguments:
            parsed, err = _parse_args(self.arguments)
            self.args = parsed
            self.error = err or None

@dataclass
class ChatTurn:
    """Một lượt trả lời của model: văn bản và/hoặc các lời gọi tool, kèm Usage của lần gọi."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    finish_reason: Optional[str] = None

    def __iter__(self):
        """Cho phép unpack kiểu tuple: content, usage = call_llm(...) khi chat văn bản."""
        return iter((self.content, self.usage))
    
    def to_assistant_message(self) -> dict[str, Any]:
        """Tự biến thành dict OpenAI để append thẳng vào messages của Agent."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False) if not tc.error else "{}"
                    },
                    **({"extra_content": tc.extra_content} if tc.extra_content else {}),
                }
                for tc in self.tool_calls
            ]
        return msg
    
_CRITIC_CLIENT: Optional[OpenAI] = None

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"


def critic_endpoint() -> Optional[tuple[str, str]]:
    """(api_key, base_url) riêng cho Critic, hoặc None nếu dùng chung endpoint với Actor.

    FINCO_CRITIC_PROVIDER=gemini: khóa GEMINI_API_KEY, endpoint OpenAI-compatible của Google.
    Hoặc đặt trực tiếp FINCO_CRITIC_BASE_URL + FINCO_CRITIC_API_KEY (vLLM, OpenRouter...).
    """
    if os.getenv("FINCO_CRITIC_PROVIDER", "").strip().lower() == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("FINCO_CRITIC_PROVIDER=gemini nhưng thiếu GEMINI_API_KEY trong .env.")
        return key, os.getenv("FINCO_CRITIC_BASE_URL") or GEMINI_BASE_URL
    base, key = os.getenv("FINCO_CRITIC_BASE_URL"), os.getenv("FINCO_CRITIC_API_KEY")
    if base and key:
        return key, base
    return None


def critic_default_model() -> Optional[str]:
    """Model mặc định của Critic khi gọi với profile='critic' (None = dùng LLM_MODEL như Actor)."""
    m = os.getenv("FINCO_CRITIC_MODEL")
    if m:
        return m
    if os.getenv("FINCO_CRITIC_PROVIDER", "").strip().lower() == "gemini":
        return GEMINI_DEFAULT_MODEL
    return None


def get_client(profile: Optional[str] = None) -> OpenAI:
    """profile='critic': client riêng cho Critic nếu có cấu hình (critic_endpoint), không thì dùng chung."""
    global _CLIENT, _CRITIC_CLIENT
    if profile == "critic":
        ep = critic_endpoint()
        if ep is not None:
            if _CRITIC_CLIENT is None:
                _CRITIC_CLIENT = OpenAI(api_key=ep[0], base_url=ep[1], max_retries=0, timeout=DEFAULT_TIMEOUT)
            return _CRITIC_CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError("Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env.")

        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=DEFAULT_TIMEOUT
        )
        
    return _CLIENT

def merge_usage(*usages: Usage) -> Usage:
    return Usage(
        calls=sum(u.calls for u in usages),
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        max_prompt=max((u.max_prompt for u in usages), default=0),
    )

def load_prompt(filename: str) -> str:
    """Nội dung file prompt trong thư mục prompts/."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file prompt: {path}")
    return path.read_text(encoding="utf-8").strip()

def _daily_quota_wait(e: Exception) -> Optional[float]:
    """Số giây server yêu cầu chờ nếu lỗi là hết hạn mức token/NGÀY; None nếu không phải lỗi đó."""
    msg = str(e)
    if "per day" not in msg and "(TPD)" not in msg:
        return None
    m = _TRY_AGAIN.search(msg)
    if not m or not m.group(1):
        return float("inf")  # hết hạn mức ngày nhưng không biết bao giờ hồi
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def _create_with_retry(
    request: dict[str, Any],
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
    profile: Optional[str] = None,
):
    client = get_client(profile) if profile else get_client()
    delay = initial_delay

    attempt = 0
    daily_waits = 0
    while attempt <= max_retries:
        try:
            return client.chat.completions.create(**request)
        
        except _NO_RETRY as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            err_name = f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__
            print(f"[llm] Lỗi cố định {err_name}: {e}. Dừng ngay!", file=sys.stderr)
            raise

        except _RETRY as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            err_name = f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__

            daily = _daily_quota_wait(e)
            if daily is not None:
                if daily > MAX_DAILY_WAIT or daily_waits >= MAX_DAILY_WAITS:
                    shown = "không rõ" if daily == float("inf") else f"{daily / 60:.0f} phút"
                    print(f"[llm] Hết hạn mức token/ngày, cần chờ {shown} (> {MAX_DAILY_WAIT / 60:.0f} phút): dừng.", file=sys.stderr)
                    raise RuntimeError(f"Gọi LLM thất bại: hết hạn mức token/ngày ({err_name}), cần chờ {shown}: {e}") from e
                daily_waits += 1
                wait = daily + random.uniform(1.0, 3.0)
                print(f"[llm] Hết hạn mức token/ngày. Chờ đúng {wait:.0f}s rồi thử lại (không tính vào số lần retry)...", file=sys.stderr)
                time.sleep(wait)
                continue  # không tăng attempt: chờ đủ lâu thì lần này là lần thử "mới"

            if attempt == max_retries:
                print(f"[llm] Đã thử lại {max_retries} lần nhưng vẫn thất bại: {e}", file=sys.stderr)
                raise RuntimeError(f"Gọi LLM thất bại sau {max_retries} lần thử lại do lỗi {err_name}: {e}") from e

            headers = getattr(getattr(e, "response", None), "headers", None) or {}
            try:
                retry_after = float(headers.get("retry-after", -1))
            except (ValueError, TypeError):
                retry_after = -1.0

            base = min(retry_after, MAX_RETRY_AFTER) if retry_after >= 0 else min(max_delay, delay)
            wait = base + random.uniform(0.1, 0.5)

            print(f"[llm] Gặp lỗi tạm thời {err_name}. Chờ {wait:.2f}s...", file=sys.stderr)
            time.sleep(wait)
            delay = min(max_delay, delay * 2.0)
            attempt += 1

    raise RuntimeError("Gọi LLM thất bại: vòng lặp kết thúc bất thường.")


def call_llm(
    messages: list[dict[str, Any]],            
    tools: Optional[list[dict[str, Any]]] = None, 
    model: Optional[str] = None,               
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
    stop: Optional[list[str]] = None,
    profile: Optional[str] = None,
) -> ChatTurn:
    if profile == "critic" and not model and critic_endpoint() is not None:
        model = critic_default_model()
    request: dict[str, Any] = {
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if stop:
        request["stop"] = stop
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    response = _create_with_retry(
        request, max_retries=max_retries, initial_delay=initial_delay, max_delay=max_delay, profile=profile
    )

    if not getattr(response, "choices", None):
        raise EmptyResponseError("LLM trả về phản hồi không có choices.")
    
    choice = response.choices[0]
    message = choice.message
    content = (message.content or "").strip()

    if getattr(choice, "finish_reason", None) == "length":
        print("[llm] Cảnh báo: phản hồi bị cắt do max_tokens (finish_reason='length').", file=sys.stderr)

    raw_tool_calls = getattr(message, "tool_calls", None) or []
    
    if not content and not raw_tool_calls:
        hint = " (bị cắt do max_tokens, hãy tăng max_tokens)" if choice.finish_reason == "length" else ""
        raise EmptyResponseError(f"LLM trả về nội dung rỗng, finish_reason={choice.finish_reason!r}{hint}.")

    tool_calls: list[ToolCall] = [
        ToolCall(
            id=getattr(tc, "id", "") or f"call_{i}",
            name=tc.function.name,
            arguments=tc.function.arguments or "{}",
            extra_content=_extra_content(tc),
        )
        for i, tc in enumerate(raw_tool_calls)
    ]

    u = getattr(response, "usage", None)
    prompt = getattr(u, "prompt_tokens", 0) if u else 0
    completion = getattr(u, "completion_tokens", 0) if u else 0
    usage = Usage(calls=1, prompt_tokens=prompt, completion_tokens=completion, max_prompt=prompt)

    return ChatTurn(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=getattr(choice, "finish_reason", None)
    )

call_llm_tools = call_llm