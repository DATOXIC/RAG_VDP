"""Client gọi LLM qua OpenAI-compatible API (Groq), có Retry + Exponential Backoff + Jitter (W2).

Đọc file từ trên xuống:
  1. get_client / load_prompt   : khóa API từ .env, đọc prompt từ prompts/*.txt
  2. _create_with_retry         : NƠI DUY NHẤT quyết định lỗi nào retry
  3. call_llm                   : hỏi đáp thuần văn bản -> (nội dung, Usage)
  4. call_llm_tools             : cho model gọi tool (function calling) -> ChatTurn

Quy tắc retry: lỗi tạm thời (429, 5xx, timeout, mất kết nối) thì chờ rồi thử lại, ưu tiên header
Retry-After; lỗi cố định (401, 400, 403, 404) thì raise ngay. Chỉ log ra stderr: stdout dành cho MCP.
"""

import os
import random
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
DEFAULT_TIMEOUT = 60.0    # giây cho một lần gọi, để một lần treo không làm treo cả lượt
MAX_RETRY_AFTER = 60.0    # Retry-After lớn hơn mức này thì chỉ chờ tối đa bằng mức này

_TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_PERMANENT = (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError)
_CLIENT: Optional[OpenAI] = None


class EmptyResponseError(RuntimeError):
    """LLM trả về phản hồi không dùng được (không có choices, hoặc nội dung rỗng)."""


class MalformedToolCallError(RuntimeError):
    """Model sinh lời gọi tool sai định dạng và server từ chối (Groq: 400 "tool_use_failed").

    Khác lỗi 400 thường: nhắc model thử lại thường sửa được, nên agent bắt riêng.
    """


@dataclass
class ToolCall:
    """Một lời gọi tool do model yêu cầu. arguments là chuỗi JSON thô, CHƯA parse."""
    id: str
    name: str
    arguments: str


@dataclass
class ChatTurn:
    """Một lượt trả lời của model: văn bản và/hoặc các lời gọi tool, kèm Usage của lần gọi."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    finish_reason: Optional[str] = None


# --------------------------------------------------------------------------- 1. Cấu hình
def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def _describe(e: Exception) -> str:
    """Tên lỗi kèm mã HTTP nếu có."""
    status = getattr(e, "status_code", None)
    return f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__


def get_client() -> OpenAI:
    """OpenAI client dùng chung (singleton).

    max_retries=0: tắt retry ngầm của SDK, để _create_with_retry là nơi duy nhất quyết định retry
    (nếu không số request thật bị nhân lên). timeout hạ từ 10 phút của SDK xuống DEFAULT_TIMEOUT.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError("Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env.")
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=DEFAULT_TIMEOUT)
    return _CLIENT


def load_prompt(filename: str) -> str:
    """Nội dung file prompt trong thư mục prompts/."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def merge_usage(*usages: Usage) -> Usage:
    """Tổng Usage của nhiều lần gọi: calls, prompt, completion được CỘNG; max_prompt lấy GIÁ TRỊ LỚN NHẤT."""
    return Usage(
        calls=sum(u.calls for u in usages),
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        max_prompt=max((u.max_prompt for u in usages), default=0),
    )


# --------------------------------------------------------------------------- 2. Retry
def _retry_wait(e: Exception, delay: float, max_delay: float) -> float:
    """Giây chờ trước lần thử tiếp: ưu tiên header Retry-After, không có thì dùng backoff; cộng jitter."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        retry_after = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        retry_after = -1.0
    base = min(retry_after, MAX_RETRY_AFTER) if retry_after >= 0 else min(max_delay, delay)
    return base + random.uniform(0.1, 0.5)


def _create_with_retry(request: dict[str, Any], max_retries: int, initial_delay: float, max_delay: float):
    """chat.completions.create có retry; trả phản hồi thô. Raise ngay với lỗi cố định."""
    client = get_client()
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request)
        except _PERMANENT as e:
            _log(f"Lỗi cố định {_describe(e)}: {e}. Dừng ngay, không retry.")
            raise
        except _TRANSIENT as e:
            if attempt == max_retries:
                _log(f"Đã thử lại {max_retries} lần nhưng vẫn thất bại {_describe(e)}: {e}")
                raise RuntimeError(
                    f"Gọi LLM thất bại sau {max_retries} lần thử lại do lỗi {_describe(e)}: {e}"
                ) from e
            wait = _retry_wait(e, delay, max_delay)
            _log(f"Lỗi tạm thời {_describe(e)} ở lần thử {attempt + 1}/{max_retries}. Chờ {wait:.2f}s...")
            time.sleep(wait)
            delay = min(max_delay, delay * 2.0)
    raise RuntimeError("Gọi LLM thất bại: vòng lặp retry kết thúc bất thường.")  # không tới được


def _usage_of(response) -> Usage:
    """Usage của MỘT lần gọi (calls=1). Lần thử lại thất bại không đếm vì không tốn token."""
    u = getattr(response, "usage", None)
    prompt = u.prompt_tokens if u else 0
    return Usage(calls=1, prompt_tokens=prompt, completion_tokens=u.completion_tokens if u else 0, max_prompt=prompt)


def _first_choice(response):
    if not getattr(response, "choices", None):
        raise EmptyResponseError("LLM trả về phản hồi không có choices.")
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        _log("Cảnh báo: phản hồi bị cắt do max_tokens (finish_reason='length').")
    return choice


# --------------------------------------------------------------------------- 3. Hỏi đáp thuần văn bản
def call_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
    stop: Optional[list[str]] = None,
) -> tuple[str, Usage]:
    """Gọi LLM, trả (nội dung, Usage của lần gọi).

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}].
    Raises: lỗi cố định (không retry); RuntimeError khi hết retry; EmptyResponseError khi nội dung rỗng.
    """
    request: dict[str, Any] = {
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout": timeout,
    }
    if stop:
        request["stop"] = stop
    response = _create_with_retry(request, max_retries, initial_delay, max_delay)

    # Phân tích NGOÀI vòng retry để EmptyResponseError không bị coi là lỗi tạm thời.
    choice = _first_choice(response)
    content = choice.message.content or ""
    if not content.strip():
        hint = " (bị cắt do max_tokens, hãy tăng max_tokens)" if choice.finish_reason == "length" else ""
        raise EmptyResponseError(f"LLM trả về nội dung rỗng, finish_reason={choice.finish_reason!r}{hint}.")
    return content, _usage_of(response)


# --------------------------------------------------------------------------- 4. Gọi tool
def call_llm_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
) -> ChatTurn:
    """Như call_llm nhưng model được gọi tool; trả ChatTurn (văn bản và/hoặc tool_calls).

    messages có thể chứa role "tool" và assistant kèm tool_calls (dict thuần, định dạng OpenAI).
    Nội dung rỗng KHÔNG phải lỗi khi có tool_calls.
    Raises: MalformedToolCallError khi server từ chối lời gọi tool sai định dạng; còn lại như call_llm.
    """
    request: dict[str, Any] = {
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages, "tools": tools, "tool_choice": "auto",
        "temperature": temperature, "max_tokens": max_tokens, "timeout": timeout,
    }
    try:
        response = _create_with_retry(request, max_retries, initial_delay, max_delay)
    except BadRequestError as e:
        if getattr(e, "code", None) == "tool_use_failed" or "tool_use_failed" in str(e):
            raise MalformedToolCallError(f"Model gọi tool sai định dạng: {e}") from e
        raise

    choice = _first_choice(response)
    message = choice.message
    tool_calls = [
        ToolCall(id=getattr(tc, "id", "") or f"call_{i}", name=tc.function.name, arguments=tc.function.arguments or "")
        for i, tc in enumerate(getattr(message, "tool_calls", None) or [])
    ]
    return ChatTurn(
        content=(message.content or "").strip(),
        tool_calls=tool_calls,
        usage=_usage_of(response),
        finish_reason=getattr(choice, "finish_reason", None),
    )
