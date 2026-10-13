"""Client gọi mô hình LLM qua OpenAI-compatible API (Groq) với cơ chế Retry chuẩn W2.

Tuân thủ giáo trình Tuần 2 (W2) & PLAN.md:
1. Đọc khóa API từ .env (python-dotenv), không hard-code key.
2. Retry + Exponential Backoff + Jitter cho các lỗi tạm thời (429 RateLimit, 5xx, timeout,
   connection). Nếu server gửi header Retry-After thì chờ theo đó.
3. KHÔNG retry các lỗi cố định (401 Auth, 400 Bad Request, 403, 404).
4. Đo và trả về dataclass Usage(calls, prompt_tokens, completion_tokens, max_prompt)
   cho MỘT lần gọi. Dùng merge_usage() để gộp thành tổng của một câu hỏi.
5. Chỉ ghi log ra stderr: giữ stdout sạch cho giao thức MCP stdio.

Quy ước lỗi: call_llm RAISE khi lỗi cố định, hết số lần retry, hoặc phản hồi rỗng
(EmptyResponseError). Agent phải bắt các ngoại lệ này để một câu hỏi lỗi không làm
chết cả lượt chạy.
"""

import os
import random
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

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

# Nạp biến môi trường từ .env
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

PROMPTS_DIR = REPO_ROOT / "prompts"

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# Header Retry-After lớn hơn mức này thì chỉ chờ tối đa bằng mức này.
MAX_RETRY_AFTER = 60.0

# Giới hạn thời gian chờ một lần gọi (giây). Đặt cả ở client lẫn từng request để không bao giờ treo.
DEFAULT_TIMEOUT = 60.0

_CLIENT: Optional[OpenAI] = None


class Message(TypedDict):
    """Một tin nhắn gửi cho LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class EmptyResponseError(RuntimeError):
    """LLM trả về phản hồi không dùng được (không có choices, hoặc nội dung rỗng)."""


class MalformedToolCallError(RuntimeError):
    """Model sinh lời gọi tool sai định dạng và server từ chối (Groq: 400 "tool_use_failed").

    Khác lỗi 400 thông thường: thử lại với một lời nhắc thường sửa được, nên agent bắt riêng.
    """


def _log(msg: str) -> None:
    """Ghi log ra stderr để bảo toàn stdout cho MCP stdio."""
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def _describe(e: Exception) -> str:
    """Tên lỗi kèm mã HTTP nếu có, để log lỗi HTTP rõ ràng."""
    status = getattr(e, "status_code", None)
    return f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__


def get_client() -> OpenAI:
    """Khởi tạo và tái sử dụng OpenAI client (singleton).

    max_retries=0: SDK mặc định tự retry 2 lần bên trong. Tắt nó để call_llm / call_llm_tools là
    nơi DUY NHẤT quyết định retry, nếu không số request thật bị nhân lên.
    timeout: SDK mặc định chờ 10 phút; hạ về DEFAULT_TIMEOUT để một lần gọi treo không làm treo cả lượt.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError(
                "Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env."
            )
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=DEFAULT_TIMEOUT)
    return _CLIENT


def load_prompt(filename: str) -> str:
    """Đọc file prompt từ thư mục prompts/.

    Args:
        filename: Tên file (vd: "agent_system.txt").
    Returns:
        Nội dung chuỗi của file prompt.
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(filename: str, **values: str) -> str:
    """Đọc prompt rồi điền các chỗ trống ${tên} bằng string.Template (W2).

    Chỗ trống không có giá trị được giữ nguyên (safe_substitute), không báo lỗi.
    Dùng để đưa danh sách tool lấy từ list_tools() vào prompt thay vì viết cứng.
    """
    return string.Template(load_prompt(filename)).safe_substitute(values)


def merge_usage(*usages: Usage) -> Usage:
    """Gộp Usage của nhiều lần gọi thành tổng cho một câu hỏi.

    calls, prompt_tokens, completion_tokens được CỘNG; max_prompt lấy GIÁ TRỊ LỚN NHẤT.
    Luôn trả đối tượng mới, không sửa các Usage đầu vào.
    """
    return Usage(
        calls=sum(u.calls for u in usages),
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        max_prompt=max((u.max_prompt for u in usages), default=0),
    )


def _retry_after_seconds(e: Exception) -> Optional[float]:
    """Đọc header Retry-After (giây) từ phản hồi lỗi, hoặc None nếu không có/không hợp lệ."""
    headers = getattr(getattr(e, "response", None), "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _retry_wait(e: Exception, delay: float, max_delay: float) -> float:
    """Thời gian chờ trước lần thử tiếp: ưu tiên Retry-After, không có thì dùng backoff."""
    retry_after = _retry_after_seconds(e)
    base = min(retry_after, MAX_RETRY_AFTER) if retry_after is not None else min(max_delay, delay)
    return base + random.uniform(0.1, 0.5)


def _parse_response(response) -> tuple[str, Usage]:
    """Rút nội dung và Usage từ phản hồi. Raise EmptyResponseError nếu không dùng được."""
    if not getattr(response, "choices", None):
        raise EmptyResponseError("LLM trả về phản hồi không có choices.")

    choice = response.choices[0]
    content = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None)

    if not content.strip():
        hint = " (bị cắt do max_tokens, hãy tăng max_tokens)" if finish == "length" else ""
        raise EmptyResponseError(f"LLM trả về nội dung rỗng, finish_reason={finish!r}{hint}.")
    if finish == "length":
        _log("Cảnh báo: phản hồi bị cắt do max_tokens (finish_reason='length').")

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    usage = Usage(
        calls=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        max_prompt=prompt_tokens,
    )
    return content, usage


def _create_with_retry(
    request: dict[str, Any],
    max_retries: int,
    initial_delay: float,
    max_delay: float,
):
    """Gọi chat.completions.create với Retry + Exponential Backoff + Jitter; trả phản hồi thô.

    Nơi DUY NHẤT trong repo quyết định lỗi nào retry (call_llm và call_llm_tools cùng dùng).
    """
    client = get_client()
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request)

        # --- CÁC LỖI CỐ ĐỊNH: KHÔNG ĐƯỢC RETRY (NÉM LỖI NGAY) ---
        except (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError) as e:
            _log(f"Lỗi cố định {_describe(e)}: {e}. Dừng ngay, không retry.")
            raise

        # --- CÁC LỖI TẠM THỜI: RETRY VỚI EXPONENTIAL BACKOFF + JITTER ---
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            if attempt == max_retries:
                _log(f"Đã thử lại {max_retries} lần nhưng vẫn thất bại {_describe(e)}: {e}")
                raise RuntimeError(
                    f"Gọi LLM thất bại sau {max_retries} lần thử lại do lỗi {_describe(e)}: {e}"
                ) from e

            wait = _retry_wait(e, delay, max_delay)
            _log(
                f"Lỗi tạm thời {_describe(e)} ở lần thử {attempt + 1}/{max_retries}. "
                f"Chờ {wait:.2f}s trước khi thử lại..."
            )
            time.sleep(wait)
            delay = min(max_delay, delay * 2.0)

        except Exception as e:
            _log(f"Lỗi không xác định {_describe(e)}: {e}")
            raise

    # Không thể tới đây (vòng lặp luôn return hoặc raise), giữ để kiểu trả về nhất quán.
    raise RuntimeError("Gọi LLM thất bại: vòng lặp retry kết thúc bất thường.")


def call_llm(
    messages: list[Message],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
    stop: Optional[list[str]] = None,
) -> tuple[str, Usage]:
    """Gọi LLM an toàn có bọc Retry + Exponential Backoff + Jitter và trả về Usage.

    Args:
        messages: Danh sách Message {"role": "system"|"user"|"assistant", "content": "..."}.
        model: Tên mô hình (nếu None sẽ lấy LLM_MODEL từ .env).
        temperature: Độ ngẫu nhiên (mặc định 0.0 để suy luận xác định cho thẩm định).
        max_tokens: Số token tối đa cho câu trả lời.
        timeout: Giới hạn thời gian chờ phản hồi (giây).
        max_retries: Số lần thử lại tối đa cho các lỗi tạm thời.
        initial_delay: Thời gian chờ ban đầu (giây).
        max_delay: Thời gian chờ tối đa giữa các lần thử lại (giây).
        stop: Chuỗi dừng, ví dụ ["Observation:"] để mô hình không tự bịa Observation.

    Returns:
        tuple[str, Usage]: (nội_dung_phản_hồi, Usage của lần gọi này). Usage.calls = 1;
        các lần thử lại thất bại không được đếm vì không tốn token.

    Raises:
        AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError:
            Lỗi cố định (không retry).
        RuntimeError: Hết số lần retry cho lỗi tạm thời.
        EmptyResponseError: Phản hồi rỗng hoặc không có choices.
    """
    request: dict[str, Any] = {
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if stop:
        request["stop"] = stop

    response = _create_with_retry(request, max_retries, initial_delay, max_delay)
    # Phân tích ngoài vòng retry để EmptyResponseError không bị coi là lỗi tạm thời.
    return _parse_response(response)


# ---------------------------------------------------------------------------
# GỌI LLM VỚI TOOL (native function calling): dùng bởi vòng lặp agent
# ---------------------------------------------------------------------------
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


def _is_tool_use_failed(e: BadRequestError) -> bool:
    return getattr(e, "code", None) == "tool_use_failed" or "tool_use_failed" in str(e)


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
    """Như call_llm nhưng cho phép model gọi tool; trả ChatTurn (văn bản và/hoặc tool_calls).

    Dùng chung bộ retry với call_llm. Khác call_llm ở chỗ:
    - messages có thể chứa role "tool" và assistant kèm tool_calls (dict thuần theo định dạng OpenAI).
    - Nội dung rỗng KHÔNG phải lỗi khi có tool_calls; agent tự quyết định khi cả hai đều rỗng.

    Raises:
        MalformedToolCallError: server từ chối vì lời gọi tool sai định dạng (nên nhắc model thử lại).
        Các lỗi khác giống call_llm.
    """
    request: dict[str, Any] = {
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    try:
        response = _create_with_retry(request, max_retries, initial_delay, max_delay)
    except BadRequestError as e:
        if _is_tool_use_failed(e):
            raise MalformedToolCallError(f"Model gọi tool sai định dạng: {e}") from e
        raise

    if not getattr(response, "choices", None):
        raise EmptyResponseError("LLM trả về phản hồi không có choices.")

    choice = response.choices[0]
    message = choice.message
    tool_calls = [
        ToolCall(
            id=getattr(tc, "id", "") or f"call_{i}",
            name=tc.function.name,
            arguments=tc.function.arguments or "",
        )
        for i, tc in enumerate(getattr(message, "tool_calls", None) or [])
    ]

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    finish = getattr(choice, "finish_reason", None)
    if finish == "length":
        _log("Cảnh báo: phản hồi bị cắt do max_tokens (finish_reason='length').")

    return ChatTurn(
        content=(message.content or "").strip(),
        tool_calls=tool_calls,
        usage=Usage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_prompt=prompt_tokens,
        ),
        finish_reason=finish,
    )
