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
from pathlib import Path
from typing import Literal, Optional, TypedDict

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

_CLIENT: Optional[OpenAI] = None


class Message(TypedDict):
    """Một tin nhắn gửi cho LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class EmptyResponseError(RuntimeError):
    """LLM trả về phản hồi không dùng được (không có choices, hoặc nội dung rỗng)."""


def _log(msg: str) -> None:
    """Ghi log ra stderr để bảo toàn stdout cho MCP stdio."""
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def _describe(e: Exception) -> str:
    """Tên lỗi kèm mã HTTP nếu có, để log lỗi HTTP rõ ràng."""
    status = getattr(e, "status_code", None)
    return f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__


def get_client() -> OpenAI:
    """Khởi tạo và tái sử dụng OpenAI client (singleton).

    max_retries=0: SDK mặc định tự retry 2 lần bên trong. Tắt nó để call_llm là nơi
    DUY NHẤT quyết định retry, nếu không số request thật bị nhân lên.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError(
                "Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env."
            )
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
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


def call_llm(
    messages: list[Message],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    timeout: float = 60.0,
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
    client = get_client()
    target_model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    extra = {"stop": stop} if stop else {}

    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **extra,
            )

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
            continue

        except Exception as e:
            _log(f"Lỗi không xác định {_describe(e)}: {e}")
            raise

        # Phân tích ngoài khối try để EmptyResponseError không bị coi là lỗi tạm thời.
        return _parse_response(response)

    # Không thể tới đây (vòng lặp luôn return hoặc raise), giữ để kiểu trả về nhất quán.
    raise RuntimeError("Gọi LLM thất bại: vòng lặp retry kết thúc bất thường.")
