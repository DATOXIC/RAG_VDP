"""Client gọi mô hình LLM qua OpenAI-compatible API (Groq) với cơ chế Retry chuẩn W2.

Tuân thủ nghiêm ngặt giáo trình Tuần 2 (W2) & PLAN.md:
1. Đọc khóa API từ .env (python-dotenv), không hard-code key.
2. Retry + Exponential Backoff + Jitter cho các lỗi tạm thời (429 RateLimit, 5xx, timeout, connection).
3. KHÔNG retry các lỗi cố định (401 Auth, 400 Bad Request, 404 Not Found).
4. Đo và trả về dataclass Usage(calls, prompt_tokens, completion_tokens, max_prompt).
5. Chỉ ghi log ra stderr: giữ stdout sạch cho giao thức MCP stdio.
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

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

# Cấu hình mặc định từ .env
DEFAULT_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

_CLIENT: Optional[OpenAI] = None


def _log(msg: str) -> None:
    """Ghi log ra stderr để bảo toàn stdout cho MCP stdio."""
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def get_client() -> OpenAI:
    """Khởi tạo và tái sử dụng OpenAI client (singleton)."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError(
                "Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env."
            )
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
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


def call_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    timeout: float = 60.0,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
) -> tuple[str, Usage]:
    """Gọi LLM an toàn có bọc Retry + Exponential Backoff + Jitter và trả về Usage.

    Args:
        messages: Danh sách tin nhắn dạng [{"role": "system"|"user"|"assistant", "content": "..."}].
        model: Tên mô hình (nếu None sẽ lấy LLM_MODEL từ .env).
        temperature: Độ ngẫu nhiên (mặc định 0.0 để suy luận xác định cho thẩm định).
        max_tokens: Số token tối đa cho câu trả lời.
        timeout: Giới hạn thời gian chờ phản hồi (giây).
        max_retries: Số lần thử lại tối đa cho các lỗi tạm thời.
        initial_delay: Thời gian chờ ban đầu (giây).
        max_delay: Thời gian chờ tối đa giữa các lần thử lại (giây).

    Returns:
        tuple[str, Usage]: (nội_dung_phản_hồi, đối_tượng_Usage_đo_lường_token).

    Raises:
        AuthenticationError: Sai API key (không retry).
        BadRequestError: Lỗi tham số/context (không retry).
        RuntimeError: Khi vượt quá số lần retry cho phép.
    """
    client = get_client()
    target_model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL

    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            content = response.choices[0].message.content or ""

            # Đo lường token usage
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            usage = Usage(
                calls=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                max_prompt=prompt_tokens,
            )

            return content, usage

        # --- CÁC LỖI CỐ ĐỊNH: KHÔNG ĐƯỢC RETRY (NÉM LỖI NGAY) ---
        except (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError) as e:
            _log(f"Lỗi cố định ({type(e).__name__}): {e}. Dừng ngay không retry.")
            raise

        # --- CÁC LỖI TẠM THỜI: RETRY VỚI EXPONENTIAL BACKOFF + JITTER ---
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            if attempt == max_retries:
                _log(f"Đã thử lại {max_retries} lần nhưng vẫn thất bại ({type(e).__name__}): {e}")
                raise RuntimeError(
                    f"Gọi LLM thất bại sau {max_retries} lần thử lại do lỗi {type(e).__name__}: {e}"
                ) from e

            # Tính toán thời gian ngủ: Exponential Backoff + Full Jitter ngẫu nhiên
            jitter = random.uniform(0.1, 0.5)
            sleep_time = min(max_delay, delay) + jitter
            _log(
                f"Lỗi tạm thời ({type(e).__name__}) ở lần thử {attempt + 1}/{max_retries}. "
                f"Đang chờ {sleep_time:.2f}s trước khi thử lại..."
            )
            time.sleep(sleep_time)
            delay = min(max_delay, delay * 2.0)

        except Exception as e:
            _log(f"Lỗi không xác định ({type(e).__name__}): {e}")
            raise
