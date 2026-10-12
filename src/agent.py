"""Kết quả một lượt chạy của agent.

Nguyên tắc: LLM chỉ trả văn bản, nên KHÔNG tin vào đối tượng mà agent "kể lại". Nguồn sự thật duy
nhất của một lượt chạy là `trace`: mỗi lời gọi tool một `LogLine` (tên tool, args đã parse, kết quả
chuỗi). Bài nộp cho verifier được dựng từ trace bằng `submission_from_trace`, không gán tay.
"""
import json

from dataclasses import dataclass, field
from typing import Optional
from finco.llm import get_client, load_prompt, DEFAULT_MODEL
from finco.types import LogLine, Usage
from eval.verifier import AgentSubmission, submission_from_trace


@dataclass
class AgentStep:
    """Một bước ReAct để hiển thị/gỡ lỗi (Thought chỉ có ở đây, LogLine không lưu)."""
    step_number: int
    thought: str                     # Suy nghĩ của LLM
    tool_name: Optional[str] = None  # Tool được gọi (nếu có)
    tool_args: dict = field(default_factory=dict)
    observation: str = ""            # Kết quả tool trả về


@dataclass
class AgentRunResult:
    """Hộp đen của một lượt chạy: câu trả lời cuối + trace + mức dùng LLM."""
    app_id: str
    final_answer: str                                    # Văn bản Answer cuối (chỉ để hiển thị)
    trace: list[LogLine] = field(default_factory=list)   # Mỗi lời gọi tool một dòng (ghi cả lời gọi lỗi)
    steps: list[AgentStep] = field(default_factory=list)
    usage: Optional[Usage] = None
    success: bool = True                                 # False nếu hết MAX_STEPS hoặc LLM lỗi
    error_message: Optional[str] = None

    @property
    def tool_calls(self) -> list[str]:
        """Tên các tool đã gọi theo thứ tự (kể cả lời gọi lỗi), để thống kê."""
        return [line.tool for line in self.trace]

    def to_submission(self) -> AgentSubmission:
        """Bài nộp cho verifier, dựng từ trace: quyết định, mã lý do, trích dẫn, DTI, lãi suất."""
        return submission_from_trace(self.app_id, self.trace)

class Agent:

    def __init__(self, model: Optional[str] = None, max_steps: int = 8, verbose: bool = True):
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = load_prompt("agent_system.txt")
        self.client_llm = get_client()
        
    def _safe_parse_args(self, raw_args: str) -> tuple[dict, Optional[str]]:
        """Phòng vệ W6: Bắt lỗi cú pháp JSON, thiếu ngoặc không làm chết run."""
        try:
            return json.loads(raw_args or "{}"), None
        except json.JSONDecodeError as e:
            # Thử tự động vá lỗi thiếu dấu ngoặc nhọn
            cleaned = (raw_args or "").strip()
            if not cleaned.endswith("}"):
                try:
                    return json.loads(cleaned + "}"), None
                except Exception:
                    pass
            return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})"