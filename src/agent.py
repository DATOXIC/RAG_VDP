from dataclasses import dataclass, field
from typing import Optional
from finco.types import Decision, Usage
from eval.verifier import AgentSubmission

@dataclass
class AgentStep:
    """Ghi nhận chi tiết 1 bước suy nghĩ và hành động của Agent."""
    step_number: int
    thought: str                     # Suy nghĩ của LLM
    tool_name: Optional[str] = None  # Tool được gọi (nếu có)
    tool_args: dict = field(default_factory=dict)
    observation: str = ""            # Kết quả tool trả về

@dataclass
class AgentRunResult:
    """
    HỘP ĐEN LƯỢT CHẠY CỦA AGENT:
    Gói gọn 100% mọi thứ Agent đã làm trong một lần thẩm định hồ sơ!
    """
    app_id: str
    final_answer: str                         # Câu trả lời cuối cùng
    decision: Optional[Decision] = None       # Quyết định chính thức (đã lưu DB)
    steps: list[AgentStep] = field(default_factory=list) # Toàn bộ quá trình ReAct
    tool_calls: list[str] = field(default_factory=list)  # Danh sách tên các tool đã gọi
    calculated_dti: Optional[float] = None    # Con số DTI tính được
    used_rate: Optional[float] = None         # Lãi suất đã dùng
    usage: Optional[Usage] = None             # Số token và số lần gọi LLM
    success: bool = True                      # Chạy trọn vẹn hay bị lỗi/timeout
    error_message: Optional[str] = None

    def to_submission(self) -> AgentSubmission:
        """Tự động chuyển đổi thành bài thi nộp cho Verifier chỉ bằng 1 nốt nhạc!"""
        return AgentSubmission(
            app_id=self.app_id,
            decision=self.decision.decision if self.decision else "NONE",
            reason=self.decision.reason if self.decision else self.final_answer,
            cited_docs=self.decision.cited_docs if self.decision else [],
            tool_calls=self.tool_calls,
            calculated_dti=self.calculated_dti,
            used_rate=self.used_rate,
        )