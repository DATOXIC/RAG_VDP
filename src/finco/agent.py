from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from eval.verifier import AgentSubmission, submission_from_trace
from finco.llm import DEFAULT_MODEL, ChatTurn, call_llm_tools, load_prompt, merge_usage
from finco.mcp_client import Client, get_default_server_params, to_openai_tools
from finco.types import LogLine, Usage

# --------------------------------------------------------------------------- 1. Hằng số
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"

MAX_STEPS = 12              # số lần gọi LLM tối đa mỗi lượt (quy trình dài nhất cần khoảng 9)
WRITE_TOOL = "submit_decision"
VERIFY_TOOL = "get_decision"
ANSWER_TOOL = "__answer__"  # dòng log cuối mỗi câu: câu trả lời + usage (kể cả câu không gọi tool)

LlmFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], str], ChatTurn]


# --------------------------------------------------------------------------- 2. Kết quả một lượt
@dataclass
class AgentRunResult:
    """Kết quả một lượt: câu trả lời cuối + trace (mỗi lời gọi tool một LogLine) + mức dùng LLM."""
    app_id: str
    run_id: str = ""
    question_id: str = ""
    question: str = ""
    final_answer: str = ""
    success: bool = True
    error_message: Optional[str] = None
    trace: list[LogLine] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    answer_log: Optional[LogLine] = None

    @property
    def tool_calls(self) -> list[str]:
        return [line.tool for line in self.trace]

    def to_submission(self) -> AgentSubmission:
        """Bài nộp cho verifier, dựng từ trace."""
        return submission_from_trace(self.app_id, self.trace)

    def save_log(self, filepath: Optional[str | Path] = None) -> Path:
        """Ghi trace + dòng __answer__ ra .jsonl. Mặc định logs/run_<run_id>.jsonl; truyền cùng
        một filepath cho nhiều lượt để gộp vào một file."""
        path = Path(filepath) if filepath else LOGS_DIR / f"run_{self.run_id}.jsonl"
        return save_trace_to_jsonl(self.trace + ([self.answer_log] if self.answer_log else []), path)


def save_trace_to_jsonl(trace: list[LogLine], log_file: Path | str) -> Path:
    """Nối các dòng log vào file .jsonl (tạo thư mục nếu chưa có)."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in trace:
            f.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")
    return path


