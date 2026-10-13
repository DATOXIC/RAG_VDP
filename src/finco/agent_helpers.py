"""Các kiểu dữ liệu và hàm phụ trợ cho Agent thẩm định FinCo.

Bao gồm:
- Hằng số và Regex cấu hình vòng lặp
- Dataclass: AgentStep (hiển thị), AgentRunResult (kết quả bao bọc)
- Hàm phụ trợ: vá JSON, bóc tách lỗi, lưu vết JSONL, kiểm tra Read-Your-Writes
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from eval.verifier import AgentSubmission, submission_from_trace
from finco.llm import ChatTurn
from finco.types import LogLine, Usage

# --------------------------------------------------------------------------- 1. Hằng số
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"

MAX_STEPS = 12     # số lần gọi LLM tối đa mỗi lượt (quy trình dài nhất cần khoảng 9)
MAX_NUDGES = 2     # số lần nhắc model sửa một lượt kết thúc chưa hợp lệ
MAX_SEARCHES = 5   # số lần tra cứu chính sách tối đa mỗi lượt (chống xoay vòng truy vấn)

SEARCH_TOOL = "search_policy"
WRITE_TOOL = "submit_decision"
VERIFY_TOOL = "get_decision"
ANSWER_TOOL = "__answer__"  # dòng log cuối mỗi câu: câu trả lời + usage (kể cả câu không gọi tool)

_APP_ID = re.compile(r"APP-\d+", re.IGNORECASE)
_TEXT_ACTION = re.compile(r"^\s*(Action|Action Input)\s*:", re.IGNORECASE | re.MULTILINE)

LlmFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], str], ChatTurn]


# --------------------------------------------------------------------------- 2. Kết quả một lượt
@dataclass
class AgentStep:
    """Một bước để người đọc (Thought chỉ có ở đây, LogLine không lưu)."""
    step_number: int
    thought: str
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    observation: str = ""
    is_error: bool = False


@dataclass
class AgentRunResult:
    """Hộp đen của một lượt: câu trả lời cuối + trace + mức dùng LLM."""
    app_id: str
    run_id: str = ""
    question_id: str = ""
    question: str = ""
    final_answer: str = ""
    success: bool = True
    error_message: Optional[str] = None
    trace: list[LogLine] = field(default_factory=list)     # mỗi lời gọi tool một dòng
    steps: list[AgentStep] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    answer_log: Optional[LogLine] = None

    @property
    def tool_calls(self) -> list[str]:
        return [line.tool for line in self.trace]

    @property
    def decision_written(self) -> bool:
        return decision_status(self.trace, self.app_id)[0]

    @property
    def decision_verified(self) -> bool:
        return decision_status(self.trace, self.app_id)[1]

    def to_submission(self) -> AgentSubmission:
        """Bài nộp cho verifier, dựng từ trace."""
        return submission_from_trace(self.app_id, self.trace)

    def save_log(self, filepath: Optional[str | Path] = None) -> Path:
        """Ghi trace + dòng __answer__ ra .jsonl. Mặc định logs/run_<run_id>.jsonl; truyền cùng
        một filepath cho nhiều lượt để gộp vào một file."""
        path = Path(filepath) if filepath else LOGS_DIR / f"run_{self.run_id}.jsonl"
        return save_trace_to_jsonl(self.trace + ([self.answer_log] if self.answer_log else []), path)


# --------------------------------------------------------------------------- 3. Hàm phụ nhỏ
def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def save_trace_to_jsonl(trace: list[LogLine], log_file: Path | str) -> Path:
    """Nối các dòng log vào file .jsonl (tạo thư mục nếu chưa có)."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in trace:
            f.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")
    return path


def _is_error(text: str) -> bool:
    return text.lstrip().startswith("Error:")


def decision_status(trace: list[LogLine], app_id: str) -> tuple[bool, bool, bool]:
    """(đã ghi, đã đọc lại sau khi ghi, hồ sơ không tồn tại), chỉ tính lời gọi THÀNH CÔNG đúng hồ sơ."""
    target = app_id.strip().upper()
    written = verified = looked_up = lookup_failed = False

    for line in trace:
        if str(line.args.get("app_id", "")).strip().upper() != target:
            continue

        ok = not _is_error(line.result)

        if line.tool == "get_application":
            looked_up, lookup_failed = looked_up or ok, lookup_failed or not ok

        elif line.tool == WRITE_TOOL and ok:
            written, verified = True, False  # ghi lại thì phải đọc lại lần nữa

        elif line.tool == VERIFY_TOOL and ok and written:
            verified = True

    return written, verified, lookup_failed and not looked_up


def _safe_parse_args(raw: str) -> tuple[dict, Optional[str]]:
    """Parse JSON tham số tool, không bao giờ raise: trả (dict, None) hoặc ({}, thông báo lỗi)."""
    raw = (raw or "").strip() or "{}"
    
    value: Any = None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        for patched in (raw + "}", raw + '"}'):  # model nhỏ hay thiếu ngoặc/nháy đóng: vá một lần
            try:
                value = json.loads(patched)
                break
            except json.JSONDecodeError:
                continue
        else:
            return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})."
    if not isinstance(value, dict):
        return {}, ('Error: Tham số gọi tool phải là một đối tượng JSON dạng {"tên": giá_trị}, '
                    f"nhận được {type(value).__name__}.")
    return value, None


def _assistant_message(turn: ChatTurn) -> dict[str, Any]:
    """Tin nhắn assistant đưa lại vào lịch sử; tham số luôn là JSON hợp lệ (rác làm server từ chối)."""
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name,
                          "arguments": json.dumps(_safe_parse_args(tc.arguments)[0], ensure_ascii=False)}}
            for tc in turn.tool_calls
        ]
    return msg


def _result_text(result: Any) -> str:
    """Văn bản từ kết quả call_tool của MCP; lỗi phía server luôn bắt đầu bằng "Error:"."""

    lines = []

    for c in (result.content or []):
        text_val = getattr(c, "text", None)

        if isinstance(text_val, str):
            lines.append(text_val)

    text = "\n".join(lines).strip()

    if getattr(result, "is_error", False) and not _is_error(text):
        text = f"Error: {text or 'công cụ báo lỗi không kèm nội dung.'}"

    return text or "(Công cụ không trả về nội dung: không có kết quả phù hợp.)"


def _repeat_search_error(args: dict, prior_queries: list[str]) -> str:
    """"Error: ..." nếu truy vấn đã chạy hoặc đã hết MAX_SEARCHES, ngược lại chuỗi rỗng."""

    query = " ".join(str(args.get("query", "")).lower().split())
    
    if query in prior_queries:
        return ("Error: Truy vấn này đã được tra cứu ở bước trước và kết quả không đổi. Hãy dựa vào các kết quả "
                "đã có để trả lời; nếu chúng không chứa thông tin cần thiết thì trả lời bằng câu từ chối.")
    
    if len(prior_queries) >= MAX_SEARCHES:
        return (f"Error: Đã tra cứu {len(prior_queries)} lần, đủ giới hạn. Không tra cứu thêm: hãy trả lời "
                "từ các kết quả đã có, hoặc từ chối nếu không có thông tin.")
    return ""


def prior_searches(result: AgentRunResult) -> list[str]:
    """Các truy vấn search_policy đã chạy thành công trong lượt này (đã chuẩn hóa khoảng trắng/hoa thường)."""
    return [" ".join(str(s.tool_args.get("query", "")).lower().split())
            for s in result.steps if s.tool_name == SEARCH_TOOL and not s.is_error]


async def execute_agent_tool(
    mcp: Any,
    tc: Any,
    step: int,
    tool_names: set[str],
    result: AgentRunResult,
    verbose: bool = True,
) -> dict[str, Any]:
    """Parse tham số, gọi tool qua MCP, ghi AgentStep + LogLine; trả tin nhắn role 'tool'."""
    thought = ""
    args, output = _safe_parse_args(tc.arguments)
    if output is None:
        if tc.name not in tool_names:
            output = f"Error: Công cụ '{tc.name}' không tồn tại. Công cụ hợp lệ: {', '.join(sorted(tool_names))}."
        else:
            thought = str(args.pop("thought", "") or "")
            if verbose:
                print(f"\n[Bước {step}]\nThought: {thought or '(không có)'}\nAction : {tc.name}({args})")
            output = (_repeat_search_error(args, prior_searches(result)) if tc.name == SEARCH_TOOL else "")
            if not output:
                try:
                    output = _result_text(await mcp.call_tool(tc.name, args))
                except Exception as e:
                    output = f"Error: Gọi công cụ '{tc.name}' thất bại ({type(e).__name__}: {e})"
    if verbose:
        print(f"Observation: {output[:160]}{'...' if len(output) > 160 else ''}")

    result.steps.append(AgentStep(step, thought or "(không có)", tc.name, args, output, _is_error(output)))
    result.trace.append(LogLine(
        ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id,
        question_id=result.question_id, step=step, tool=tc.name, args=args, result=output,
        usage=result.usage,
    ))
    return {"role": "tool", "tool_call_id": tc.id, "content": output}


def check_completion_issue(turn: ChatTurn, require_decision: bool, result: AgentRunResult) -> Optional[str]:
    """Lượt kết thúc (không còn tool_calls) có hợp lệ không? Trả về thông báo nhắc nếu chưa, None nếu ổn."""
    if _TEXT_ACTION.search(turn.content):
        return ("Bạn vừa viết lời gọi công cụ dưới dạng văn bản, hệ thống KHÔNG thực thi văn bản. "
                "Hãy gọi công cụ bằng function calling.")
    if require_decision:
        written, verified, lookup_failed = decision_status(result.trace, result.app_id)

        if lookup_failed:
            return "LOOKUP_FAILED"
        
        if not written:
            return (f"Chưa hoàn tất: hồ sơ {result.app_id} chưa có quyết định được ghi. "
                    f"Hãy làm tiếp quy trình, gọi {WRITE_TOOL} rồi {VERIFY_TOOL}, sau đó mới trả lời.")
        
        if not verified:
            return (f"Chưa hoàn tất: bạn đã ghi quyết định nhưng chưa kiểm chứng. "
                    f"Hãy gọi {VERIFY_TOOL}('{result.app_id}') và đối chiếu kết quả trước khi trả lời.")
    
    if not turn.content:
        return "Phản hồi rỗng. Hãy gọi công cụ cần thiết hoặc đưa ra câu trả lời."
    return None


def create_answer_log(result: AgentRunResult) -> LogLine:
    """Tạo dòng log __answer__ cuối mỗi lượt."""
    return LogLine(
        ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id,
        question_id=result.question_id, step=result.steps[-1].step_number if result.steps else 0,
        tool=ANSWER_TOOL, args={"success": result.success, "error": result.error_message},
        result=result.final_answer or result.error_message or "", usage=result.usage,
    )
