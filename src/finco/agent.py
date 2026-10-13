"""Agent thẩm định FinCo: vòng lặp tự viết, gọi tool qua MCP bằng native function calling.

Vòng lặp (hàm Agent._loop), đọc từ trên xuống là hiểu cả file:

    gọi LLM
      |-- có tool_calls -> chạy từng tool qua MCP, đưa kết quả lại cho LLM, lặp
      |-- không còn tool_calls -> kiểm tra đã làm đủ chưa
              |-- đủ -> kết thúc
              |-- thiếu (chưa ghi/đọc lại quyết định, rỗng...) -> nhắc tối đa MAX_NUDGES lần

Quy ước:
- run() không bao giờ raise: lỗi LLM, tool, MCP đều dừng gọn, trace vẫn được giữ.
- Tool lỗi trả chuỗi "Error: ..." cho model tự sửa, không làm chết lượt chạy.
- Chưa submit_decision rồi get_decision thành công thì chưa tính là xong (đề mục 2.7).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from eval.verifier import AgentSubmission, submission_from_trace
from finco.llm import (
    DEFAULT_MODEL,
    ChatTurn,
    MalformedToolCallError,
    ToolCall,
    call_llm_tools,
    load_prompt,
    merge_usage,
)
from finco.mcp_client import Client, get_default_server_params, to_openai_tools
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
    text = "\n".join(c.text for c in (result.content or []) if isinstance(getattr(c, "text", None), str)).strip()
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


# --------------------------------------------------------------------------- 4. Agent
class Agent:
    """Agent thẩm định hồ sơ vay / hỏi đáp chính sách, dùng các tool của FinCo MCP server."""

    def __init__(self, model: Optional[str] = None, max_steps: int = MAX_STEPS,
                 verbose: bool = True, llm: Optional[LlmFn] = None):
        """llm: hàm (messages, tools, model) -> ChatTurn. Mặc định gọi LLM thật; test truyền hàm giả."""
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = load_prompt("agent_system.txt")
        self._llm: LlmFn = llm or (lambda messages, tools, model: call_llm_tools(messages, tools, model=model))

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

    # ---- điểm vào
    async def run(self, app_id: str, question: Optional[str] = None,
                  question_id: Optional[str] = None, require_decision: Optional[bool] = None) -> AgentRunResult:
        """Chạy một lượt, không raise.

        app_id: mã hồ sơ ("APP-014") hoặc nhãn tùy ý cho câu hỏi đáp ("Q").
        question: None = yêu cầu thẩm định và ghi quyết định hồ sơ app_id.
        require_decision: bắt buộc ghi + đọc lại; None = tự suy (True khi không có question và app_id là APP-xxx).
        """
        app_id = app_id.strip()
        query = question or (f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo "
                             "và ghi nhận quyết định vào hệ thống.")
        if require_decision is None:
            require_decision = question is None and bool(_APP_ID.fullmatch(app_id))

        result = AgentRunResult(app_id=app_id, run_id=new_run_id(),
                                question_id=question_id or app_id, question=query)
        try:
            async with Client(get_default_server_params()) as mcp:
                tools = to_openai_tools((await mcp.list_tools()).tools)
                await self._loop(mcp, tools, query, require_decision, result)
        except Exception as e:  # không khởi động được MCP, mất kết nối...
            self._fail(result, f"{type(e).__name__}: {e}")

        result.answer_log = LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id,
            question_id=result.question_id, step=result.steps[-1].step_number if result.steps else 0,
            tool=ANSWER_TOOL, args={"success": result.success, "error": result.error_message},
            result=result.final_answer or result.error_message or "", usage=replace(result.usage),
        )
        return result

    def run_sync(self, *args, **kwargs) -> AgentRunResult:
        """Bản đồng bộ của run(), tiện cho script và REPL."""
        return asyncio.run(self.run(*args, **kwargs))

    # ---- vòng lặp
    async def _loop(self, mcp: Client, tools: list[dict], query: str,
                    require_decision: bool, result: AgentRunResult) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        tool_names = {t["function"]["name"] for t in tools}
        nudges = 0

        for step in range(1, self.max_steps + 1):
            turn: Optional[ChatTurn] = None
            try:
                # Luồng riêng: retry/backoff của llm.py dùng time.sleep, không được chặn event loop MCP.
                turn = await asyncio.to_thread(self._llm, messages, tools, self.model)
                problem = None
            except MalformedToolCallError as e:
                problem = f"Lời gọi công cụ trước không hợp lệ (JSON sai), hãy gọi lại với tham số đúng cú pháp. ({e})"
            except Exception as e:  # hết retry, lỗi xác thực, phản hồi rỗng...
                self._fail(result, f"Lỗi gọi LLM: {type(e).__name__}: {e}")
                return

            if turn is not None:
                result.usage = merge_usage(result.usage, turn.usage)
                if turn.tool_calls:  # model muốn gọi tool: chạy hết rồi lặp
                    messages.append(_assistant_message(turn))
                    for tc in turn.tool_calls:
                        messages.append(await self._run_tool(mcp, tc, step, tool_names, result))
                    continue
                if require_decision and decision_status(result.trace, result.app_id)[2]:
                    result.final_answer = turn.content
                    self._fail(result, f"Không đọc được hồ sơ {result.app_id}, chưa ghi quyết định.")
                    return
                problem = self._completion_problem(turn, require_decision, result)
                if problem is None:
                    result.final_answer = turn.content
                    self._say(f"\n[AI Answer]:\n{turn.content}\n")
                    return

            # Có vấn đề: nhắc model, hết lượt nhắc thì thất bại.
            if nudges >= MAX_NUDGES:
                result.final_answer = turn.content if turn else ""
                self._fail(result, problem)
                return
            nudges += 1
            self._say(f"[Nhắc lại] {problem}")
            if turn is not None and turn.content:
                messages.append(_assistant_message(turn))
            messages.append({"role": "user", "content": problem})

        self._fail(result, f"Vượt quá MAX_STEPS ({self.max_steps}) mà chưa có câu trả lời cuối.")

    def _fail(self, result: AgentRunResult, message: str) -> None:
        result.success, result.error_message = False, message
        self._say(f"\n[Lỗi] {message}")

    def _completion_problem(self, turn: ChatTurn, require_decision: bool, result: AgentRunResult) -> Optional[str]:
        """Lượt kết thúc (không còn tool_calls) có hợp lệ không? Trả lời nhắc nếu chưa, None nếu ổn."""
        if _TEXT_ACTION.search(turn.content):
            return ("Bạn vừa viết lời gọi công cụ dưới dạng văn bản, hệ thống KHÔNG thực thi văn bản. "
                    "Hãy gọi công cụ bằng function calling.")
        if require_decision:
            written, verified, _ = decision_status(result.trace, result.app_id)
            if not written:
                return (f"Chưa hoàn tất: hồ sơ {result.app_id} chưa có quyết định được ghi. "
                        f"Hãy làm tiếp quy trình, gọi {WRITE_TOOL} rồi {VERIFY_TOOL}, sau đó mới trả lời.")
            if not verified:
                return (f"Chưa hoàn tất: bạn đã ghi quyết định nhưng chưa kiểm chứng. "
                        f"Hãy gọi {VERIFY_TOOL}('{result.app_id}') và đối chiếu kết quả trước khi trả lời.")
        if not turn.content:
            return "Phản hồi rỗng. Hãy gọi công cụ cần thiết hoặc đưa ra câu trả lời."
        return None

    # ---- chạy một tool
    async def _run_tool(self, mcp: Client, tc: ToolCall, step: int, tool_names: set[str],
                        result: AgentRunResult) -> dict[str, Any]:
        """Parse tham số, gọi tool qua MCP, ghi AgentStep + LogLine; trả tin nhắn role "tool". Không raise."""
        thought = ""
        args, output = _safe_parse_args(tc.arguments)
        if output is None:
            if tc.name not in tool_names:
                output = f"Error: Công cụ '{tc.name}' không tồn tại. Công cụ hợp lệ: {', '.join(sorted(tool_names))}."
            else:
                thought = str(args.pop("thought", "") or "")
                self._say(f"\n[Bước {step}]\nThought: {thought or '(không có)'}\nAction : {tc.name}({args})")
                output = (_repeat_search_error(args, self._prior_searches(result)) if tc.name == SEARCH_TOOL else "")
                if not output:
                    try:
                        output = _result_text(await mcp.call_tool(tc.name, args))
                    except Exception as e:  # tool/MCP lỗi không được làm chết cả lượt
                        output = f"Error: Gọi công cụ '{tc.name}' thất bại ({type(e).__name__}: {e})"
        self._say(f"Observation: {output[:160]}{'...' if len(output) > 160 else ''}")

        result.steps.append(AgentStep(step, thought or "(không có)", tc.name, args, output, _is_error(output)))
        result.trace.append(LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id,
            question_id=result.question_id, step=step, tool=tc.name, args=args, result=output,
            usage=replace(result.usage),  # bản chụp tại bước này
        ))
        return {"role": "tool", "tool_call_id": tc.id, "content": output}

    @staticmethod
    def _prior_searches(result: AgentRunResult) -> list[str]:
        """Các truy vấn search_policy đã chạy thành công trong lượt này (đã chuẩn hóa khoảng trắng/hoa thường)."""
        return [" ".join(str(s.tool_args.get("query", "")).lower().split())
                for s in result.steps if s.tool_name == SEARCH_TOOL and not s.is_error]
