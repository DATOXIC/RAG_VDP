"""Agent thẩm định FinCo: vòng lặp tự viết, gọi tool qua MCP bằng native function calling.

Thiết kế:
- Vòng lặp Orchestration viết tay (không framework): gọi LLM -> có tool_calls thì thực thi qua MCP
  rồi đưa kết quả lại -> không còn tool_calls thì kết thúc. Giới hạn MAX_STEPS.
- Danh sách tool lấy động từ MCP server (list_tools); schema đổi sang định dạng OpenAI/Groq và được
  tiêm thêm tham số 'thought' để model nói lý do trước mỗi lời gọi (phục vụ audit).
- Không bao giờ raise ra ngoài run(): lỗi tool, lỗi LLM, lỗi MCP đều dừng gọn, trace vẫn được giữ.
- Ép quy trình ghi + đọc lại (mục 2.7 của đề): chưa submit_decision rồi get_decision thành công thì
  chưa được coi là hoàn tất; nhắc model tối đa MAX_NUDGES lần rồi đánh dấu thất bại.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"

# Số lần gọi LLM tối đa cho một lượt. Quy trình dài nhất cần ~9 lần (đọc hồ sơ, 4 lần tra cứu,
# tính DTI, ghi, đọc lại, trả lời); chừa dư cho lỗi tool và lời nhắc.
MAX_STEPS = 12
# Số lần nhắc model sửa một lượt kết thúc sai (chưa ghi/kiểm chứng, lệnh viết thành văn bản...).
MAX_NUDGES = 2

# Cặp công cụ ghi + kiểm chứng (hợp đồng nghiệp vụ, mục 2.7). Schema và mô tả vẫn lấy từ list_tools();
# hằng số này chỉ để vòng lặp biết "đã ghi" và "đã đọc lại" là hai lời gọi nào.
WRITE_TOOL = "submit_decision"
VERIFY_TOOL = "get_decision"
# Tên giả của dòng log cuối mỗi câu: lưu câu trả lời + USAGE của câu đó (kể cả câu không gọi tool nào).
ANSWER_TOOL = "__answer__"

_APP_ID = re.compile(r"APP-\d+", re.IGNORECASE)
_TEXT_ACTION = re.compile(r"^\s*(Action|Action Input)\s*:", re.IGNORECASE | re.MULTILINE)

LlmFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], str], ChatTurn]


@dataclass
class AgentStep:
    """Một bước để hiển thị/gỡ lỗi cho người (Thought chỉ có ở đây, LogLine không lưu)."""
    step_number: int
    thought: str                     # Suy nghĩ của LLM
    tool_name: Optional[str] = None  # Tool được gọi (nếu có)
    tool_args: dict = field(default_factory=dict)
    observation: str = ""            # Kết quả tool trả về
    is_error: bool = False           # Lời gọi này trả lỗi ("Error: ...")


@dataclass
class AgentRunResult:
    """Hộp đen của một lượt chạy: câu trả lời cuối + trace + mức dùng LLM."""
    app_id: str
    final_answer: str                                    # Văn bản trả lời cuối (chỉ để hiển thị)
    trace: list[LogLine] = field(default_factory=list)   # Mỗi lời gọi tool một dòng (ghi cả lời gọi lỗi)
    steps: list[AgentStep] = field(default_factory=list)
    usage: Optional[Usage] = None
    success: bool = True                                 # False nếu hết MAX_STEPS, chưa ghi/kiểm chứng hoặc lỗi
    error_message: Optional[str] = None
    run_id: str = ""                                     # Duy nhất mỗi lượt chạy
    question_id: str = ""                                # Mã câu hỏi (testset) hoặc mã phiên REPL
    question: str = ""
    answer_log: Optional[LogLine] = None                 # Dòng log cuối: câu trả lời + usage của câu này

    @property
    def tool_calls(self) -> list[str]:
        """Tên các tool đã gọi theo thứ tự (kể cả lời gọi lỗi), để thống kê."""
        return [line.tool for line in self.trace]

    @property
    def decision_written(self) -> bool:
        """Đã ghi quyết định thành công cho hồ sơ này chưa."""
        return _decision_state(self.trace, self.app_id)[0]

    @property
    def decision_verified(self) -> bool:
        """Đã đọc lại thành công quyết định sau khi ghi chưa (Read-Your-Writes)."""
        return _decision_state(self.trace, self.app_id)[1]

    def save_log(self, filepath: Optional[str | Path] = None) -> Path:
        """Ghi vết chạy của lượt này ra .jsonl để kiểm toán.

        Mặc định mỗi lượt một file logs/run_<run_id>.jsonl. Truyền cùng một filepath cho nhiều
        lượt để gộp chúng vào một file (ví dụ cả một lần chạy eval).
        """
        path = Path(filepath) if filepath else LOGS_DIR / f"run_{self.run_id or new_run_id()}.jsonl"
        lines = list(self.trace) + ([self.answer_log] if self.answer_log else [])
        return save_trace_to_jsonl(lines, path)

    def to_submission(self) -> AgentSubmission:
        """Bài nộp cho verifier, dựng từ trace: quyết định, mã lý do, trích dẫn, DTI, lãi suất."""
        return submission_from_trace(self.app_id, self.trace)


def new_run_id() -> str:
    """Mã lượt chạy duy nhất, đọc được và sắp xếp được theo thời gian."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def save_trace_to_jsonl(trace: list[LogLine], log_file: Path | str) -> Path:
    """Nối các dòng log vào file .jsonl (tạo thư mục nếu chưa có). Mỗi dòng một đối tượng JSON."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in trace:
            f.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")
    return path


def _is_error(text: str) -> bool:
    return text.lstrip().startswith("Error:")


def _decision_state(trace: list[LogLine], app_id: str) -> tuple[bool, bool, bool]:
    """(đã ghi, đã đọc lại sau khi ghi, tra cứu hồ sơ thất bại) tính trên các lời gọi thành công.

    Chỉ lời gọi THÀNH CÔNG cho đúng hồ sơ mới tính: một get_decision báo lỗi không phải bằng chứng
    kiểm chứng. Đã đọc lại phải xảy ra SAU lần ghi thành công.
    """
    target = app_id.strip().upper()
    written = verified = False
    lookup_ok = lookup_failed = False
    for line in trace:
        same_app = str(line.args.get("app_id", "")).strip().upper() == target
        ok = not _is_error(line.result)
        if line.tool == "get_application" and same_app:
            lookup_ok = lookup_ok or ok
            lookup_failed = lookup_failed or not ok
        elif line.tool == WRITE_TOOL and same_app and ok:
            written, verified = True, False  # ghi lại thì phải đọc lại lần nữa
        elif line.tool == VERIFY_TOOL and same_app and ok and written:
            verified = True
    return written, verified, lookup_failed and not lookup_ok


def _safe_parse_args(raw_args: str) -> tuple[dict, Optional[str]]:
    """Parse JSON tham số tool, không bao giờ raise. Trả (dict, None) hoặc ({}, thông báo lỗi)."""
    raw = (raw_args or "").strip() or "{}"
    value: Any = None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        # Model nhỏ hay thiếu ngoặc/nháy đóng: thử vá một lần, không đoán thêm.
        for patched in (raw + "}", raw + '"}'):
            try:
                value = json.loads(patched)
                break
            except json.JSONDecodeError:
                continue
        else:
            return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})."
    if not isinstance(value, dict):
        return {}, (
            "Error: Tham số gọi tool phải là một đối tượng JSON dạng {\"tên\": giá_trị}, "
            f"nhận được {type(value).__name__}."
        )
    return value, None


def _assistant_message(turn: ChatTurn) -> dict[str, Any]:
    """Dựng tin nhắn assistant sạch (dict thuần) để đưa lại vào lịch sử hội thoại."""
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    # Đưa lại JSON hợp lệ: tham số hỏng làm server từ chối cả hội thoại.
                    "arguments": json.dumps(_safe_parse_args(tc.arguments)[0], ensure_ascii=False),
                },
            }
            for tc in turn.tool_calls
        ]
    return msg


def _result_text(result: Any) -> str:
    """Rút văn bản từ kết quả call_tool của MCP; lỗi phía server luôn bắt đầu bằng "Error:"."""
    parts = [c.text for c in (getattr(result, "content", None) or []) if isinstance(getattr(c, "text", None), str)]
    text = "\n".join(parts).strip()
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    if is_error and not _is_error(text):
        text = f"Error: {text or 'công cụ báo lỗi không kèm nội dung.'}"
    return text or "(Công cụ không trả về nội dung: không có kết quả phù hợp.)"


class Agent:
    """Agent thẩm định hồ sơ vay / hỏi đáp chính sách, dùng các tool của FinCo MCP server."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        verbose: bool = True,
        llm: Optional[LlmFn] = None,
    ):
        """llm: hàm (messages, tools, model) -> ChatTurn; mặc định gọi LLM thật qua llm.call_llm_tools.
        Thay bằng hàm giả khi test để không tốn API."""
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = load_prompt("agent_system.txt")
        self._llm: LlmFn = llm or (lambda messages, tools, model: call_llm_tools(messages, tools, model=model))

    # ------------------------------------------------------------------ helpers
    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)

    async def _call_llm(self, messages: list[dict], tools: list[dict]) -> ChatTurn:
        """Gọi LLM trong luồng riêng: retry/backoff dùng time.sleep không được chặn event loop của MCP."""
        return await asyncio.to_thread(self._llm, messages, tools, self.model)

    def _completion_problem(
        self, turn: ChatTurn, require_decision: bool, trace: list[LogLine], app_id: str
    ) -> Optional[str]:
        """Lượt kết thúc (không còn tool_calls) có hợp lệ không? Trả lời nhắc nếu chưa, None nếu ổn."""
        if _TEXT_ACTION.search(turn.content):
            return (
                "Bạn vừa viết lời gọi công cụ dưới dạng văn bản, hệ thống KHÔNG thực thi văn bản. "
                "Hãy gọi công cụ bằng function calling."
            )
        if require_decision:
            written, verified, lookup_failed = _decision_state(trace, app_id)
            if lookup_failed:
                return None  # không có hồ sơ thì không có gì để ghi: kết thúc, lỗi được báo ở result
            if not written:
                return (
                    f"Chưa hoàn tất: hồ sơ {app_id} chưa có quyết định được ghi. Hãy làm tiếp quy trình, "
                    f"gọi {WRITE_TOOL} rồi {VERIFY_TOOL}, sau đó mới trả lời."
                )
            if not verified:
                return (
                    f"Chưa hoàn tất: bạn đã ghi quyết định nhưng chưa kiểm chứng. Hãy gọi {VERIFY_TOOL}"
                    f"('{app_id}') và đối chiếu kết quả trước khi trả lời."
                )
        if not turn.content:
            return "Phản hồi rỗng. Hãy gọi công cụ cần thiết hoặc đưa ra câu trả lời."
        return None

    async def _execute_tool(
        self,
        mcp_client: Client,
        tool_call: ToolCall,
        step: int,
        result: AgentRunResult,
        usage: Usage,
        tool_names: set[str],
    ) -> dict[str, Any]:
        """Parse tham số, gọi tool qua MCP, ghi AgentStep + LogLine vào result; trả tin nhắn role "tool".

        Không raise: mọi lỗi thành chuỗi "Error: ..." trả lại cho model để nó tự sửa.
        """
        name = tool_call.name
        args, parse_err = _safe_parse_args(tool_call.arguments)
        thought = ""

        if parse_err:
            output = parse_err
        elif name not in tool_names:
            output = f"Error: Công cụ '{name}' không tồn tại. Công cụ hợp lệ: {', '.join(sorted(tool_names))}."
        else:
            thought = str(args.pop("thought", "") or "")
            self._say(f"\n[Bước {step}]\nThought: {thought or '(không có)'}\nAction : {name}({args})")
            try:
                output = _result_text(await mcp_client.call_tool(name, args))
            except Exception as e:  # tool/MCP lỗi không được làm chết cả lượt
                output = f"Error: Gọi công cụ '{name}' thất bại ({type(e).__name__}: {e})"
        self._say(f"Observation: {output[:160]}{'...' if len(output) > 160 else ''}")

        result.steps.append(
            AgentStep(
                step_number=step,
                thought=thought or "(không có)",
                tool_name=name,
                tool_args=args,
                observation=output,
                is_error=_is_error(output),
            )
        )
        result.trace.append(
            LogLine(
                ts=datetime.now().isoformat(timespec="seconds"),
                run_id=result.run_id,
                question_id=result.question_id,
                step=step,
                tool=name,
                args=args,
                result=output,
                usage=replace(usage),  # bản chụp tại bước này, không phải đối tượng dùng chung
            )
        )
        return {"role": "tool", "tool_call_id": tool_call.id, "content": output}

    # ------------------------------------------------------------------ main loop
    async def run(
        self,
        app_id: str,
        question: Optional[str] = None,
        question_id: Optional[str] = None,
        require_decision: Optional[bool] = None,
    ) -> AgentRunResult:
        """Chạy một lượt. Không raise: kết quả (kể cả thất bại) nằm trong AgentRunResult.

        Args:
            app_id: Mã hồ sơ đang xét ("APP-014"), hoặc nhãn tùy ý cho câu hỏi đáp ("Q").
            question: Câu gửi cho agent; None = yêu cầu thẩm định và ghi nhận hồ sơ app_id.
            question_id: Mã ghi vào log (id testset, mã phiên); mặc định app_id.
            require_decision: Bắt buộc ghi + đọc lại quyết định trước khi coi là xong. None = tự suy:
                True khi không có question và app_id có dạng APP-xxx.
        """
        app_id = app_id.strip()
        query = question or (
            f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        )
        if require_decision is None:
            require_decision = question is None and bool(_APP_ID.fullmatch(app_id))

        result = AgentRunResult(
            app_id=app_id,
            final_answer="",
            run_id=new_run_id(),
            question_id=question_id or app_id,
            question=query,
        )
        result.usage = Usage(calls=0, prompt_tokens=0, completion_tokens=0, max_prompt=0)

        try:
            async with Client(get_default_server_params()) as mcp_client:
                tools = to_openai_tools((await mcp_client.list_tools()).tools)
                tool_names = {t["function"]["name"] for t in tools}
                await self._loop(mcp_client, tools, tool_names, query, require_decision, result)
        except Exception as e:  # không khởi động được MCP, mất kết nối, lỗi LLM không cứu được...
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"
            self._say(f"\n[Lỗi] {result.error_message}")

        if result.success and require_decision:
            written, verified, lookup_failed = _decision_state(result.trace, result.app_id)
            if lookup_failed and not written:
                result.success = False
                result.error_message = f"Không đọc được hồ sơ {result.app_id}, chưa ghi quyết định."
            elif not (written and verified):
                result.success = False
                result.error_message = (
                    "Kết thúc khi chưa ghi quyết định." if not written
                    else "Đã ghi quyết định nhưng chưa kiểm chứng bằng đọc lại."
                )

        usage = result.usage
        result.answer_log = LogLine(
            ts=datetime.now().isoformat(timespec="seconds"),
            run_id=result.run_id,
            question_id=result.question_id,
            step=(result.steps[-1].step_number if result.steps else 0),
            tool=ANSWER_TOOL,
            args={"success": result.success, "error": result.error_message},
            result=result.final_answer or (result.error_message or ""),
            usage=replace(usage),
        )
        return result

    async def _loop(
        self,
        mcp_client: Client,
        tools: list[dict],
        tool_names: set[str],
        query: str,
        require_decision: bool,
        result: AgentRunResult,
    ) -> None:
        """Vòng lặp orchestration. Cập nhật result tại chỗ, kể cả usage tích lũy sau mỗi lần gọi LLM."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        nudges = 0

        for step in range(1, self.max_steps + 1):
            try:
                turn = await self._call_llm(messages, tools)
            except MalformedToolCallError as e:
                if nudges >= MAX_NUDGES:
                    result.success, result.error_message = False, f"Model liên tục gọi tool sai định dạng: {e}"
                    return
                nudges += 1
                self._say(f"[Cảnh báo] {e}")
                messages.append({
                    "role": "user",
                    "content": "Lời gọi công cụ trước không hợp lệ (JSON sai). Hãy gọi lại công cụ với "
                               "tham số là một đối tượng JSON đúng cú pháp.",
                })
                continue
            except Exception as e:  # hết retry, lỗi xác thực, phản hồi rỗng...
                result.success, result.error_message = False, f"Lỗi gọi LLM: {type(e).__name__}: {e}"
                self._say(f"\n[Lỗi] {result.error_message}")
                return

            result.usage = merge_usage(result.usage, turn.usage)

            # TH 1: model không gọi tool nữa -> kiểm tra rồi kết thúc (hoặc nhắc sửa)
            if not turn.tool_calls:
                problem = self._completion_problem(turn, require_decision, result.trace, result.app_id)
                if problem and nudges < MAX_NUDGES:
                    nudges += 1
                    self._say(f"[Nhắc lại] {problem}")
                    if turn.content:
                        messages.append(_assistant_message(turn))
                    messages.append({"role": "user", "content": problem})
                    continue
                result.final_answer = turn.content
                if problem:  # đã hết lượt nhắc mà vẫn chưa hợp lệ
                    result.success, result.error_message = False, problem
                self._say(f"\n[AI Answer]:\n{result.final_answer}\n")
                return

            # TH 2: model yêu cầu gọi tool
            messages.append(_assistant_message(turn))
            for tool_call in turn.tool_calls:
                messages.append(
                    await self._execute_tool(mcp_client, tool_call, step, result, result.usage, tool_names)
                )

        result.success = False
        result.error_message = f"Vượt quá MAX_STEPS ({self.max_steps}) mà chưa có câu trả lời cuối."

    def run_sync(
        self,
        app_id: str,
        question: Optional[str] = None,
        question_id: Optional[str] = None,
        require_decision: Optional[bool] = None,
    ) -> AgentRunResult:
        """Chạy đồng bộ, tiện cho script và REPL."""
        return asyncio.run(self.run(app_id, question, question_id, require_decision))
