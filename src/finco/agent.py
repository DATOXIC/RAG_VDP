"""Agent thẩm định FinCo: vòng lặp ReAct viết tay (W6), gọi công cụ qua MCP server (W7).

    LLM trả lời gì?
      - có tool_calls  -> chạy từng tool qua MCP, đưa kết quả (Observation) vào lịch sử, lặp lại
      - không có       -> đó là câu trả lời cuối, dừng
    Quá MAX_STEPS lần gọi LLM mà chưa có câu trả lời thì dừng và báo thất bại.

Đọc file từ trên xuống: 1 hằng số, 2 kết quả một lượt, 3 Agent (run -> _run_tool -> _call).
"""

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


def _parse_args(raw: str) -> tuple[dict, str]:
    """Tham số tool (chuỗi JSON thô) -> (dict, "") hoặc ({}, "Error: ...") để model tự sửa."""
    try:
        args = json.loads((raw or "").strip() or "{}")
    except json.JSONDecodeError as e:
        return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})."
    if not isinstance(args, dict):
        return {}, "Error: Tham số gọi tool phải là một đối tượng JSON dạng {\"tên\": giá_trị}."
    return args, ""


def _tool_output(res: Any) -> str:
    """Văn bản từ kết quả call_tool của MCP; lỗi phía server luôn bắt đầu bằng "Error:"."""
    text = "\n".join(c.text for c in (res.content or []) if isinstance(getattr(c, "text", None), str)).strip()
    if getattr(res, "is_error", False) and not text.startswith("Error:"):
        text = f"Error: {text or 'công cụ báo lỗi không kèm nội dung.'}"
    return text or "(Công cụ không trả về nội dung: không có kết quả phù hợp.)"


# --------------------------------------------------------------------------- 3. Agent
class Agent:
    def __init__(self, model: Optional[str] = None, max_steps: int = MAX_STEPS,
                 verbose: bool = True, llm: Optional[LlmFn] = None):
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = load_prompt("agent_system.txt")
        self._llm = llm or (lambda messages, tools, model: call_llm_tools(messages, tools, model=model))

    def run_sync(self, *args, **kwargs) -> AgentRunResult:
        """Chạy đồng bộ, tiện cho CLI chat và test."""
        return asyncio.run(self.run(*args, **kwargs))

    async def run(self, app_id: str, question: Optional[str] = None,
                  question_id: Optional[str] = None) -> AgentRunResult:
        """Một lượt: mở MCP server -> vòng lặp ReAct -> kết quả. Không raise: lỗi nằm trong result."""
        app_id = app_id.strip()
        query = question or f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        result = AgentRunResult(app_id=app_id, run_id=f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
                                question_id=question_id or app_id, question=query)
        try:
            async with Client(get_default_server_params()) as mcp:
                tools = to_openai_tools((await mcp.list_tools()).tools)
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ]
                for step in range(1, self.max_steps + 1):
                    try:  # thread riêng: lời gọi LLM chặn, không được làm nghẽn kết nối MCP
                        turn = await asyncio.to_thread(self._llm, messages, tools, self.model)
                    except Exception as e:
                        result.success, result.error_message = False, f"Lỗi gọi LLM: {type(e).__name__}: {e}"
                        break
                    result.usage = merge_usage(result.usage, turn.usage)

                    if not turn.tool_calls:  # câu trả lời cuối
                        result.final_answer = turn.content
                        if not turn.content:
                            result.success, result.error_message = False, "LLM trả về phản hồi rỗng."
                        elif self.verbose:
                            print(f"\n[AI Answer]:\n{turn.content}\n")
                        break

                    calls = [(tc, *_parse_args(tc.arguments)) for tc in turn.tool_calls]
                    messages.append({
                        "role": "assistant", "content": turn.content or None,
                        "tool_calls": [  # tham số luôn là JSON hợp lệ: rác trong lịch sử làm server từ chối
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.name, "arguments": json.dumps(args, ensure_ascii=False)}}
                            for tc, args, _ in calls
                        ],
                    })
                    for tc, args, error in calls:
                        output = error or await self._run_tool(mcp, tc.name, args, step, result)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                else:  # hết MAX_STEPS mà chưa break
                    result.success = False
                    result.error_message = f"Vượt quá MAX_STEPS ({self.max_steps}) mà chưa có câu trả lời cuối."
        except Exception as e:  # không mở được MCP server, mất kết nối giữa chừng...
            result.success, result.error_message = False, f"{type(e).__name__}: {e}"

        result.answer_log = LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id, question_id=result.question_id,
            step=result.trace[-1].step if result.trace else 0, tool=ANSWER_TOOL,
            args={"success": result.success, "error": result.error_message},
            result=result.final_answer or result.error_message or "", usage=result.usage,
        )
        return result

    async def _run_tool(self, mcp: Any, name: str, args: dict, step: int, result: AgentRunResult) -> str:
        """Chạy một tool. Ghi thành công thì code TỰ đọc lại, không phụ thuộc model nhớ gọi get_decision."""
        thought = str(args.pop("thought", "") or "")  # 'thought' dành cho người đọc, không gửi cho tool
        if self.verbose:
            print(f"\n[Bước {step}]\nThought: {thought or '(không có)'}\nAction : {name}({args})")
        output = await self._call(mcp, name, args, step, result)

        if name == WRITE_TOOL and not output.startswith("Error:"):
            readback = await self._call(mcp, VERIFY_TOOL, {"app_id": args.get("app_id", "")}, step, result)
            output += f"\n\n[Hệ thống tự đọc lại bằng {VERIFY_TOOL}]\n{readback}"
        return output

    async def _call(self, mcp: Any, name: str, args: dict, step: int, result: AgentRunResult) -> str:
        """Gọi tool qua MCP (lỗi nào cũng thành chuỗi "Error: ...") và ghi một LogLine vào trace."""
        try:
            output = _tool_output(await mcp.call_tool(name, args))
        except Exception as e:
            output = f"Error: Gọi công cụ '{name}' thất bại ({type(e).__name__}: {e})"
        if self.verbose:
            print(f"Observation: {output[:160]}{'...' if len(output) > 160 else ''}")
        result.trace.append(LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), run_id=result.run_id,
            question_id=result.question_id, step=step, tool=name, args=args, result=output, usage=result.usage,
        ))
        return output
