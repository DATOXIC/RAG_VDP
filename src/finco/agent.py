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
from datetime import datetime
from typing import Any, Optional

from finco.agent_helpers import (
    ANSWER_TOOL,
    LOGS_DIR,
    MAX_NUDGES,
    MAX_SEARCHES,
    MAX_STEPS,
    SEARCH_TOOL,
    VERIFY_TOOL,
    WRITE_TOOL,
    AgentRunResult,
    AgentStep,
    LlmFn,
    _APP_ID,
    _TEXT_ACTION,
    _assistant_message,
    _is_error,
    _repeat_search_error,
    _result_text,
    _safe_parse_args,
    decision_status,
    new_run_id,
    save_trace_to_jsonl,
)
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
from finco.types import LogLine

# Tái xuất khẩu (re-export) để đảm bảo 100% tương thích ngược với code và unit tests cũ
__all__ = [
    "Agent",
    "AgentStep",
    "AgentRunResult",
    "save_trace_to_jsonl",
    "ANSWER_TOOL",
    "MAX_NUDGES",
    "MAX_STEPS",
    "MAX_SEARCHES",
    "_safe_parse_args",
]


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
            result=result.final_answer or result.error_message or "", usage=result.usage,
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
            usage=result.usage,  # bản chụp tại bước này
        ))
        return {"role": "tool", "tool_call_id": tc.id, "content": output}

    @staticmethod
    def _prior_searches(result: AgentRunResult) -> list[str]:
        """Các truy vấn search_policy đã chạy thành công trong lượt này (đã chuẩn hóa khoảng trắng/hoa thường)."""
        return [" ".join(str(s.tool_args.get("query", "")).lower().split())
                for s in result.steps if s.tool_name == SEARCH_TOOL and not s.is_error]
