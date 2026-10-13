"""Agent Thẩm định FinCo - Kiến trúc chuẩn Tuần 6 (ReAct) + Tuần 7 (MCP Client).

Luồng chính (đúng như bài học Week 6 & Week 7):
1. Kết nối MCP Server (Tuần 7) -> Lấy danh sách tools động.
2. Vòng lặp ReAct (Tuần 6):
   - Gọi LLM với messages và tools.
   - Nếu LLM gọi tool -> Thực thi qua MCP Server, đưa Observation vào messages và lặp lại.
   - Nếu LLM không gọi tool -> Đưa ra câu trả lời cuối cùng (Final Answer).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from finco.agent_helpers import (
    ANSWER_TOOL,
    MAX_NUDGES,
    MAX_SEARCHES,
    MAX_STEPS,
    _APP_ID,
    AgentRunResult,
    AgentStep,
    LlmFn,
    _assistant_message,
    _safe_parse_args,
    check_completion_issue,
    create_answer_log,
    execute_agent_tool,
    new_run_id,
    save_trace_to_jsonl,
)
from finco.llm import (
    DEFAULT_MODEL,
    MalformedToolCallError,
    call_llm_tools,
    load_prompt,
    merge_usage,
)
from finco.mcp_client import Client, get_default_server_params, to_openai_tools

# Re-exports để đảm bảo 100% tương thích ngược với unit tests và các module khác
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
    """Agent thẩm định ReAct chuẩn Tuần 6, gọi công cụ qua MCP Server Tuần 7."""

    def __init__(self, model: Optional[str] = None, max_steps: int = MAX_STEPS,
                 verbose: bool = True, llm: Optional[LlmFn] = None):
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = load_prompt("agent_system.txt")
        self._llm = llm or (lambda msgs, tls, mdl: call_llm_tools(msgs, tls, model=mdl))

    def run_sync(self, *args, **kwargs) -> AgentRunResult:
        """Chạy đồng bộ tiện lợi cho CLI chat và test."""
        return asyncio.run(self.run(*args, **kwargs))

    async def run(self, app_id: str, question: Optional[str] = None,
                  question_id: Optional[str] = None, require_decision: Optional[bool] = None) -> AgentRunResult:
        """Vòng đời thẩm định: Mở MCP (Tuần 7) -> Vòng lặp ReAct (Tuần 6) -> Trả kết quả."""
        app_id = app_id.strip()
        query = question or f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        if require_decision is None:
            require_decision = question is None and bool(_APP_ID.fullmatch(app_id))

        result = AgentRunResult(app_id=app_id, run_id=new_run_id(), question_id=question_id or app_id, question=query)

        # -------------------------------------------------------------
        # BƯỚC 1: KẾT NỐI MCP SERVER (CHUẨN BÀI HỌC TUẦN 7)
        # -------------------------------------------------------------
        try:
            async with Client(get_default_server_params()) as mcp:
                tools = to_openai_tools((await mcp.list_tools()).tools)
                tool_names = {t["function"]["name"] for t in tools}

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ]
                nudges = 0

                # -------------------------------------------------------------
                # BƯỚC 2: VÒNG LẶP REACT CHUẨN BÀI HỌC TUẦN 6
                # -------------------------------------------------------------
                for step in range(1, self.max_steps + 1):
                    # 1. Gọi LLM (chạy thread riêng để không block MCP)
                    try:
                        turn = await asyncio.to_thread(self._llm, messages, tools, self.model)
                    except MalformedToolCallError as e:
                        turn = None
                        problem = f"Lời gọi công cụ trước không hợp lệ (JSON sai), hãy gọi lại với tham số đúng cú pháp. ({e})"
                    except Exception as e:
                        result.success, result.error_message = False, f"Lỗi gọi LLM: {type(e).__name__}: {e}"
                        break

                    if turn:
                        result.usage = merge_usage(result.usage, turn.usage)

                        # TH 1: LLM YÊU CẦU GỌI TOOLS (ACTION)
                        if turn.tool_calls:
                            messages.append(_assistant_message(turn))
                            for tc in turn.tool_calls:
                                tool_msg = await execute_agent_tool(mcp, tc, step, tool_names, result, self.verbose)
                                messages.append(tool_msg)
                            continue

                        # TH 2: LLM ĐƯA RA CÂU TRẢ LỜI CUỐI (FINAL ANSWER)
                        problem = check_completion_issue(turn, require_decision, result)
                        if problem == "LOOKUP_FAILED":
                            result.final_answer = turn.content
                            result.success, result.error_message = False, f"Không đọc được hồ sơ {result.app_id}, chưa ghi quyết định."
                            break

                        if problem is None:
                            result.final_answer = turn.content
                            if self.verbose:
                                print(f"\n[AI Answer]:\n{turn.content}\n")
                            break

                    # Nếu có vấn đề (quên ghi DB, format sai): Nhắc model sửa (tối đa MAX_NUDGES)
                    if nudges >= MAX_NUDGES:
                        result.success, result.error_message = False, problem
                        result.final_answer = turn.content if turn else ""
                        break

                    nudges += 1

                    if self.verbose:
                        print(f"[Nhắc lại] {problem}")

                    if turn and turn.content:
                        messages.append(_assistant_message(turn))
                        
                    messages.append({"role": "user", "content": problem})

                else:  # Hết vòng for (step > max_steps) mà chưa break
                    if not result.final_answer and result.success:
                        result.success, result.error_message = False, f"Vượt quá MAX_STEPS ({self.max_steps}) mà chưa có câu trả lời cuối."

        except Exception as e:
            result.success, result.error_message = False, f"{type(e).__name__}: {e}"

        # -------------------------------------------------------------
        # BƯỚC 3: ĐÓNG DẤU LOG VÀ TRẢ VỀ KẾT QUẢ
        # -------------------------------------------------------------
        result.answer_log = create_answer_log(result)
        return result
