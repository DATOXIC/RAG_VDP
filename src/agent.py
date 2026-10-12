"""Kết quả một lượt chạy của agent.

Nguyên tắc: LLM chỉ trả văn bản, nên KHÔNG tin vào đối tượng mà agent "kể lại". Nguồn sự thật duy
nhất của một lượt chạy là `trace`: mỗi lời gọi tool một `LogLine` (tên tool, args đã parse, kết quả
chuỗi). Bài nộp cho verifier được dựng từ trace bằng `submission_from_trace`, không gán tay.
"""
from datetime import datetime
import json

from dataclasses import dataclass, field
from typing import Any, Optional
from finco.llm import get_client, load_prompt, DEFAULT_MODEL
from finco.types import LogLine, Usage
from finco.mcp_client import Client, get_default_server_params, to_openai_tools
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

    def _init_messages(self, query: str) -> list[dict[str, str]]:
        """1. Chuẩn bị ngữ cảnh hội thoại ban đầu."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

    def _call_llm_step(self, messages: list[dict], tools: list[dict], usage: Usage) -> Any:
        """2. Gọi LLM và tích lũy số lượng token tiêu thụ."""
        response = self.client_llm.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )

        if response.usage:
            usage.calls += 1
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
            usage.max_prompt = max(usage.max_prompt, response.usage.prompt_tokens)

        return response.choices[0].message
    
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

    def _safe_parse_args(self, raw_args: str) -> tuple[dict, Optional[str]]:
        """3. Phòng vệ W6: Bắt lỗi JSON hỏng, thiếu ngoặc không làm chết run."""
        try:
            return json.loads(raw_args or "{}"), None
        except json.JSONDecodeError as e:
            cleaned = (raw_args or "").strip()
            if not cleaned.endswith("}"):
                try:
                    return json.loads(cleaned + "}"), None
                except Exception:
                    pass
            return {}, f"Error: Tham số gọi tool không phải JSON hợp lệ ({e})"
        
    async def _execute_tool(
        self,
        mcp_client: Client,
        tool_call: Any,
        step: int,
        run_id: str,
        app_id: str,
        usage: Usage,
    ) -> tuple[AgentStep, LogLine, dict]:
        """4. Bóc tách suy nghĩ 'thought', gọi tool qua MCP và đóng gói log."""
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        args, parse_err = self._safe_parse_args(raw_args)

        if parse_err:
            tool_output = parse_err
            thought = "Lỗi cú pháp khi tạo tham số JSON"
        else:
            # Bóc tách suy nghĩ ReAct ra khỏi tham số
            thought = args.pop("thought", "Không có suy nghĩ")
            if self.verbose:
                print(f"\n[Bước {step}]")
                print(f"Thought: {thought}")
                print(f"Action : {tool_name}({args})")
            # Gọi thực thi qua MCP Server
            mcp_res = await mcp_client.call_tool(tool_name, args)
            tool_output = mcp_res.content[0].text if mcp_res.content else ""
            if self.verbose:
                print(f"👁️ Observation: {tool_output[:120]}...")
        # Tạo đối tượng bước hiển thị (cho người)
        step_obj = AgentStep(
            step_number=step,
            thought=thought,
            tool_name=tool_name,
            tool_args=args,
            observation=tool_output,
        )

        # Tạo dòng kiểm toán (cho máy & Verifier)
        log_obj = LogLine(
            ts=datetime.now().isoformat(timespec="seconds"),
            run_id=run_id,
            question_id=app_id,
            step=step,
            tool=tool_name,
            args=args,
            result=tool_output,
            usage=usage,
        )
        # Tin nhắn phản hồi trả về cho LLM
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_output,
        }
        return step_obj, log_obj, tool_msg

    async def run(self, app_id: str, question: Optional[str] = None) -> AgentRunResult:
        """Nhạc trưởng điều phối toàn bộ vòng lặp thẩm định."""
        run_id = f"run_{app_id}"
        query = question or f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        result_obj = AgentRunResult(app_id=app_id, final_answer="")
        trace: list[LogLine] = []
        steps: list[AgentStep] = []
        usage = Usage(calls=0, prompt_tokens=0, completion_tokens=0, max_prompt=0)
        # 1. Mở kết nối MCP Server
        async with Client(get_default_server_params()) as mcp_client:
            tools_for_llm = to_openai_tools((await mcp_client.list_tools()).tools)
            messages = self._init_messages(query)
            step_count = 0
            # 2. Vòng lặp Orchestration tự viết (Mục 2.2 & 2.6 của Thầy)
            while step_count < self.max_steps:
                step_count += 1
                msg = self._call_llm_step(messages, tools_for_llm, usage)
                # TH 1: LLM ĐÃ KẾT LUẬN XONG (Không gọi tool nữa)
                if not msg.tool_calls:
                    result_obj.final_answer = msg.content or ""
                    if self.verbose:
                        print(f"\n[AI Answer]:\n{result_obj.final_answer}\n")
                    break
                # TH 2: LLM YÊU CẦU GỌI TOOL
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    step_obj, log_obj, tool_msg = await self._execute_tool(
                        mcp_client, tool_call, step_count, run_id, app_id, usage
                    )
                    steps.append(step_obj)
                    trace.append(log_obj)
                    messages.append(tool_msg)
            # Kiểm tra nếu hết vòng lặp mà chưa kết luận
            if step_count >= self.max_steps and not result_obj.final_answer:
                result_obj.success = False
                result_obj.error_message = f"Vượt quá MAX_STEPS ({self.max_steps})"
        # 3. Đóng gói kết quả
        result_obj.trace = trace
        result_obj.steps = steps
        result_obj.usage = usage
        return result_obj

    def run_sync(self, app_id: str, question: Optional[str] = None) -> AgentRunResult:
        """Chạy đồng bộ tiện lợi cho script chạy thử."""
        import asyncio
        return asyncio.run(self.run(app_id, question))