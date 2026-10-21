from __future__ import annotations

import asyncio
import json
import os
import re
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

MAX_STEPS = int(os.getenv("FINCO_MAX_STEPS", "15"))  # số lần gọi LLM tối đa mỗi lượt (quy trình dài nhất ~9, +3 cho vòng Critic); model yếu hay lặp nên có thể nâng
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
    trace: list[LogLine] = field(default_factory=list) # Cứ gọi tool là thực hiện lưu lại LogLine
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    answer_log: Optional[LogLine] = None
    seen_calls: dict[str, int] = field(default_factory=dict, repr=False)  # (tool, args) -> bước gọi thành công đầu tiên

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


# Sơ đồ vòng lặp (Agent.run):
# 1. Khởi tạo Client MCP + Kết quả (AgentRunResult)
# 2. Discovery danh sách tools từ Server
# 3. Tạo tool schema (OpenAI format)
# ------------------------------------------------------
# [VÒNG LẶP: for step in range(MAX_STEPS)]
#    4. Gọi API LLM -> Nhận ChatTurn
#       - NẾU KHÔNG CÓ TOOL CALL -> Lấy final_answer -> BREAK (Dừng vòng lặp)
#       - NẾU CÓ TOOL CALL -> Tiếp tục:
#    5. Lưu yêu cầu của AI vào message: messages.append(turn.to_assistant_message())
#    6. Thực hiện tool:
#       - Chạy tool qua MCP
#       - Ghi LogLine vào result.trace (để tí nữa chấm điểm)
#       - Nếu là 'submit_decision' -> Tự đọc lại bằng 'get_decision'
#    7. Đưa kết quả tool vào message: messages.append({"role": "tool", ...})
# 8. Chuyển trace thành AgentSubmission để chấm điểm (result.to_submission())


def is_underwrite_intent(app_id: Optional[str], question: Optional[str]) -> bool:
    """Xác định lượt chạy có phải là thẩm định hồ sơ cần ghi nhận quyết định hay không."""
    
    # Chuẩn hóa chuỗi câu hỏi
    q = (question or "").strip().lower()
    
    # Kiểm tra xem có mã hồ sơ (APP-xxx) ở tham số app_id hoặc nằm trong câu hỏi không
    has_app_id = bool(app_id and app_id.strip()) or bool(re.search(r"\bapp-\d+\b", q))
    
    # 0. Nếu hoàn toàn không có mã hồ sơ nào -> Chắc chắn 100% là hỏi đáp chính sách (QA)
    if not has_app_id:
        return False

    # 1. Có app_id nhưng không truyền câu hỏi riêng (None hoặc chuỗi rỗng) -> Mặc định thẩm định
    if not q:
        return True

    # 2. Các câu hỏi có ý định loại trừ (ví dụ C01, C02: chỉ tính toán/kiểm tra, không ghi quyết định)
    neg_patterns = [
        "chỉ cần", "chỉ tính", "chỉ kiểm tra", "chỉ tra cứu", "chỉ hỏi",
        "chưa cần ghi", "không cần ghi", "không ghi", "chưa ghi",
        "không thẩm định", "chưa thẩm định", "không gọi submit"
    ]
    if any(neg in q for neg in neg_patterns):
        return False

    # 3. Các câu hỏi yêu cầu thẩm định rõ ràng (ví dụ A01, A02)
    pos_patterns = [
        "thẩm định", "ghi nhận quyết định", "xét duyệt", 
        "phê duyệt hồ sơ", "duyệt hồ sơ", "ra quyết định", "ghi quyết định"
    ]
    if any(k in q for k in pos_patterns):
        return True

    # 4. Các câu hỏi còn lại (dù có mã APP-xxx nhưng chỉ hỏi thông tin lẻ, hoặc tra cứu T01, S01, V01, N01): Đều là QA
    return False


class Agent:
    def __init__(
        self, 
        model: Optional[str] = None, 
        max_steps: int = MAX_STEPS,
        verbose: bool = True, 
        llm: Optional[LlmFn] = None
    ):

        """Khởi tạo: nạp sẵn cấu hình và prompt vào bộ nhớ một lần duy nhất."""
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.verbose = verbose
        self.underwrite_prompt = load_prompt("agent_underwrite.txt")
        self.qa_prompt = load_prompt("agent_qa.txt")
        self.system_prompt = self.underwrite_prompt
        self.dedup_tools = os.getenv("FINCO_DEDUP_TOOLS", "0").strip() == "1"  # cờ bật/tắt tính năng chống gọi trùng lặp
        
        if llm is not None:
            self._llm = llm
        else:

            def default_llm(messages, tools, model):
                return call_llm_tools(messages, tools, model=model)
                
            self._llm = default_llm

    def run_sync(self, *args, **kwargs) -> AgentRunResult:
        return asyncio.run(self.run(*args, **kwargs))

    async def _run_tool(self, mcp: Any, name: str, args: dict, step: int, result: AgentRunResult) -> str:
        """Thực thi 1 tool: bóc thought, gọi MCP, ghi LogLine và tự đọc lại nếu là submit_decision."""
        
        # THOUGHT
        thought = str(args.pop("thought", "") or "")
        if self.verbose:
            print(f"\n[Bước {step}]\nThought: {thought or '(không có)'}\nAction : {name}({args})")

        # KIỂM TRA TRÙNG LẶP
        call_key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        first_step = result.seen_calls.get(call_key)

        # Trùng key + KHÔNG PHẢI WRITE_TOOL / VERIFY_TOOL --> Chặn
        if self.dedup_tools and first_step is not None and name not in (WRITE_TOOL, VERIFY_TOOL):
            output = (
                f"Bạn đã gọi {name} với đúng tham số này ở bước {first_step}; kết quả không đổi (xem lại phía trên). "
                "Không gọi lặp: hãy chuyển sang bước kế tiếp của quy trình."
            )
            if self.verbose:
                print(f"Observation: [bỏ qua lời gọi lặp] {output[:120]}")

            result.trace.append(
                LogLine(
                ts=datetime.now().isoformat(timespec="seconds"), 
                run_id=result.run_id,
                question_id=result.question_id, 
                step=step, 
                tool=name,
                args=args,
                result=output, 
                usage=result.usage,
            ))
            return output

        try:
            res = await mcp.call_tool(name, args)
            output = res.content[0].text if res.content else ""

            if res.is_error and not output.startswith("Error:"):  # vd. "Unknown tool: ..." do MCP sinh ra
                output = f"Error: {output or 'công cụ báo lỗi không kèm nội dung.'}"
                
        except Exception as e:
            output = f"Error: Gọi công cụ '{name}' thất bại ({type(e).__name__}: {e})"

        if self.verbose:
            print(f"Observation: {output[:160]}{'...' if len(output) > 160 else ''}")

        if not output.startswith("Error:"):
            result.seen_calls.setdefault(call_key, step)

        # Thực hiện tool xong -> Ghi lại Log
        result.trace.append(LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), 
            run_id=result.run_id,
            question_id=result.question_id, 
            step=step, 
            tool=name, 
            args=args, 
            result=output, 
            usage=result.usage,
        ))

        if name == WRITE_TOOL and not output.startswith("Error:"):
            readback_args = {"app_id": args.get("app_id", "")}
            rb_res = await mcp.call_tool(VERIFY_TOOL, readback_args)
            readback_text = rb_res.content[0].text if rb_res.content else ""
            
            result.trace.append(LogLine(
                ts=datetime.now().isoformat(timespec="seconds"), 
                run_id=result.run_id,
                question_id=result.question_id, 
                step=step, 
                tool=VERIFY_TOOL, 
                args=readback_args, 
                result=readback_text, 
                usage=result.usage,
            ))
            output += f"\n\n[Hệ thống tự đọc lại bằng {VERIFY_TOOL}]\n{readback_text}"

        return output

    async def run(
        self, 
        app_id: str, 
        question: Optional[str] = None,
        question_id: Optional[str] = None
    ) -> AgentRunResult:
        """  """

        app_id = app_id.strip()

        is_underwrite = is_underwrite_intent(app_id, question)

        query = question or f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        
        result = AgentRunResult(
            app_id=app_id, 
            run_id=f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
            question_id=question_id or app_id, 
            question=query
        )

        try:
            # 1. Kết nối MCP Server & Discovery tools
            async with Client(get_default_server_params()) as mcp:

                # DISCOVERY
                all_tools = to_openai_tools((await mcp.list_tools()).tools)

                # LỌC TOOLS
                if is_underwrite:
                    tools = all_tools
                else:
                    tools = [ t for t in all_tools if t.get("function", {}).get("name")  not in (WRITE_TOOL, VERIFY_TOOL) ]

                # LỌC PROMPT
                active_prompt = self.underwrite_prompt if is_underwrite else self.qa_prompt
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": active_prompt},
                    {"role": "user", "content": query},
                ]


                # 2. Vòng lặp ReAct
                for step in range(1, self.max_steps + 1):

                    try:
                        turn = await asyncio.to_thread(self._llm, messages, tools, self.model)
                    except Exception as e:
                        result.success= False
                        result.error_message = f"Lỗi gọi LLM: {type(e).__name__}: {e}"
                        break
                    
                    result.usage = merge_usage(result.usage, turn.usage)

                    # ĐIỀU KIỆN DỪNG: Model không gọi tool nào nữa -> Đã có câu trả lời cuối
                    if not turn.tool_calls:

                        has_submitted = any(
                            line.tool == WRITE_TOOL and not line.result.startswith("Error:") 
                            for line in result.trace
                        )

                        # NUDGE nhắc nhở LLM
                        if is_underwrite and step < self.max_steps:
                            has_submitted = any(line for line in result.trace if line.tool == WRITE_TOOL and not line.result.startswith("Error:"))

                            if not has_submitted:
                                messages.append({
                                    "role": "user", 
                                    "content": f"Chưa hoàn tất: Hồ sơ {result.app_id} chưa được ghi nhận quyết định thành công. Hãy gọi {WRITE_TOOL} trước khi trả lời."
                                })
                                continue

                        
                        result.final_answer = turn.content
                        
                        # if is is_underwrite and result.final_answer and not bool(re.search(r"CP-\d+",result.final_answer)):
                        #     sub = result.to_submission()
                        #     if sub.cited_docs:
                        #         docs_str = ", ".join("[{d}]" for d in sub.cited_docs)
                        #         result.final_answer += f"\n\n**Căn cứ pháp lý áp dụng:**{docs_str}"  

                        if not turn.content:
                            result.success = False
                            result.error_message = "LLM trả về phản hồi rỗng."

                        elif self.verbose:
                            print(f"\n[AI Answer]:\n{turn.content}\n")
                        break

                    # NẾU CÓ TOOL CALL:
                    # Lưu lượt nói của model vào messages 
                    messages.append(turn.to_assistant_message())

                    # Thực thi từng tool
                    for tc in turn.tool_calls:
                        output = tc.error or await self._run_tool(mcp, tc.name, tc.args, step, result)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                
                else:
                    result.success = False
                    result.error_message = f"Vượt quá MAX_STEPS ({self.max_steps}) mà chưa có câu trả lời cuối."
        except Exception as e:
            result.success= False 
            result.error_message = f"{type(e).__name__}: {e}"

        # 3. Ghi dòng tổng kết cuối cùng
        result.answer_log = LogLine(
            ts=datetime.now().isoformat(timespec="seconds"), 
            run_id=result.run_id, 
            question_id=result.question_id,
            step=result.trace[-1].step if result.trace else 0, 
            tool=ANSWER_TOOL,
            args={"success": result.success, "error": result.error_message},
            result=result.final_answer or result.error_message or "", 
            usage=result.usage,
        )
        return result
 
        