"""Test vòng lặp Agent bằng LLM GIẢ (kịch bản định sẵn): không tốn API, không phụ thuộc mạng.

MCP server thật vẫn được spawn qua stdio và ghi vào DB sandbox, nên kiểm tra được cả đường
agent -> MCP client -> server -> tools -> SQLite. Chỉ phần "model nghĩ gì" là giả.

Chạy từ thư mục gốc repo:
    python -m unittest tests.test_agent -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.sandbox_db import isolated_db_sandbox  # noqa: E402
from eval.verifier import load_trace, submission_from_trace  # noqa: E402
from finco.agent import ANSWER_TOOL, Agent, _parse_args  # noqa: E402
from finco.llm import ChatTurn, ToolCall  # noqa: E402
from finco.types import Usage  # noqa: E402

APP = "APP-003"  # DTI 63,04% ở lãi 17% -> reject DTI_EXCEEDED (đáp án chuẩn)
_ids = iter(range(10_000))


def call(name: str, raw: str | None = None, **args) -> ToolCall:
    """Một lời gọi tool; raw cho phép đưa chuỗi arguments hỏng."""
    payload = raw if raw is not None else json.dumps({"thought": "kịch bản test", **args}, ensure_ascii=False)
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=payload)


def turn(*calls: ToolCall, content: str = "", prompt: int = 100, completion: int = 10) -> ChatTurn:
    return ChatTurn(
        content=content,
        tool_calls=list(calls),
        usage=Usage(calls=1, prompt_tokens=prompt, completion_tokens=completion, max_prompt=prompt),
    )


class Script:
    """LLM giả: mỗi lần được gọi trả (hoặc ném) phần tử kế tiếp của kịch bản, và ghi lại những gì nó thấy."""

    def __init__(self, *items):
        self.items = list(items)
        self.seen: list[list[dict]] = []  # bản chụp messages ở từng lần gọi

    def __call__(self, messages, tools, model):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        if not self.items:
            raise AssertionError("Kịch bản hết nhưng agent vẫn gọi LLM")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def good_decision_script(app_id=APP, tail="Đã ghi và kiểm chứng."):
    """Quy trình đúng: đọc hồ sơ -> tính DTI -> ghi (code tự đọc lại) -> trả lời."""
    return [
        turn(call("get_application", app_id=app_id)),
        turn(call("calc_dti", app_id=app_id, annual_rate="0.17")),
        turn(call(
            "submit_decision", app_id=app_id, decision="reject", reason="DTI vượt trần",
            cited_docs=["CP-01 v1", "CP-05"], reason_codes=["DTI_EXCEEDED"],
        )),
        turn(content=tail),
    ]


class AgentTestBase(unittest.TestCase):
    def run_agent(self, *items, app_id=APP, question=None, max_steps=12):
        script = Script(*items)
        agent = Agent(max_steps=max_steps, verbose=False, llm=script)
        with isolated_db_sandbox():
            result = agent.run_sync(app_id, question=question)
        return result, script


class TestHappyPath(AgentTestBase):
    def test_full_assessment(self):
        result, script = self.run_agent(*good_decision_script())

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(result.tool_calls, ["get_application", "calc_dti", "submit_decision", "get_decision"])
        self.assertEqual(result.final_answer, "Đã ghi và kiểm chứng.")
        self.assertEqual(len(script.seen), 4)

        sub = result.to_submission()
        self.assertEqual(sub.decision, "reject")
        self.assertEqual(sub.reason_codes, ["DTI_EXCEEDED"])
        self.assertAlmostEqual(sub.calculated_dti, 0.6304)
        self.assertAlmostEqual(sub.used_rate, 0.17)

    def test_thought_is_not_sent_to_tools(self):
        result, _ = self.run_agent(*good_decision_script())
        self.assertNotIn("thought", result.trace[0].args)

    def test_usage_is_summed_and_snapshotted_per_line(self):
        result, _ = self.run_agent(*good_decision_script())
        self.assertEqual(result.usage, Usage(calls=4, prompt_tokens=400, completion_tokens=40, max_prompt=100))
        # mỗi dòng log mang usage tại thời điểm của nó, không phải đối tượng dùng chung
        self.assertEqual([l.usage.calls for l in result.trace], [1, 2, 3, 3])
        self.assertEqual(result.answer_log.usage.calls, 4)

    def test_run_ids_are_unique(self):
        a, _ = self.run_agent(*good_decision_script())
        b, _ = self.run_agent(*good_decision_script())
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertTrue(all(l.run_id == a.run_id for l in a.trace))

    def test_system_prompt_and_tool_schemas_reach_llm(self):
        _, script = self.run_agent(*good_decision_script())
        first = script.seen[0]
        self.assertEqual(first[0]["role"], "system")
        self.assertNotIn("${", first[0]["content"])  # không còn chỗ trống chưa điền
        self.assertIn(APP, first[1]["content"])

    def test_save_log_writes_trace_plus_answer_line(self):
        result, _ = self.run_agent(*good_decision_script())
        with tempfile.TemporaryDirectory() as tmp:
            path = result.save_log(Path(tmp) / "run.jsonl")
            rows = load_trace(path)
            self.assertEqual([r["tool"] for r in rows],
                             ["get_application", "calc_dti", "submit_decision", "get_decision", ANSWER_TOOL])
            self.assertEqual(rows[-1]["usage"]["calls"], 4)
            self.assertEqual(rows[-1]["result"], "Đã ghi và kiểm chứng.")
            # verifier đọc lại từ file ra cùng kết quả, và bỏ qua dòng __answer__
            sub = submission_from_trace(APP, rows)
            self.assertEqual(sub.decision, "reject")
            self.assertNotIn(ANSWER_TOOL, sub.tool_calls)

class TestReadBack(AgentTestBase):
    """Ghi quyết định thành công thì CODE tự đọc lại; model không cần (và không bị bắt) gọi get_decision."""

    def test_code_reads_back_after_successful_submit(self):
        script = good_decision_script()
        result, llm = self.run_agent(*script)
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(result.tool_calls, ["get_application", "calc_dti", "submit_decision", "get_decision"])
        self.assertEqual(result.trace[3].step, result.trace[2].step)  # cùng bước với lần ghi
        self.assertEqual(result.to_submission().decision, "reject")
        # model thấy kết quả đọc lại ngay trong phản hồi của submit_decision
        tool_msg = [m for m in llm.seen[3] if m["role"] == "tool"][-1]["content"]
        self.assertIn("get_decision", tool_msg)
        self.assertIn("reject", tool_msg)

    def test_failed_submit_is_not_read_back(self):
        bad = turn(call(
            "submit_decision", app_id=APP, decision="approve", reason="x",
            cited_docs=["CP-01 v1"], reason_codes=["DTI_EXCEEDED"],  # mã không khớp decision
        ))
        script = good_decision_script()
        script.insert(2, bad)
        result, _ = self.run_agent(*script)
        self.assertTrue(result.success, result.error_message)
        self.assertTrue(result.trace[2].result.startswith("Error:"))
        # lần ghi lỗi không kích hoạt đọc lại; chỉ lần ghi thành công mới có get_decision theo sau
        self.assertEqual(result.tool_calls,
                         ["get_application", "calc_dti", "submit_decision", "submit_decision", "get_decision"])
        self.assertEqual(result.to_submission().decision, "reject")  # lần ghi thành công mới là bài nộp

    def test_qa_mode_needs_no_decision(self):
        result, _ = self.run_agent(
            turn(content="Không có thông tin này trong tài liệu chính sách của FinCo."),
            app_id="Q", question="Vay mua ô tô lãi bao nhiêu?",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.trace, [])
        self.assertEqual(result.usage.calls, 1)  # câu không gọi tool nào vẫn có USAGE (dòng __answer__)
        self.assertEqual(result.answer_log.tool, ANSWER_TOOL)


class TestRobustness(AgentTestBase):
    def test_unknown_tool_is_an_error_observation_not_a_crash(self):
        script = [turn(call("drop_database", app_id=APP))] + good_decision_script()
        result, _ = self.run_agent(*script)
        self.assertTrue(result.success, result.error_message)
        self.assertTrue(result.trace[0].result.startswith("Error:"))

    def test_broken_json_args_are_reported_to_model(self):
        script = [turn(call("get_application", raw='{"app_id": APP-003'))] + good_decision_script()
        result, llm = self.run_agent(*script)
        self.assertTrue(result.success, result.error_message)
        # tham số hỏng không tới tool (không có dòng log); model nhận Error: ở lượt sau
        self.assertEqual(result.tool_calls, ["get_application", "calc_dti", "submit_decision", "get_decision"])
        sent_back = [m for m in llm.seen[1] if m["role"] == "tool"][0]["content"]
        self.assertTrue(sent_back.startswith("Error:"))

    def test_unknown_application_is_an_error_observation(self):
        result, _ = self.run_agent(
            turn(call("get_application", app_id="APP-999")),
            turn(content="Không tìm thấy hồ sơ APP-999."),
            app_id="APP-999",
        )
        self.assertTrue(result.trace[0].result.startswith("Error:"))
        self.assertNotIn("submit_decision", result.tool_calls)

    def test_llm_failure_keeps_trace_and_does_not_raise(self):
        script = [turn(call("get_application", app_id=APP)), RuntimeError("hết retry: 429")]
        result, _ = self.run_agent(*script)
        self.assertFalse(result.success)
        self.assertIn("Lỗi gọi LLM", result.error_message)
        self.assertEqual(result.tool_calls, ["get_application"])
        self.assertEqual(result.usage.calls, 1)
        self.assertEqual(result.answer_log.args["success"], False)

    def test_empty_answer_fails(self):
        result, _ = self.run_agent(turn(content=""), app_id="Q", question="Câu hỏi?")
        self.assertFalse(result.success)

    def test_max_steps_stops_cleanly(self):
        loop = [turn(call("get_application", app_id=APP)) for _ in range(5)]
        result, llm = self.run_agent(*loop, max_steps=3)
        self.assertFalse(result.success)
        self.assertIn("MAX_STEPS", result.error_message)
        self.assertEqual(len(llm.seen), 3)
        self.assertEqual(len(result.trace), 3)

    def test_parallel_tool_calls_are_all_answered(self):
        script = [
            turn(call("get_application", app_id=APP), call("get_application", app_id="APP-001")),
            turn(content="Xong."),
        ]
        result, llm = self.run_agent(*script, app_id="Q", question="So sánh hai hồ sơ")
        self.assertEqual(result.tool_calls, ["get_application", "get_application"])
        tool_msgs = [m for m in llm.seen[1] if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)

    def test_broken_args_are_not_echoed_back_in_history(self):
        script = [turn(call("get_application", raw="{kèm rác")), turn(content="ok")]
        _, llm = self.run_agent(*script, app_id="Q", question="?")
        assistant = [m for m in llm.seen[1] if m["role"] == "assistant"][0]
        json.loads(assistant["tool_calls"][0]["function"]["arguments"])  # phải là JSON hợp lệ


class TestParseArgs(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_parse_args('{"a": 1}'), ({"a": 1}, ""))

    def test_empty_means_no_args(self):
        self.assertEqual(_parse_args(""), ({}, ""))
        self.assertEqual(_parse_args("   "), ({}, ""))

    def test_garbage_is_error_not_exception(self):
        args, err = _parse_args("không phải json")
        self.assertEqual(args, {})
        self.assertTrue(err.startswith("Error:"))

    def test_non_dict(self):
        for raw in ("[1, 2]", '"abc"', "42", "null"):
            args, err = _parse_args(raw)
            self.assertEqual(args, {}, raw)
            self.assertTrue(err.startswith("Error:"), raw)


if __name__ == "__main__":
    unittest.main()
