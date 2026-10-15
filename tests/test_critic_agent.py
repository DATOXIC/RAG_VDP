"""Test critic-agent (agent thứ hai có vòng lặp + tool đọc riêng). LLM và tool đọc đều giả: không tốn API.

Chạy từ thư mục gốc repo:  python -m unittest tests.test_critic_agent -v
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.sandbox_db import isolated_db_sandbox  # noqa: E402
from finco import critic as critic_mod  # noqa: E402
from finco import tools  # noqa: E402
from finco.critic_agent import CriticAgent, READ_ONLY_TOOLS, TOOL_SPECS, Toolbox  # noqa: E402
from finco.llm import ChatTurn, ToolCall  # noqa: E402
from finco.types import Chunk  # noqa: E402

APP = "APP-003"          # nộp 2025-02-20, DTI 63,04% ở lãi 17% -> reject DTI_EXCEEDED
DATE = "2025-02-20"
APP_TEXT = "Mã hồ sơ: APP-003\nTuổi: 40\nNgày nộp hồ sơ: 2025-02-20"
DTI_TEXT = "=== KẾT QUẢ TÍNH DTI HỒ SƠ APP-003 ===\n- Tỷ lệ nợ trên thu nhập (DTI): 0.6304 (63.04%)"
CP01 = "[CP-01 v1]\nTrần DTI tối đa là 50% thu nhập đủ điều kiện."

AGREE = {"expected_decision": "reject", "expected_reason_codes": ["DTI_EXCEEDED"],
         "reason_consistent": True, "issues": []}


def call(name, **args):
    return ToolCall(id=f"c_{name}_{len(args)}", name=name, arguments=json.dumps(args))


def turn(content="", *calls):
    return ChatTurn(content=content, tool_calls=list(calls))


def final(d=AGREE):
    return turn(json.dumps(d, ensure_ascii=False))


class ScriptedLlm:
    """Trả lần lượt các ChatTurn định sẵn; ghi lại tools mà mỗi lần gọi được cấp."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.tools_seen = []
        self.messages_seen = []

    def __call__(self, messages, tools_):
        self.tools_seen.append(tools_)
        self.messages_seen.append(list(messages))
        t = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        if isinstance(t, Exception):
            raise t
        return t


def hit(doc_id="CP-01", version="v1", text=CP01):
    return SimpleNamespace(chunk=Chunk(f"{doc_id}-{version}-01", doc_id, version, "2025-01-01", "Mục 1", text))


def toolbox(searches=None):
    log = {"search": [], "app": 0, "dti": 0}

    def get_app(app_id):
        log["app"] += 1
        return APP_TEXT, DATE

    def search(query, as_of, top_k):
        log["search"].append((query, as_of, top_k))
        return "kết quả", [hit()]

    def dti(app_id, rate):
        log["dti"] += 1
        return DTI_TEXT

    return Toolbox(get_app, search, dti), log


FULL = [
    turn("", call("get_application", app_id=APP)),
    turn("", call("search_policy", query="CP-01 chính sách cho vay")),
    turn("", call("calc_dti", app_id=APP, annual_rate="0.17")),
]


class AgentTestBase(unittest.TestCase):
    def setUp(self):
        tools.reset_session_state()
        self.addCleanup(tools.reset_session_state)
        self.addCleanup(tools.set_critic, None)

    def make(self, *turns, max_steps=6):
        llm = ScriptedLlm(*turns)
        box, log = toolbox()
        agent = CriticAgent(step_fn=llm, toolbox=box, prompt="critic", max_steps=max_steps)
        tools.set_critic(agent)
        return agent, llm, log

    def gather_actor(self):
        tools.get_application(APP)
        critic_mod.record_retrieval(APP, [hit("CP-01"), hit("CP-02"), hit("CP-03")], DATE)
        tools.calc_dti(APP, "0.17")

    def submit(self, decision="reject", codes=("DTI_EXCEEDED",),
               cited=("CP-01 v1", "CP-02 v1", "CP-03 v1", "CP-05 v1"), reason="DTI 63,04% vượt trần"):
        return tools.submit_decision(APP, decision, reason, list(cited), list(codes))


class TestIsolationAndTools(AgentTestBase):
    def test_exposes_only_read_tools(self):
        self.assertEqual({s.name for s in TOOL_SPECS}, set(READ_ONLY_TOOLS))
        self.assertNotIn("submit_decision", READ_ONLY_TOOLS)

    def test_critic_investigates_independently_and_passes(self):
        agent, llm, log = self.make(*FULL, final())
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertTrue(out.startswith("Thành công"), out)
        self.assertEqual(log["app"], 1)
        self.assertEqual(log["dti"], 1)
        self.assertEqual(log["search"][0][1], DATE)   # tự áp ngày nộp hồ sơ
        self.assertEqual(agent.last_run.outcome, "pass")
        self.assertEqual(agent.last_run.calls, 4)

    def test_critic_does_not_receive_actor_evidence(self):
        agent, llm, _ = self.make(*FULL, final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.submit()
        first_user = llm.messages_seen[0][1]["content"]
        self.assertNotIn("Trần DTI", first_user)
        self.assertNotIn("Tuổi: 40", first_user)
        self.assertIn("decision: reject", first_user)

    def test_critic_does_not_touch_actor_session_state(self):
        _, _, _ = self.make(*FULL, final())
        with isolated_db_sandbox():
            tools.calc_dti("APP-004", "0.17")           # Actor đã DTI hồ sơ khác
            self.gather_actor()
            before = critic_mod.get_evidence(APP).dti_result_raw
            active = tools._ACTIVE_APP_ID
            self.submit()
        self.assertEqual(active, APP)
        self.assertTrue(before)

    def test_search_requires_application_first(self):
        agent, _, log = self.make(turn("", call("search_policy", query="CP-01")), final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.submit()
        self.assertEqual(log["search"], [])
        self.assertIn("get_application", agent.last_run.steps[0]["result"])

    def test_unknown_and_write_tools_are_refused(self):
        agent, _, _ = self.make(turn("", call("submit_decision", app_id=APP)), *FULL, final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.submit()
        self.assertIn("không tồn tại", agent.last_run.steps[0]["result"])

    def test_cannot_inspect_other_application(self):
        agent, _, log = self.make(turn("", call("get_application", app_id="APP-009")), *FULL, final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.submit()
        self.assertIn("chỉ được thẩm tra", agent.last_run.steps[0]["result"])
        self.assertEqual(log["app"], 1)

    def test_repeated_call_is_not_executed_twice(self):
        agent, _, log = self.make(
            turn("", call("get_application", app_id=APP)),
            turn("", call("get_application", app_id=APP)),
            *FULL[1:], final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.submit()
        self.assertEqual(log["app"], 1)


class TestVerdicts(AgentTestBase):
    def test_grounded_disagreement_blocks(self):
        bad = {"expected_decision": "approve", "expected_reason_codes": ["APPROVED"], "reason_consistent": True,
               "issues": [{"detail": "DTI đạt trần", "evidence_quote": "Trần DTI tối đa là 50%"}]}
        self.make(*FULL, final(bad))
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertIn("CRITIC_REJECT [critic] lần 1/2", out)
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))

    def test_quote_from_actor_only_evidence_is_not_grounded(self):
        # Câu này chỉ có trong bằng chứng của Actor, critic chưa từng nhận từ tool của mình.
        bad = {"expected_decision": "approve", "expected_reason_codes": ["APPROVED"], "reason_consistent": True,
               "issues": [{"detail": "bịa", "evidence_quote": "câu này critic chưa hề thấy từ công cụ"}]}
        self.make(*FULL, final(bad))
        with isolated_db_sandbox():
            self.gather_actor()
            self.assertTrue(self.submit().startswith("Thành công"))

    def test_tier1_blocks_without_spending_critic_tokens(self):
        agent, llm, _ = self.make(final())
        with isolated_db_sandbox():
            tools.calc_dti(APP, "0.17")   # chưa get_application / search_policy
            out = self.submit()
        self.assertIn("CRITIC_REJECT [evidence]", out)
        self.assertEqual(llm.tools_seen, [])

    def test_step_budget_forces_final_answer_without_tools(self):
        agent, llm, _ = self.make(
            turn("", call("get_application", app_id=APP)),
            turn("", call("calc_dti", app_id=APP, annual_rate="0.17")),
            turn("", call("search_policy", query="a")),
            final(), max_steps=4)
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertTrue(out.startswith("Thành công"), out)
        self.assertIsNone(llm.tools_seen[-1])        # bước cuối không còn tool
        self.assertEqual(agent.last_run.calls, 4)

    def test_never_concluding_fails_closed(self):
        self.make(turn("", call("get_application", app_id=APP)), max_steps=3)
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertIn("CRITIC_REJECT [error]", out)
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))

    def test_llm_failure_fails_closed(self):
        self.make(RuntimeError("API sập"))
        with isolated_db_sandbox():
            self.gather_actor()
            self.assertIn("CRITIC_REJECT [error]", self.submit())

    def test_garbage_json_is_reprompted_once(self):
        self.make(*FULL, turn("không phải json"), final())
        with isolated_db_sandbox():
            self.gather_actor()
            self.assertTrue(self.submit().startswith("Thành công"))

    def test_tool_exception_does_not_kill_the_loop(self):
        llm = ScriptedLlm(turn("", call("get_application", app_id=APP)), final())

        def boom(app_id):
            raise OSError("db hỏng")

        box = Toolbox(boom, lambda q, a, k: ("", []), lambda a, r: "")
        tools.set_critic(CriticAgent(step_fn=llm, toolbox=box, prompt="critic"))
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
        self.assertTrue(out.startswith("Thành công") or "CRITIC_REJECT" in out, out)


class TestDtiIsSelfComputed(AgentTestBase):
    def test_conclusion_without_calc_dti_is_sent_back(self):
        agent, llm, log = self.make(
            turn("", call("get_application", app_id=APP)),
            turn("", call("search_policy", query="CP-01")),
            final(),                                               # gật đầu khi chưa tự tính DTI
            turn("", call("calc_dti", app_id=APP, annual_rate="0.17")),
            final())
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertTrue(out.startswith("Thành công"), out)
        self.assertEqual(log["dti"], 1)
        self.assertIn("PHẢI tự gọi calc_dti", llm.messages_seen[3][-1]["content"])

    def test_never_computing_dti_fails_closed(self):
        self.make(turn("", call("get_application", app_id=APP)), final())
        with isolated_db_sandbox():
            self.gather_actor()
            out = self.submit()
            self.assertIn("CRITIC_REJECT [error]", out)
            self.assertIn("chưa tự tính DTI", out)

    def test_only_dti_dependent_decisions_require_it(self):
        P = critic_mod.Proposal
        need = CriticAgent._needs_dti
        self.assertTrue(need(P(APP, "approve", "", [], ["APPROVED"])))
        self.assertTrue(need(P(APP, "escalate", "", [], ["ESCALATE_MANAGER"])))
        self.assertTrue(need(P(APP, "reject", "", [], ["dti_exceeded"])))
        self.assertFalse(need(P(APP, "reject", "", [], ["AGE"])))
        self.assertFalse(need(P(APP, "need_info", "", [], ["MISSING_DOCS"])))


class TestPureHelpers(unittest.TestCase):
    def test_compute_dti_matches_calc_dti_and_has_no_side_effects(self):
        from finco.store import db
        tools.reset_session_state()
        with isolated_db_sandbox():
            app = db.get_application(APP)
            pure = tools.compute_dti(app, "0.17")
            self.assertNotIn(APP, tools._DTI_DONE)
            self.assertEqual(critic_mod.get_evidence(APP).dti_result_raw, "")
            self.assertEqual(tools.calc_dti(APP, "0.17"), pure)
            self.assertIn(APP, tools._DTI_DONE)
        tools.reset_session_state()

    def test_render_application_matches_get_application(self):
        from finco.store import db
        with isolated_db_sandbox():
            self.assertEqual(tools.get_application(APP), tools.render_application(db.get_application(APP)))
        tools.reset_session_state()


if __name__ == "__main__":
    unittest.main()
