"""Test Critic nhúng trong submit_decision (Mẫu 2: Tool-Embedded Critic / Reflexion).

LLM critic được thay bằng hàm giả, bằng chứng tra cứu được tiêm trực tiếp: không tốn API,
không cần index/embedding. Chạy từ thư mục gốc repo:
    python -m unittest tests.test_critic -v
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
from finco.critic import Critic  # noqa: E402
from finco.types import Chunk  # noqa: E402

APP = "APP-003"          # nộp 2025-02-20, DTI 63,04% ở lãi 17% -> reject DTI_EXCEEDED
DATE = "2025-02-20"
ESCALATE = dict(decision="escalate", reason="Chuyển cấp xem xét", reason_codes=["ESCALATE_MANAGER"])


def hit(doc_id, version="v1", text="Nội dung điều khoản."):
    c = Chunk(f"{doc_id}-{version}-01", doc_id, version, "2025-01-01", "Mục 1", f"[{doc_id} {version}]\n{text}")
    return SimpleNamespace(chunk=c)


class FakeLLM:
    """Trả lần lượt các phản hồi định sẵn (dict -> JSON, Exception -> raise)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


AGREE = {"expected_decision": "reject", "expected_reason_codes": ["DTI_EXCEEDED"],
         "reason_consistent": True, "issues": []}


class CriticTestBase(unittest.TestCase):
    def setUp(self):
        tools.reset_session_state()
        self.addCleanup(tools.reset_session_state)
        self.addCleanup(tools.set_critic, None)

    def use(self, *responses) -> FakeLLM:
        llm = FakeLLM(*responses)
        tools.set_critic(Critic(llm=llm, prompt="critic"))
        return llm

    def gather(self, docs=("CP-01", "CP-02", "CP-03"), as_of=DATE, dti=True):
        tools.get_application(APP)
        critic_mod.record_retrieval(APP, [hit(d) for d in docs], as_of)
        if dti:
            tools.calc_dti(APP, "0.17")

    def submit(self, decision="reject", codes=("DTI_EXCEEDED",), cited=("CP-01 v1", "CP-02 v1", "CP-03 v1", "CP-05 v1"),
               reason="DTI 63,04% vượt trần"):
        return tools.submit_decision(APP, decision, reason, list(cited), list(codes))


class TestEvidenceTier(CriticTestBase):
    def test_no_evidence_is_rejected_without_calling_llm(self):
        llm = self.use(AGREE)
        with isolated_db_sandbox():
            tools.calc_dti(APP, "0.17")  # qua cổng DTI nhưng chưa get_application / search_policy
            out = self.submit()
        self.assertTrue(out.startswith("Error: CRITIC_REJECT [evidence]"), out)
        self.assertEqual(llm.calls, 0)

    def test_missing_required_doc_is_named(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.gather(docs=("CP-01", "CP-03"))  # thiếu CP-02
            out = self.submit(cited=("CP-01 v1", "CP-03 v1", "CP-05 v1"))
        self.assertIn("Chưa tra cứu CP-02", out)

    def test_search_with_wrong_date_does_not_count(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.gather(as_of="2024-01-01")
            out = self.submit()
        self.assertIn("CP-01", out)
        self.assertIn("CRITIC_REJECT [evidence]", out)

    def test_citing_unseen_version_is_rejected(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.gather()
            out = self.submit(cited=("CP-01 v2", "CP-02 v1", "CP-03 v1", "CP-05 v1"))
        self.assertIn("phiên bản đã thấy của CP-01 là v1", out)

    def test_decision_stays_unwritten_after_rejection(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.submit()
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))


class TestLlmTier(CriticTestBase):
    def test_agreement_writes_decision_and_clears_evidence(self):
        llm = self.use(AGREE)
        with isolated_db_sandbox():
            self.gather()
            out = self.submit()
            self.assertTrue(out.startswith("Thành công"), out)
            self.assertIn("DTI_EXCEEDED", tools.get_decision(APP))
        self.assertEqual(llm.calls, 1)
        self.assertEqual(critic_mod.get_evidence(APP).app_data_raw, "")

    def test_grounded_disagreement_blocks(self):
        bad = {"expected_decision": "approve", "expected_reason_codes": ["APPROVED"], "reason_consistent": True,
               "issues": [{"detail": "Tuổi trong phạm vi, DTI đạt", "evidence_quote": "Tuổi: 40"}]}
        self.use(bad)
        with isolated_db_sandbox():
            self.gather()
            out = self.submit()
            self.assertIn("CRITIC_REJECT [critic] lần 1/2", out)
            self.assertIn('căn cứ: "Tuổi: 40"', out)
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))

    def test_ungrounded_disagreement_is_only_a_warning(self):
        bad = {"expected_decision": "approve", "expected_reason_codes": ["APPROVED"], "reason_consistent": True,
               "issues": [{"detail": "bịa", "evidence_quote": "câu này không hề có trong bằng chứng"}]}
        self.use(bad)
        with isolated_db_sandbox():
            self.gather()
            self.assertTrue(self.submit().startswith("Thành công"))

    def test_llm_failure_fails_closed(self):
        self.use(RuntimeError("API sập"))
        with isolated_db_sandbox():
            self.gather()
            out = self.submit()
            self.assertIn("CRITIC_REJECT [error]", out)
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))

    def test_garbage_json_is_retried_once_then_fails_closed(self):
        llm = self.use("không phải json")
        with isolated_db_sandbox():
            self.gather()
            out = self.submit()
        self.assertIn("CRITIC_REJECT [error]", out)
        self.assertEqual(llm.calls, 2)

    def test_garbage_then_valid_json_passes(self):
        self.use("rác", AGREE)
        with isolated_db_sandbox():
            self.gather()
            self.assertTrue(self.submit().startswith("Thành công"))

    def test_markdown_fenced_json_is_accepted(self):
        self.use("```json\n" + json.dumps(AGREE) + "\n```")
        with isolated_db_sandbox():
            self.gather()
            self.assertTrue(self.submit().startswith("Thành công"))


class TestCircuitBreaker(CriticTestBase):
    def trip(self):
        self.gather()
        self.assertIn("lần 1/2", self.submit(cited=("CP-01 v1",)))
        out = self.submit(cited=("CP-01 v1",))
        self.assertIn("lần 2/2", out)
        self.assertIn("escalate", out)

    def test_only_escalate_manager_is_accepted_after_breaker(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.trip()
            self.assertTrue(self.submit().startswith("Error:"))  # vẫn reject -> bị từ chối
            self.assertTrue(tools.get_decision(APP).startswith("Error:"))

            out = tools.submit_decision(APP, cited_docs=["CP-01 v1"], **ESCALATE)
            self.assertTrue(out.startswith("Thành công"), out)
            stored = tools.get_decision(APP)
            self.assertIn("ESCALATE", stored.upper())
            self.assertIn("CRITIC_UNRESOLVED", stored)

    def test_escalate_after_breaker_does_not_need_dti(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            tools.get_application(APP)  # chưa tra gì, chưa calc_dti
            for _ in range(2):  # need_info không cần DTI nên tới được Critic và bị chặn vì thiếu bằng chứng
                out = self.submit(decision="need_info", codes=("MISSING_DOCS",), cited=("CP-02 v1",))
                self.assertIn("CRITIC_REJECT", out)
            out = tools.submit_decision(APP, cited_docs=["CP-02 v1"], **ESCALATE)
            self.assertTrue(out.startswith("Thành công"), out)

    def test_breaker_is_per_application(self):
        self.use(AGREE)
        with isolated_db_sandbox():
            self.trip()
            self.assertEqual(critic_mod.get_evidence("APP-004").critic_rejections, 0)


class TestCriticOffByDefault(CriticTestBase):
    def test_tools_work_without_critic(self):
        with isolated_db_sandbox():
            tools.calc_dti(APP, "0.17")
            self.assertTrue(self.submit(cited=("CP-01 v1",)).startswith("Thành công"))


if __name__ == "__main__":
    unittest.main()
