"""Test chấm điểm của eval/run_eval.py: hoàn toàn offline (không agent, không LLM, không MCP).

Chạy từ thư mục gốc repo:
    python -m unittest tests.test_run_eval -v
"""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval import run_eval as ev  # noqa: E402
from finco.agent import save_trace_to_jsonl  # noqa: E402
from finco.types import LogLine, TestCase, Usage  # noqa: E402

EXPECTED = ev.load_expected()


def step(tool, args=None, result="ok", qid="Q1", n=1):
    return {"ts": "t", "run_id": "r", "question_id": qid, "step": n, "tool": tool,
            "args": args or {}, "result": result,
            "usage": {"calls": n, "prompt_tokens": 10 * n, "completion_tokens": n, "max_prompt": 10 * n},
            "cache_hit": False}


def record(trace=(), answer="", success=True, calls=3, qid="Q1"):
    return ev.RunRecord(
        question_id=qid, trace=list(trace), final_answer=answer,
        usage=Usage(calls=calls, prompt_tokens=300, completion_tokens=30, max_prompt=150),
        success=success, complete=True,
    )


def qa_case(**kw):
    base = dict(id="Q1", question="?", type="tra cứu", expected_answer="Phí **2%** trên dư nợ gốc.",
                expected_docs=["CP-03"], is_no_answer=False)
    base.update(kw)
    return TestCase(**base)


SEARCH_CP03 = "[CP-03 v1 · 2024-01-01]\n4.1. Trả nợ trước hạn trong 12 tháng đầu: phí 2% trên dư nợ gốc."


class TestNormalize(unittest.TestCase):
    def test_thousand_separators_and_decimals(self):
        self.assertEqual(ev.normalize("6.417.491 đồng"), ev.normalize("6,417,491 đồng"))
        self.assertIn("6417491", ev.normalize("PMT = 6.417.491 VNĐ"))
        self.assertEqual(ev.normalize("15,5 %"), "15.5%")
        self.assertIn(ev.normalize("24,6"), ev.normalize("DTI = 24,62%"))

    def test_decimal_is_not_mistaken_for_thousands(self):
        self.assertIn("0.2462", ev.normalize("DTI 0,2462"))

    def test_key_facts(self):
        self.assertEqual(ev.key_facts("Gồm **CCCD**, **HDLD** và sao kê."), ["CCCD", "HDLD"])
        self.assertEqual(ev.key_facts("không đánh dấu"), [])

    def test_citations_in_text(self):
        got = ev.cited_in_text("Theo [CP-01 v2] và CP-05, cũng xem cp-03_v1.")
        self.assertEqual(got, ["CP-01@v2", "CP-05", "CP-03@v1"])


class TestRefusal(unittest.TestCase):
    def test_marker_is_a_refusal(self):
        self.assertTrue(ev.is_refusal(ev.REFUSAL_MARKER + ". Đã tra CP-01."))

    def test_lenient_only_for_no_answer(self):
        text = "Tài liệu không đề cập đến vay mua ô tô."
        self.assertFalse(ev.is_refusal(text))
        self.assertTrue(ev.is_refusal(text, lenient=True))


class TestScoreQuestions(unittest.TestCase):
    def test_no_answer_correct_refusal(self):
        case = qa_case(id="N1", type="no_answer", is_no_answer=True, expected_docs=[], expected_answer="từ chối")
        s = ev.score_question(case, record([step("search_policy", result="Không tìm thấy")],
                                           answer=ev.REFUSAL_MARKER + "."))
        self.assertTrue(s.passed and s.refused)

    def test_no_answer_hallucinated_answer_fails(self):
        case = qa_case(id="N1", type="no_answer", is_no_answer=True, expected_docs=[], expected_answer="từ chối")
        s = ev.score_question(case, record(answer="Lãi suất vay mua nhà là 8%/năm."))
        self.assertFalse(s.passed)
        self.assertIn("không từ chối", s.notes)

    def test_no_answer_that_writes_a_decision_fails(self):
        case = qa_case(id="N1", type="no_answer", is_no_answer=True, expected_docs=[], expected_answer="từ chối")
        trace = [step("submit_decision", {"app_id": "APP-001"}, "Thành công: đã ghi")]
        s = ev.score_question(case, record(trace, answer=ev.REFUSAL_MARKER))
        self.assertFalse(s.passed)

    def test_answerable_pass(self):
        s = ev.score_question(
            qa_case(), record([step("search_policy", result=SEARCH_CP03)], answer="Phí 2 % trên dư nợ gốc [CP-03 v1].")
        )
        self.assertTrue(s.passed, s.notes)
        self.assertTrue(s.facts_ok and s.citation_ok and s.grounded)

    def test_missing_fact_fails(self):
        s = ev.score_question(qa_case(), record([step("search_policy", result=SEARCH_CP03)],
                                                answer="Có thu phí trả nợ trước hạn [CP-03]."))
        self.assertFalse(s.passed)
        self.assertIn("thiếu ý", s.notes)

    def test_wrong_version_citation_fails(self):
        case = qa_case(id="V1", expected_answer="Tối đa **60 tuổi**.", expected_docs=["CP-01@v1"])
        trace = [step("search_policy", result="[CP-01 v2 · 2025-04-01] tuổi 22 đến 58")]
        s = ev.score_question(case, record(trace, answer="Tối đa 60 tuổi theo [CP-01 v2]."))
        self.assertFalse(s.passed)
        self.assertFalse(s.citation_ok)

    def test_false_refusal_fails(self):
        s = ev.score_question(qa_case(), record(answer=ev.REFUSAL_MARKER + ". 2% [CP-03]"))
        self.assertFalse(s.passed)
        self.assertIn("từ chối nhầm", s.notes)

    def test_fabricated_citation_is_flagged(self):
        trace = [step("search_policy", result=SEARCH_CP03)]
        s = ev.score_question(qa_case(), record(trace, answer="Phí 2% theo [CP-03] và [CP-99]."))
        self.assertFalse(s.grounded)
        self.assertIn("CP-99", s.notes)

    def test_failed_run_never_passes(self):
        s = ev.score_question(qa_case(), record(answer="Phí 2% [CP-03]", success=False))
        self.assertFalse(s.passed)

    def test_incomplete_log_is_reported(self):
        rec = ev.RunRecord(question_id="Q1", trace=[], complete=False)
        s = ev.score_question(qa_case(), rec)
        self.assertFalse(s.passed)
        self.assertIn("__answer__", s.error)

    def test_tool_errors_are_counted(self):
        trace = [step("get_application", result="Error: không tìm thấy"), step("search_policy", result=SEARCH_CP03)]
        s = ev.score_question(qa_case(), record(trace, answer="2% [CP-03]"))
        self.assertEqual((s.tool_calls, s.tool_errors), (2, 1))


class TestActionScoring(unittest.TestCase):
    def action_case(self, app_id="APP-003", decision="reject"):
        return TestCase(id="A1", question="Thẩm định", type="hành động", expected_answer="", expected_docs=[],
                        expected_decision=decision, target_app_id=app_id)

    def perfect_trace(self):
        exp = EXPECTED["APP-003"]
        dti = f"- Tỷ lệ nợ trên thu nhập (DTI): {exp.dti} ({exp.dti * 100:.2f}%)"
        return [
            step("get_application", {"app_id": "APP-003"}, "Mã hồ sơ: APP-003"),
            step("search_policy", {"query": "x", "as_of_date": "2025-02-20"}, "[CP-01 v1 · 2024-01-01] DTI tối đa 50%"),
            step("calc_dti", {"app_id": "APP-003", "annual_rate": str(exp.annual_rate)}, dti),
            step("submit_decision", {
                "app_id": "APP-003", "decision": "reject", "reason": "DTI vượt trần",
                "cited_docs": list(exp.cited), "reason_codes": ["DTI_EXCEEDED"],
            }, "Thành công: đã ghi"),
            step("get_decision", {"app_id": "APP-003"}, "=== QUYẾT ĐỊNH ==="),
        ]

    def test_perfect_run_passes(self):
        s = ev.score_question(self.action_case(), record(self.perfect_trace(), answer="Từ chối vì DTI vượt trần."),
                              EXPECTED["APP-003"])
        self.assertTrue(s.passed, s.notes)
        self.assertTrue(s.decision_ok and s.root_cause_ok and s.verify_ok and s.numbers_ok)
        self.assertTrue(s.wrote_decision)

    def test_wrong_decision_fails(self):
        trace = self.perfect_trace()
        trace[3]["args"]["decision"] = "approve"
        s = ev.score_question(self.action_case(), record(trace, answer="Duyệt."), EXPECTED["APP-003"])
        self.assertFalse(s.passed)
        self.assertFalse(s.decision_ok)

    def test_missing_read_back_fails(self):
        s = ev.score_question(self.action_case(), record(self.perfect_trace()[:-1], answer="Xong."),
                              EXPECTED["APP-003"])
        self.assertFalse(s.passed)
        self.assertFalse(s.verify_ok)

    def test_answer_line_in_trace_is_ignored(self):
        trace = self.perfect_trace() + [step(ev.ANSWER_TOOL, result="Xong")]
        s = ev.score_question(self.action_case(), record(trace, answer="Xong"), EXPECTED["APP-003"])
        self.assertTrue(s.passed, s.notes)


class TestLogRoundTrip(unittest.TestCase):
    def test_records_from_log(self):
        usage = Usage(calls=2, prompt_tokens=200, completion_tokens=20, max_prompt=120)
        lines = [
            LogLine("t", "r1", "T01", 1, "search_policy", {"query": "x"}, SEARCH_CP03, Usage(1, 100, 10, 100)),
            LogLine("t", "r1", "T01", 1, ev.ANSWER_TOOL, {"success": True, "error": None}, "Phí 2% [CP-03]", usage),
            LogLine("t", "r2", "N01", 0, ev.ANSWER_TOOL, {"success": False, "error": "Lỗi gọi LLM"}, "", usage),
            LogLine("t", "r3", "S01", 1, "search_policy", {}, "x", usage),  # chạy dở: không có dòng __answer__
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            save_trace_to_jsonl(lines, path)
            with open(path, "a", encoding="utf-8") as f:
                f.write("{dòng hỏng\n")
            recs = ev.records_from_log(path)
        self.assertEqual(set(recs), {"T01", "N01", "S01"})
        self.assertTrue(recs["T01"].success and recs["T01"].complete)
        self.assertEqual(recs["T01"].usage, usage)
        self.assertEqual(len(recs["T01"].trace), 1)
        self.assertFalse(recs["N01"].success)
        self.assertEqual(recs["N01"].error, "Lỗi gọi LLM")
        self.assertFalse(recs["S01"].complete)

    def test_score_only_end_to_end(self):
        usage = Usage(calls=1, prompt_tokens=100, completion_tokens=10, max_prompt=100)
        lines = [
            LogLine("t", "r", "T02", 1, "search_policy", {}, SEARCH_CP03, usage),
            LogLine("t", "r", "T02", 1, ev.ANSWER_TOOL, {"success": True, "error": None}, "2% trên nợ gốc [CP-03 v1]", usage),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "eval.jsonl"
            save_trace_to_jsonl(lines, log)
            out = Path(tmp) / "out"
            buf = StringIO()
            with redirect_stdout(buf):
                code = ev.main(["--score-only", str(log), "--ids", "T02", "--out-dir", str(out)])
            self.assertEqual(code, 0, buf.getvalue())
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["questions"], summary["passed"]), (1, 1))
            self.assertTrue((out / "per_question.csv").exists())


class TestTestsetFile(unittest.TestCase):
    """tests/testset.jsonl phải thỏa yêu cầu của đề và khớp dữ liệu thật."""

    @classmethod
    def setUpClass(cls):
        cls.cases = ev.load_cases(ev.TESTSET_PATH)

    def test_size_and_no_answer_quota(self):
        self.assertGreaterEqual(len(self.cases), 12)
        self.assertGreaterEqual(sum(c.is_no_answer for c in self.cases), 4)

    def test_ids_unique(self):
        ids = [c.id for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_action_cases_have_ground_truth(self):
        for c in self.cases:
            if c.expected_decision:
                self.assertIn(c.target_app_id, EXPECTED, c.id)
                self.assertEqual(c.expected_decision, EXPECTED[c.target_app_id].decision, c.id)

    def test_answerable_cases_have_checkable_facts(self):
        for c in self.cases:
            if not c.is_no_answer and not c.expected_decision:
                self.assertTrue(ev.key_facts(c.expected_answer), f"{c.id} chưa đánh dấu ý chính **...**")
                self.assertTrue(c.expected_docs, f"{c.id} chưa có expected_docs")

    def test_no_answer_cases_expect_nothing(self):
        for c in self.cases:
            if c.is_no_answer:
                self.assertEqual(c.expected_docs, [], c.id)
                self.assertIsNone(c.expected_decision, c.id)

    def test_target_apps_exist(self):
        with open(ev.APPLICATIONS_PATH, encoding="utf-8") as f:
            apps = {a["app_id"] for a in json.load(f)}
        for c in self.cases:
            if c.target_app_id:
                self.assertIn(c.target_app_id, apps, c.id)

    def test_expected_facts_match_ground_truth_numbers(self):
        """Các con số đặt trong expected_answer khớp expected_decisions.json (chống lệch khi đổi dữ liệu)."""
        by_id = {c.id: c for c in self.cases}
        c01 = by_id["C01"]
        self.assertIn(ev.normalize(ev.key_facts(c01.expected_answer)[0]), ev.normalize(f"{EXPECTED['APP-001'].dti * 100:.2f}"))
        c02 = by_id["C02"]
        self.assertEqual(ev.key_facts(c02.expected_answer), [str(EXPECTED["APP-014"].monthly_payment)])


if __name__ == "__main__":
    unittest.main()


def test_cited_in_text_accepts_unicode_hyphens():
    """Model hay viết 'CP‑02' bằng U+2011; bộ chấm phải coi như 'CP-02'."""
    from eval.run_eval import cited_in_text

    assert cited_in_text("quy định trong CP‑02 v1 và CP–01") == ["CP-02@v1", "CP-01"]
