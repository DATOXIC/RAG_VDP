"""Test cho eval.verifier (cầu nối trace -> AgentSubmission, trích dẫn, đơn vị) và phần parse của eval.judge.

Chạy từ thư mục gốc repo:
    python -m unittest tests.test_verifier -v

Không gọi mạng, không cần LLM.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import (  # noqa: E402
    ExpectedDecision,
    normalize_doc_id,
    submission_from_trace,
    verify_citations,
    verify_decision_record,
    verify_numeric_math,
    verify_root_cause,
)

DTI_RESULT = (
    "=== KẾT QUẢ TÍNH DTI HỒ SƠ APP-015 ===\n"
    "- Tỷ lệ nợ trên thu nhập (DTI): 0.4709 (47.09%)\n"
)


def step(tool, args, result):
    return {"tool": tool, "args": args, "result": result}


def good_trace(app_id="APP-015"):
    return [
        step("get_application", {"app_id": app_id}, "Mã hồ sơ: " + app_id),
        step("search_policy", {"query": "dti", "as_of_date": "2025-05-10"}, "[CP-01 v2] ..."),
        step("calc_dti", {"app_id": app_id, "annual_rate": "0.17"}, DTI_RESULT),
        step(
            "submit_decision",
            {"app_id": app_id, "decision": "Reject ", "reason": "DTI vượt 45%", "cited_docs": ["CP-01 v2", "CP-03"],
             "reason_codes": ["dti_exceeded"]},
            "Thành công: Đã ghi nhận quyết định",
        ),
        step("get_decision", {"app_id": app_id}, "=== QUYẾT ĐỊNH ĐANG LƯU ==="),
    ]


EXPECT = ExpectedDecision(
    app_id="APP-015", decision="reject", policy_version="v2", reasons=["DTI_EXCEEDED"],
    cited=["CP-01@v2", "CP-03"], annual_rate=0.17, monthly_payment=0, qualified_income=0,
    dti=0.4709, max_dti=0.45,
)


class SubmissionFromTrace(unittest.TestCase):
    def test_happy_path(self):
        sub = submission_from_trace("app-015", good_trace())
        self.assertEqual(sub.decision, "reject")
        self.assertEqual(sub.cited_docs, ["CP-01 v2", "CP-03"])
        self.assertAlmostEqual(sub.calculated_dti, 0.4709)
        self.assertAlmostEqual(sub.used_rate, 0.17)
        res = verify_decision_record(sub, EXPECT)
        self.assertTrue(res.is_overall_pass, res.error_message)

    def test_failed_submit_is_not_a_decision(self):
        trace = good_trace()
        trace[3]["result"] = "Error: cited_docs không được rỗng."
        sub = submission_from_trace("APP-015", trace)
        self.assertEqual(sub.decision, "NONE")
        self.assertFalse(verify_decision_record(sub, EXPECT).trace_valid)

    def test_verify_must_come_after_submit_and_match_app(self):
        trace = good_trace()
        trace[-1]["args"] = {"app_id": "APP-999"}
        self.assertFalse(verify_decision_record(submission_from_trace("APP-015", trace), EXPECT).read_after_write)

        trace = good_trace()
        trace.insert(0, trace.pop())  # get_decision đứng trước submit
        self.assertFalse(verify_decision_record(submission_from_trace("APP-015", trace), EXPECT).read_after_write)

    def test_rate_as_percent_string(self):
        trace = good_trace()
        trace[2]["args"]["annual_rate"] = "17"
        self.assertAlmostEqual(submission_from_trace("APP-015", trace).used_rate, 0.17)

    def test_garbage_steps_do_not_crash(self):
        sub = submission_from_trace("APP-015", [{}, {"tool": None}, {"tool": "calc_dti", "args": "x", "result": None}])
        self.assertEqual(sub.decision, "NONE")


class RootCause(unittest.TestCase):
    def test_from_trace_and_case_insensitive(self):
        sub = submission_from_trace("APP-015", good_trace())
        self.assertEqual(sub.reason_codes, ["DTI_EXCEEDED"])
        self.assertTrue(verify_decision_record(sub, EXPECT).root_cause_correct)

    def test_wrong_or_missing_code_fails_overall(self):
        for codes in (["AGE"], []):
            trace = good_trace()
            trace[3]["args"]["reason_codes"] = codes
            res = verify_decision_record(submission_from_trace("APP-015", trace), EXPECT)
            self.assertFalse(res.root_cause_correct)
            self.assertFalse(res.is_overall_pass)
            self.assertIn("mã lý do", res.error_message)

    def test_extra_codes_not_penalised_and_string_accepted(self):
        self.assertTrue(verify_root_cause(["AGE", "DTI_EXCEEDED"], ["DTI_EXCEEDED"])["root_cause_correct"])
        self.assertTrue(verify_root_cause("dti_exceeded", ["DTI_EXCEEDED"])["root_cause_correct"])

    def test_no_expected_code_never_passes(self):
        self.assertFalse(verify_root_cause(["APPROVED"], [])["root_cause_correct"])


class Citations(unittest.TestCase):
    def test_normalize_variants(self):
        for raw in ("CP-01 v2", "CP-01@v2", "cp-01_v2", "CP-01 (v2)", "CP-01v2"):
            self.assertEqual(normalize_doc_id(raw), "CP-01@v2", raw)
        self.assertEqual(normalize_doc_id("CP-02"), "CP-02")

    def test_wrong_version_is_a_miss(self):
        r = verify_citations(["CP-01 v1", "CP-03"], ["CP-01@v2", "CP-03"])
        self.assertEqual(r["citation_recall"], 0.5)
        self.assertEqual(r["citation_precision"], 0.5)
        self.assertFalse(r["citation_valid"])

    def test_unversioned_expectation_accepts_any_version(self):
        r = verify_citations(["CP-02 v1"], ["CP-02"])
        self.assertTrue(r["citation_valid"])

    def test_missing_version_when_expected(self):
        self.assertFalse(verify_citations(["CP-01"], ["CP-01@v2"])["citation_valid"])


class Numeric(unittest.TestCase):
    def test_percent_units(self):
        r = verify_numeric_math(47.09, 0.4709, 17.0, 0.17)
        self.assertTrue(r["dti_correct"] and r["rate_correct"])

    def test_wrong_dti(self):
        self.assertFalse(verify_numeric_math(0.40, 0.4709, 0.17, 0.17)["dti_correct"])


class JudgeParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from eval import judge
        except ImportError as e:  # thiếu openai/dotenv trong môi trường test
            raise unittest.SkipTest(f"không import được eval.judge: {e}")
        cls.judge = judge

    def test_string_false_is_false(self):
        self.assertFalse(self.judge._require_bool({"k": "false"}, "k"))
        self.assertTrue(self.judge._require_bool({"k": True}, "k"))

    def test_bad_bool_raises(self):
        for data in ({}, {"k": None}, {"k": "maybe"}, {"k": 1}):
            with self.assertRaises(ValueError):
                self.judge._require_bool(data, "k")

    def test_extract_json_with_prose_and_fence(self):
        self.assertEqual(self.judge._extract_json_block('Kết quả: {"a": 1} xong'), {"a": 1})
        self.assertEqual(self.judge._extract_json_block('```json\n{"a": 2}\n```'), {"a": 2})
        with self.assertRaises(Exception):
            self.judge._extract_json_block("không có json")


if __name__ == "__main__":
    unittest.main()
