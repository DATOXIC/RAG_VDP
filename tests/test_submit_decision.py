"""Test tool submit_decision + get_decision với reason_codes, trên DB sandbox (không đụng data/finco.db).

Chạy từ thư mục gốc repo:
    python -m unittest tests.test_submit_decision -v
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.sandbox_db import isolated_db_sandbox  # noqa: E402
from finco import tools  # noqa: E402


class SubmitDecisionReasonCodes(unittest.TestCase):
    def setUp(self):
        tools._DTI_DONE.clear()
        self.addCleanup(tools._DTI_DONE.clear)

    def submit(self, app_id="APP-003", decision="reject", codes=("DTI_EXCEEDED",)):
        return tools.submit_decision(app_id, decision, "DTI vượt ngưỡng", ["CP-01 v1"], list(codes))

    def test_roundtrip(self):
        with isolated_db_sandbox():
            tools.calc_dti("APP-003", "0.17")
            self.assertTrue(self.submit().startswith("Thành công"))
            out = tools.get_decision("APP-003")
            self.assertIn("DTI_EXCEEDED", out)

    def test_rejects_bad_codes(self):
        with isolated_db_sandbox():
            self.assertTrue(self.submit(codes=()).startswith("Error:"))
            self.assertTrue(self.submit(codes=("NOT_A_CODE",)).startswith("Error:"))
            # mã hợp lệ nhưng sai trạng thái
            self.assertTrue(self.submit(decision="approve", codes=("DTI_EXCEEDED",)).startswith("Error:"))
            # chưa có gì được ghi
            self.assertTrue(tools.get_decision("APP-003").startswith("Error:"))

    def test_lowercase_codes_normalised(self):
        with isolated_db_sandbox():
            tools.calc_dti("APP-003", "0.17")
            self.assertTrue(self.submit(codes=("dti_exceeded",)).startswith("Thành công"))


class DtiRequiredBeforeDecision(unittest.TestCase):
    """Quyết định dựa trên DTI chỉ được ghi sau khi calc_dti thành công cho đúng hồ sơ."""

    def setUp(self):
        tools._DTI_DONE.clear()
        self.addCleanup(tools._DTI_DONE.clear)

    def submit(self, app_id, decision, codes):
        return tools.submit_decision(app_id, decision, "Lý do", ["CP-01 v1"], list(codes))

    def test_dti_based_decisions_are_blocked_without_calc_dti(self):
        with isolated_db_sandbox():
            for decision, codes in (("approve", ["APPROVED"]), ("escalate", ["ESCALATE_MANAGER"]),
                                    ("reject", ["DTI_EXCEEDED"])):
                out = self.submit("APP-003", decision, codes)
                self.assertTrue(out.startswith("Error:") and "calc_dti" in out, (decision, out))
            self.assertTrue(tools.get_decision("APP-003").startswith("Error:"))  # chưa ghi gì

    def test_calc_dti_for_another_application_does_not_count(self):
        with isolated_db_sandbox():
            tools.calc_dti("APP-001", "0.155")
            self.assertTrue(self.submit("APP-003", "reject", ["DTI_EXCEEDED"]).startswith("Error:"))

    def test_failed_calc_dti_does_not_count(self):
        with isolated_db_sandbox():
            tools.calc_dti("APP-003", "khong-phai-so")
            self.assertTrue(self.submit("APP-003", "reject", ["DTI_EXCEEDED"]).startswith("Error:"))

    def test_decisions_before_dti_step_need_no_calc_dti(self):
        with isolated_db_sandbox():
            self.assertTrue(self.submit("APP-003", "reject", ["OVERDUE"]).startswith("Thành công"))

class TestSmartTools(unittest.TestCase):
    """Kiểm tra các tiện ích thông minh trong tools.py: tự động gán ngày nộp và căn cứ CP-05."""

    def setUp(self):
        tools._DTI_DONE.clear()
        tools._CURRENT_APP_DATE.clear()
        tools._ACTIVE_APP_ID = None
        self.addCleanup(tools._DTI_DONE.clear)
        self.addCleanup(tools._CURRENT_APP_DATE.clear)

    def test_calc_dti_includes_cp05_reference(self):
        with isolated_db_sandbox():
            res = tools.calc_dti("APP-003", "0.17")
            self.assertIn("CP-05 v1", res)

    def test_search_policy_auto_injects_submission_date(self):
        with isolated_db_sandbox():
            # Chưa get_application -> search_policy không auto-fill
            self.assertIsNone(tools._ACTIVE_APP_ID)

            # Đọc hồ sơ APP-001 (ngày nộp: 2025-02-10)
            tools.get_application("APP-001")
            self.assertEqual(tools._ACTIVE_APP_ID, "APP-001")
            self.assertEqual(tools._CURRENT_APP_DATE.get("APP-001"), "2025-02-10")

            # search_policy không truyền as_of_date -> tự động dùng ngày của APP-001 (2025-02-10, thuộc CP-01 v1)
            out = tools.search_policy("điều kiện cho vay", as_of_date=None)
            self.assertIn("CP-01 v1", out)
            self.assertIn("50%", out)  # CP-01 v1 quy định DTI tối đa 50%

            # Truyền "None" dạng chuỗi -> cũng được xử lý tự động điền ngày (trả về CP-01 v1)
            out_str_none = tools.search_policy("điều kiện cho vay", as_of_date="None")
            self.assertIn("CP-01 v1", out_str_none)


if __name__ == "__main__":
    unittest.main()


