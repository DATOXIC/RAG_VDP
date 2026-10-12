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
    def submit(self, app_id="APP-003", decision="reject", codes=("DTI_EXCEEDED",)):
        return tools.submit_decision(app_id, decision, "DTI vượt ngưỡng", ["CP-01 v1"], list(codes))

    def test_roundtrip(self):
        with isolated_db_sandbox():
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
            self.assertTrue(self.submit(codes=("dti_exceeded",)).startswith("Thành công"))


if __name__ == "__main__":
    unittest.main()
