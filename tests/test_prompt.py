"""Kiểm tra prompts/agent_system.txt: đúng thiết kế, và các truy vấn gợi ý thật sự ra đúng chính sách gốc.

Prompt và tầng truy xuất do hai người sửa: đổi tài liệu hay chunking có thể làm truy vấn gợi ý trượt sang
tài liệu diễn giải (CP-15/CP-16) mà không ai hay. Chạy offline (embedding cục bộ), không gọi LLM.

    python -m unittest tests.test_prompt -v
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.scoring import REFUSAL_MARKER  # noqa: E402
from finco.llm import load_prompt  # noqa: E402
from finco.rag.retriever import retrieve  # noqa: E402
from finco.types import REASON_CODES  # noqa: E402

TOOLS = ("search_policy", "get_application", "calc_dti", "submit_decision", "get_decision")

# (đoạn nhận diện truy vấn, ngày nộp, tài liệu phải có trong kết quả top_k mặc định của tool)
EXPECTATIONS = [
    ("Chính sách cho vay", "2025-02-10", "CP-01 v1"),
    ("Chính sách cho vay", "2025-05-10", "CP-01 v2"),
    ("Quy định chứng từ", "2025-05-10", "CP-02 v1"),
    ("Biểu lãi suất", "2025-05-10", "CP-03 v1"),
    ("Phân cấp phê duyệt", "2025-05-10", "CP-04 v1"),
    ("nhân viên FinCo", "2025-06-20", "CP-06 v1"),
    ("nợ quá hạn", "2025-08-01", "CP-06 v1"),
]


class TestPromptDesign(unittest.TestCase):
    """Prompt đúng thiết kế: function calling, không chứa số liệu chính sách."""

    @classmethod
    def setUpClass(cls):
        cls.text = load_prompt("agent_system.txt")

    def test_missing_prompt_file(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("khong_ton_tai.txt")

    def test_no_hardcoded_policy_numbers(self):
        self.assertIsNone(re.search(r"\d+\s*(%|triệu|tháng|tuổi)", self.text), "prompt chứa con số chính sách")

    def test_uses_function_calling_not_text_actions(self):
        self.assertIn("function calling", self.text)
        self.assertIn("thought", self.text)
        self.assertNotIn("Action:", self.text)
        self.assertNotIn("Observation:", self.text)

    def test_mentions_every_tool_and_reason_code(self):
        for name in TOOLS + tuple(REASON_CODES):
            self.assertIn(name, self.text)

    def test_refusal_sentence_matches_eval_marker(self):
        self.assertIn(REFUSAL_MARKER, self.text)


class TestSuggestedQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = []
        for line in load_prompt("agent_system.txt").splitlines():
            if "Truy vấn gợi ý:" in line:
                cls.queries += re.findall(r'"([^"]+)"', line)

    def test_prompt_has_suggested_queries(self):
        self.assertGreaterEqual(len(self.queries), 6)

    def test_every_suggested_query_reaches_its_source_document(self):
        for fragment, as_of, expected_doc in EXPECTATIONS:
            matching = [q for q in self.queries if fragment in q]
            self.assertTrue(matching, f"prompt không có truy vấn gợi ý chứa {fragment!r}")
            for q in matching:
                out = retrieve(q, as_of_date=as_of, top_k=3)
                heads = re.findall(r"^\[(CP-\d+ v\d+) ·", out, flags=re.MULTILINE)
                self.assertIn(expected_doc, heads, f"{q!r} @ {as_of} chỉ trả {heads}")


if __name__ == "__main__":
    unittest.main()
