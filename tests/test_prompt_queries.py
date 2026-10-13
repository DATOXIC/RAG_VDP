"""Các truy vấn gợi ý trong prompts/agent_system.txt phải thật sự truy xuất ra đúng chính sách gốc.

Prompt và tầng truy xuất là hai thứ do hai người sửa: đổi tài liệu hoặc chunking có thể làm các
truy vấn gợi ý trượt sang tài liệu diễn giải (CP-15/CP-16) mà không ai hay. Test này bắt lỗi đó.
Chạy offline (embedding cục bộ), không gọi LLM.

    python -m unittest tests.test_prompt_queries -v
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from finco.llm import load_prompt  # noqa: E402
from finco.rag.retriever import retrieve  # noqa: E402

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


def suggested_queries() -> list[str]:
    text = load_prompt("agent_system.txt")
    queries = []
    for line in text.splitlines():
        if "Truy vấn gợi ý:" in line:
            queries += re.findall(r'"([^"]+)"', line)
    return queries


class TestSuggestedQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = suggested_queries()

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
