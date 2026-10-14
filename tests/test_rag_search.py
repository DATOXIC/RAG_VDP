"""Test BM25 và hybrid (bm25.py, hybrid.py).

Chạy từ thư mục gốc repo, chỉ dùng thư viện chuẩn (unittest):
    python -m unittest discover -s tests -v

Các test dùng chunk tự dựng và embed giả nên chạy nhanh, không cần tải bge-m3.
Riêng lớp TestRealIndex đọc index thật nhưng chỉ dùng BM25.
"""

import sys
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finco.rag.bm25 import BM25Index, is_valid_at, tokenize  # noqa: E402
from finco.rag.dense import VectorStore  # noqa: E402
from finco.rag.hybrid import HybridSearcher, min_max  # noqa: E402
from finco.rag.retriever import retrieve, retrieve_hits  # noqa: E402
from finco.types import Chunk, SearchHit  # noqa: E402


def make_chunk(cid, text, eff="2024-01-01", exp=None, doc="CP-01", ver="v1"):
    return Chunk(
        chunk_id=cid, doc_id=doc, version=ver, effective_date=eff,
        section="s", text=text, expiry_date=exp,
    )


# 6 chunk: đủ nhiều để IDF của "dti" (có ở 2/6 chunk) dương.
CHUNKS = [
    make_chunk("V1", "DTI tối đa là 50%", eff="2024-01-01", exp="2025-03-31", ver="v1"),
    make_chunk("V2", "DTI tối đa là 45%", eff="2025-04-01", ver="v2"),
    make_chunk("RATE", "Lãi suất 15,5% mỗi năm cho kỳ hạn 13 đến 24 tháng", doc="CP-03"),
    make_chunk("N1", "Giờ làm việc của phòng giải ngân", doc="CP-08"),
    make_chunk("N2", "Khiếu nại được tiếp nhận qua tổng đài", doc="CP-12"),
    make_chunk("N3", "Thẻ tín dụng có hạn mức theo thu nhập", doc="CP-09"),
]
# V1 và V2 cùng vector: dense không phân biệt được hai phiên bản (như dữ liệu thật).
VECTORS = np.array(
    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]],
    dtype=np.float32,
)
VECTORS = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)


def fake_embed(texts):
    """Câu hỏi nhắc tới DTI thì giống V1/V2, ngược lại giống RATE."""
    out = [np.array([1, 0, 0, 0] if "dti" in t.lower() else [0, 1, 0, 0], dtype=np.float32) for t in texts]
    return np.stack(out)


def make_store():
    s = VectorStore()
    s.chunks = CHUNKS
    s.vectors = VECTORS
    return s


class TestTokenize(unittest.TestCase):
    def test_percent_emits_number_too(self):
        self.assertEqual(tokenize("45%"), ["45%", "45"])

    def test_decimal_kept_whole(self):
        self.assertIn("15,5%", tokenize("lãi 15,5%/năm"))
        self.assertIn("15,5", tokenize("lãi 15,5%/năm"))

    def test_markdown_and_punctuation_stripped(self):
        toks = tokenize("**45%**. [CP-01 v2 | hiệu lực]")
        self.assertIn("45%", toks)
        self.assertNotIn("**45%**.", toks)
        self.assertNotIn("|", toks)

    def test_lowercase(self):
        self.assertEqual(tokenize("DTI"), ["dti"])

    def test_nfd_equals_nfc(self):
        composed = "lãi suất"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(tokenize(decomposed), tokenize(composed))

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("  ,.  "), [])


class TestValidAt(unittest.TestCase):
    v1, v2 = CHUNKS[0], CHUNKS[1]

    def test_boundary_last_day_of_v1(self):
        self.assertTrue(is_valid_at(self.v1, "2025-03-31"))
        self.assertFalse(is_valid_at(self.v2, "2025-03-31"))

    def test_boundary_first_day_of_v2(self):
        self.assertFalse(is_valid_at(self.v1, "2025-04-01"))
        self.assertTrue(is_valid_at(self.v2, "2025-04-01"))

    def test_before_any_effective_date(self):
        self.assertFalse(is_valid_at(self.v1, "2023-12-31"))
        self.assertFalse(is_valid_at(self.v2, "2023-12-31"))

    def test_open_ended_chunk_stays_valid(self):
        self.assertTrue(is_valid_at(self.v2, "2099-01-01"))

    def test_matches_dense_valid_mask(self):
        """Hai chỗ lọc ngày (bm25 và dense) không được lệch nhau."""
        store = make_store()
        for d in ("2023-12-31", "2024-01-01", "2025-03-31", "2025-04-01", "2030-01-01"):
            expected = store._valid_mask(d).tolist()
            actual = [is_valid_at(c, d) for c in CHUNKS]
            self.assertEqual(actual, expected, msg=d)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.ix = BM25Index(CHUNKS)

    def ids(self, query, **kw):
        return [h.chunk.chunk_id for h in self.ix.search(query, **kw)]

    def test_exact_keyword_found(self):
        self.assertEqual(self.ids("lãi suất kỳ hạn", top_k=1), ["RATE"])

    def test_date_filter_selects_version(self):
        self.assertEqual(self.ids("DTI tối đa", as_of_date="2025-03-31"), ["V1"])
        self.assertEqual(self.ids("DTI tối đa", as_of_date="2025-04-01"), ["V2"])

    def test_without_filter_both_versions(self):
        self.assertEqual(set(self.ids("DTI tối đa")), {"V1", "V2"})

    def test_number_token_matches(self):
        self.assertEqual(self.ids("45%", top_k=1), ["V2"])

    def test_empty_and_unmatched_query(self):
        self.assertEqual(self.ids(""), [])
        self.assertEqual(self.ids("zzzz qqqq"), [])

    def test_top_k_zero(self):
        self.assertEqual(self.ids("DTI", top_k=0), [])

    def test_date_with_no_valid_chunk(self):
        self.assertEqual(self.ids("DTI", as_of_date="2000-01-01"), [])

    def test_hit_fields(self):
        hit = self.ix.search("DTI tối đa", top_k=1)[0]
        self.assertGreater(hit.score, 0)
        self.assertEqual(hit.score, hit.bm25_score)
        self.assertIsNone(hit.dense_score)

    def test_scores_descending(self):
        scores = [h.score for h in self.ix.search("DTI tối đa lãi suất")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_requires_chunks(self):
        with self.assertRaises(ValueError):
            BM25Index([])


class TestMinMax(unittest.TestCase):
    def test_range(self):
        np.testing.assert_allclose(min_max(np.array([1.0, 2.0, 3.0])), [0.0, 0.5, 1.0])

    def test_all_equal_positive_gives_ones(self):
        np.testing.assert_allclose(min_max(np.array([5.0, 5.0])), [1.0, 1.0])

    def test_all_zero_stays_zero(self):
        np.testing.assert_allclose(min_max(np.array([0.0, 0.0])), [0.0, 0.0])

    def test_empty(self):
        self.assertEqual(min_max(np.array([])).size, 0)


@patch("finco.rag.hybrid.embed", side_effect=fake_embed)
class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.hy = HybridSearcher(self.store, BM25Index(CHUNKS))

    def ids(self, query, **kw):
        return [h.chunk.chunk_id for h in self.hy.search(query, **kw)]

    def test_date_filter_applies(self, _):
        self.assertEqual(self.ids("DTI tối đa", top_k=2, as_of_date="2025-03-31")[0], "V1")
        self.assertNotIn("V2", self.ids("DTI tối đa", top_k=6, as_of_date="2025-03-31"))
        self.assertEqual(self.ids("DTI tối đa", top_k=1, as_of_date="2025-04-01"), ["V2"])

    def test_hit_has_all_scores(self, _):
        hit = self.hy.search("DTI tối đa", top_k=1)[0]
        self.assertIsNotNone(hit.dense_score)
        self.assertIsNotNone(hit.bm25_score)
        self.assertTrue(0.0 <= hit.score <= 1.0)

    def test_empty_query_skips_embedding(self, embed_mock):
        self.assertEqual(self.hy.search("   "), [])
        embed_mock.assert_not_called()

    def test_top_k_zero_and_empty_date(self, _):
        self.assertEqual(self.hy.search("DTI", top_k=0), [])
        self.assertEqual(self.hy.search("DTI", as_of_date="2000-01-01"), [])

    def test_weights_override_changes_ranking_source(self, _):
        """Chỉ BM25 (w_dense=0) cho cùng thứ tự với BM25 thuần."""
        bm25_order = [h.chunk.chunk_id for h in BM25Index(CHUNKS).search("lãi suất DTI", top_k=3)]
        hybrid_order = self.ids("lãi suất DTI", top_k=3, w_bm25=1.0, w_dense=0.0)
        self.assertEqual(hybrid_order, bm25_order)

    def test_mismatched_chunk_lists_rejected(self, _):
        with self.assertRaises(ValueError):
            HybridSearcher(self.store, BM25Index(list(reversed(CHUNKS))))

    def test_empty_store_rejected(self, _):
        with self.assertRaises(ValueError):
            HybridSearcher(VectorStore(), BM25Index(CHUNKS))


class TestRealIndex(unittest.TestCase):
    """Index thật: lọc ngày ở ranh giới 31/3 và 1/4 của CP-01 (chỉ BM25, không cần mô hình)."""

    @classmethod
    def setUpClass(cls):
        index_dir = REPO_ROOT / "data" / "index"
        if not (index_dir / "chunks.json").exists():
            raise unittest.SkipTest("Chưa có data/index/")
        store = VectorStore()
        store.load(index_dir)
        cls.ix = BM25Index(store.chunks)

    def top1(self, query, date):
        return self.ix.search(query, top_k=1, as_of_date=date)[0].chunk

    def test_dti_limit_switches_version_at_boundary(self):
        q = "Tỷ lệ nợ trên thu nhập tối đa là bao nhiêu?"
        before = self.top1(q, "2025-03-31")
        after = self.top1(q, "2025-04-01")
        self.assertEqual((before.doc_id, before.version), ("CP-01", "v1"))
        self.assertEqual((after.doc_id, after.version), ("CP-01", "v2"))

    def test_filtered_results_never_include_other_version(self):
        for d, banned in (("2025-03-31", "v2"), ("2025-04-01", "v1")):
            for h in self.ix.search("giới hạn tuổi số tiền kỳ hạn DTI", top_k=10, as_of_date=d):
                if h.chunk.doc_id == "CP-01":
                    self.assertNotEqual(h.chunk.version, banned)


class TestRetrieveHits(unittest.TestCase):
    """Kiểm tra retrieve_hits trả về (text, hits) sạch phục vụ Zero-Regex."""

    def test_empty_query_returns_error_and_empty_hits(self):
        text, hits = retrieve_hits("   ")
        self.assertTrue(text.startswith("Error:"))
        self.assertEqual(hits, [])

    def test_invalid_date_returns_error_and_empty_hits(self):
        text, hits = retrieve_hits("Lãi suất", as_of_date="sai-ngay")
        self.assertTrue(text.startswith("Error:"))
        self.assertEqual(hits, [])

    def test_valid_search_returns_text_and_structured_hits(self):
        text, hits = retrieve_hits("quy định tuổi và số tiền vay CP-01", as_of_date="2025-01-15", top_k=2)
        self.assertFalse(text.startswith("Error:"))
        self.assertGreater(len(hits), 0)
        self.assertTrue(all(isinstance(h, SearchHit) for h in hits))
        # Có thể lấy trực tiếp doc_id và version không cần regex
        doc_ids = {h.chunk.doc_id for h in hits}
        self.assertIn("CP-01", doc_ids)
        # retrieve() tương thích ngược với retrieve_hits()
        legacy_text = retrieve("quy định tuổi và số tiền vay CP-01", as_of_date="2025-01-15", top_k=2)
        self.assertEqual(text, legacy_text)


if __name__ == "__main__":
    unittest.main()
