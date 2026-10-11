"""Tìm kiếm hybrid: trộn điểm BM25 (từ khóa) và cosine (ngữ nghĩa).

Dựa trên w4/ex2_1_hybrid_search.py của lớp (0.3 BM25 + 0.7 vector, min-max), khác ở:
1. Lọc as_of_date TRƯỚC, rồi mới chuẩn hóa: điểm min/max chỉ tính trên chunk hiệu lực.
2. Chuẩn hóa min-max cho CẢ HAI thang (bản mẫu chỉ chuẩn hóa BM25).
3. Không print; trả list[SearchHit] cùng hợp đồng với dense và BM25.
"""

from typing import Optional

import numpy as np

from finco.rag.bm25 import BM25Index
from finco.rag.dense import VectorStore, embed
from finco.types import SearchHit
from finco.filter import bit_mask


W_BM25 = 0.3
W_DENSE = 0.7


def min_max(scores: np.ndarray) -> np.ndarray:
    """Chuẩn hóa về [0, 1]. Mọi điểm bằng nhau: toàn 1 nếu dương, toàn 0 nếu bằng 0.

    Trường hợp toàn 0 xảy ra với BM25 khi truy vấn không khớp từ nào; trả 0 để
    không cộng điểm giả cho mọi chunk.
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi == lo:
        return np.ones_like(scores) if hi > 0 else np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridSearcher:
    """Trộn VectorStore (dense) và BM25Index trên cùng một tập chunk."""

    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        w_bm25: float = W_BM25,
        w_dense: float = W_DENSE,
    ):
        if store.vectors is None or not store.chunks:
            raise ValueError("VectorStore chưa có dữ liệu.")
        # Hai chỉ mục phải cùng chunk và cùng thứ tự, nếu không điểm bị ghép sai.
        if [c.chunk_id for c in store.chunks] != [c.chunk_id for c in bm25.chunks]:
            raise ValueError("VectorStore và BM25Index không cùng danh sách chunk.")
        self.store = store
        self.bm25 = bm25
        self.w_bm25 = w_bm25
        self.w_dense = w_dense

    def search(
        self,
        query: str,
        top_k: int = 5,
        as_of_date: Optional[str] = None,
        w_bm25: Optional[float] = None,
        w_dense: Optional[float] = None,
    ) -> list[SearchHit]:
        """Tìm top_k chunk theo điểm trộn.

        Args:
            query: Câu truy vấn.
            top_k: Số kết quả tối đa.
            as_of_date: "YYYY-MM-DD". Có thì chỉ xét chunk hiệu lực tại ngày này.
            w_bm25, w_dense: Ghi đè trọng số (dùng cho thí nghiệm).
        Returns:
            Danh sách SearchHit giảm dần theo điểm trộn; rỗng nếu truy vấn rỗng
            hoặc không chunk nào hiệu lực. score là điểm trộn, dense_score và
            bm25_score là điểm thô.
        """
        if top_k <= 0 or not query.strip():
            return []
        wb = self.w_bm25 if w_bm25 is None else w_bm25
        wd = self.w_dense if w_dense is None else w_dense

        candidates = np.flatnonzero(bit_mask(as_of_date)) # Mảng 0 và 1
        if candidates.size == 0:
            return []

        # Điểm thô 
        raw_bm25 = self.bm25.get_scores(query)[candidates]
        raw_dense = (self.store.vectors @ embed([query])[0])[candidates]

        # Chuẩn hóa -> Fusion 
        fused = wb * min_max(raw_bm25) + wd * min_max(raw_dense)

        order = np.argsort(fused)[::-1][:top_k]
        return [
            SearchHit(
                chunk=self.store.chunks[candidates[j]],
                score=float(fused[j]),
                dense_score=float(raw_dense[j]),
                bm25_score=float(raw_bm25[j]),
            )
            for j in order
        ]
