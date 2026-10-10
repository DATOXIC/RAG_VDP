"""Tìm kiếm từ khóa BM25 trên các chunk chính sách.

Cùng hợp đồng với dense.VectorStore.search: nhận (query, top_k, as_of_date),
trả list[SearchHit]. Không in ra stdout (MCP stdio dùng stdout làm giao thức).

Quyết định thiết kế:
1. Tách từ bằng regex, giữ chữ, số (kể cả số thập phân) và dấu %. Không dùng
   thư viện tách từ tiếng Việt vì đề giới hạn thư viện.
2. Dựng chỉ mục trên chunk.text (đã gồm header "[CP-01 v2 | ...]"), nên hỏi theo
   mã tài liệu cũng khớp được.
3. Không bỏ dấu tiếng Việt: "lãi" và "lai" là hai từ khác nhau.
4. Lọc theo as_of_date cùng điều kiện với dense._valid_mask. IDF được tính một lần
   trên toàn bộ chunk (cả v1 lẫn v2), lọc ngày áp dụng khi xếp hạng.
"""

import re
import unicodedata
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from finco.types import Chunk, SearchHit

# Số (có thể thập phân, kèm %) hoặc một từ. Số đứng trước để "15,5%" không bị tách.
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?|\w+")


def tokenize(text: str) -> list[str]:
    """Tách văn bản thành danh sách token chữ thường.

    - Chuẩn hóa NFC để dấu tiếng Việt tổ hợp không bị tách rời khỏi chữ cái.
    - "45%" cho ra cả "45%" và "45", để truy vấn "45 phần trăm" vẫn khớp.
    - Dấu câu và ký hiệu Markdown (**, [, ], |) bị bỏ.
    """
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower()):
        tokens.append(tok)
        if tok.endswith("%") and len(tok) > 1:
            tokens.append(tok[:-1])
    return tokens


def is_valid_at(chunk: Chunk, as_of_date: str) -> bool:
    """Chunk còn hiệu lực tại as_of_date (so sánh chuỗi ISO YYYY-MM-DD).

    Phải giữ cùng điều kiện với VectorStore._valid_mask trong dense.py.
    """
    return chunk.effective_date <= as_of_date and (
        chunk.expiry_date is None or as_of_date <= chunk.expiry_date
    )


class BM25Index:
    """Chỉ mục BM25 in-memory trên danh sách chunk."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("BM25Index cần ít nhất một chunk.")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def get_scores(self, query: str) -> np.ndarray:
        """Điểm BM25 thô của mọi chunk (cùng thứ tự với self.chunks).

        Truy vấn không có token nào cho ra toàn số 0.
        """
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def valid_mask(self, as_of_date: Optional[str]) -> np.ndarray:
        """Mặt nạ chunk còn hiệu lực. as_of_date=None nghĩa là không lọc."""
        if as_of_date is None:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([is_valid_at(c, as_of_date) for c in self.chunks], dtype=bool)

    def search(
        self,
        query: str,
        top_k: int = 5,
        as_of_date: Optional[str] = None,
    ) -> list[SearchHit]:
        """Tìm top_k chunk khớp từ khóa nhất.

        Args:
            query: Câu truy vấn.
            top_k: Số kết quả tối đa.
            as_of_date: "YYYY-MM-DD". Có thì chỉ xét chunk hiệu lực tại ngày này.
        Returns:
            Danh sách SearchHit, điểm giảm dần. Chỉ gồm chunk có điểm > 0 (khớp
            ít nhất một từ), nên có thể ít hơn top_k hoặc rỗng.
        """
        if top_k <= 0:
            return []

        scores = self.get_scores(query)
        candidates = np.flatnonzero(self.valid_mask(as_of_date))
        if candidates.size == 0:
            return []

        order = candidates[np.argsort(scores[candidates])[::-1][:top_k]]
        return [
            SearchHit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                bm25_score=float(scores[i]),
            )
            for i in order
            if scores[i] > 0
        ]
