"""Các hàm tiện ích lọc chunk theo ngày hiệu lực dùng chung cho Dense và BM25."""

from datetime import datetime
from typing import Optional

import numpy as np

from finco.types import Chunk


def validate_date(date_str: str) -> bool:
    """Kiểm tra chuỗi ngày có đúng định dạng YYYY-MM-DD hay không."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except (ValueError, AttributeError):
        return False


def is_valid_at(chunk: Chunk, as_of_date: str) -> bool:
    """Kiểm tra chunk còn hiệu lực tại as_of_date (so sánh chuỗi ISO YYYY-MM-DD)."""
    return chunk.effective_date <= as_of_date and (
        chunk.expiry_date is None or as_of_date <= chunk.expiry_date
    )


def valid_mask(chunks: list[Chunk], as_of_date: Optional[str] = None) -> np.ndarray:
    """Tạo mảng Boolean: True nếu chunk còn hiệu lực tại as_of_date.

    Nếu as_of_date là None: không lọc (trả về toàn True).
    """
    if as_of_date is None:
        return np.ones(len(chunks), dtype=bool)
    return np.array([is_valid_at(c, as_of_date) for c in chunks], dtype=bool)


def get_top_valid_indices(
    scores: np.ndarray,
    chunks: list[Chunk],
    as_of_date: Optional[str] = None,
    top_k: int = 5,
) -> np.ndarray:
    """Lọc và lấy top_k index của chunk hợp lệ có điểm cao nhất."""
    if top_k <= 0 or scores.size == 0 or len(chunks) == 0:
        return np.empty(0, dtype=int)

    candidates = np.arange(len(chunks))

    if as_of_date is not None:
        mask = valid_mask(chunks, as_of_date)
        candidates = candidates[mask]

    if candidates.size == 0:
        return np.empty(0, dtype=int)

    candidate_scores = scores[candidates]
    sorted_local_indices = np.argsort(candidate_scores)[::-1][:top_k]
    return candidates[sorted_local_indices]
