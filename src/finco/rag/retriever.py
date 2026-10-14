"""Giao diện truy xuất duy nhất cho tool search_policy (vai C bọc lại).

    retrieve(query, as_of_date=None, top_k=3) -> str

Hợp đồng (PLAN.md mục 6.4):
- Không bao giờ raise: mọi lỗi trả chuỗi bắt đầu bằng "Error: ...".
- Chế độ truy vấn lấy từ biến môi trường RETRIEVAL_MODE ("dense" | "hybrid",
  mặc định "hybrid"), KHÔNG phải tham số của tool.
- as_of_date là NGÀY NỘP HỒ SƠ (YYYY-MM-DD), quyết định phiên bản chính sách.
  None nghĩa là không biết ngày: chỉ tìm trong các đoạn chưa hết hạn
  (expiry_date is None), không phụ thuộc đồng hồ máy.
- top_k bị chặn trong [1, 5]. Kết quả gom theo tài liệu, có nguồn, cắt theo
  ngân sách token.
- Chỉ ghi log ra stderr: stdout phải sạch cho MCP stdio.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from finco.rag.bm25 import BM25Index
from finco.rag.chunker import chunk_all, count_tokens
from finco.rag.dense import VectorStore
from finco.rag.hybrid import HybridSearcher
from finco.rag.loader import load_all_policies
from finco.types import SearchHit

# Đường dẫn tuyệt đối từ vị trí file: MCP server do client spawn, cwd không chắc là gốc repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
POLICIES_DIR = REPO_ROOT / "data" / "policies"
INDEX_DIR = REPO_ROOT / "data" / "index"

MODES = ("dense", "hybrid")
DEFAULT_MODE = "hybrid"
MAX_TOP_K = 5
MAX_CONTEXT_TOKENS = 1500

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Engine:
    """Bộ ba (dense, bm25, hybrid) dựng trên cùng một tập chunk."""

    def __init__(self, store: VectorStore):
        self.store = store
        self.bm25 = BM25Index(store.chunks)
        self.hybrid = HybridSearcher(store, self.bm25)

    def search(self, mode: str, query: str, top_k: int, as_of_date: Optional[str]):
        if mode == "dense":
            return self.store.search(query, top_k=top_k, as_of_date=as_of_date)
        return self.hybrid.search(query, top_k=top_k, as_of_date=as_of_date)


_ENGINES: Optional[tuple[_Engine, _Engine]] = None  # (đầy đủ, chỉ chunk chưa hết hạn)


def _load_engines() -> tuple[_Engine, _Engine]:
    """Nạp index một lần. Kiểm tra index khớp dữ liệu hiện tại (corpus_hash)."""
    global _ENGINES
    if _ENGINES is None:
        expected = chunk_all(load_all_policies(POLICIES_DIR))
        full = VectorStore()
        full.load(INDEX_DIR, expected_chunks=expected)

        # "Hiện hành" = chưa có expiry_date. Dựng riêng một chỉ mục con để việc
        # chuẩn hóa điểm hybrid chỉ tính trên các chunk này (lọc trước, chấm sau).
        keep = [i for i, c in enumerate(full.chunks) if c.expiry_date is None]
        current = VectorStore()
        current.chunks = [full.chunks[i] for i in keep]
        current.vectors = full.vectors[keep]

        _ENGINES = (_Engine(full), _Engine(current))
        _log(f"retriever: nạp {len(full.chunks)} chunk, {len(keep)} chunk hiện hành.")
    return _ENGINES


def reset() -> None:
    """Xóa cache index (dùng trong test hoặc sau khi build lại index)."""
    global _ENGINES
    _ENGINES = None


def _validate_date(as_of_date: Optional[str]) -> Optional[str]:
    """Trả thông báo lỗi nếu as_of_date sai định dạng, ngược lại None."""
    if as_of_date is None:
        return None
    if not isinstance(as_of_date, str) or not _DATE_RE.match(as_of_date):
        return f"Error: as_of_date phải có dạng YYYY-MM-DD, nhận được {as_of_date!r}."
    try:
        datetime.strptime(as_of_date, "%Y-%m-%d")
    except ValueError:
        return f"Error: as_of_date {as_of_date!r} không phải ngày hợp lệ."
    return None


def _body(text: str) -> str:
    """Bỏ dòng header "[CP-01 v2 | ...] ..." đầu chunk, giữ phần nội dung."""
    head, sep, rest = text.partition("\n")
    return rest.strip() if sep and head.startswith("[") else text.strip()


def _format(hits: list[SearchHit], budget: int) -> str:
    """Gom theo (doc_id, version), giữ thứ tự xếp hạng, cắt theo ngân sách token."""
    groups: dict[tuple[str, str], list[SearchHit]] = {}
    for h in hits:
        groups.setdefault((h.chunk.doc_id, h.chunk.version), []).append(h)

    blocks: list[str] = []
    used = 0
    for (doc_id, version), group in groups.items():
        first = group[0].chunk
        parts = [f"[{doc_id} {version} · {first.effective_date}]"]
        for h in group:
            piece = f"{h.chunk.section}\n{_body(h.chunk.text)}"
            cost = count_tokens(piece)
            if blocks or len(parts) > 1:  # luôn giữ đoạn đầu tiên, dù vượt ngân sách
                if used + cost > budget:
                    continue
            parts.append(piece)
            used += cost
        if len(parts) > 1:
            blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def retrieve_hits(
    query: str, as_of_date: Optional[str] = None, top_k: int = 3
) -> tuple[str, list[SearchHit]]:
    """Tìm các đoạn chính sách liên quan và trả về cả văn bản lẫn danh sách SearchHit có cấu trúc.

    Args:
        query: Câu hỏi, tiếng Việt.
        as_of_date: Ngày nộp hồ sơ "YYYY-MM-DD". Bỏ trống nếu không gắn với
            hồ sơ nào: khi đó chỉ tìm trong các đoạn chính sách đang hiện hành.
        top_k: Số đoạn tối đa (1..5).
    Returns:
        tuple (văn_bản_kết_quả, danh_sách_hits):
        - Nếu có lỗi: trả về (chuỗi lỗi "Error: ...", [])
        - Nếu không tìm thấy: trả về (thông báo không tìm thấy, [])
        - Nếu thành công: trả về (văn bản đã format kèm nguồn, danh sách hits nguyên bản)
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return ("Error: query không được rỗng.", [])
        err = _validate_date(as_of_date)
        if err:
            return (err, [])
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            return (f"Error: top_k phải là số nguyên, nhận được {top_k!r}.", [])
        top_k = max(1, min(top_k, MAX_TOP_K))

        mode = os.environ.get("RETRIEVAL_MODE", DEFAULT_MODE).strip().lower()
        if mode not in MODES:
            return (f"Error: RETRIEVAL_MODE={mode!r} không hợp lệ, chỉ nhận {', '.join(MODES)}.", [])

        full, current = _load_engines()
        if as_of_date is None:
            hits = current.search(mode, query, top_k, None)
        else:
            hits = full.search(mode, query, top_k, as_of_date)

        if not hits:
            when = f"tại ngày {as_of_date}" if as_of_date else "hiện hành"
            return (f"Không tìm thấy đoạn chính sách nào phù hợp {when}.", [])
        return (_format(hits, MAX_CONTEXT_TOKENS), hits)
    except Exception as e:  # tool không bao giờ raise
        _log(f"retriever: lỗi {type(e).__name__}: {e}")
        return (f"Error: truy xuất thất bại ({type(e).__name__}: {e}).", [])


def retrieve(query: str, as_of_date: Optional[str] = None, top_k: int = 3) -> str:
    """Tìm các đoạn chính sách liên quan và trả về văn bản có nguồn.

    Args:
        query: Câu hỏi, tiếng Việt.
        as_of_date: Ngày nộp hồ sơ "YYYY-MM-DD". Bỏ trống nếu không gắn với
            hồ sơ nào: khi đó chỉ tìm trong các đoạn chính sách đang hiện hành.
        top_k: Số đoạn tối đa (1..5).
    Returns:
        Văn bản dạng "[CP-01 v2 · 2025-04-01]\\n<mục>\\n<nội dung>"; hoặc một câu
        thông báo không tìm thấy; hoặc chuỗi bắt đầu bằng "Error: ".
    """
    text, _ = retrieve_hits(query=query, as_of_date=as_of_date, top_k=top_k)
    return text