from finco.types import Chunk
from datetime import datetime
import os
from pathlib import Path
from typing import Optional
import tiktoken
from finco.rag.bm25 import BM25Index
from finco.rag.dense import VectorStore
from finco.rag.hybrid import HybridSearcher

PROJECT_ROOT = Path(__file__).parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
MAX_CONTEXT_TOKENS = 1500

_ENC = tiktoken.get_encoding("cl100k_base")
_SEARCHER: Optional[HybridSearcher] = None
_STORE: Optional[VectorStore] = None

def _get_searcher():
    """Khởi tạo một lần và tái sử dụng bộ tìm kiếm (Hybrid hoặc Dense)."""
    global _SEARCHER, _STORE
    if _STORE is None:
        _STORE = VectorStore()
        _STORE.load(INDEX_DIR)
    
    mode = os.environ.get("RETRIEVAL_MODE", "hybrid").lower()
    if mode == "dense":
        return _STORE
    
    if _SEARCHER is None:
        bm25 = BM25Index(_STORE.chunks)
        _SEARCHER = HybridSearcher(_STORE, bm25)
    return _SEARCHER
    
def _count_tokens(text: str) -> int:
    """Đếm số token"""
    return len(_ENC.encode(text))

def _validate_date(date_str: str) -> bool:
    """Kiểm tra chuỗi ngày có đúng định dạng YYYY-MM-DD hay không."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except (ValueError, AttributeError):
        return False


def search_policy(query: str, as_of_date: Optional[str] = None, top_k: int = 5) -> str:
    """
    1. Kiểm tra Query
    2. Kiểm tra as_of_date
    3. Ép tra top_k
    4. Tìm kiếm
    5. Đóng gói kết quả và quản lý Token
    """
    try:
        if not query or not query.strip():
            return "Error: Query không được để trống"
        query = query.strip()

        if as_of_date is not None:
            if not _validate_date(as_of_date):
                return f"Error: as_of_date:{as_of_date} không đúng định dạng"
            as_of_date = as_of_date.strip()

        top_k = min(1,min(int(top_k),5))

        searcher = _get_searcher()
        hits = searcher.search(query,top_k,as_of_date) # List[Hits]

        if not hits:
            return f"Không tìm thấy Policy nào phù hợp với Query và as_of_date:{as_of_date}"

        output: list[str] = []
        count = 0
        for hit in hits:
            chunk = hit.chunk
            header = f"[{chunk.doc_id} {chunk.version}. {chunk.effective_date}]"
            text = f"{header} {chunk.section}\n{chunk.text}"
            token = _count_tokens(text)

            if output and (count + token > MAX_CONTEXT_TOKENS):
                break

            output.append(text)
            count += token

        return "\n\n---\n\n".join(output)
    
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi tìm kiếm chính sách: {e}"
