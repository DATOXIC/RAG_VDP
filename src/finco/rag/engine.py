import os
from pathlib import Path
from typing import Optional, Union
from finco.rag.dense import VectorStore
from finco.rag.bm25 import BM25Index
from finco.rag.hybrid import HybridSearcher
from finco.rag.chunker import chunk_all
from finco.rag.loader import load_all_policies

# =========== PATH =============
INDEX_DIR = Path(__file__).parents[3] / "data" / "index"
POLICIES_DIR = Path(__file__).resolve().parents[3] / "data" / "policies"

#=========== BIẾN GLOBAL --> NẠP LƯỜI ============
"""
Lazy Loading: Cần thì mới nạp    
Cơ chế: Sử dụng biến GLOBAL để cố định trong RAM
"""
_STORE: Optional[VectorStore] = None
_BM25: Optional[BM25Index] = None
_HYBRID: Optional[HybridSearcher] = None 

def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()

        #Kiểm tra xem có giống với INDEX không --> 
        expected = chunk_all(load_all_policies(POLICIES_DIR))
        _STORE.load(INDEX_DIR,expected_chunks=expected)
    return _STORE

def get_bm25() -> BM25Index:
    global _BM25
    if _BM25 is None:
        store = get_vector_store()
        _BM25 = BM25Index(store.chunks)
    return _BM25

def get_searcher() -> Union[HybridSearcher, VectorStore]:
    """Trả về bộ tìm kiếm dựa theo biến môi trường RETRIEVAL_MODE ('dense' hoặc 'hybrid')."""
    global _HYBRID
    mode = os.environ.get("RETRIEVAL_MODE", "hybrid").lower()
    
    if mode == "dense":
        return get_vector_store()
    
    if _HYBRID is None:
        _HYBRID = HybridSearcher(get_vector_store(), get_bm25())
    return _HYBRID

def reset_engines():
    """Hàm dọn dẹp"""
    global _STORE, _BM25, _HYBRID
    _STORE = None
    _BM25 = None
    _HYBRID = None
