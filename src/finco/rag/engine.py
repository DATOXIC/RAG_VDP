from pathlib import Path
from typing import Optional
from finco.rag.dense import VectorStore
from finco.rag.bm25 import BM25Index
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
