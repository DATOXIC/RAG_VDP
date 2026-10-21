from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional

from finco.types import Chunk, SearchHit
from finco.rag.filter import get_top_valid_indices, valid_mask
import numpy as np

# 1. Cố định mô hình embedding cho toàn hệ thống
MODEL_NAME = "BAAI/bge-m3"
# Biến toàn cục lưu trữ mô hình để không nạp lại nhiều lần

_MODEL_INSTANCE = None  # SentenceTransformer, nạp lười để import module nhanh


def _log(msg: str) -> None:
    """Ghi log ra stderr: stdout phải sạch cho MCP stdio."""
    print(msg, file=sys.stderr, flush=True)


def corpus_hash(chunks: list[Chunk]) -> str:
    """Băm nội dung và toàn bộ metadata của chunk theo thứ tự vector.

    Ngày hết hạn không nằm trong text nhưng ảnh hưởng trực tiếp đến lọc phiên
    bản. JSON có thứ tự khóa cố định giúp hash ổn định khi lưu và nạp lại.
    """
    payload = json.dumps(
        [asdict(c) for c in chunks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer

        try:
            _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME, local_files_only=True) # Lấy sẵn nếu có
        except Exception:
            _log(f"Chưa có '{MODEL_NAME}' trong cache, đang tải về (lần đầu, ~2 GB)...")
            _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME) # Nếu chưa thì tải

    return _MODEL_INSTANCE


def embed(texts: list[str]) -> np.ndarray:
    """Nhúng danh sách văn bản thành ma trận vector NumPy (đã chuẩn hóa L2).
    Args:
        texts: Danh sách các đoạn văn bản cần nhúng.
    Returns:
        np.ndarray: Ma trận 2D kích thước (N, 1024), kiểu np.float32.
    """

    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model = get_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True, # Chuẩn hóa độ dài
        show_progress_bar=False,
        batch_size=16,
    )
    return np.asarray(embeddings, dtype=np.float32)


class VectorStore:
    """Kho lưu trữ và tìm kiếm vector in-memory bằng NumPy."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: Optional[np.ndarray] = None

    def build(self, chunks: list[Chunk]) -> None:
        """Nạp danh sách chunks và nhúng thành ma trận vector."""
        if not chunks:
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.vectors = embed(texts)

        _log(f"Đã nhúng và lưu trữ {len(self.chunks)} chunks (shape: {self.vectors.shape}).")

    def _valid_mask(self, as_of_date: str) -> np.ndarray:
        """Tạo mảng Boolean: True nếu chunk còn hiệu lực tại as_of_date."""
        return valid_mask(self.chunks, as_of_date)

    def search(self, query: str, top_k: int = 5, as_of_date: Optional[str] = None) -> list[SearchHit]:
        """Tìm top_k chunk gần nhất theo cosine.
        Args:
            query: Câu truy vấn.
            top_k: Số kết quả tối đa.
            as_of_date: "YYYY-MM-DD". Nếu có, chỉ xét chunk có hiệu lực tại ngày này
                (lọc trước khi xếp hạng). None = không lọc.
        Returns:
            Danh sách SearchHit, điểm giảm dần; rỗng nếu không chunk nào hiệu lực.
        """
        if self.vectors is None or not self.chunks:
            raise ValueError("VectorStore chưa có dữ liệu!")
        if top_k <= 0 or not query.strip():
            return []

        query_vector = embed([query])[0]
        scores = np.dot(self.vectors, query_vector)
        
        order = get_top_valid_indices(scores, self.chunks, as_of_date, top_k)

        return [
            SearchHit(
                self.chunks[i],
                score = float(scores[i]),
                dense_score= float(scores[i])
            )
            for i in order
        ]


    def save(self, index_dir: str | Path = "data/index") -> None:
        """Lưu ma trận vector và chunks và metadata ra ổ cứng."""

        if self.vectors is None or not self.chunks:
            raise ValueError("VectorStore đang rỗng, không thể lưu!")

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lưu VECTOR
        np.save(out_dir / "vectors.npy", self.vectors)

        # Lưu CHUNKS
        chunks_data = [asdict(c) for c in self.chunks]
        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Lưu METADATA
        meta = {
            "model_id": MODEL_NAME,
            "num_chunks": len(self.chunks),
            "vector_dim": int(self.vectors.shape[1]),
            "corpus_hash": corpus_hash(self.chunks),
            "created_at": datetime.now().isoformat(),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _log(f"Đã lưu index thành công vào thư mục: {out_dir}")

    def load(
        self,
        index_dir: str | Path = "data/index",
        expected_chunks: Optional[list[Chunk]] = None,
    ) -> None:
        """Nạp index từ ổ cứng và kiểm tra tính nhất quán.

        Args:
            index_dir: Thư mục chứa vectors.npy, chunks.json, meta.json.
            expected_chunks: Nếu truyền vào (chunk hiện tại từ chunk_all), so mã băm
                để phát hiện index cũ so với dữ liệu/chunker mới.
        Raises:
            FileNotFoundError: Thiếu file index.
            ValueError: Lệch mô hình, lệch số lượng/chiều vector, hoặc index đã cũ.
        """
        in_dir = Path(index_dir)

        # CHECK
        for name in ("vectors.npy", "chunks.json", "meta.json"):
            if not (in_dir / name).exists():
                raise FileNotFoundError(
                    f"Không tìm thấy '{name}' trong: {in_dir}. Hãy build lại index."
                )

        # Nạp METADA
        with open(in_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_id") != MODEL_NAME:
            raise ValueError(
                f"Lệch mô hình: Index được tạo bởi '{meta.get('model_id')}', "
                f"nhưng hệ thống đang chạy '{MODEL_NAME}'!"
            )

        # Nạp Vector
        vectors = np.load(in_dir / "vectors.npy")

        # Nạp Chunks
        with open(in_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks = [Chunk(**item) for item in json.load(f)]

        
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Index hỏng: {vectors.shape[0] if vectors.ndim == 2 else '?'} vector "
                f"nhưng {len(chunks)} chunk. Hãy build lại index."
            )
        
        if meta.get("vector_dim") != vectors.shape[1] or meta.get("num_chunks") != len(chunks):
            raise ValueError("Index hỏng: meta.json không khớp dữ liệu. Hãy build lại index.")

        # Kiểm tra cả metadata/nội dung đã lưu trước khi đối chiếu nguồn hiện tại.
        if meta.get("corpus_hash") != corpus_hash(chunks):
            raise ValueError("Index hỏng: corpus_hash không khớp chunks.json. Hãy build lại index.")

        # 3. Phát hiện index cũ so với dữ liệu hiện tại
        if expected_chunks is not None and meta.get("corpus_hash") != corpus_hash(expected_chunks):
            raise ValueError(
                "Index đã cũ so với dữ liệu/chunker hiện tại. Hãy build lại index."
            )
        
        self.vectors = vectors
        self.chunks = chunks    
        _log(f"Đã nạp thành công {len(self.chunks)} chunks từ index offline.")
