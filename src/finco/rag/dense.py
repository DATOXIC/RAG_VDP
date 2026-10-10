from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Union

from finco.types import Chunk, SearchHit
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Cố định mô hình embedding cho toàn hệ thống
MODEL_NAME = "BAAI/bge-m3"

# Biến toàn cục lưu trữ mô hình để không nạp lại nhiều lần
_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Nạp và tái sử dụng mô hình embedding ở chế độ local offline."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME, local_files_only=True)
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
        normalize_embeddings=True,
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
        print(f" Đã nhúng và lưu trữ {len(self.chunks)} chunks (Shape: {self.vectors.shape}).")

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        
        if self.vectors is None or not self.chunks:
            raise ValueError("VectorStore chưa có dữ liệu!")

        query_vector = embed([query])[0]

        scores = np.dot(self.vectors, query_vector)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for i in top_indices:
            results.append(SearchHit(
                chunk = self.chunks[i],
                score = float(scores[i])
            ))

        return results

    def save(self, index_dir: str | Path = "data/index") -> None:
        """Lưu ma trận vector và chunks và metadata ra ổ cứng."""
        if self.vectors is None or not self.chunks:
            raise ValueError("VectorStore đang rỗng, không thể lưu!")
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "vectors.npy", self.vectors)

        chunks_data = [asdict(c) for c in self.chunks]
        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        meta = {
            "model_id": MODEL_NAME,
            "num_chunks": len(self.chunks),
            "vector_dim": int(self.vectors.shape[1]),
            "created_at": datetime.now().isoformat(),
        }

        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f" Đã lưu index thành công vào thư mục: {out_dir}")

    def load(self, index_dir: str | Path = "data/index") -> None:

        in_dir = Path(index_dir)
        if not (in_dir / "vectors.npy").exists() or not (
            in_dir / "chunks.json"
        ).exists():
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu index trong: {in_dir}"
            )

        # 1. Kiểm tra đối soát model_id (tránh nạp nhầm model khác)
        if (in_dir / "meta.json").exists():
            with open(in_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("model_id") != MODEL_NAME:
                raise ValueError(
                    f"Lệch mô hình: Index được tạo bởi '{meta.get('model_id')}', "
                    f"nhưng hệ thống đang chạy '{MODEL_NAME}'!"
                )

        # 2. Nạp vector bằng NumPy
        self.vectors = np.load(in_dir / "vectors.npy")
        # 3. Nạp chunks và tái tạo lại đối tượng Chunk
        with open(in_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk(**item) for item in chunks_data]
        print(
            f" Đã nạp thành công {len(self.chunks)} chunks từ index offline!"
        )
