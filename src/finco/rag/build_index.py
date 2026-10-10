"""Dựng lại chỉ mục vector (data/index/) từ data/policies/.
Quy trình: đọc tài liệu -> chunk -> nhúng -> lưu -> nạp lại để tự kiểm chứng.
Chỉ ghi log ra stderr: stdout phải sạch cho MCP stdio.
"""

import sys
from pathlib import Path

from finco.rag.chunker import chunk_all
from finco.rag.dense import VectorStore
from finco.rag.loader import load_all_policies

POLICIES_DIR = Path("data")/"policies"
INDEX_DIR = Path("data")/"index"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_index(
    policies_dir: Path = POLICIES_DIR,
    index_dir: Path = INDEX_DIR,
) -> int:
    """Dựng và lưu index, rồi nạp lại để chắc nó khớp với dữ liệu hiện tại.

    Returns:
        Số chunk đã lập chỉ mục.
    Raises:
        ValueError: Không có chunk nào, hoặc index vừa lưu không nạp lại khớp được.
    """
    docs = load_all_policies(policies_dir)
    chunks = chunk_all(docs)
    if not chunks:
        raise ValueError(f"Không tạo được chunk nào từ: {policies_dir}")
    _log(f"Đã đọc {len(docs)} tài liệu, tách thành {len(chunks)} chunk.")

    store = VectorStore()
    store.build(chunks)
    store.save(index_dir)

    # Kiểm chứng: nạp lại bằng đúng cơ chế mà lúc chạy thật sẽ dùng.
    # Lệch model_id / số chunk / corpus_hash đều ném ValueError tại đây.
    VectorStore().load(index_dir, expected_chunks=chunks)
    _log("Kiểm chứng xong: index nạp lại khớp dữ liệu hiện tại.")
    return len(chunks)


def main() -> None:
    n = build_index()
    _log(f"Hoàn tất: {n} chunk -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
