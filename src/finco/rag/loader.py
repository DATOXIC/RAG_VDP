"""
Nạp Policy thành các đối tượng Document
"""
from pathlib import Path
from typing import Union

from finco.types import Document

REQUIRED_FRONT_MATTER_KEYS = (
    "doc_id",
    "title",
    "version",
    "effective_date",
    "owner",
)

def load_document(file_path: Union[str, Path]) -> Document:
    """Đọc một file chính sách .md, tách front matter và trả về đối tượng Document.
    Returns:
        Document: Đối tượng chứa metadata và nội dung thân văn bản.
    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu định dạng front matter không hợp lệ hoặc thiếu khóa bắt buộc.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file tài liệu: {path}")

    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    if not lines or lines[0].strip() != "---":
        raise ValueError(f"File '{path}' không hợp lệ: Dòng đầu tiên phải bắt đầu bằng '---' để mở front matter.")

    end_fm_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_fm_idx = i
            break

    if end_fm_idx == -1:
        raise ValueError(f"File '{path}' không hợp lệ: Không tìm thấy dòng '---' đóng front matter.")

    fm_lines = lines[1:end_fm_idx]
    body = "\n".join(lines[end_fm_idx + 1 :]).strip()

    # 3. Phân tích từng dòng front matter theo dấu : đầu tiên
    metadata: dict[str, str] = {}
    for line in fm_lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue

        key, sep, val = line_clean.partition(":")
        if not sep:
            continue

        key_clean = key.strip()

        # Cắt bỏ comment inline nếu có (ví dụ: CP-01 v1 # tùy chọn)
        val_clean = val.split("#", 1)[0].strip()

        # Khóa tùy chọn nếu giá trị rỗng thì coi là None
        if val_clean:
            metadata[key_clean] = val_clean

    missing_keys = [k for k in REQUIRED_FRONT_MATTER_KEYS if not metadata.get(k)]
    if missing_keys:
        raise ValueError(f"File '{path}' thiếu khóa bắt buộc trong front matter: {', '.join(missing_keys)}")

    return Document(
        doc_id=metadata["doc_id"],
        title=metadata["title"],
        version=metadata["version"],
        effective_date=metadata["effective_date"],
        owner=metadata["owner"],
        body=body,
        source_path=str(path.resolve()),
        expiry_date=metadata.get("expiry_date") or None,
        supersedes=metadata.get("supersedes") or None,
        status=metadata.get("status") or None,
    )


def load_all_policies(policies_dir: Union[str, Path] = "data/policies") -> list[Document]:
    """Đọc toàn bộ các file chính sách .md trong thư mục và kiểm tra tính duy nhất.
    Args:
        policies_dir: Thư mục chứa các file .md chính sách (mặc định data/policies).
    Returns:
        list[Document]: Danh sách các tài liệu chính sách đã tải.
    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
        ValueError: Nếu có hai file trùng cặp (doc_id, version).
    """
    dir_path = Path(policies_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Thư mục chính sách không tồn tại: {dir_path}")

    md_files = sorted(dir_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"Không tìm thấy file chính sách .md nào trong thư mục: {dir_path}")

    documents: list[Document] = []
    seen_pairs: dict[tuple[str, str], Path] = {}

    for file_path in md_files:
        doc = load_document(file_path)
        pair = (doc.doc_id, doc.version)

        if pair in seen_pairs:
            raise ValueError(
                f"Phát hiện trùng lặp tài liệu: Cặp doc_id='{doc.doc_id}', version='{doc.version}' "
                f"đã xuất hiện tại '{seen_pairs[pair]}' và bị trùng ở '{file_path}'."
            )

        seen_pairs[pair] = file_path
        documents.append(doc)

    return documents
