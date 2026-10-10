import re
import tiktoken
from finco.types import Chunk, Document

# ========= CONFIG ===============
TOTAL_MAX_TOKENS = 430
# Regex nhận diện dòng bắt đầu điều khoản (ví dụ: "1.1.", "3.4.")
CLAUSE_RE = re.compile(r"^\d+\.\d+\.")
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Ước lượng số lượng token của đoạn văn bản bằng cl100k_base."""
    return len(_enc.encode(text))


def _split_sections(doc: Document) -> list[tuple[str, str]]:
    """Tách thân bài Markdown thành các mục theo tiêu đề '## '.

    Bỏ qua phần tiêu đề cấp 1 '#' ở đầu file.
    Returns:
        list[tuple[str, str]]: Danh sách các cặp (tên_section, nội_dung_body).
    """
    lines = doc.body.splitlines()
    sections: list[tuple[str, str]] = []
    current_section = None
    current_lines: list[str] = []

    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith("## "):
            if current_section is not None:
                body_text = "\n".join(current_lines).strip()
                if body_text:
                    sections.append((current_section, body_text))
            current_section = clean_line[3:].strip()
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    if current_section is not None:
        body_text = "\n".join(current_lines).strip()
        if body_text:
            sections.append((current_section, body_text))

    return sections


def _split_blocks(body: str) -> list[str]:

    lines = body.splitlines()
    blocks: list[str] = []
    current_block_lines: list[str] = []

    for line in lines:
        clean_line = line.strip()
        if CLAUSE_RE.match(clean_line):
            if current_block_lines:
                block_text = "\n".join(current_block_lines).strip()
                if block_text:
                    blocks.append(block_text)
            current_block_lines = [line]
        else:
            current_block_lines.append(line)

    if current_block_lines:
        block_text = "\n".join(current_block_lines).strip()
        if block_text:
            blocks.append(block_text)

    # Nếu văn bản không dùng đánh số điều khoản, coi toàn bộ body là 1 block
    if not blocks and body.strip():
        blocks.append(body.strip())

    return blocks


def _split_large_block(block: str, max_body_tokens: int, overlap_tokens: int = 40) -> list[str]:

    lines = block.splitlines()
    parts: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        test_text = "\n".join(current_lines + [line]).strip()

        if count_tokens(test_text) <= max_body_tokens or not current_lines:
            current_lines.append(line)
        else:
            # Đóng phần hiện tại
            part_str = "\n".join(current_lines).strip()
            if part_str:
                parts.append(part_str)

            # Overlap nội bộ: lấy vài dòng cuối của phần trước
            overlap_lines: list[str] = []
            overlap_count = 0
            for prev_line in reversed(current_lines):
                t_count = count_tokens(prev_line)
                if overlap_count + t_count <= overlap_tokens:
                    overlap_lines.insert(0, prev_line)
                    overlap_count += t_count
                else:
                    break

            current_lines = overlap_lines + [line]

    if current_lines:
        part_str = "\n".join(current_lines).strip()
        if not (parts and part_str == parts[-1]):
            parts.append(part_str)

    return parts if parts else [block]


def _pack_blocks_clean(blocks: list[str], max_body_tokens: int) -> list[str]:
    """Gom các điều khoản sạch sẽ theo ranh giới tự nhiên (KHÔNG OVERLAP giữa các Block).

    Nếu có block nào đơn lẻ quá dài, tách riêng block đó bằng _split_large_block.
    """
    if not blocks:
        return []

    # 1. Nếu toàn bộ section vừa vặn trong ngân sách thân bài -> Giữ nguyên 1 phần
    full_text = "\n\n".join(blocks).strip()
    if count_tokens(full_text) <= max_body_tokens:
        return [full_text]

    # 2. Gom tham lam sạch sẽ theo từng điều khoản
    parts: list[str] = []
    current_blocks: list[str] = []

    for block in blocks:
        block_tokens = count_tokens(block)

        # Trường hợp A: Bản thân block này vượt quá ngân sách thân bài
        if block_tokens > max_body_tokens:
            # Nếu đang có các block gom dở trước đó, đóng phần đó lại
            if current_blocks:
                parts.append("\n\n".join(current_blocks).strip())
                current_blocks = []

            # Chia nhỏ nội bộ chính block khổng lồ này
            sub_parts = _split_large_block(block, max_body_tokens=max_body_tokens)
            parts.extend(sub_parts)
            continue

        # Trường hợp B: Block bình thường -> Thử gom vào current_blocks
        test_text = "\n\n".join(current_blocks + [block]).strip() if current_blocks else block
        if count_tokens(test_text) <= max_body_tokens:
            current_blocks.append(block)
        else:
            # Vượt ngưỡng: Đóng phần hiện tại
            if current_blocks:
                parts.append("\n\n".join(current_blocks).strip())
            # Bắt đầu phần mới sạch sẽ từ block hiện tại (Overlap giữa các block = 0!)
            current_blocks = [block]

    if current_blocks:
        parts.append("\n\n".join(current_blocks).strip())

    return parts


def _build_header(doc: Document, section: str, part: int, total: int) -> str:
    """Dựng dòng header chuẩn hóa, có hậu tố (phần x/y) khi có nhiều hơn 1 phần."""
    if total > 1:
        return (
            f"[{doc.doc_id} {doc.version} | hiệu lực từ {doc.effective_date}] "
            f"{doc.title} – {section} (phần {part}/{total})"
        )
    return (
        f"[{doc.doc_id} {doc.version} | hiệu lực từ {doc.effective_date}] "
        f"{doc.title} – {section}"
    )


def chunk_document(doc: Document) -> list[Chunk]:
    """Phân đoạn một tài liệu Document thành danh sách các Chunk chuẩn hóa."""
    sections = _split_sections(doc)
    chunks: list[Chunk] = []
    chunk_index = 0

    for section_title, section_body in sections:
        # Ước lượng độ dài Header tối đa (kèm hậu tố phần x/y)
        sample_header = _build_header(doc, section_title, part=9, total=9)
        header_tokens = count_tokens(sample_header)

        # Ngân sách token dành riêng cho phần thân: trừ đi Header và 5 token dự phòng
        max_body_tokens = max(100, TOTAL_MAX_TOKENS - header_tokens - 5)

        # Bóc tách điều khoản và gom các phần sạch sẽ
        blocks = _split_blocks(section_body)
        parts = _pack_blocks_clean(blocks, max_body_tokens=max_body_tokens)
        total_parts = len(parts)

        for part_idx, part_text in enumerate(parts, start=1):
            header = _build_header(doc, section_title, part=part_idx, total=total_parts)
            full_text = f"{header}\n{part_text}"
            chunk_id = f"{doc.doc_id}-{doc.version}-{chunk_index:02d}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    version=doc.version,
                    effective_date=doc.effective_date,
                    section=section_title,  # Giữ nguyên tên gốc để gom nhóm
                    text=full_text,
                    expiry_date=doc.expiry_date,
                )
            )
            chunk_index += 1

    return chunks


def chunk_all(docs: list[Document]) -> list[Chunk]:
    """Phân đoạn toàn bộ danh sách tài liệu Document thành các Chunk."""
    if not docs:
        return []

    all_chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    for doc in docs:
        doc_chunks = chunk_document(doc)
        for c in doc_chunks:
            if c.chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Phát hiện trùng lặp mã chunk_id='{c.chunk_id}' tại {doc.doc_id} {doc.version}!"
                )
            seen_chunk_ids.add(c.chunk_id)
        all_chunks.extend(doc_chunks)

    return all_chunks
