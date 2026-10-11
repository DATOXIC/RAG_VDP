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
    lines = doc.body.splitlines() # Tách thành list[str] khi gặp \n

    header_section: str = None
    body_section: list[str] = []
    map: list[tuple[str,str]] = []
    for l in lines:
        line = l.strip() # Chuẩn hóa 2 đầu

        if line.startswith("## "):
            if header_section is not None:
                body = "\n".join(body_section).strip()
                if body:
                    map.append((header_section,body))
            header_section=line[3:].strip()
            body_section = []
        else:
            if header_section is not None:
                body_section.append(l)

    if header_section is not None:
        body = "\n".join(body_section).strip()
        if body:
            map.append((header_section,body))

    return map


def _split_blocks(body: str) -> list[str]:
    """
    Chia theo chỉ mục nhỏ VD(1.1 1.2 1.3) 
    """
    lines = body.splitlines()
    buffer: list[str] = []
    blocks: list[str] = []
    for l in lines:
        line = l.strip()

        if CLAUSE_RE.match(line):
            if buffer :
                temp = "\n".join(buffer).strip()
                if temp:
                    blocks.append(temp)
                    buffer =[]
            buffer.append(l)
        else:
            buffer.append(l)

    if buffer:
        temp = "\n".join(buffer).strip()
        if temp:
            blocks.append(temp)
            buffer =[]

    return blocks



def _split_large_block(block: str, max_body_tokens: int, overlap_tokens: int = 40) -> list[str]:

    lines = block.splitlines()
    buffer: list[str] = []
    result: list[str] = []
    for line in lines:
        test = "\n".join(buffer + [line]).strip()
        if count_tokens(test) <= max_body_tokens or not buffer:
            buffer.append(line)
        else:
            part_text = "\n".join(buffer).strip()
            if part_text:
                result.append(part_text)

            # Lấy overlap
            overlap: list[str] = []
            count = 0
            for i in reversed(buffer):
                temp_count = count_tokens(i)
                if count + temp_count <= overlap_tokens:
                    count += temp_count
                    overlap.insert(0,i)
                else:
                    break

            buffer = overlap + [line]

    if buffer:
        part_text = "\n".join(buffer).strip()
        if part_text:
            result.append(part_text)

    return result


def _pack_blocks_clean(blocks: list[str], max_body_tokens: int) -> list[str]:
    """Gom các điều khoản sạch sẽ theo ranh giới tự nhiên (KHÔNG OVERLAP giữa các Block).

    Nếu có block nào đơn lẻ quá dài, tách riêng block đó bằng _split_large_block.
    """
    if not blocks:
        return []

    full_text = "\n\n".join(blocks).strip()
    if count_tokens(full_text) <= max_body_tokens:
        return [full_text]

    buffer: list[str] = []
    result: list[str] = []

    for block in blocks:
        if count_tokens(block) > max_body_tokens:
            if buffer:
                result.append("\n\n".join(buffer).strip())
                buffer = []
            result.extend(_split_large_block(block,max_body_tokens))
            continue

        test = "\n\n".join(buffer + [block]).strip()
        if count_tokens(test) > max_body_tokens:
            result.append("\n\n".join(buffer))
            buffer = [block]
        else:
            buffer.append(block)

    if buffer:
        result.append("\n".join(buffer))

    return result



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
        # Ước lượng độ dài Header tối đa 
        sample_header = _build_header(doc, section_title, part=9, total=9)
        header_tokens = count_tokens(sample_header)

        max_body_tokens = max(100, TOTAL_MAX_TOKENS - header_tokens - 5)

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
                    section=section_title,  
                    text=full_text,
                    expiry_date=doc.expiry_date,
                )
            )
            chunk_index += 1

    # return chunks


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
