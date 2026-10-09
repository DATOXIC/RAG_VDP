from finco.types import Document
from finco.types import Chunk


def chunk_document(doc: Document) -> list[Chunk]:
    lines = doc.body.splitlines()

    chunks: list[Chunk] = []
    current_section = None
    current_lines: list[str] = []
    chunk_index = 0

    for line in lines:
        clean_line = line.strip()

        if clean_line.startswith("## "):

            if current_section is not None:
                body_text = "\n".join(current_lines).strip()

                if body_text: 
                    header = f"[{doc.doc_id} {doc.version} | hiệu lực từ {doc.effective_date}] {doc.title} – {current_section}"

                    full_text = f"{header}\n{body_text}"
                    
                    chunk_id = f"{doc.doc_id}-{doc.version}-{chunk_index:02d}"

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            version=doc.version,
                            effective_date=doc.effective_date,
                            section=current_section,
                            text=full_text,
                            expiry_date=doc.expiry_date,
                        )
                    )
                    chunk_index += 1

            current_section = clean_line[3:].strip()
            current_lines = []  
        else:
            if current_section is not None:
                current_lines.append(line)


    if current_section is not None:
        body_text = "\n".join(current_lines).strip()
        if body_text:
            header = f"[{doc.doc_id} {doc.version} | hiệu lực từ {doc.effective_date}] {doc.title} – {current_section}"
            full_text = f"{header}\n{body_text}"
            chunk_id = f"{doc.doc_id}-{doc.version}-{chunk_index:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    version=doc.version,
                    effective_date=doc.effective_date,
                    section=current_section,
                    text=full_text,
                    expiry_date=doc.expiry_date,
                )
            )
            
    return chunks

def chunk_all(docs: list[Document]) -> list[Chunk]:
    if not docs:
        return []

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for doc in docs:
        doc_chunks = chunk_document(doc)

        for c in doc_chunks:
            if c.chunk_id in seen:
                raise ValueError(
                    f"Phát hiện trùng lặp mã chunk_id='{c.chunk_id}' "
                    f"tại tài liệu {doc.doc_id} {doc.version}!"
                )
            chunks.append(c)
            
            seen.add(c.chunk_id)
    return chunks

    
            

