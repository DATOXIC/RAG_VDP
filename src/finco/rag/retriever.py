#======================= KHÔNG CÒN SỬ DỤNG ---> THAY THẾ BẰNG tools.py ======================

# from typing import Optional
# from finco.filter import validate_date
# from finco.rag.chunker import count_tokens
# from finco.rag.engine import get_searcher
# from finco.types import SearchHit

# MAX_TOP_K = 5
# MAX_CONTEXT_TOKENS = 1500


# def _format_hits(hits: list[SearchHit], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
#     """Gom các chunk thành chuỗi Markdown và kiểm soát ngân sách token."""
#     blocks: list[str] = []
#     used_tokens = 0

#     for hit in hits:
#         c = hit.chunk
#         header = f"[{c.doc_id} {c.version} · {c.effective_date}]"
#         text = f"{header} {c.section}\n{c.text}"
#         tokens = count_tokens(text)

#         # Nếu vượt quá ngân sách token thì dừng lại (nhưng luôn giữ ít nhất 1 chunk đầu)
#         if blocks and (used_tokens + tokens > max_tokens):
#             break

#         blocks.append(text)
#         used_tokens += tokens

#     return "\n\n---\n\n".join(blocks)


# def retrieve(query: str, as_of_date: Optional[str] = None, top_k: int = 3) -> str:
#     """Hàm duy nhất thực hiện toàn bộ quy trình tìm kiếm chính sách."""
#     try:
#         if not query or not query.strip():
#             return "Error: query không được rỗng."

#         if as_of_date is not None and not validate_date(as_of_date):
#             return f"Error: as_of_date '{as_of_date}' không đúng định dạng YYYY-MM-DD."

#         top_k = max(1, min(int(top_k), MAX_TOP_K))

#         # 1. Lấy searcher từ engine.py (Lazy load, tự nhận diện biến môi trường)
#         searcher = get_searcher()

#         # 2. Tìm kiếm (đã tự động lọc ngày qua filter.py)
#         hits = searcher.search(query=query.strip(), top_k=top_k, as_of_date=as_of_date)

#         if not hits:
#             when = f"tại ngày {as_of_date}" if as_of_date else "hiện hành"
#             return f"Không tìm thấy đoạn chính sách nào phù hợp {when}."

#         # 3. Đóng gói kết quả dạng chuỗi cho LLM
#         return _format_hits(hits)

#     except Exception as e:
#         return f"Error: Truy xuất thất bại: {e}"