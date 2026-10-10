from finco.types import Chunk
from datetime import datetime
import os
from pathlib import Path
from typing import Optional
import tiktoken
from finco.rag.bm25 import BM25Index
from finco.rag.dense import VectorStore
from finco.rag.hybrid import HybridSearcher
from finco.store import db

PROJECT_ROOT = Path(__file__).parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
MAX_CONTEXT_TOKENS = 1500

_ENC = tiktoken.get_encoding("cl100k_base")
_SEARCHER: Optional[HybridSearcher] = None
_STORE: Optional[VectorStore] = None

def _get_searcher():
    """Khởi tạo một lần và tái sử dụng bộ tìm kiếm (Hybrid hoặc Dense)."""
    global _SEARCHER, _STORE
    if _STORE is None:
        _STORE = VectorStore()
        _STORE.load(INDEX_DIR)
    
    mode = os.environ.get("RETRIEVAL_MODE", "hybrid").lower()
    if mode == "dense":
        return _STORE
    
    if _SEARCHER is None:
        bm25 = BM25Index(_STORE.chunks)
        _SEARCHER = HybridSearcher(_STORE, bm25)
    return _SEARCHER
    
def _count_tokens(text: str) -> int:
    """Đếm số token"""
    return len(_ENC.encode(text))

def _validate_date(date_str: str) -> bool:
    """Kiểm tra chuỗi ngày có đúng định dạng YYYY-MM-DD hay không."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except (ValueError, AttributeError):
        return False


def search_policy(query: str, as_of_date: Optional[str] = None, top_k: int = 5) -> str:
    """
    1. Kiểm tra Query
    2. Kiểm tra as_of_date
    3. Ép tra top_k
    4. Tìm kiếm
    5. Đóng gói kết quả và quản lý Token
    """
    try:
        if not query or not query.strip():
            return "Error: Query không được để trống"
        query = query.strip()

        if as_of_date is not None:
            if not _validate_date(as_of_date):
                return f"Error: as_of_date:{as_of_date} không đúng định dạng"
            as_of_date = as_of_date.strip()

        top_k = max(1,min(int(top_k),5))

        searcher = _get_searcher()
        hits = searcher.search(query,top_k,as_of_date) # List[Hits]

        if not hits:
            return f"Không tìm thấy Policy nào phù hợp với Query và as_of_date:{as_of_date}"

        output: list[str] = []
        count = 0
        for hit in hits:
            chunk = hit.chunk
            header = f"[{chunk.doc_id} {chunk.version}. {chunk.effective_date}]"
            text = f"{header} {chunk.section}\n{chunk.text}"
            token = _count_tokens(text)

            if output and (count + token > MAX_CONTEXT_TOKENS):
                break

            output.append(text)
            count += token

        return "\n\n---\n\n".join(output)
    
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi tìm kiếm chính sách: {e}"



def get_application(app_id: str) -> str:
    """
    1. Kiểm tra app_id rỗng
    2. Lấy dữ liệu từ DB
    3. Kiểm tra thông tin có đủ hay không
    4. Định dạng văn bản cho LLM
    """
    try:
        if not app_id or not app_id.strip():
            return "Error: app_id không được để trống"

        app_id = app_id.strip()

        app = db.get_application(app_id)

        if app.existing_debt_monthly is None:
            debt = "Chưa khai báo (None)"
        else:
            debt = f"{app.existing_debt_monthly:,} VND" # cặp ":," trong fstring giúp phân cách số cho dễ đọc

        docs = ", ".join(app.docs_submitted) if app.docs_submitted else "Không có"

        info = [
                f"Mã hồ sơ: {app.app_id}",
                f"Tên khách hàng: {app.applicant_name}",
                f"Tuổi: {app.age}",
                f"Sản phẩm vay: {app.product}",
                f"Số tiền đề nghị vay: {app.amount:,} VNĐ",
                f"Kỳ hạn vay: {app.term_months} tháng",
                f"Loại nguồn thu nhập: {app.income_type} ({'Hưởng lương' if app.income_type == 'salary' else 'Tự kinh doanh'})",
                f"Thu nhập chính hàng tháng: {app.monthly_income:,} VNĐ",
                f"Thu nhập phụ hàng tháng: {app.supplementary_income:,} VNĐ",
                f"Nghĩa vụ nợ hiện tại mỗi tháng: {debt}",
                f"Là nhân viên FinCo: {'Có' if app.is_employee else 'Không'}",
                f"Có nợ quá hạn >30 ngày trong 12 tháng: {'Có' if app.overdue_30d_12m else 'Không'}",
                f"Chứng từ đã nộp: {docs}",
                f"Ngày nộp hồ sơ: {app.submitted_date}",
                f"Ghi chú hồ sơ: {app.note if app.note else 'Không có'}",
            ]
        return "\n".join(info)
    
    except db.AppNotFound:
        return f"Error: Không tìm thấy hồ sơ với mã '{app_id}'."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi lấy hồ sơ: {e}"