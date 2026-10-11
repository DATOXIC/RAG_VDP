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
from finco.types import Decision
from finco.filter import _validate_date
from finco.rag.chunker import count_tokens

PROJECT_ROOT = Path(__file__).parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
MAX_CONTEXT_TOKENS = 1500

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
            token = count_tokens(text)

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

        app_id = app_id.strip().upper()

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


def calc_dti(app_id: str, annual_rate: str) -> str:
    """ Tính DTI để biết người ngày dành bao nhiêu % thu nhập để trả nợ

    Args:
        annual_rate: Lãi suất năm
    
    1. Kiểm tra Khai báo nợ chưa
    2. Tính thu nhập 
    3. Tính khoản nợ phải trả mỗi tháng
    4. Tỉ lệ DTI
    """
    try:
        if not app_id or not app_id.strip():
            return "Error: app_id không được để trống"
        app_id = app_id.strip().upper()

        app = db.get_application(app_id)

        if app.existing_debt_monthly is None:
            return (
                f"Error: Hồ sơ {app_id} chưa khai báo nghĩa vụ nợ hiện tại "
                "(existing_debt_monthly là None) theo CP-02. Không thể tính DTI, "
                "cần chuyển sang quyết định need_info."
            )

        rate = float(annual_rate)
        if rate <= 0:
            return f"Error: annual_rate phải lớn hơn 0, nhận được {rate}."
        if rate > 1.0:
            rate = rate / 100.0  # Ví dụ: 15.5 -> 0.155

        if app.income_type == "salary": # Làm công
            qualified_income = round(app.monthly_income + 0.7 * app.supplementary_income)
            income_desc = f"{app.monthly_income:,} + 70% × {app.supplementary_income:,}"
        else:  # Làm tư
            qualified_income = round(0.8 * app.monthly_income + 0.5 * app.supplementary_income)
            income_desc = f"80% × {app.monthly_income:,} + 50% × {app.supplementary_income:,}"

        if qualified_income <= 0:
            return f"Error: Thu nhập đủ điều kiện bằng 0 hoặc âm ({qualified_income:,} VNĐ), không thể tính DTI."

        P = app.amount # Khoản vay
        n = app.term_months # Kỳ hạn vay
        r = rate / 12 # Lãi suất tháng

        PMT = round(P * r / (1.0 - (1.0 + r) ** (-n)))

        # Tính DTI
        existing_debt = app.existing_debt_monthly
        total_monthly_debt = PMT + existing_debt
        dti = round(total_monthly_debt / qualified_income, 4)
        dti_percent = round(dti * 100, 2)
        
        formula = (
            f"DTI = ({PMT:,} [PMT] + {existing_debt:,} [Nợ cũ]) / "
            f"{qualified_income:,} [Thu nhập] = {dti_percent}%"
        )
        result_text = [
            f"=== KẾT QUẢ TÍNH DTI HỒ SƠ {app_id} ===",
            f"- Số tiền vay: {P:,} VNĐ, kỳ hạn: {n} tháng, lãi suất: {rate * 100:.1f}%/năm",
            f"- Khoản trả hàng tháng của khoản vay mới (PMT): {PMT:,} VNĐ",
            f"- Thu nhập đủ điều kiện (CP-05): {qualified_income:,} VNĐ (Công thức: {income_desc})",
            f"- Nghĩa vụ nợ hiện tại: {existing_debt:,} VNĐ",
            f"- Tổng nghĩa vụ nợ hàng tháng: {total_monthly_debt:,} VNĐ",
            f"- Tỷ lệ nợ trên thu nhập (DTI): {dti} ({dti_percent}%)",
            f"- Diễn giải công thức: {formula}",
        ]
        return "\n".join(result_text)
    
    except db.AppNotFound:
        return f"Error: Không tìm thấy hồ sơ với mã '{app_id}'."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi tính DTI: {e}"


def submit_decision(app_id: str, decision: str, reason: str, cited_docs: list[str]) -> str:
    """
    Args:
        app_id: Mã hồ sơ vay (ví dụ: 'APP-001').
        decision: Trạng thái: 'approve' | 'reject' | 'escalate' | 'need_info'.
        reason: Lý do chi tiết nêu rõ các căn cứ điều kiện.
        cited_docs: Danh sách tài liệu chính sách làm căn cứ (ví dụ: ['CP-01 v1', 'CP-03']).

    1. Kiểm tra app_id
    2. Kiểm tra decision có hợp lệ
    3. Kiểm tra reason
    4. Kiểm tra cited_docs không được rỗng, Đây là bằng chứng quan trọng
    5. Tạo Class Decision và ghi vào DTB
    """
    try:
        # 1. Kiểm tra đầu vào
        if not app_id or not app_id.strip():
            return "Error: app_id không được để trống."
        
        clean_app_id = app_id.strip().upper()

        if not decision or not decision.strip():
            return "Error: decision không được để trống."
        
        clean_decision = decision.strip().lower()

        if clean_decision not in ("approve", "reject", "escalate", "need_info"):
            return (
                f"Error: decision '{decision}' không hợp lệ. Phải là một trong: "
                "'approve', 'reject', 'escalate', 'need_info'."
            )
        
        if not reason or not reason.strip():
            return "Error: reason (lý do) không được để trống."
        
        # 2. Kiểm tra cited_docs bắt buộc không được rỗng
        if not cited_docs or not isinstance(cited_docs, list) or len(cited_docs) == 0:
            return "Error: cited_docs không được rỗng. Quyết định phải trích dẫn ít nhất một chính sách làm căn cứ."
        
        # Làm sạch danh sách cited_docs
        clean_cited = [str(d).strip() for d in cited_docs if str(d).strip()]
        if not clean_cited:
            return "Error: cited_docs không chứa mã tài liệu hợp lệ nào."
        
        # 3. Tạo đối tượng Decision và gọi DB để ghi
        dec = Decision(
            app_id=clean_app_id,
            decision=clean_decision,
            reason=reason.strip(),
            cited_docs=clean_cited,
            decided_at=datetime.now().isoformat(timespec="seconds"),
        )

        db.insert_decision(dec)
        return (
            f"Thành công: Đã ghi nhận quyết định '{clean_decision}' cho hồ sơ {clean_app_id}.\n"
            f"- Căn cứ trích dẫn: {', '.join(clean_cited)}\n"
            f"- Lý do: {reason.strip()}"
            f" BƯỚC TIẾP THEO BẮT BUỘC: Bạn phải gọi get_decision('{clean_app_id}') "
            f"để xác nhận quyết định đã được lưu an toàn trước khi đưa ra Final Answer."
        )
    except db.AppNotFound:
        return f"Error: Không tìm thấy hồ sơ với mã '{app_id}'."
    except db.DecisionExists:
        return f"Error: Hồ sơ '{app_id}' đã có quyết định trong hệ thống, không thể ghi đè."
    except db.InvalidDecision as e:
        return f"Error: Quyết định không hợp lệ: {e}"
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi ghi quyết định: {e}"


def get_decision(app_id: str) -> str:
    """Đọc lại quyết định thẩm định đang lưu trong cơ sở dữ liệu để kiểm chứng.
    Khi nào dùng:
        Bắt buộc dùng ngay sau submit_decision để tự kiểm chứng xem quyết định
        đã được lưu chính xác vào cơ sở dữ liệu hay chưa (Read-Your-Writes).
    Khi nào KHÔNG dùng:
        Không dùng khi hồ sơ chưa từng được xử lý quyết định.
    Args:
        app_id: Mã hồ sơ vay cần kiểm tra (ví dụ: 'APP-001').
    Returns:
        Chuỗi văn bản chi tiết quyết định đang lưu trong DB, hoặc 'Error: ...' nếu chưa có.
    """
    try:
        if not app_id or not app_id.strip():
            return "Error: app_id không được để trống."
        
        clean_app_id = app_id.strip().upper()
        dec = db.get_decision(clean_app_id)

        info = [
            f"=== QUYẾT ĐỊNH ĐANG LƯU CHO HỒ SƠ {dec.app_id} ===",
            f"- Trạng thái quyết định: {dec.decision.upper()}",
            f"- Căn cứ chính sách (cited_docs): {', '.join(dec.cited_docs)}",
            f"- Thời điểm ghi nhận: {dec.decided_at}",
            f"- Lý do chi tiết: {dec.reason}",
        ]
        return "\n".join(info)

    except db.DecisionNotFound:
        return f"Error: Hồ sơ '{app_id}' chưa có quyết định nào trong hệ thống."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi đọc quyết định: {e}"