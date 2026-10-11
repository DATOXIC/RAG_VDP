"""
Đánh giá kết quả của LLM so với Expectation Result:
1. Độ chính xác quyết định
2. Độ chuẩn xác trích dẫn
3. Kiểm tra chuỗi gọi Tool
4. Kiểm tra tính toán các con số
"""
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class VerifyResult:
    """Phiếu kết quả sát hạch chi tiết đáp ứng đủ 4 tiêu chí của PLAN.md."""
    app_id: str
    # ----------------------------------------------------
    # TIÊU CHÍ 1: ĐỘ CHÍNH XÁC QUYẾT ĐỊNH (Decision Accuracy)
    # ----------------------------------------------------
    decision_correct: bool          # Đúng quyết định duyệt/từ chối không
    actual_decision: Optional[str]  # Quyết định AI đưa ra (approve/reject/...)
    expected_decision: str         # Quyết định chuẩn của đáp án
    # ----------------------------------------------------
    # TIÊU CHÍ 2: ĐỘ CHUẨN XÁC TRÍCH DẪN (Citation Accuracy)
    # ----------------------------------------------------
    citation_recall: float         # Có trích dẫn đủ luật bắt buộc không (0.0 -> 1.0)
    citation_precision: float      # Có bị trích dẫn thừa tài liệu rác không (0.0 -> 1.0)
    citation_valid: bool          # Đạt chuẩn trích dẫn hay không (Recall >= 0.8)
    # ----------------------------------------------------
    # TIÊU CHÍ 3: KIỂM TRA CHUỖI GỌI TOOL (Tool Call Trace)
    # ----------------------------------------------------
    has_search_policy: bool        # Có tra cứu chính sách trước khi quyết định không
    has_calc_dti: bool             # Có gọi tool tính toán nợ không (không được tính mồm)
    read_after_write: bool         # Có gọi get_decision sau khi submit_decision không
    trace_valid: bool              # Quy trình gọi tool có tuân thủ 100% không
    # ----------------------------------------------------
    # TIÊU CHÍ 4: KIỂM TRA TÍNH TOÁN CÁC CON SỐ (Numeric Math)
    # ----------------------------------------------------
    rate_correct: bool             # Lãi suất tra cứu và dùng tính toán có đúng không
    dti_correct: bool              # Con số DTI tính ra có khớp đáp án chuẩn không (sai số <= 0.5%)
    actual_dti: Optional[float]    # Con số DTI mà AI tính được
    expected_dti: float            # Con số DTI chuẩn trong đề thi
    # ----------------------------------------------------
    # TỔNG KẾT
    # ----------------------------------------------------
    is_overall_pass: bool          # ĐỖ TOÀN DIỆN (Thỏa mãn cả 4 tiêu chí trên)
    error_message: str = ""        # Liệt kê chi tiết tiêu chí nào bị trượt để debug

@dataclass
class 

def verify_decision_record():
    """
    Nhận vào một bài làm và một đáp án
    Returns:
        Kết quả 
    """
    ...