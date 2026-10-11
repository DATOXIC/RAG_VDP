"""
Đánh giá kết quả của LLM so với Expectation Result:
1. Độ chính xác quyết định
2. Độ chuẩn xác trích dẫn
3. Kiểm tra chuỗi gọi Tool
4. Kiểm tra tính toán các con số
"""
from dataclasses import dataclass, asdict, field
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
class ExpectedDecision:
    """Đáp án chuẩn của Ban giám khảo cho một hồ sơ."""
    app_id: str
    decision: str                     # "approve", "reject", "escalate", "need_info"
    policy_version: str               # "v1" hoặc "v2"
    reasons: list[str]                # Ví dụ: ["APPROVED"], ["DTI_EXCEEDED"]
    cited: list[str]                  # Ví dụ: ["CP-01@v1", "CP-02", "CP-03"]
    annual_rate: float                # Lãi suất chuẩn (ví dụ: 0.155)
    monthly_payment: int              # Khoản trả hàng tháng chuẩn
    qualified_income: int             # Thu nhập hợp lệ chuẩn
    dti: float                        # Tỷ lệ DTI chuẩn (ví dụ: 0.2462)
    max_dti: float                    # Ngưỡng trần DTI (ví dụ: 0.5)

    @classmethod
    def from_dict(cls, data: dict) -> "ExpectedDecision":
        """Chuyển đổi từ dict JSON sang Object có kiểu dữ liệu rõ ràng."""
        return cls(
            app_id=data["app_id"],
            decision=data["decision"],
            policy_version=data.get("policy_version", ""),
            reasons=data.get("reasons", []),
            cited=data.get("cited", []),
            annual_rate=float(data.get("annual_rate", 0.0)),
            monthly_payment=int(data.get("monthly_payment", 0)),
            qualified_income=int(data.get("qualified_income", 0)),
            dti=float(data.get("dti", 0.0)),
            max_dti=float(data.get("max_dti", 0.0)),
        )

@dataclass
class AgentSubmission:
    """Bài thi thực tế nộp về từ Agent thẩm định."""
    app_id: str
    decision: str                                 # Quyết định AI đưa ra
    reason: str                                   # Lời giải thích của AI
    cited_docs: list[str] = field(default_factory=list) # Danh sách luật trích dẫn
    tool_calls: list[str] = field(default_factory=list) # Thứ tự các tool đã gọi
    calculated_dti: Optional[float] = None        # DTI mà AI tính được (nếu có)
    used_rate: Optional[float] = None             # Lãi suất AI dùng trong calc_dti

def normalize_doc_id(doc_id: str) -> str:
    """Chuẩn hóa 'CP-01@v1' -> 'CP-01' để so sánh công bằng."""
    return doc_id.split("@")[0].strip().upper()

def verify_citations(actual_cited: list[str], expected_cited: list[str], min_recall: float = 0.8) -> dict:
    """
    Logic:

    """

    actual_set = {normalize_doc_id(d) for d in actual_cited}
    expect_set = {normalize_doc_id(d) for d in expected_cited}

    if not expect_set:
        return {
            "precision": 1.0, # Lấy chính xác bao nhiêu và nhầm bao nhiêu
            "recall": 1.0,    # Lấy được bao nhiêu trong cái đúng
            "missing": [],
            "extra": list(actual_set),
            "is_valid": True,
        }

    hits = actual_set.intersection(expect_set)

    precision = len(hits) / len(actual_set) if actual_set else 0.0
    recall = len(hits) / len(expect_set)
    missing = list(expect_set - actual_set)
    extra = list(actual_set - expect_set)

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "missing": missing,
        "extra": extra,
        "is_valid": recall >= min_recall,
    } 

def verify_tool_trace(tool_calls: list[str]) -> dict:
    """
    1. Kiểm tra có search_policy ?
    2. Kiểm tra có calc_dti ?
    3. Kiểm tra có get_decision sau khi submit_decision hay không
    """
    has_search = "search_policy" in tool_calls
    has_calc = "calc_dti" in tool_calls

    read_after_write = False
    if "submit_decision" in tool_calls:
        cur = tool_calls.index("submit_decision")

        if "get_decision" in tool_calls[cur+1:]:
            read_after_write = True

    trace_valid = has_search and read_after_write

    return {
        "has_search": has_search,
        "has_dti": has_calc,
        "read_after_write": read_after_write,
        "is_valid": trace_valid,
    }

def verify_numeric_math(
    actual_dti: Optional[float], 
    expected_dti: float,
    used_rate: Optional[float], 
    expected_rate: float,
    dti_tolerance: float = 0.005,
    rate_tolerance: float = 0.001
) -> dict:
    """Kiểm tra độ chính xác của các phép tính toán tài chính."""
    # Nếu hồ sơ không yêu cầu tính DTI (expected_dti <= 0, vd: reject do tuổi hoặc need_info do thiếu giấy tờ)
    if expected_dti <= 0:
        return {
            "dti_correct": True,
            "rate_correct": True,
            "is_valid": True,
        }

    # Kiểm tra DTI khi đề thi yêu cầu tính DTI
    dti_correct = False
    if actual_dti is not None:
        dti_correct = abs(actual_dti - expected_dti) <= dti_tolerance

    # Kiểm tra Lãi suất khi đề thi có lãi suất
    rate_correct = True
    if expected_rate > 0:
        rate_correct = (used_rate is not None and abs(used_rate - expected_rate) <= rate_tolerance)

    return {
        "dti_correct": dti_correct,
        "rate_correct": rate_correct,
        "is_valid": dti_correct and rate_correct,  
    }

def verify_decision_record(actual: AgentSubmission, expect: ExpectedDecision) -> VerifyResult:
    """
    Nhận vào một bài làm thực tế (actual) và một đáp án chuẩn (expect).
    Chấm điểm cả 4 tiêu chí và trả về đối tượng VerifyResult.
    """
    app_id = actual.app_id

    # 1. TIÊU CHÍ 1: So khớp quyết định (approve / reject / escalate / need_info)
    actual_decision = actual.decision.strip().lower() if actual.decision else None
    expected_decision = expect.decision.strip().lower()
    decision_correct = (actual_decision == expected_decision)

    # 2. TIÊU CHÍ 2: Chấm điểm trích dẫn luật
    citation_res = verify_citations(actual.cited_docs, expect.cited)

    # 3. TIÊU CHÍ 3: Kiểm tra chuỗi gọi tool (Quy trình Read-Your-Writes)
    trace_res = verify_tool_trace(actual.tool_calls)

    # 4. TIÊU CHÍ 4: Kiểm tra tính toán con số (DTI & Lãi suất)
    math_res = verify_numeric_math(
        actual_dti=actual.calculated_dti,
        expected_dti=expect.dti,
        used_rate=actual.used_rate,
        expected_rate=expect.annual_rate,
    )

    # ĐỖ TOÀN DIỆN khi thỏa mãn cả 4 tiêu chí:
    is_overall_pass = (
        decision_correct
        and citation_res["is_valid"]
        and trace_res["is_valid"]
        and math_res["is_valid"]
    )

    # Thu thập chi tiết lỗi cụ thể để ghi vào error_message (phục vụ debug)
    errors: list[str] = []
    if not decision_correct:
        errors.append(f"Sai quyết định (AI: '{actual_decision}' != Chuẩn: '{expected_decision}')")

    if not citation_res["is_valid"]:
        missing_str = ", ".join(citation_res["missing"])
        errors.append(f"Thiếu luật trích dẫn: [{missing_str}] (Recall: {citation_res['recall']})")

    if not trace_res["is_valid"]:
        trace_reasons = []
        if not trace_res["has_search"]:
            trace_reasons.append("thiếu search_policy")
        if not trace_res["read_after_write"]:
            trace_reasons.append("vi phạm Read-Your-Writes (thiếu get_decision sau submit)")
        errors.append(f"Lỗi quy trình: {', '.join(trace_reasons)}")

    if not math_res["dti_correct"]:
        errors.append(f"Sai DTI (AI: {actual.calculated_dti} != Chuẩn: {expect.dti})")
        
    if not math_res["rate_correct"] and actual.used_rate is not None:
        errors.append(f"Dùng sai lãi suất (AI: {actual.used_rate} != Chuẩn: {expect.annual_rate})")

    return VerifyResult(
        app_id=app_id,
        decision_correct=decision_correct,
        actual_decision=actual_decision,
        expected_decision=expected_decision,
        citation_recall=citation_res["recall"],
        citation_precision=citation_res["precision"],
        citation_valid=citation_res["is_valid"],
        has_search_policy=trace_res["has_search"],
        has_calc_dti=trace_res["has_dti"],
        read_after_write=trace_res["read_after_write"],
        trace_valid=trace_res["is_valid"],
        rate_correct=math_res["rate_correct"],
        dti_correct=math_res["dti_correct"],
        actual_dti=actual.calculated_dti,
        expected_dti=expect.dti,
        is_overall_pass=is_overall_pass,
        error_message="; ".join(errors),
    )