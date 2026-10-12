"""
Định nghĩa các kiểu dữ liệu và dataclass dùng chung trong hệ thống FinCo.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

@dataclass
class Application:
    """Hồ sơ vay vốn khách hàng (ánh xạ từ dữ liệu data/applications.json).
    Attributes:
        app_id: Mã định danh duy nhất của hồ sơ (vd: "APP-001").
        applicant_name: Họ và tên người nộp đơn vay.
        age: Tuổi người vay tại ngày nộp hồ sơ.
        product: Tên gói sản phẩm vay (vd: "FinCo Easy").
        amount: Số tiền đề nghị vay (đơn vị: VNĐ, số nguyên).
        term_months: Kỳ hạn vay tính theo tháng (vd: 24, 36).
        income_type: Loại nguồn thu nhập ("salary" - hưởng lương, "self_employed" - tự kinh doanh).
        monthly_income: Thu nhập chính bình quân hàng tháng (đơn vị: VNĐ).
        supplementary_income: Thu nhập phụ bình quân hàng tháng (đơn vị: VNĐ).
        existing_debt_monthly: Nghĩa vụ nợ hiện tại mỗi tháng (VNĐ).
            Giá trị None biểu thị chưa khai báo (khác với 0 VNĐ - không có nợ).
        is_employee: Khách hàng có phải là nhân viên nội bộ FinCo hay không.
        overdue_30d_12m: Có khoản nợ quá hạn trên 30 ngày trong 12 tháng gần nhất hay không.
        docs_submitted: Danh sách mã chứng từ đã nộp (vd: ["CCCD", "HDLD", "SK_LUONG_3T"]).
        submitted_date: Ngày nộp hồ sơ theo định dạng "YYYY-MM-DD".
        note: Ghi chú bổ sung cho hồ sơ vay.
    """
    app_id: str
    applicant_name: str
    age: int
    product: str
    amount: int
    term_months: int
    income_type: Literal["salary", "self_employed"]
    monthly_income: int
    supplementary_income: int
    existing_debt_monthly: Optional[int]
    is_employee: bool
    overdue_30d_12m: bool
    submitted_date: str
    docs_submitted: list[str] = field(default_factory=list)
    note: str = ""


# Mã lý do có cấu trúc đi kèm mỗi quyết định: tập đóng, khớp tests/expected_decisions.json.
# Dùng để eval so tập hợp với đáp án chuẩn mà không cần LLM đọc văn bản reason.
REASON_CODES_BY_DECISION: dict[str, tuple[str, ...]] = {
    "approve": ("APPROVED",),
    "reject": ("AGE", "OVERDUE", "AMOUNT_RANGE", "TERM_RANGE", "DTI_EXCEEDED"),
    "need_info": ("MISSING_DOCS", "MISSING_DEBT_DECLARATION"),
    "escalate": ("ESCALATE_MANAGER", "ESCALATE_COUNCIL", "ESCALATE_EMPLOYEE"),
}
REASON_CODES: tuple[str, ...] = tuple(c for codes in REASON_CODES_BY_DECISION.values() for c in codes)


@dataclass
class Decision:
    """Quyết định thẩm định đã ghi cho một hồ sơ (bảng decisions; mỗi hồ sơ một quyết định).
    Attributes:
        app_id: Mã hồ sơ được quyết định (khóa chính).
        decision: Trạng thái: "approve", "reject", "escalate" hoặc "need_info".
        reason: Lý do, nêu rõ điều kiện làm căn cứ.
        decided_at: Thời điểm ghi quyết định.
        cited_docs: Tài liệu chính sách làm căn cứ (vd: ["CP-01 v2", "CP-05"]); không được rỗng.
        reason_codes: Mã lý do có cấu trúc (vd: ["DTI_EXCEEDED"]), thuộc REASON_CODES và khớp trạng thái.
    """
    app_id: str
    decision: Literal["approve", "reject", "escalate", "need_info"]
    reason: str
    decided_at: str
    cited_docs: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class Document:
    """Một file tài liệu chính sách sau khi đọc front matter (data/policies/*.md), chưa tách chunk.
    Attributes:
        doc_id: Mã tài liệu (vd: "CP-01"). Cặp (doc_id, version) là duy nhất.
        title: Tên tài liệu.
        version: Phiên bản (vd: "v1", "v2").
        effective_date: Ngày bắt đầu hiệu lực, "YYYY-MM-DD".
        owner: Đơn vị sở hữu tài liệu.
        body: Phần Markdown sau front matter.
        source_path: Đường dẫn file nguồn (để truy lỗi).
        expiry_date: Ngày cuối còn hiệu lực. None = vẫn còn hiệu lực.
        supersedes: Phiên bản bị thay thế (vd: "CP-01 v1").
        status: "active" hoặc "superseded".
    """
    doc_id: str
    title: str
    version: str
    effective_date: str
    owner: str
    body: str
    source_path: str
    expiry_date: Optional[str] = None
    supersedes: Optional[str] = None
    status: Optional[str] = None


@dataclass
class Chunk:
    """Một mục "## " của tài liệu: đơn vị được embedding và tìm kiếm.
    Mang theo metadata của tài liệu mẹ vì lọc theo ngày hiệu lực chạy trên chunk.
    Attributes:
        chunk_id: Mã duy nhất (vd: "CP-01-v2-03").
        doc_id: Mã tài liệu mẹ.
        version: Phiên bản tài liệu mẹ.
        effective_date: Ngày hiệu lực của tài liệu mẹ, "YYYY-MM-DD".
        section: Tiêu đề mục "##".
        text: Nội dung sẽ được embedding, gồm dòng đầu "[CP-01 v2 | hiệu lực từ ...]".
        expiry_date: Ngày cuối còn hiệu lực của tài liệu mẹ. None = còn hiệu lực.
    """
    chunk_id: str
    doc_id: str
    version: str
    effective_date: str
    section: str
    text: str
    expiry_date: Optional[str] = None


@dataclass
class SearchHit:
    """Một kết quả truy xuất: chunk kèm điểm số.
    Attributes:
        chunk: Chunk tìm được.
        score: Điểm cuối cùng dùng để xếp hạng (sau chuẩn hóa/trộn nếu là hybrid).
        dense_score: Điểm cosine thô
        bm25_score: Điểm BM25 thô
    """
    chunk: Chunk
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None


@dataclass
class DtiResult:
    """Kết quả tool calc_dti.
    Attributes:
        monthly_payment: Khoản trả hàng tháng của khoản vay mới (VNĐ, làm tròn).
        qualified_income: Thu nhập đủ điều kiện theo CP-05 (VNĐ).
        dti: Tỷ lệ nợ trên thu nhập, dạng số thập phân (0.4709 = 47,09%).
        formula: Chuỗi mô tả cách tính, để agent trích dẫn và verifier đối chiếu.
    """
    monthly_payment: int
    qualified_income: int
    dti: float
    formula: str


@dataclass
class Usage:
    """Mức dùng LLM của một câu hỏi (yêu cầu USAGE trong logs).
    Attributes:
        calls: Số lần gọi LLM.
        prompt_tokens: Tổng token đầu vào.
        completion_tokens: Tổng token đầu ra.
        max_prompt: Số token đầu vào lớn nhất của một lần gọi.
    """
    calls: int
    prompt_tokens: int
    completion_tokens: int
    max_prompt: int


@dataclass
class LogLine:
    """Một dòng log .jsonl cho mỗi bước của agent.
    Attributes:
        ts: Thời điểm, ISO 8601.
        run_id: Mã lần chạy.
        question_id: Mã câu hỏi (id trong testset, hoặc mã phiên REPL).
        step: Số thứ tự bước trong vòng lặp.
        tool: Tên tool được gọi.
        args: Tham số truyền cho tool. Tool do MCP khám phá động nên tham số khác nhau
            theo từng tool, đây là chỗ duy nhất chấp nhận dict.
        result: Kết quả tool dạng chuỗi (lỗi bắt đầu bằng "Error: ").
        usage: Mức dùng LLM tích lũy.
        cache_hit: Có dùng kết quả cache hay không.
    """
    ts: str
    run_id: str
    question_id: str
    step: int
    tool: str
    args: dict[str, Any]
    result: str
    usage: Usage
    cache_hit: bool = False


@dataclass
class TestCase:
    """Một dòng của tests/testset.jsonl.
    Attributes:
        id: Mã câu hỏi.
        question: Câu hỏi gửi cho agent.
        type: Loại câu (tra cứu, tổng hợp, phiên bản, tính toán, thiếu dữ liệu, no_answer, hành động).
        expected_answer: Đáp án mong đợi (văn bản).
        expected_docs: Tài liệu phải được trích (vd: ["CP-01 v2"]).
        is_no_answer: True nếu chính sách không có câu trả lời và agent phải từ chối.
        expected_decision: Quyết định mong đợi nếu là câu thẩm định, ngược lại None.
        target_app_id: Hồ sơ được hỏi tới, nếu có.
    """
    id: str
    question: str
    type: str
    expected_answer: str
    expected_docs: list[str] = field(default_factory=list)
    is_no_answer: bool = False
    expected_decision: Optional[Literal["approve", "reject", "escalate", "need_info"]] = None
    target_app_id: Optional[str] = None