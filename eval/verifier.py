"""
Đánh giá kết quả của LLM so với Expectation Result:
1. Độ chính xác quyết định
2. Độ chuẩn xác trích dẫn
3. Kiểm tra chuỗi gọi Tool
4. Kiểm tra tính toán các con số

--> Đánh giá vào những phần cố định của Decision
"""
import csv
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Optional

from finco.types import Decision

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = REPO_ROOT / "tests" / "expected_decisions.json"

@dataclass
class VerifyResult:
    """Phiếu kết quả sát hạch chi tiết đáp ứng đủ 4 tiêu chí của PLAN.md."""
    app_id: str
    # ----------------------------------------------------
    # TIÊU CHÍ 1: ĐỘ CHÍNH XÁC QUYẾT ĐỊNH (Decision Accuracy)
    # ----------------------------------------------------
    decision_correct: bool = False          # Đúng quyết định duyệt/từ chối không
    actual_decision: Optional[str] = None   # Quyết định AI đưa ra (approve/reject/...)
    expected_decision: str = "N/A"          # Quyết định chuẩn của đáp án
    # ----------------------------------------------------
    # TIÊU CHÍ 2: ĐỘ CHUẨN XÁC TRÍCH DẪN (Citation Accuracy)
    # ----------------------------------------------------
    citation_recall: float = 0.0            # Có trích dẫn đủ luật bắt buộc không (0.0 -> 1.0)
    citation_precision: float = 0.0         # Có bị trích dẫn thừa tài liệu rác không (0.0 -> 1.0)
    citation_valid: bool = False            # Đạt chuẩn trích dẫn hay không (Recall >= 0.8)
    # ----------------------------------------------------
    # TIÊU CHÍ 3: KIỂM TRA CHUỖI GỌI TOOL (Tool Call Trace)
    # ----------------------------------------------------
    has_search_policy: bool = False         # Có tra cứu chính sách trước khi quyết định không
    has_calc_dti: bool = False              # Có gọi tool tính toán nợ không (không được tính mồm)
    read_after_write: bool = False          # Có gọi get_decision sau khi submit_decision không
    trace_valid: bool = False               # Quy trình gọi tool có tuân thủ 100% không
    # ----------------------------------------------------
    # TIÊU CHÍ 4: KIỂM TRA TÍNH TOÁN CÁC CON SỐ (Numeric Math)
    # ----------------------------------------------------
    rate_correct: bool = False              # Lãi suất tra cứu và dùng tính toán có đúng không
    dti_correct: bool = False               # Con số DTI tính ra có khớp đáp án chuẩn không (sai số <= 0.5%)
    actual_dti: Optional[float] = None     # Con số DTI mà AI tính được
    expected_dti: float = 0.0               # Con số DTI chuẩn trong đề thi
    # ----------------------------------------------------
    # TỔNG KẾT
    # ----------------------------------------------------
    is_overall_pass: bool = False           # ĐỖ TOÀN DIỆN (Thỏa mãn cả 4 tiêu chí trên)
    error_message: str = ""                 # Liệt kê chi tiết tiêu chí nào bị trượt để debug

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
    cited_docs:     list[str] = field(default_factory=list) # Danh sách luật trích dẫn
    tool_calls: list[str] = field(default_factory=list) # Thứ tự các tool đã gọi
    calculated_dti: Optional[float] = None        # DTI mà AI tính được (nếu có)
    used_rate: Optional[float] = None             # Lãi suất AI dùng trong calc_dti

    @classmethod
    def from_decision(
        cls, 
        decision: Decision, 
        tool_calls: list[str], 
        calculated_dti: Optional[float] = None, 
        used_rate: Optional[float] = None
    ) -> "AgentSubmission":
        """Khởi tạo bài nộp trực tiếp từ đối tượng Decision + vết gọi tool."""
        return cls(
            app_id=decision.app_id,
            decision=decision.decision,
            reason=decision.reason,
            cited_docs=decision.cited_docs,
            tool_calls=tool_calls,
            calculated_dti=calculated_dti,
            used_rate=used_rate,
        )

def normalize_doc_id(doc_id: str) -> str:
    """Chuẩn hóa 'CP-01@v1' -> 'CP-01' để so sánh công bằng."""
    return doc_id.split("@")[0].strip().upper()


def verify_decision(actual_dec: Optional[str], expected_dec: str) -> dict:
    """TIÊU CHÍ 1: So khớp quyết định (approve / reject / escalate / need_info)."""
    actual = actual_dec.strip().lower() if actual_dec else None
    expected = expected_dec.strip().lower()
    return {
        "decision_correct": (actual == expected),
        "actual_decision": actual,
        "expected_decision": expected,
    }


def verify_citations(actual_cited: list[str], expected_cited: list[str], min_recall: float = 0.8) -> dict:
    """TIÊU CHÍ 2: Chấm điểm trích dẫn luật (Recall & Precision)."""
    actual_set = {normalize_doc_id(d) for d in actual_cited if d and d.strip()}
    expect_set = {normalize_doc_id(d) for d in expected_cited if d and d.strip()}

    if not expect_set:
        return {
            "citation_recall": 1.0,
            "citation_precision": 1.0,
            "citation_valid": True,
        }

    hits = actual_set.intersection(expect_set)
    precision = len(hits) / len(actual_set) if actual_set else 0.0
    recall = len(hits) / len(expect_set)

    return {
        "citation_recall": round(recall, 2),
        "citation_precision": round(precision, 2),
        "citation_valid": recall >= min_recall,
    }


def verify_tool_trace(tool_calls: list[str]) -> dict:
    """TIÊU CHÍ 3: Kiểm tra chuỗi gọi tool (Quy trình Read-Your-Writes)."""
    has_search = "search_policy" in tool_calls
    has_calc = "calc_dti" in tool_calls

    read_after_write = False
    if "submit_decision" in tool_calls:
        last_submit_idx = max(i for i, call in enumerate(tool_calls) if call == "submit_decision")
        read_after_write = "get_decision" in tool_calls[last_submit_idx + 1:]

    return {
        "has_search_policy": has_search,
        "has_calc_dti": has_calc,
        "read_after_write": read_after_write,
        "trace_valid": has_search and read_after_write,
    }


def verify_numeric_math(
    actual_dti: Optional[float], 
    expected_dti: float,
    used_rate: Optional[float], 
    expected_rate: float,
    dti_tolerance: float = 0.005,
    rate_tolerance: float = 0.001
) -> dict:
    """TIÊU CHÍ 4: Kiểm tra độ chính xác của các phép tính toán tài chính (DTI & Lãi suất)."""
    # Nếu hồ sơ không yêu cầu tính DTI (vd: loại do tuổi hoặc thiếu giấy tờ)
    if expected_dti <= 0:
        return {
            "dti_correct": True,
            "rate_correct": True,
            "actual_dti": actual_dti,
            "expected_dti": expected_dti,
        }

    # Kiểm tra DTI
    dti_correct = (actual_dti is not None and abs(actual_dti - expected_dti) <= dti_tolerance)

    # Kiểm tra Lãi suất
    rate_correct = True
    if expected_rate > 0:
        rate_correct = (used_rate is not None and abs(used_rate - expected_rate) <= rate_tolerance)

    return {
        "dti_correct": dti_correct,
        "rate_correct": rate_correct,
        "actual_dti": actual_dti,
        "expected_dti": expected_dti,
    }


def verify_decision_record(actual: AgentSubmission, expect: ExpectedDecision) -> VerifyResult:
    """
    Nhận vào một bài làm thực tế (actual) và một đáp án chuẩn (expect).
    Chấm điểm 4 khối tiêu chí độc lập và đóng gói kết quả bằng toán tử unpacking **.
    """
    # 1. Chấm điểm 4 khối tiêu chí độc lập
    c1 = verify_decision(actual.decision, expect.decision)
    c2 = verify_citations(actual.cited_docs, expect.cited)
    c3 = verify_tool_trace(actual.tool_calls)
    c4 = verify_numeric_math(
        actual_dti=actual.calculated_dti,
        expected_dti=expect.dti,
        used_rate=actual.used_rate,
        expected_rate=expect.annual_rate,
    )

    # 2. Đỗ toàn diện khi thỏa mãn đồng thời cả 4 tiêu chí
    is_overall_pass = (
        c1["decision_correct"]
        and c2["citation_valid"]
        and c3["trace_valid"]
        and c4["dti_correct"]
        and c4["rate_correct"]
    )

    # 3. Thu thập thông điệp lỗi chi tiết (phục vụ debug)
    errors: list[str] = []
    if not c1["decision_correct"]:
        errors.append(f"Sai quyết định (AI: '{c1['actual_decision']}' != Chuẩn: '{c1['expected_decision']}')")

    if not c2["citation_valid"]:
        errors.append(f"Thiếu luật trích dẫn (Recall: {c2['citation_recall']:.2f} < 0.8)")

    if not c3["trace_valid"]:
        trace_reasons = []
        if not c3["has_search_policy"]:
            trace_reasons.append("thiếu search_policy")
        if not c3["read_after_write"]:
            if "submit_decision" not in actual.tool_calls:
                trace_reasons.append("chưa gọi submit_decision")
            else:
                trace_reasons.append("vi phạm Read-Your-Writes (thiếu get_decision sau submit)")
        errors.append(f"Lỗi quy trình: {', '.join(trace_reasons)}")

    if not c4["dti_correct"]:
        errors.append(f"Sai DTI (AI: {c4['actual_dti']} != Chuẩn: {c4['expected_dti']})")

    if not c4["rate_correct"]:
        if actual.used_rate is None:
            errors.append(f"Thiếu thông tin lãi suất sử dụng (Chuẩn: {expect.annual_rate})")
        else:
            errors.append(f"Dùng sai lãi suất (AI: {actual.used_rate} != Chuẩn: {expect.annual_rate})")

    # 4. Trả về VerifyResult qua unpacking ** (gọn gàng, khớp 100% thuộc tính)
    return VerifyResult(
        app_id=actual.app_id,
        **c1,
        **c2,
        **c3,
        **c4,
        is_overall_pass=is_overall_pass,
        error_message="; ".join(errors),
    )

def verify_all_decisions(
    submissions: list[AgentSubmission], 
    expected_path: Path = EXPECTED_PATH
) -> list[VerifyResult]:

    with open(expected_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    expected_map = {
        item["app_id"]: ExpectedDecision.from_dict(item) 
        for item in raw_data
    }

    results: list[VerifyResult] = []

    for actual in submissions:
        app_id = actual.app_id
        if app_id in expected_map:
            expect = expected_map[app_id]
            results.append(verify_decision_record(actual, expect))
        else:
            # Ghi nhận bài thi có mã hồ sơ không hợp lệ
            results.append(
                VerifyResult(
                    app_id=app_id,
                    actual_decision=actual.decision,
                    actual_dti=actual.calculated_dti,
                    error_message=f"Mã hồ sơ {app_id} không tồn tại trong bộ đề thi",
                )
            )
    return results



# ========================AI-USAGE =================================

def calculate_summary_metrics(results: list[VerifyResult]) -> dict[str, float]:
    """Tính toán các chỉ số phần trăm đánh giá tổng thể theo PLAN.md."""
    total = len(results)
    if total == 0:
        return {}
    return {
        "total_cases": total,
        "pass_count": sum(1 for r in results if r.is_overall_pass),
        "overall_pass_rate": round(sum(1 for r in results if r.is_overall_pass) / total * 100, 2),
        "decision_accuracy": round(sum(1 for r in results if r.decision_correct) / total * 100, 2),
        "citation_recall_avg": round(sum(r.citation_recall for r in results) / total * 100, 2),
        "trace_valid_rate": round(sum(1 for r in results if r.trace_valid) / total * 100, 2),
        "math_valid_rate": round(sum(1 for r in results if (r.dti_correct and r.rate_correct)) / total * 100, 2),
    }


def print_summary_table(results: list[VerifyResult]) -> None:
    """In bảng tổng kết kết quả chấm thi trực quan ra console."""
    metrics = calculate_summary_metrics(results)
    print("\n" + "=" * 90)
    print(f"{'BẢNG KẾT QUẢ SÁT HẠCH AGENT THẨM ĐỊNH (FINCO VERIFIER)':^90}")
    print("=" * 90)
    print(f"{'Mã HS':<9} | {'Quyết định':<12} | {'Trích dẫn':<10} | {'Trace':<8} | {'Toán (DTI/Lãi)':<15} | {'Kết quả':<8} | {'Lỗi ghi nhận'}")
    print("-" * 90)
    for r in results:
        dec_str = "ĐÚNG" if r.decision_correct else f"SAI ({r.actual_decision})"
        cite_str = f"R={r.citation_recall:.2f}"
        trace_str = "ĐẠT" if r.trace_valid else "LỖI"
        math_str = "ĐẠT" if (r.dti_correct and r.rate_correct) else "SAI"
        status_str = "PASS [V]" if r.is_overall_pass else "FAIL [X]"
        err_str = r.error_message if r.error_message else "-"
        print(f"{r.app_id:<9} | {dec_str:<12} | {cite_str:<10} | {trace_str:<8} | {math_str:<15} | {status_str:<8} | {err_str}")
    print("=" * 90)
    print(f"Tổng số hồ sơ: {metrics.get('total_cases', 0)} | Số ca ĐỖ: {metrics.get('pass_count', 0)}")
    print(f"-> Tỷ lệ Đỗ toàn diện (Overall Pass) : {metrics.get('overall_pass_rate', 0)}%")
    print(f"-> Độ chính xác quyết định (Accuracy): {metrics.get('decision_accuracy', 0)}%")
    print(f"-> Độ phủ trích dẫn TB (Recall)      : {metrics.get('citation_recall_avg', 0)}%")
    print(f"-> Tuân thủ quy trình gọi tool (Trace): {metrics.get('trace_valid_rate', 0)}%")
    print(f"-> Độ chuẩn xác con số (Math)        : {metrics.get('math_valid_rate', 0)}%")
    print("=" * 90 + "\n")
def export_results_csv(results: list[VerifyResult], output_path: Path = REPO_ROOT / "eval" / "out" / "agent_eval_results.csv") -> None:
    """Xuất kết quả chi tiết ra file CSV phục vụ báo cáo bảo vệ đồ án."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "app_id", "is_overall_pass", "decision_correct", "actual_decision", "expected_decision",
        "citation_recall", "citation_precision", "citation_valid",
        "has_search_policy", "has_calc_dti", "read_after_write", "trace_valid",
        "dti_correct", "actual_dti", "expected_dti", "rate_correct", "error_message"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            writer.writerow({k: d[k] for k in fieldnames})
    print(f"[Verifier] Đã xuất báo cáo đánh giá chi tiết ra: {output_path}")


# if __name__ == "__main__":
#     import sys
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8")

#     # Chạy thử nghiệm giả lập bộ trọng tài trên toàn bộ 21 đề thi
#     with open(EXPECTED_PATH, encoding="utf-8") as f:
#         ground_truth = json.load(f)

#     print(f"\n[FinCo Verifier] Chạy kiểm thử tự động trên 21 hồ sơ trong {EXPECTED_PATH.name}...")

#     # Giả lập thí sinh nộp bài chuẩn 100% (Mock Perfect Agent)
#     mock_submissions = [
#         AgentSubmission(
#             app_id=item["app_id"],
#             decision=item["decision"],
#             reason="Thẩm định đạt yêu cầu chính sách FinCo",
#             cited_docs=item.get("cited", []),
#             tool_calls=["search_policy", "calc_dti", "submit_decision", "get_decision"],
#             calculated_dti=item.get("dti"),
#             used_rate=item.get("annual_rate"),
#         )
#         for item in ground_truth
#     ]

#     # Cố tình tạo 1 hồ sơ bị lỗi (APP-003) để kiểm chứng bộ bắt lỗi
#     mock_submissions[2].decision = "approve"  # Chuẩn là reject (DTI vượt trần)
#     mock_submissions[2].tool_calls = ["search_policy", "calc_dti"]  # Thiếu submit & get_decision

#     results = verify_all_decisions(mock_submissions)
#     print_summary_table(results)
#     export_results_csv(results)
