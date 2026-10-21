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
import re
from typing import Any, Iterable, Mapping, Optional, Union

from finco.types import LogLine

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
    # TIÊU CHÍ 1b: ĐÚNG ĐIỂM NGHẼN (Root Cause), so tập mã lý do, không dùng LLM
    # ----------------------------------------------------
    root_cause_correct: bool = False        # Mã lý do chuẩn có nằm trong reason_codes của agent không
    actual_reason_codes: list[str] = field(default_factory=list)    # Mã lý do agent ghi
    expected_reason_codes: list[str] = field(default_factory=list)  # Mã lý do chuẩn (expected.reasons)
    
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
    reason_codes: list[str] = field(default_factory=list)  # Mã lý do có cấu trúc (submit_decision)
    tool_calls: list[str] = field(default_factory=list) # Thứ tự các tool đã gọi
    calculated_dti: Optional[float] = None        # DTI mà AI tính được (nếu có)
    used_rate: Optional[float] = None             # Lãi suất AI dùng trong calc_dti


# ---------------------------------------------------------------------------
# CẦU NỐI: TRACE (LogLine / dòng jsonl) -> AgentSubmission
# LLM chỉ trả văn bản. Thứ đáng tin là những gì agent đã GỌI qua tool: tên tool, args (JSON đã
# parse ở biên tool) và kết quả tool (chuỗi do code format). Verifier chấm trên những thứ đó.
# ---------------------------------------------------------------------------
TraceStep = Union[LogLine, Mapping[str, Any]]
_DTI_LINE = re.compile(r"\(DTI\):\s*([0-9]*\.?[0-9]+)")


def load_trace(path: Path, question_id: Optional[str] = None) -> list[dict]:
    """Đọc file log .jsonl; nếu có question_id thì chỉ giữ các dòng của câu đó. Bỏ qua dòng hỏng."""
    steps: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and (question_id is None or row.get("question_id") == question_id):
                steps.append(row)
    return steps


def _step_fields(step: TraceStep) -> tuple[str, dict, str]:
    """Lấy (tool, args, result) từ LogLine hoặc dict, chịu được dữ liệu thiếu/sai kiểu."""
    
    if isinstance(step, LogLine):
        tool, args, result = step.tool, step.args, step.result
    else:
        tool, args, result = step.get("tool"), step.get("args"), step.get("result")
    return (
        str(tool or ""),
        args if isinstance(args, dict) else {},
        result if isinstance(result, str) else "" if result is None else str(result),
    )


def _as_fraction(value: Optional[float]) -> Optional[float]:
    """15.5 (phần trăm) -> 0.155; 0.155 giữ nguyên. Cùng quy ước với tool calc_dti."""
    if value is None:
        return None
    return value / 100.0 if value > 1.0 else value


def _clean_codes(value: Any) -> list[str]:
    """Chuẩn hóa danh sách mã lý do: chuỗi đơn -> list, strip, viết hoa, bỏ trùng, giữ thứ tự."""
    
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(c).strip().upper() for c in value if str(c).strip()))


def submission_from_trace(app_id: str, steps: Iterable[TraceStep]) -> AgentSubmission:
    """Dựng bài nộp từ trace của MỘT hồ sơ.
    - Chỉ tính lời gọi tool THÀNH CÔNG (kết quả không bắt đầu bằng "Error:"): lời gọi lỗi không
      phải bằng chứng agent đã tra cứu / ghi / kiểm chứng.
    - submit_decision, get_decision, calc_dti phải đúng app_id đang chấm.
    - Quyết định, lý do, trích dẫn lấy từ args của lần submit_decision thành công cuối cùng.
    - DTI lấy từ kết quả calc_dti cuối (chuỗi do code sinh, không phải văn bản LLM); lãi suất
      lấy từ args agent truyền vào calc_dti.
    """
    
    target = app_id.strip().upper() # App_id muốn tìm
    tool_calls: list[str] = []
    submit_args: Optional[dict] = None
    dti: Optional[float] = None
    rate: Optional[float] = None

    for step in steps:
        tool, args, result = _step_fields(step)
        
        # "__answer__" là dòng ghi câu trả lời + usage cuối mỗi câu (không phải lời gọi tool).
        if not tool or tool.startswith("__") or result.lstrip().startswith("Error:"):
            continue

        if tool in ("submit_decision", "get_decision", "calc_dti"):
            if str(args.get("app_id", "")).strip().upper() != target:
                continue

        tool_calls.append(tool)

        if tool == "submit_decision":
            submit_args = args

        elif tool == "calc_dti":
            m = _DTI_LINE.search(result)
            if m:
                dti = float(m.group(1))
                try:
                    rate = _as_fraction(float(str(args.get("annual_rate")).strip().rstrip("%").strip().replace(",", ".")))
                except (TypeError, ValueError):
                    rate = None

    # Nếu Actor không submit
    if submit_args is None:
        return AgentSubmission(
            app_id=target, 
            decision="NONE", 
            reason="", 
            tool_calls=tool_calls,
            calculated_dti=dti, 
            used_rate=rate,
        )

    cited = submit_args.get("cited_docs")
    if isinstance(cited, str):
        cited = [cited]

    return AgentSubmission(
        app_id=target,
        decision=str(submit_args.get("decision", "")).strip().lower(),
        reason=str(submit_args.get("reason", "")).strip(),
        cited_docs=[str(c).strip() for c in cited if str(c).strip()] if isinstance(cited, list) else [],
        reason_codes=_clean_codes(submit_args.get("reason_codes")),
        tool_calls=tool_calls,
        calculated_dti=dti,
        used_rate=rate,
    )


# ---------------------------------------------------------------------------
# TRÍCH DẪN: "CP-01 v2" / "CP-01@v2" / "cp-01_v2" / "CP-01 (v2)" -> ("CP-01", "v2")
# ---------------------------------------------------------------------------
_CITATION = re.compile(r"([A-Za-z]+-\d+)(?:[\s@_\-(]*[vV](\d+))?")


def parse_citation(text: str) -> Optional[tuple[str, Optional[str]]]:
    """Trả (doc_id, version) hoặc None nếu không nhận ra. version None = không nêu phiên bản."""
    m = _CITATION.search(text or "")
    if not m:
        return None
    return m.group(1).upper(), (f"v{int(m.group(2))}" if m.group(2) else None)


def normalize_doc_id(doc_id: str) -> str:
    """Chuẩn hóa về 'CP-01@v2' (hoặc 'CP-01' nếu không nêu phiên bản)."""
    parsed = parse_citation(doc_id)
    if parsed is None:
        return (doc_id or "").strip().upper()
    doc, ver = parsed
    return f"{doc}@{ver}" if ver else doc


def _citation_hits(actual: str, expected: str) -> bool:
    """Trích dẫn actual có thỏa kỳ vọng expected không.
    Đáp án nêu phiên bản (CP-01@v2) thì actual phải đúng phiên bản đó; đáp án không nêu (CP-02)
    thì chỉ cần đúng mã tài liệu."""
    a, e = parse_citation(actual), parse_citation(expected)
    if a is None or e is None or a[0] != e[0]:
        return False
    return e[1] is None or a[1] == e[1]


def verify_decision(actual_dec: Optional[str], expected_dec: str) -> dict:
    """TIÊU CHÍ 1: So khớp quyết định (approve / reject / escalate / need_info)."""
    actual = actual_dec.strip().lower() if actual_dec else None
    expected = expected_dec.strip().lower()
    return {
        "decision_correct": (actual == expected),
        "actual_decision": actual,
        "expected_decision": expected,
    }


def verify_root_cause(actual_codes: list[str], expected_codes: list[str]) -> dict:
    """TIÊU CHÍ 1b: Mã lý do chuẩn phải nằm trong mã agent ghi (so tập hợp).
    Mã thừa không bị trừ điểm: quy trình cho phép phát hiện nhiều điều kiện không đạt cùng lúc."""
    actual = _clean_codes(actual_codes)
    expected = _clean_codes(expected_codes)
    return {
        "root_cause_correct": bool(expected) and set(expected) <= set(actual), # A ⊆ B
        "actual_reason_codes": actual,
        "expected_reason_codes": expected,
    }


def verify_citations(actual_cited: list[str], expected_cited: list[str], min_recall: float = 0.8) -> dict:
    """TIÊU CHÍ 2: Chấm điểm trích dẫn luật (Recall & Precision)."""

    actual_list = list(dict.fromkeys(normalize_doc_id(d) for d in actual_cited if d and d.strip()))
    expect_list = list(dict.fromkeys(normalize_doc_id(d) for d in expected_cited if d and d.strip()))

    if not expect_list:
        return {
            "citation_recall": 1.0,
            "citation_precision": 1.0,
            "citation_valid": True,
        }

    # Recall: bao nhiêu mục đáp án được trích đúng (đúng cả phiên bản nếu đáp án nêu phiên bản)
    # Precision: bao nhiêu mục agent trích có khớp một mục đáp án
    recall_hits = sum(1 for e in expect_list if any(_citation_hits(a, e) for a in actual_list))
    precision_hits = sum(1 for a in actual_list if any(_citation_hits(a, e) for e in expect_list))
    precision = precision_hits / len(actual_list) if actual_list else 0.0
    recall = recall_hits / len(expect_list)

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

    # Chuẩn hóa đơn vị: 47.09 (%) và 0.4709 đều được hiểu là 47,09%
    actual_dti = _as_fraction(actual_dti)
    used_rate = _as_fraction(used_rate)

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
    c1b = verify_root_cause(actual.reason_codes, expect.reasons)
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
        and c1b["root_cause_correct"]
        and c2["citation_valid"]
        and c3["trace_valid"]
        and c4["dti_correct"]
        and c4["rate_correct"]
    )

    # 3. Thu thập thông điệp lỗi chi tiết (phục vụ debug)
    errors: list[str] = []

    if not c1["decision_correct"]:
        errors.append(f"Sai quyết định (AI: '{c1['actual_decision']}' != Chuẩn: '{c1['expected_decision']}')")

    if not c1b["root_cause_correct"]:
        errors.append(
            f"Sai mã lý do (AI: {c1b['actual_reason_codes'] or 'không có'} "
            f"!= Chuẩn: {c1b['expected_reason_codes']})"
        )

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
        **c1b,
        **c2,
        **c3,
        **c4,
        is_overall_pass=is_overall_pass,
        error_message="; ".join(errors),
    )

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
        "root_cause_accuracy": round(sum(1 for r in results if r.root_cause_correct) / total * 100, 2),
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
    print(f"-> Đúng điểm nghẽn (Root cause)      : {metrics.get('root_cause_accuracy', 0)}%")
    print(f"-> Độ phủ trích dẫn TB (Recall)      : {metrics.get('citation_recall_avg', 0)}%")
    print(f"-> Tuân thủ quy trình gọi tool (Trace): {metrics.get('trace_valid_rate', 0)}%")
    print(f"-> Độ chuẩn xác con số (Math)        : {metrics.get('math_valid_rate', 0)}%")
    print("=" * 90 + "\n")
    
def export_results_csv(results: list[VerifyResult], output_path: Path = REPO_ROOT / "eval" / "out" / "agent_eval_results.csv") -> None:
    """Xuất kết quả chi tiết ra file CSV phục vụ báo cáo bảo vệ đồ án."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "app_id", "is_overall_pass", "decision_correct", "actual_decision", "expected_decision",
        "root_cause_correct", "actual_reason_codes", "expected_reason_codes",
        "citation_recall", "citation_precision", "citation_valid",
        "has_search_policy", "has_calc_dti", "read_after_write", "trace_valid",
        "dti_correct", "actual_dti", "expected_dti", "rate_correct", "error_message"
    ]
    
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            row = {k: d[k] for k in fieldnames}
            row["actual_reason_codes"] = "|".join(d["actual_reason_codes"])
            row["expected_reason_codes"] = "|".join(d["expected_reason_codes"])
            writer.writerow(row)
    print(f"[Verifier] Đã xuất báo cáo đánh giá chi tiết ra: {output_path}")


