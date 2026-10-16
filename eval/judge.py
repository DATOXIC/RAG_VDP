"""
Module Giám khảo LLM (eval/judge.py) - Nhiệm vụ 3 của Luồng 3 (PLAN.md).
Thẩm định định tính trường `reason` trong quyết định của Agent:
1. is_factual: Số liệu và thông tin có trung thực, không bịa đặt không.
2. is_logically_sound: Lập luận có dẫn dắt hợp lý tới quyết định không.

Điểm nghẽn (root cause) KHÔNG do giám khảo LLM chấm: verifier so tập `reason_codes` của agent với
`expect.reasons` bằng code (eval/verifier.py: verify_root_cause).
"""

from dataclasses import dataclass
import json
import os

from finco.types import Application
from eval.verifier import ExpectedDecision, AgentSubmission
import re
from finco.llm import call_llm

# ---------------------------------------------------------------------------
# 1. BẢN SYSTEM PROMPT HOÀN THIỆN CHO GIÁM KHẢO
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Bạn là Trưởng ban Kiểm soát Thẩm định Tín dụng của ngân hàng FinCo.
Nhiệm vụ của bạn là thẩm định tính TRUNG THỰC, HỢP LÝ và ĐÚNG ĐẮN trong Lời giải thích (reason) của chuyên viên AI.

[DỮ LIỆU THAM CHIẾU CHUẨN CỦA ĐỀ THI]:
- Hồ sơ khách hàng: {applicant_info}
- Quyết định chuẩn của Ban giám khảo: {expected_decision}
- Các chỉ số tài chính chuẩn: {expected_metrics}
- Trích đoạn chính sách áp dụng: {policy_context}

[BÀI LÀM CỦA THÍ SINH]:
- Quyết định thí sinh đưa ra: {actual_decision}
- Lời giải thích của thí sinh: {agent_reason}

[QUY TẮC PHÁN XÉT CHI TIẾT]:
1. is_factual (Đúng sự thật):
   - Mọi số liệu (DTI, lãi suất, thu nhập, tuổi...) và thông tin nếu có nêu trong reason PHẢI KHỚP với hồ sơ và số liệu chuẩn.
   - Nếu thí sinh bịa ra thông tin không có thật (nợ xấu ngoài hệ thống, CIC, tài sản đảm bảo, hoặc nêu sai số) -> Đánh dấu FALSE.

2. is_logically_sound (Nhất quán logic):
   - Lời giải thích có thực sự hỗ trợ và dẫn dắt hợp lý tới quyết định {actual_decision} hay không?
   - Không được có mâu thuẫn nội tại (ví dụ: khen khách hàng đủ mọi điều kiện nhưng lại ra quyết định từ chối).

3. is_valid:
   - TRUE khi và chỉ khi cả 2 tiêu chí trên (is_factual, is_logically_sound) ĐỀU LÀ TRUE.

[ĐỊNH DẠNG ĐẦU RA]:
Chỉ trả về một đối tượng JSON duy nhất (không bọc trong markdown hay viết thêm văn bản giải thích):
{{
  "is_factual": true,
  "is_logically_sound": true,
  "is_valid": true,
  "critique": "Nhận xét ngắn gọn 1-2 câu giải thích vì sao đạt hoặc chỉ rõ điểm sai sót"
}}
"""

# ---------------------------------------------------------------------------
# 2. KHUÔN MẪU DỮ LIỆU KẾT QUẢ ĐÁNH GIÁ (DATACLASS)
# ---------------------------------------------------------------------------
@dataclass
class JudgeResult:
    """Phiếu kết quả thẩm định định tính từ Giám khảo LLM."""
    app_id: str
    is_valid: bool = False                # Đạt toàn diện (True khi cả 2 đều True)
    is_factual: bool = False              # Các số liệu/dữ kiện trong reason có đúng thật không
    is_logically_sound: bool = False      # Logic lý do -> quyết định có hợp lý không
    critique: str = ""                    # Nhận xét ngắn gọn của Giám khảo
    judge_error: bool = False             # True = giám khảo lỗi (mạng, JSON hỏng...), KHÔNG phải agent sai


# ---------------------------------------------------------------------------
# 3. HÀM CHUẨN BỊ DỮ LIỆU ĐẦU VÀO CHO PROMPT
# ---------------------------------------------------------------------------
def format_judge_inputs(
    app: Application,
    expect: ExpectedDecision,
    actual: AgentSubmission,
    policy_context: str = ""
) -> dict[str, str]:
    """
    Trích xuất và định dạng toàn bộ thông tin hồ sơ vay, đáp án chuẩn và bài làm
    thành từ điển biến để điền vào template SYSTEM_PROMPT.
    """
    # 1. Định dạng thông tin khách hàng
    debt_str = f"{app.existing_debt_monthly:,} VNĐ" if app.existing_debt_monthly is not None else "Chưa khai báo (None)"
    docs_str = ", ".join(app.docs_submitted) if app.docs_submitted else "Không có"
    applicant_info = (
        f"Mã HS: {app.app_id}, Tên: {app.applicant_name}, Tuổi: {app.age}, "
        f"Khoản vay đề nghị: {app.amount:,} VNĐ kỳ hạn {app.term_months} tháng. "
        f"Nguồn thu: {app.income_type} (Chính: {app.monthly_income:,} VNĐ, Phụ: {app.supplementary_income:,} VNĐ). "
        f"Nghĩa vụ nợ hiện tại: {debt_str}. "
        f"Nợ quá hạn >30 ngày: {'Có' if app.overdue_30d_12m else 'Không'}. "
        f"Nhân viên FinCo: {'Có' if app.is_employee else 'Không'}. "
        f"Chứng từ nộp: {docs_str}."
    )

    # 2. Định dạng các chỉ số tài chính chuẩn từ đáp án
    metrics = []
    if expect.dti > 0:
        metrics.append(f"DTI chuẩn: {expect.dti * 100:.2f}% (Trần DTI: {expect.max_dti * 100:.1f}%)")
    if expect.annual_rate > 0:
        metrics.append(f"Lãi suất chuẩn: {expect.annual_rate * 100:.1f}%/năm")
    if expect.qualified_income > 0:
        metrics.append(f"Thu nhập hợp lệ: {expect.qualified_income:,} VNĐ")
    if expect.monthly_payment > 0:
        metrics.append(f"Trả góp hàng tháng (PMT): {expect.monthly_payment:,} VNĐ")

    expected_metrics = "; ".join(metrics) if metrics else "Không yêu cầu tính toán chỉ số tài chính"

    # 3. Ngữ cảnh chính sách (nếu trống thì liệt kê danh sách tài liệu chuẩn)
    if not policy_context:
        policy_context = f"Tài liệu quy định căn cứ: {', '.join(expect.cited)}"

    return {
        "applicant_info": applicant_info,
        "expected_decision": expect.decision,
        "expected_metrics": expected_metrics,
        "policy_context": policy_context,
        "actual_decision": actual.decision,
        "agent_reason": actual.reason if actual.reason else "(Không có lời giải thích)",
    }


def build_judge_prompt(
    app: Application,
    expect: ExpectedDecision,
    actual: AgentSubmission,
    policy_context: str = ""
) -> str:
    """Ráp toàn bộ dữ liệu vào SYSTEM_PROMPT hoàn chỉnh để gửi cho LLM."""
    inputs = format_judge_inputs(app, expect, actual, policy_context)
    return SYSTEM_PROMPT.format(**inputs)



def _extract_json_block(text: str) -> dict:
    """Bóc tách đối tượng JSON từ phản hồi của LLM: chịu được code fence và văn bản thừa quanh JSON."""
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start:end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Giám khảo không trả về một đối tượng JSON")
    return data


def _require_bool(data: dict, key: str) -> bool:
    """Lấy trường boolean. Chấp nhận true/false và chuỗi "true"/"false"; thiếu hoặc kiểu lạ thì lỗi.
    Không dùng bool(): bool("false") là True và sẽ chấm sai âm thầm."""
    value = data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"trường '{key}' không phải boolean: {value!r}")


def judge_single_decision(
    app: Application,
    expect: ExpectedDecision,
    actual: AgentSubmission,
    policy_context: str = ""
) -> JudgeResult:
    """Gọi LLM Giám khảo để thẩm định định tính lời giải thích (reason) của một hồ sơ."""
    prompt = build_judge_prompt(app, expect, actual, policy_context)
    
    try:
        # JUDGE_MODEL: nên khác model của agent để giám khảo không tự chấm bài của chính mình.
        content, usage = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=os.environ.get("JUDGE_MODEL") or None,
            temperature=0.0,
            max_tokens=500,
        )
        data = _extract_json_block(content)

        is_factual = _require_bool(data, "is_factual")
        is_logically_sound = _require_bool(data, "is_logically_sound")
        is_valid = is_factual and is_logically_sound

        return JudgeResult(
            app_id=actual.app_id,
            is_factual=is_factual,
            is_logically_sound=is_logically_sound,
            is_valid=is_valid,
            critique=data.get("critique", ""),
        )
    except Exception as e:
        return JudgeResult(
            app_id=actual.app_id,
            is_factual=False,
            is_logically_sound=False,
            is_valid=False,
            critique=f"Lỗi Giám khảo LLM: {e}",
            judge_error=True,
        )
