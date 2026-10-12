"""
Module Giám khảo LLM (eval/judge.py) - Nhiệm vụ 3 của Luồng 3 (PLAN.md).
Thẩm định định tính trường `reason` trong quyết định của Agent:
1. is_factual: Số liệu và thông tin có trung thực, không bịa đặt không.
2. identifies_root_cause: Có bắt đúng điểm nghẽn chính của đề thi không.
3. is_logically_sound: Lập luận có dẫn dắt hợp lý tới quyết định không.
"""

from dataclasses import dataclass
from pathlib import Path
import json

from finco.store import db
from finco.types import Application
from eval.verifier import ExpectedDecision, AgentSubmission, EXPECTED_PATH
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
- Điểm nghẽn cốt lõi chuẩn: {expected_reasons}
- Các chỉ số tài chính chuẩn: {expected_metrics}
- Trích đoạn chính sách áp dụng: {policy_context}

[BÀI LÀM CỦA THÍ SINH]:
- Quyết định thí sinh đưa ra: {actual_decision}
- Lời giải thích của thí sinh: {agent_reason}

[QUY TẮC PHÁN XÉT CHI TIẾT]:
1. is_factual (Đúng sự thật):
   - Mọi số liệu (DTI, lãi suất, thu nhập, tuổi...) và thông tin nếu có nêu trong reason PHẢI KHỚP với hồ sơ và số liệu chuẩn.
   - Nếu thí sinh bịa ra thông tin không có thật (nợ xấu ngoài hệ thống, CIC, tài sản đảm bảo, hoặc nêu sai số) -> Đánh dấu FALSE.

2. identifies_root_cause (Bắt đúng điểm nghẽn):
   - Lời giải thích có chỉ ra đúng nguyên nhân gốc trong {expected_reasons} hay không?
   - (Ví dụ: đề thi yêu cầu DTI_EXCEEDED thì reason phải nói về tỷ lệ nợ/thu nhập; đề thi yêu cầu AGE thì phải nói về độ tuổi; đề thi yêu cầu MISSING_DOCS thì phải chỉ rõ thiếu chứng từ gì).
   - Nếu giải thích lan man, bắt sai nguyên nhân -> Đánh dấu FALSE.

3. is_logically_sound (Nhất quán logic):
   - Lời giải thích có thực sự hỗ trợ và dẫn dắt hợp lý tới quyết định {actual_decision} hay không?
   - Không được có mâu thuẫn nội tại (ví dụ: khen khách hàng đủ mọi điều kiện nhưng lại ra quyết định từ chối).

4. is_valid:
   - TRUE khi và chỉ khi cả 3 tiêu chí trên (is_factual, identifies_root_cause, is_logically_sound) ĐỀU LÀ TRUE.

[ĐỊNH DẠNG ĐẦU RA]:
Chỉ trả về một đối tượng JSON duy nhất (không bọc trong markdown hay viết thêm văn bản giải thích):
{{
  "is_factual": true,
  "identifies_root_cause": true,
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
    is_factual: bool = False              # Các số liệu/dữ kiện trong reason có đúng thật không
    identifies_root_cause: bool = False   # Có bắt đúng nguyên nhân cốt lõi không
    is_logically_sound: bool = False      # Logic lý do -> quyết định có hợp lý không
    is_valid: bool = False                # Đạt toàn diện (True khi cả 3 đều True)
    critique: str = ""                    # Nhận xét ngắn gọn của Giám khảo


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
        "expected_reasons": ", ".join(expect.reasons),
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
    """Bóc tách chuỗi JSON an toàn từ phản hồi của LLM (chống markdown code fence)."""
    cleaned = text.strip()
    # Loại bỏ code block ```json ... ``` nếu có
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        else:
            # Tìm cặp ngoặc { ... } rộng nhất
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
    return json.loads(cleaned)


def judge_single_decision(
    app: Application,
    expect: ExpectedDecision,
    actual: AgentSubmission,
    policy_context: str = ""
) -> JudgeResult:
    """Gọi LLM Giám khảo để thẩm định định tính lời giải thích (reason) của một hồ sơ."""
    prompt = build_judge_prompt(app, expect, actual, policy_context)
    
    try:
        content, usage = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        data = _extract_json_block(content)
        
        is_factual = bool(data.get("is_factual", False))
        identifies_root_cause = bool(data.get("identifies_root_cause", False))
        is_logically_sound = bool(data.get("is_logically_sound", False))
        is_valid = is_factual and identifies_root_cause and is_logically_sound
        
        return JudgeResult(
            app_id=actual.app_id,
            is_factual=is_factual,
            identifies_root_cause=identifies_root_cause,
            is_logically_sound=is_logically_sound,
            is_valid=is_valid,
            critique=data.get("critique", ""),
        )
    except Exception as e:
        return JudgeResult(
            app_id=actual.app_id,
            is_factual=False,
            identifies_root_cause=False,
            is_logically_sound=False,
            is_valid=False,
            critique=f"Lỗi Giám khảo LLM: {e}",
        )

# if __name__ == "__main__":
#     import sys
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8")

#     print("\n" + "=" * 80)
#     print("CHẠY THỬ NGHIỆM GIÁM KHẢO LLM (EVAL/JUDGE.PY) - 1 HỒ SƠ THẬT (APP-003)")
#     print("=" * 80)

#     # 1. Đọc dữ liệu hồ sơ và đáp án chuẩn của APP-003
#     with open(EXPECTED_PATH, encoding="utf-8") as f:
#         expected_data = json.load(f)

#     # APP-003 là hồ sơ thứ 3 trong expected_decisions.json
#     exp_003 = ExpectedDecision.from_dict(expected_data[2])
#     app_003 = db.get_application(exp_003.app_id)

#     print(f"[*] Hồ sơ: {app_003.app_id} - Khách hàng: {app_003.applicant_name}")
#     print(f"[*] Quyết định chuẩn: '{exp_003.decision}' | Điểm nghẽn chuẩn: {exp_003.reasons}")
#     print(f"[*] DTI chuẩn: {exp_003.dti * 100:.2f}% | Trần DTI: {exp_003.max_dti * 100:.1f}%\n")

#     # 2. Thử nghiệm Thí sinh A: Lời giải thích CHUẨN XÁC
#     sub_good = AgentSubmission(
#         app_id=exp_003.app_id,
#         decision="reject",
#         reason="Từ chối vì tỷ lệ nghĩa vụ nợ DTI = 63.04%, vượt quá ngưỡng trần 50% theo quy định tại CP-01 v1.",
#         cited_docs=["CP-01 v1", "CP-05"],
#     )

#     print("--- [LẦN CHẤM THI 1: BÀI LÀM CHUẨN MỰC] ---")
#     print(f"Thí sinh viết: \"{sub_good.reason}\"")
#     print("Đang gửi cho Giám khảo LLM thẩm định...")
#     res_good = judge_single_decision(app_003, exp_003, sub_good)

#     print(f"-> Đúng sự thật (is_factual)          : {res_good.is_factual}")
#     print(f"-> Đúng nguyên nhân (identifies_root) : {res_good.identifies_root_cause}")
#     print(f"-> Nhất quán logic (is_logically_sound): {res_good.is_logically_sound}")
#     print(f"-> KẾT LUẬN TOÀN DIỆN (is_valid)       : {'PASS [V]' if res_good.is_valid else 'FAIL [X]'}")
#     print(f"-> Nhận xét của Giám khảo              : {res_good.critique}\n")

#     # 3. Thử nghiệm Thí sinh B: Lời giải thích BỊA ĐẶT & BẮT SAI BỆNH
#     sub_bad = AgentSubmission(
#         app_id=exp_003.app_id,
#         decision="reject",
#         reason="Từ chối vì khách hàng 65 tuổi quá giới hạn tuổi và có nợ xấu nhóm 5 tại Vietcombank.",
#         cited_docs=["CP-01 v1"],
#     )

#     print("--- [LẦN CHẤM THI 2: BÀI LÀM BỊA ĐẶT / BẮT SAI BỆNH] ---")
#     print(f"Thí sinh viết: \"{sub_bad.reason}\"")
#     print("Đang gửi cho Giám khảo LLM thẩm định...")
#     res_bad = judge_single_decision(app_003, exp_003, sub_bad)

#     print(f"-> Đúng sự thật (is_factual)          : {res_bad.is_factual}")
#     print(f"-> Đúng nguyên nhân (identifies_root) : {res_bad.identifies_root_cause}")
#     print(f"-> Nhất quán logic (is_logically_sound): {res_bad.is_logically_sound}")
#     print(f"-> KẾT LUẬN TOÀN DIỆN (is_valid)       : {'PASS [V]' if res_bad.is_valid else 'FAIL [X]'}")
#     print(f"-> Nhận xét của Giám khảo              : {res_bad.critique}")
#     print("=" * 80 + "\n")