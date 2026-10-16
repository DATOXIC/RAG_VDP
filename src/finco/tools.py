from datetime import datetime
from typing import Optional

from finco import critic as critic_mod
from finco.rag.retriever import retrieve_hits
from finco.store import db
from finco.types import REASON_CODES, REASON_CODES_BY_DECISION, Decision

# Kéo dài mỗi lượt Chat
_CURRENT_APP_DATE: dict[str, str] = {}  # {app_id: submitted_date}
_ACTIVE_APP_ID: Optional[str] = None

# Kéo dài khi hết chương trình
_CRITIC: Optional[critic_mod.Critic] = None 


def set_critic(c: Optional[critic_mod.Critic]) -> None:
    """Bật/tắt Critic trong submit_decision. MCP server bật lúc khởi động."""
    global _CRITIC
    _CRITIC = c


def reset_session_state() -> None:
    """Xóa toàn bộ trạng thái phiên trong tiến trình (test, hoặc bắt đầu lượt eval mới)."""
    global _ACTIVE_APP_ID
    _CURRENT_APP_DATE.clear()
    _DTI_DONE.clear()
    _ACTIVE_APP_ID = None
    critic_mod.clear_evidence()


def search_policy(query: str, as_of_date: Optional[str] = None, top_k: int = 3) -> str:
    actual_k = max(1, min(int(top_k or 3), 5))
    clean_date = str(as_of_date).strip() if as_of_date is not None else ""

    if clean_date.lower() in ("none", "null", "undefined", ""):
        clean_date = _CURRENT_APP_DATE.get(_ACTIVE_APP_ID or "", "")

    as_of = clean_date or None
    text, hits = retrieve_hits(query=query, as_of_date=as_of, top_k=actual_k)

    if hits:
        critic_mod.record_retrieval(_ACTIVE_APP_ID, hits, as_of)
    return text


def render_application(app) -> str:
    """Định dạng hồ sơ thành văn bản cho LLM."""
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
    text = "\n".join(info)
    return text


def get_application(app_id: str) -> str:
    """
    1. Kiểm tra app_id rỗng
    2. Lấy dữ liệu từ DB
    3. Kiểm tra thông tin có đủ hay không
    4. Định dạng văn bản cho LLM
    """
    global _ACTIVE_APP_ID
    try:
        if not app_id or not app_id.strip():
            return "Error: app_id không được để trống"

        clean_id = app_id.strip().upper()

        app = db.get_application(clean_id)
        _CURRENT_APP_DATE[clean_id] = app.submitted_date
        _ACTIVE_APP_ID = clean_id

        text = render_application(app)
        critic_mod.record_application(clean_id, text, app.submitted_date)
        return text

    except db.AppNotFound:
        return f"Error: Không tìm thấy hồ sơ với mã '{app_id}'."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi lấy hồ sơ: {e}"


_DTI_DONE: set[str] = set()  # hồ sơ đã calc_dti thành công trong tiến trình MCP server này


def compute_dti(app, annual_rate: str) -> str:
    """Tính DTI thuần (không ghi trạng thái): dùng chung cho calc_dti và critic-agent. Lỗi trả chuỗi 'Error: ...'."""
    app_id = app.app_id
    if app.existing_debt_monthly is None:
        return (
            f"Error: Hồ sơ {app_id} chưa khai báo nghĩa vụ nợ hiện tại "
            "(existing_debt_monthly là None) theo CP-02. Không thể tính DTI, "
            "cần chuyển sang quyết định need_info."
        )

    rate_cleaned = str(annual_rate).strip().rstrip("%").strip()
    rate = float(rate_cleaned)
    if rate <= 0:
        return f"Error: annual_rate phải lớn hơn 0, nhận được {rate}."
    if rate > 1.0:
        rate = rate / 100.0  # Ví dụ: 15.5 -> 0.155 hoặc '17.0%' -> 0.17

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
        "- Căn cứ chính sách: CP-05 v1 (Quy định cách tính thu nhập đủ điều kiện và tỷ lệ DTI)",
    ]
    text = "\n".join(result_text)
    return text


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

        text = compute_dti(app, annual_rate)
        
        if not text.startswith("Error:"):
            _DTI_DONE.add(app_id)
            critic_mod.record_dti(app_id, text)
        return text
    
    except db.AppNotFound:
        return f"Error: Không tìm thấy hồ sơ với mã '{app_id}'."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi tính DTI: {e}"


def submit_decision(
    app_id: str,
    decision: str,
    reason: str,
    cited_docs: list[str],
    reason_codes: list[str],
) -> str:
    """
    Args:
        app_id: Mã hồ sơ vay (ví dụ: 'APP-001').
        decision: Trạng thái: 'approve' | 'reject' | 'escalate' | 'need_info'.
        reason: Lý do chi tiết nêu rõ các căn cứ điều kiện.
        cited_docs: Danh sách tài liệu chính sách làm căn cứ (ví dụ: ['CP-01 v1', 'CP-03']).
        reason_codes: Mã lý do có cấu trúc, phải khớp decision:
            approve -> ['APPROVED'];
            reject -> AGE | OVERDUE | AMOUNT_RANGE | TERM_RANGE | DTI_EXCEEDED;
            need_info -> MISSING_DOCS | MISSING_DEBT_DECLARATION;
            escalate -> ESCALATE_MANAGER | ESCALATE_COUNCIL | ESCALATE_EMPLOYEE.

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
        
        # 3. Kiểm tra reason_codes: tập đóng, và phải khớp với decision
        if isinstance(reason_codes, str):
            reason_codes = [reason_codes]

        if not isinstance(reason_codes, list):
            return "Error: reason_codes phải là một danh sách mã lý do."
        clean_codes = list(dict.fromkeys(str(c).strip().upper() for c in reason_codes if str(c).strip()))
        
        if not clean_codes:
            return (
                "Error: reason_codes không được rỗng. Với decision "
                f"'{clean_decision}' chọn trong: {', '.join(REASON_CODES_BY_DECISION[clean_decision])}."
            )
        unknown = [c for c in clean_codes if c not in REASON_CODES]
        if unknown:
            return f"Error: reason_codes không hợp lệ: {', '.join(unknown)}. Các mã hợp lệ: {', '.join(REASON_CODES)}."
        allowed = REASON_CODES_BY_DECISION[clean_decision]
        mismatched = [c for c in clean_codes if c not in allowed]
        if mismatched:
            return (
                f"Error: reason_codes {', '.join(mismatched)} không khớp decision '{clean_decision}'. "
                f"Với '{clean_decision}' chỉ được dùng: {', '.join(allowed)}."
            )

        ev = critic_mod.get_evidence(clean_app_id) if _CRITIC is not None else None
        tripped = ev is not None and ev.critic_rejections >= ev.max_rejections
        final_reason = reason.strip()
        # Breaker: chỉ ép escalate khi đã từng bị chặn vì thiếu bằng chứng cứng hoặc Critic bất đồng có trích dẫn.
        # Nếu cả lịch sử chỉ là "Critic không xác minh được" (lỗi hạ tầng) thì không phạt quyết định.
        escalate_only = tripped and any(s != "error" for s in ev.block_stages)
        if escalate_only:
            if clean_decision != "escalate" or clean_codes != ["ESCALATE_MANAGER"]:
                return (
                    f"Error: Critic đã chặn hồ sơ {clean_app_id} {ev.critic_rejections} lần và hết lượt sửa. "
                    "Chỉ còn được ghi decision='escalate' với reason_codes=['ESCALATE_MANAGER'] "
                    "(chuyển cấp có thẩm quyền xem xét)."
                )
            final_reason += f" [CRITIC_UNRESOLVED: quyết định tự động không được Critic xác nhận sau {ev.critic_rejections} lần kiểm tra]"
        else:
            # Quyết định dựa trên DTI (approve, escalate, reject vì DTI) chỉ được ghi sau khi đã tính DTI.
            if (clean_decision in ("approve", "escalate") or "DTI_EXCEEDED" in clean_codes) and clean_app_id not in _DTI_DONE:
                return (
                    f"Error: Chưa tính DTI cho hồ sơ {clean_app_id}. Hãy tra biểu lãi suất CP-03 lấy lãi suất theo kỳ hạn, "
                    f"gọi calc_dti, rồi mới ghi quyết định '{clean_decision}'."
                )

            if _CRITIC is not None:
                verdict = _CRITIC.review(
                    critic_mod.Proposal(clean_app_id, clean_decision, reason.strip(), clean_cited, clean_codes),
                    ev,
                    db.get_application(clean_app_id),
                )
                if verdict.ok:
                    # Đã tra nhưng quên trích dẫn: tự bổ sung (đã xác minh), không bắt Actor sửa.
                    clean_cited += [d for d in verdict.cited_add if d not in clean_cited]
                elif tripped and verdict.stage == "error":
                    final_reason += " [CRITIC_UNVERIFIED: Critic không xác minh được quyết định này; cần người xem lại]"
                else:
                    ev.critic_rejections += 1
                    ev.block_stages.append(verdict.stage)
                    left = ev.max_rejections - ev.critic_rejections
                    shown = [i if len(i) <= 170 else i[:167] + "..." for i in verdict.issues[:3]]
                    if len(verdict.issues) > 3:
                        shown.append(f"(+{len(verdict.issues) - 3} vấn đề khác)")
                    bullets = "\n".join(f"- {i}" for i in shown)
                    if left > 0:
                        tail = (
                            "Sửa rồi gọi lại submit_decision (search_policy với as_of_date = ngày nộp hồ sơ nếu thiếu "
                            f"tài liệu, hoặc sửa decision/reason_codes). Còn {left} lần sửa."
                        )
                    elif any(s != "error" for s in ev.block_stages):
                        tail = (
                            "HẾT lượt sửa. Lần gọi tiếp theo chỉ nhận decision='escalate', "
                            "reason_codes=['ESCALATE_MANAGER'] (giữ nguyên cited_docs)."
                        )
                    else:
                        tail = "Critic tạm thời không xác minh được. Gọi lại submit_decision: quyết định sẽ được ghi kèm cờ chưa xác minh."
                    return (
                        f"Error: CRITIC_REJECT [{verdict.stage}] lần {ev.critic_rejections}/{ev.max_rejections}: "
                        f"quyết định CHƯA được ghi.\n{bullets}\n{tail}"
                    )

        # 4. Tạo đối tượng Decision và gọi DB để ghi
        dec = Decision(
            app_id=clean_app_id,
            decision=clean_decision,
            reason=final_reason,
            cited_docs=clean_cited,
            reason_codes=clean_codes,
            decided_at=datetime.now().isoformat(timespec="seconds"),
        )

        db.insert_decision(dec)
        critic_mod.clear_evidence(clean_app_id)
        return (
            f"Thành công: Đã ghi nhận quyết định '{clean_decision}' cho hồ sơ {clean_app_id}.\n"
            f"- Căn cứ trích dẫn: {', '.join(clean_cited)}\n"
            f"- Lý do: {final_reason}"
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
            f"- Mã lý do (reason_codes): {', '.join(dec.reason_codes) if dec.reason_codes else 'Không có'}",
            f"- Thời điểm ghi nhận: {dec.decided_at}",
            f"- Lý do chi tiết: {dec.reason}",
        ]
        return "\n".join(info)

    except db.DecisionNotFound:
        return f"Error: Hồ sơ '{app_id}' chưa có quyết định nào trong hệ thống."
    except Exception as e:
        return f"Error: Đã xảy ra lỗi khi đọc quyết định: {e}"