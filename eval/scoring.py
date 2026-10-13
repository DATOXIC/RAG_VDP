"""Chấm điểm: hàm thuần trên log, không gọi agent, LLM hay mạng.

Luồng: log .jsonl --records_from_log--> RunRecord --score_question--> QuestionScore --summarize--> số liệu.

Nguyên tắc chấm (bằng code, không dùng LLM):
- Câu hành động / hồ sơ: verifier so quyết định, mã lý do, trích dẫn, DTI, lãi suất, cặp ghi + đọc lại
  với tests/expected_decisions.json (dựng từ trace của agent, không đọc văn bản tự do).
- Câu no_answer: đạt khi agent từ chối đúng (câu chuẩn REFUSAL_MARKER) và KHÔNG ghi quyết định.
- Câu còn lại: đạt khi câu trả lời chứa đủ các "ý chính" trong expected_answer (đánh dấu **như này**),
  trích đúng tài liệu mong đợi (đúng phiên bản nếu có "@v"), không từ chối nhầm.
- Faithfulness theo luật: mọi tài liệu mà câu trả lời trích phải xuất hiện trong kết quả công cụ.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from eval.verifier import (
    ExpectedDecision,
    parse_citation,
    submission_from_trace,
    verify_citations,
    verify_decision_record,
)
from finco.types import TestCase, Usage

# Câu mở đầu mà prompt bắt agent dùng khi tài liệu không có đáp án (phải khớp prompts/agent_system.txt).
REFUSAL_MARKER = "Không có thông tin này trong tài liệu chính sách của FinCo"
# Cách nói khác của một lời từ chối: chỉ dùng cho câu no_answer (mọi câu đều đáng nghi nếu có cụm này).
_REFUSAL_PHRASES = (
    "không có thông tin", "không tìm thấy", "không có quy định", "không đề cập",
    "không có trong tài liệu", "không được nêu", "không có dữ liệu",
)
ANSWER_TOOL = "__answer__"  # dòng log cuối mỗi câu (khớp finco.agent.ANSWER_TOOL)

# --------------------------------------------------------------------------- chuẩn hóa văn bản
# Model hay viết "CP‑02" bằng gạch nối không ngắt (U+2011) hoặc en/em dash; coi như "-".
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def normalize(text: str) -> str:
    """Đưa văn bản về dạng so khớp được: chữ thường, bỏ **, 5.347.909 -> 5347909, 15,5 -> 15.5."""
    t = unicodedata.normalize("NFC", text or "").translate(_DASHES).lower().replace("*", "")
    t = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", t)  # dấu phân cách hàng nghìn
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)             # dấu phẩy thập phân
    t = re.sub(r"\s*%", "%", t)
    return re.sub(r"\s+", " ", t).strip()




def key_facts(expected_answer: str) -> list[str]:
    """Các ý chính phải có trong câu trả lời: phần được đánh dấu **như này** trong expected_answer."""
    return [f.strip() for f in re.findall(r"\*\*(.+?)\*\*", expected_answer or "") if f.strip()]


def is_refusal(answer: str, lenient: bool = False) -> bool:
    """Câu trả lời có phải lời từ chối không. lenient=True chấp nhận cả cách nói khác (cho câu no_answer)."""
    a = normalize(answer)
    if normalize(REFUSAL_MARKER) in a:
        return True
    return lenient and any(p in a for p in _REFUSAL_PHRASES)


def cited_in_text(answer: str) -> list[str]:
    """Các trích dẫn kiểu 'CP-01 v2' / '[CP-05]' trong câu trả lời, chuẩn hóa 'CP-01@v2' / 'CP-05'."""
    out = []
    for m in re.finditer(r"CP-\d{2}(?:\s*[@_]?\s*v\d+)?", (answer or "").translate(_DASHES), flags=re.IGNORECASE):
        parsed = parse_citation(m.group(0))
        if parsed:
            doc, ver = parsed
            out.append(f"{doc}@{ver}" if ver else doc)
    return list(dict.fromkeys(out))



# --------------------------------------------------------------------------- dữ liệu một lượt
@dataclass
class RunRecord:
    """Những gì còn lại của một câu sau khi chạy: đủ để chấm lại mà không cần agent."""
    question_id: str
    trace: list[dict]            # các dòng log tool (không gồm __answer__)
    final_answer: str = ""
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    success: bool = False
    error: Optional[str] = None
    complete: bool = False       # False = log thiếu dòng __answer__ (chạy dở)


def records_from_log(path: Path) -> dict[str, RunRecord]:
    """Đọc file log .jsonl (nhiều câu) thành {question_id: RunRecord}. Bỏ qua dòng hỏng."""
    records: dict[str, RunRecord] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "question_id" not in row:
                continue
            rec = records.setdefault(str(row["question_id"]), RunRecord(question_id=str(row["question_id"]), trace=[]))
            if row.get("tool") == ANSWER_TOOL:
                u = row.get("usage") or {}
                rec.final_answer = row.get("result", "") or ""
                rec.usage = Usage(
                    calls=int(u.get("calls", 0)), prompt_tokens=int(u.get("prompt_tokens", 0)),
                    completion_tokens=int(u.get("completion_tokens", 0)), max_prompt=int(u.get("max_prompt", 0)),
                )
                args = row.get("args") or {}
                rec.success = bool(args.get("success"))
                rec.error = args.get("error")
                rec.complete = True
            else:
                rec.trace.append(row)
    return records



# --------------------------------------------------------------------------- chấm điểm
@dataclass
class QuestionScore:
    id: str
    type: str
    target_app_id: str
    passed: bool = False
    success: bool = False                    # agent chạy xong không lỗi
    decision_ok: Optional[bool] = None       # câu hành động / hồ sơ
    root_cause_ok: Optional[bool] = None
    verify_ok: Optional[bool] = None         # cặp ghi + đọc lại, có search
    numbers_ok: Optional[bool] = None        # DTI + lãi suất
    citation_ok: Optional[bool] = None
    facts_ok: Optional[bool] = None          # câu hỏi đáp có đáp án
    refusal_expected: bool = False
    refused: bool = False
    grounded: Optional[bool] = None          # mọi tài liệu được trích đều có trong kết quả công cụ
    wrote_decision: bool = False             # có gọi ghi quyết định thành công không
    tool_calls: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_prompt: int = 0
    error: str = ""
    notes: str = ""


def _is_error(text: str) -> bool:
    return str(text or "").lstrip().startswith("Error:")


def score_question(case: TestCase, rec: RunRecord, expected: Optional[ExpectedDecision] = None) -> QuestionScore:
    """Chấm một câu. Hàm thuần: cùng đầu vào, cùng kết quả; không gọi mạng."""
    s = QuestionScore(
        id=case.id, type=case.type, target_app_id=case.target_app_id or "",
        success=rec.success and rec.complete, refusal_expected=case.is_no_answer,
        tool_calls=len(rec.trace), tool_errors=sum(1 for r in rec.trace if _is_error(r.get("result", ""))),
        llm_calls=rec.usage.calls, prompt_tokens=rec.usage.prompt_tokens,
        completion_tokens=rec.usage.completion_tokens, max_prompt=rec.usage.max_prompt,
        error=rec.error or ("" if rec.complete else "log thiếu dòng __answer__ (câu chạy dở)"),
    )
    answer = rec.final_answer or ""
    s.wrote_decision = any(
        r.get("tool") == "submit_decision" and not _is_error(r.get("result", "")) for r in rec.trace
    )
    results_text = " ".join(str(r.get("result", "")) for r in rec.trace)
    notes: list[str] = []

    # Faithfulness theo luật: tài liệu trích trong câu trả lời phải có trong những gì công cụ trả về.
    seen_docs = set(re.findall(r"CP-\d+", results_text))
    cited = cited_in_text(answer)
    ungrounded = sorted({c.split("@")[0] for c in cited} - seen_docs)
    s.grounded = not ungrounded
    if ungrounded:
        notes.append(f"trích tài liệu chưa từng thấy: {', '.join(ungrounded)}")

    if case.expected_decision:  # ---- hành động / thẩm định hồ sơ
        if expected is None:
            s.notes = f"không có đáp án chuẩn cho {case.target_app_id}"
            return s
        sub = submission_from_trace(case.target_app_id or "", rec.trace)
        v = verify_decision_record(sub, expected)
        s.decision_ok, s.root_cause_ok = v.decision_correct, v.root_cause_correct
        s.citation_ok, s.verify_ok = v.citation_valid, v.trace_valid
        s.numbers_ok = v.dti_correct and v.rate_correct
        s.passed = v.is_overall_pass and s.success
        if v.error_message:
            notes.append(v.error_message)
    elif case.is_no_answer:  # ---- câu không có đáp án: phải từ chối, không ghi gì
        s.refused = is_refusal(answer, lenient=True)
        s.passed = s.success and s.refused and not s.wrote_decision
        if not s.refused:
            notes.append("không từ chối dù tài liệu không có thông tin")
        if s.wrote_decision:
            notes.append("ghi quyết định cho câu hỏi đáp")
    else:  # ---- hỏi đáp có đáp án
        s.refused = is_refusal(answer, lenient=False)
        facts = key_facts(case.expected_answer)
        norm_answer = normalize(answer)
        missing = [f for f in facts if normalize(f) not in norm_answer]
        s.facts_ok = bool(facts) and not missing and bool(answer.strip())
        if case.expected_docs:
            s.citation_ok = verify_citations(cited, case.expected_docs)["citation_valid"]
        s.passed = s.success and bool(s.facts_ok) and s.citation_ok is not False and not s.refused
        if missing:
            notes.append(f"thiếu ý: {', '.join(missing)}")
        if s.citation_ok is False:
            notes.append(f"trích dẫn chưa đủ: có {cited or 'không có'}, cần {case.expected_docs}")
        if s.refused:
            notes.append("từ chối nhầm câu có đáp án")

    s.notes = "; ".join(notes)
    return s


def summarize(scores: list[QuestionScore]) -> dict[str, Any]:
    """Số liệu tổng hợp để in và ghi summary.json."""
    def rate(items: list[bool]) -> Optional[float]:
        return round(100.0 * sum(items) / len(items), 1) if items else None

    answerable = [s for s in scores if not s.refusal_expected and s.decision_ok is None]
    actions = [s for s in scores if s.decision_ok is not None]
    no_answer = [s for s in scores if s.refusal_expected]
    calls = sum(s.tool_calls for s in scores)
    by_type: dict[str, dict[str, Any]] = {}
    for s in scores:
        t = by_type.setdefault(s.type, {"n": 0, "passed": 0})
        t["n"] += 1
        t["passed"] += int(s.passed)

    return {
        "questions": len(scores),
        "passed": sum(s.passed for s in scores),
        "pass_rate": rate([s.passed for s in scores]),
        "run_success_rate": rate([s.success for s in scores]),
        "decision_accuracy": rate([bool(s.decision_ok) for s in actions]),
        "root_cause_accuracy": rate([bool(s.root_cause_ok) for s in actions]),
        "read_after_write_rate": rate([bool(s.verify_ok) for s in actions]),
        "numbers_accuracy": rate([bool(s.numbers_ok) for s in actions]),
        "correct_refusal_rate": rate([s.refused and not s.wrote_decision for s in no_answer]),
        "false_refusal_rate": rate([s.refused for s in answerable]),
        "faithfulness_rule_rate": rate([bool(s.grounded) for s in scores if s.success]),
        "tool_error_rate": round(100.0 * sum(s.tool_errors for s in scores) / calls, 1) if calls else None,
        "avg_tool_calls": round(calls / len(scores), 2) if scores else None,
        "avg_llm_calls": round(sum(s.llm_calls for s in scores) / len(scores), 2) if scores else None,
        "total_prompt_tokens": sum(s.prompt_tokens for s in scores),
        "total_completion_tokens": sum(s.completion_tokens for s in scores),
        "max_prompt_tokens": max((s.max_prompt for s in scores), default=0),
        "by_type": by_type,
    }

