"""Chạy agent trên bộ câu hỏi và chấm điểm: `make eval`.

    python eval/run_eval.py                         # 16 câu trong tests/testset.jsonl
    python eval/run_eval.py --apps                  # thêm 21 hồ sơ trong expected_decisions.json
    python eval/run_eval.py --ids T01 A01           # chỉ chạy vài câu
    python eval/run_eval.py --pace 8                # nghỉ 8 giây giữa các câu (tránh rate limit)
    python eval/run_eval.py --resume logs/eval_X.jsonl   # chạy tiếp, bỏ qua câu đã có trong log
    python eval/run_eval.py --score-only logs/eval_X.jsonl   # chấm lại từ log, KHÔNG gọi LLM
    python eval/run_eval.py --judge                 # thêm LLM-judge (tốn thêm lượt gọi) cho câu hành động

Nguyên tắc chấm (bằng code, không dùng LLM trừ khi có --judge):
- Câu hành động / hồ sơ: verifier so quyết định, mã lý do, trích dẫn, DTI, lãi suất, cặp ghi + đọc lại
  với tests/expected_decisions.json (dựng từ trace của agent, không đọc văn bản tự do).
- Câu no_answer: đạt khi agent từ chối đúng (câu chuẩn REFUSAL_MARKER) và KHÔNG ghi quyết định.
- Câu còn lại: đạt khi câu trả lời chứa đủ các "ý chính" trong expected_answer (đánh dấu **như này**),
  trích đúng tài liệu mong đợi (đúng phiên bản nếu có "@v"), không từ chối nhầm.
- Faithfulness theo luật: mọi tài liệu mà câu trả lời trích phải xuất hiện trong kết quả công cụ.

Chấm điểm là hàm thuần trên log: --score-only cho cùng kết quả mà không tốn một lượt gọi LLM nào.
Mỗi câu chạy trên một DB sandbox sạch nên các câu không ảnh hưởng nhau và lần chạy sau không hưởng lợi
từ lần trước. Không dùng cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.verifier import (  # noqa: E402
    ExpectedDecision,
    calculate_summary_metrics,
    export_results_csv,
    parse_citation,
    print_summary_table,
    submission_from_trace,
    verify_citations,
    verify_decision_record,
)
from finco.types import TestCase, Usage  # noqa: E402

TESTSET_PATH = REPO_ROOT / "tests" / "testset.jsonl"
EXPECTED_PATH = REPO_ROOT / "tests" / "expected_decisions.json"
APPLICATIONS_PATH = REPO_ROOT / "data" / "applications.json"
LOGS_DIR = REPO_ROOT / "logs"
OUT_DIR = REPO_ROOT / "eval" / "out"

# Câu mở đầu mà prompt bắt agent dùng khi tài liệu không có đáp án (phải khớp prompts/agent_system.txt).
REFUSAL_MARKER = "Không có thông tin này trong tài liệu chính sách của FinCo"
# Cách nói khác của một lời từ chối: chỉ dùng cho câu no_answer (mọi câu đều đáng nghi nếu có cụm này).
_REFUSAL_PHRASES = (
    "không có thông tin", "không tìm thấy", "không có quy định", "không đề cập",
    "không có trong tài liệu", "không được nêu", "không có dữ liệu",
)
ANSWER_TOOL = "__answer__"
ACTION_TYPE = "hành động"


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


def record_from_result(question_id: str, result: Any) -> RunRecord:
    """AgentRunResult -> RunRecord (cùng dạng với đọc từ log)."""
    return RunRecord(
        question_id=question_id,
        trace=[asdict(line) for line in result.trace],
        final_answer=result.final_answer,
        usage=result.usage or Usage(0, 0, 0, 0),
        success=result.success,
        error=result.error_message,
        complete=True,
    )


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
    judge_valid: Optional[bool] = None
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


def score_question(
    case: TestCase,
    rec: RunRecord,
    expected: Optional[ExpectedDecision] = None,
    judge_valid: Optional[bool] = None,
) -> QuestionScore:
    """Chấm một câu. Hàm thuần: cùng đầu vào, cùng kết quả; không gọi mạng."""
    s = QuestionScore(
        id=case.id, type=case.type, target_app_id=case.target_app_id or "",
        success=rec.success and rec.complete, refusal_expected=case.is_no_answer,
        tool_calls=len(rec.trace), tool_errors=sum(1 for r in rec.trace if _is_error(r.get("result", ""))),
        llm_calls=rec.usage.calls, prompt_tokens=rec.usage.prompt_tokens,
        completion_tokens=rec.usage.completion_tokens, max_prompt=rec.usage.max_prompt,
        error=rec.error or ("" if rec.complete else "log thiếu dòng __answer__ (câu chạy dở)"),
        judge_valid=judge_valid,
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

    if judge_valid is False:
        notes.append("giám khảo LLM: lý do chưa đạt")
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


# --------------------------------------------------------------------------- nạp dữ liệu
def load_cases(testset: Path) -> list[TestCase]:
    cases = []
    with open(testset, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(TestCase(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                raise SystemExit(f"{testset}:{n}: dòng testset không hợp lệ: {e}")
    return cases


def load_expected(path: Path = EXPECTED_PATH) -> dict[str, ExpectedDecision]:
    with open(path, encoding="utf-8") as f:
        return {item["app_id"]: ExpectedDecision.from_dict(item) for item in json.load(f)}


def app_cases(expected: dict[str, ExpectedDecision]) -> list[TestCase]:
    """Mỗi hồ sơ trong đáp án chuẩn thành một câu hành động (id = mã hồ sơ)."""
    return [
        TestCase(
            id=app_id,
            question=f"Hãy thẩm định hồ sơ {app_id} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống.",
            type="hồ sơ",
            expected_answer=f"Quyết định {e.decision}",
            expected_docs=list(e.cited),
            expected_decision=e.decision,  # type: ignore[arg-type]
            target_app_id=app_id,
        )
        for app_id, e in expected.items()
    ]


# --------------------------------------------------------------------------- chạy agent
def run_case(agent: Any, case: TestCase) -> tuple[RunRecord, Any]:
    """Chạy một câu trên DB sandbox sạch. Trả (RunRecord, AgentRunResult)."""
    from eval.sandbox_db import isolated_db_sandbox

    needs_decision = bool(case.expected_decision)
    with isolated_db_sandbox():
        result = agent.run_sync(
            app_id=case.target_app_id or "Q",
            question=None if case.type == "hồ sơ" else case.question,
            question_id=case.id,
            require_decision=needs_decision,
        )
    return record_from_result(case.id, result), result


def judge_case(case: TestCase, rec: RunRecord) -> Optional[bool]:
    """LLM-judge lý do của một câu hành động. None nếu không áp dụng; judge_error coi là None (không phạt agent)."""
    from eval.judge import judge_single_decision
    from finco.types import Application

    expected = load_expected().get(case.target_app_id or "")
    if expected is None or not case.expected_decision:
        return None
    with open(APPLICATIONS_PATH, encoding="utf-8") as f:
        raw = next((a for a in json.load(f) if a["app_id"] == case.target_app_id), None)
    if raw is None:
        return None
    app = Application(**{k: raw[k] for k in Application.__dataclass_fields__ if k in raw})
    sub = submission_from_trace(case.target_app_id or "", rec.trace)
    context = "\n\n".join(r["result"] for r in rec.trace if r.get("tool") == "search_policy")
    verdict = judge_single_decision(app, expected, sub, policy_context=context)
    return None if verdict.judge_error else verdict.is_valid


# --------------------------------------------------------------------------- xuất kết quả
def write_per_question_csv(scores: list[QuestionScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(QuestionScore.__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scores:
            w.writerow(asdict(s))


def write_chart(scores: list[QuestionScore], path: Path) -> bool:
    """Biểu đồ (bonus #4): tỷ lệ đạt theo loại câu và token từng câu. False nếu thiếu matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    types = sorted({s.type for s in scores})
    rates = [100.0 * sum(s.passed for s in scores if s.type == t) / sum(1 for s in scores if s.type == t) for t in types]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(types, rates, color="#3b7dd8")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Tỷ lệ đạt (%)")
    ax1.set_title("Tỷ lệ đạt theo loại câu")
    ax1.tick_params(axis="x", rotation=30)
    ids = [s.id for s in scores]
    tokens = [s.prompt_tokens + s.completion_tokens for s in scores]
    ax2.bar(ids, tokens, color=["#2e9e5b" if s.passed else "#d8453b" for s in scores])
    ax2.set_ylabel("Token (prompt + completion)")
    ax2.set_title("Token từng câu (xanh = đạt, đỏ = trượt)")
    ax2.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def print_report(scores: list[QuestionScore], summary: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"{'ID':<8}{'Loại':<14}{'Kết quả':<9}{'Tool':>5}{'Lỗi':>5}{'LLM':>5}{'Token':>9}   Ghi chú")
    print("-" * 100)
    for s in scores:
        tokens = s.prompt_tokens + s.completion_tokens
        print(f"{s.id:<8}{s.type:<14}{'ĐẠT' if s.passed else 'TRƯỢT':<9}{s.tool_calls:>5}{s.tool_errors:>5}"
              f"{s.llm_calls:>5}{tokens:>9}   {(s.error or s.notes)[:60]}")
    print("=" * 100)
    labels = [
        ("pass_rate", "Tỷ lệ đạt toàn diện"), ("run_success_rate", "Lượt chạy hoàn tất không lỗi"),
        ("decision_accuracy", "Độ đúng quyết định"), ("root_cause_accuracy", "Đúng mã lý do"),
        ("read_after_write_rate", "Có ghi + đọc lại + tra cứu"), ("numbers_accuracy", "Đúng DTI và lãi suất"),
        ("correct_refusal_rate", "Từ chối đúng (no_answer)"), ("false_refusal_rate", "Từ chối nhầm (câu có đáp án)"),
        ("faithfulness_rule_rate", "Faithfulness theo luật (tài liệu trích có thật)"),
        ("tool_error_rate", "Tỷ lệ lời gọi tool lỗi"),
    ]
    for key, label in labels:
        v = summary.get(key)
        print(f"{label:<52}: {'n/a' if v is None else f'{v}%'}")
    print(f"{'Số lời gọi tool / câu':<52}: {summary['avg_tool_calls']}")
    print(f"{'Số lần gọi LLM / câu':<52}: {summary['avg_llm_calls']}")
    print(f"{'Token prompt / completion (tổng)':<52}: {summary['total_prompt_tokens']} / {summary['total_completion_tokens']}")
    print("=" * 100 + "\n")


# --------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chạy và chấm điểm agent FinCo.")
    ap.add_argument("--testset", type=Path, default=TESTSET_PATH)
    ap.add_argument("--apps", action="store_true", help="thêm toàn bộ hồ sơ trong expected_decisions.json")
    ap.add_argument("--only-apps", action="store_true", help="chỉ chạy các hồ sơ (bỏ testset)")
    ap.add_argument("--ids", nargs="+", help="chỉ chạy các id này")
    ap.add_argument("--model", help="tên model (mặc định LLM_MODEL trong .env)")
    ap.add_argument("--pace", type=float, default=0.0, help="giây nghỉ giữa hai câu (tránh rate limit)")
    ap.add_argument("--judge", action="store_true", help="thêm LLM-judge cho câu hành động")
    ap.add_argument("--resume", type=Path, help="log .jsonl cũ: bỏ qua câu đã xong, ghi tiếp vào file đó")
    ap.add_argument("--score-only", type=Path, help="chỉ chấm lại từ log này, không gọi LLM")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tag", default="", help="nhãn thêm vào tên file log")
    args = ap.parse_args(argv)

    expected = load_expected()
    cases: list[TestCase] = [] if args.only_apps else load_cases(args.testset)
    if args.apps or args.only_apps:
        cases += app_cases(expected)
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        print("Không có câu nào để chạy (kiểm tra --ids / --testset).")
        return 2

    # ---- thu thập RunRecord: từ log (chấm lại) hoặc chạy agent thật
    log_path: Optional[Path] = None
    model_name: Optional[str] = args.model
    records: dict[str, RunRecord] = {}
    judged: dict[str, Optional[bool]] = {}
    if args.score_only:
        log_path = args.score_only
        records = records_from_log(log_path)
        print(f"Chấm lại từ {log_path} ({len(records)} câu có trong log). Không gọi LLM.")
    else:
        if args.resume:
            log_path = args.resume
            records = {k: r for k, r in records_from_log(log_path).items() if r.complete}
            print(f"Chạy tiếp {log_path}: bỏ qua {len(records)} câu đã xong.")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = LOGS_DIR / f"eval_{stamp}{('_' + args.tag) if args.tag else ''}.jsonl"
        from finco.agent import Agent  # nhập muộn: --score-only không cần MCP/khóa API

        agent = Agent(model=args.model, verbose=False)
        model_name = agent.model
        todo = [c for c in cases if c.id not in records]
        print(f"Log: {log_path}\nSẽ chạy {len(todo)}/{len(cases)} câu (model={model_name}, pace={args.pace}s).")
        for i, case in enumerate(todo, start=1):
            started = time.time()
            rec, result = run_case(agent, case)
            result.save_log(log_path)  # ghi ngay: lỡ dừng giữa chừng vẫn --resume được
            records[case.id] = rec
            if args.judge and case.expected_decision:
                judged[case.id] = judge_case(case, rec)
            print(f"[{i}/{len(todo)}] {case.id:<8} {'ok ' if rec.success else 'LỖI'} "
                  f"{len(rec.trace)} tool, {rec.usage.calls} lần gọi LLM, {time.time() - started:.1f}s"
                  f"{'' if rec.success else '  -> ' + str(rec.error)}")
            if args.pace and i < len(todo):
                time.sleep(args.pace)

    # ---- chấm điểm
    scores: list[QuestionScore] = []
    for case in cases:
        rec = records.get(case.id) or RunRecord(question_id=case.id, trace=[], error="không có trong log")
        scores.append(score_question(case, rec, expected.get(case.target_app_id or ""), judged.get(case.id)))
    summary = summarize(scores)
    summary.update({
        "log_file": str(log_path), "model": model_name,
        "cache": "none", "generated_at": datetime.now().isoformat(timespec="seconds"),
    })

    # Bảng riêng của verifier cho các câu thẩm định hồ sơ (định dạng đã dùng trong báo cáo)
    verified = []
    for case in cases:
        if case.expected_decision and case.target_app_id in expected and case.id in records:
            sub = submission_from_trace(case.target_app_id, records[case.id].trace)
            verified.append(verify_decision_record(sub, expected[case.target_app_id]))
    if verified:
        summary["verifier"] = calculate_summary_metrics(verified)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_per_question_csv(scores, args.out_dir / "per_question.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_ok = write_chart(scores, args.out_dir / "summary.png")
    print_report(scores, summary)
    if verified:
        print_summary_table(verified)
        export_results_csv(verified, args.out_dir / "agent_eval_results.csv")

    print(f"Đã ghi: {args.out_dir / 'per_question.csv'}, {args.out_dir / 'summary.json'}"
          f"{', ' + str(args.out_dir / 'summary.png') if chart_ok else ''}")
    return 0 if summary["passed"] == summary["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
