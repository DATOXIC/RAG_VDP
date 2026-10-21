from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.scoring import QuestionScore, RunRecord, records_from_log, score_question, summarize  # noqa: E402
from eval.verifier import (  # noqa: E402
    ExpectedDecision,
    calculate_summary_metrics,
    export_results_csv,
    print_summary_table,
    submission_from_trace,
    verify_decision_record,
)
from finco.types import TestCase, Usage  # noqa: E402

TESTSET_PATH = REPO_ROOT / "tests" / "testset.jsonl"
EXPECTED_PATH = REPO_ROOT / "tests" / "expected_decisions.json"
APPLICATIONS_PATH = REPO_ROOT / "data" / "applications.json"
LOGS_DIR = REPO_ROOT / "logs"
OUT_DIR = REPO_ROOT / "eval" / "out"

MAX_INFRA_FAILS = 2


def is_infra_error(error: Optional[str]) -> bool:
    """Lỗi hạ tầng (hết hạn mức, quá TPM, mạng), không phải lỗi quyết định của agent."""
    e = (error or "").lower()
    return any(k in e for k in ("ratelimit", "rate_limit", "429", "413", "tokens per", "timeout", "connection"))



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

    with isolated_db_sandbox():
        result = agent.run_sync(
            app_id=case.target_app_id or "Q",
            question=None if case.type == "hồ sơ" else case.question,
            question_id=case.id,
        )
    return record_from_result(case.id, result), result


# --------------------------------------------------------------------------- xuất kết quả
def write_per_question_csv(scores: list[QuestionScore], path: Path) -> None:
    # Tránh gây lỗi
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
def collect_records(args: argparse.Namespace, cases: list[TestCase]) -> tuple[dict[str, RunRecord], Path, Optional[str]]:
    """Lấy RunRecord của từng câu: đọc từ log (--score-only) hoặc chạy agent thật. Trả (records, log, model)."""
    if args.score_only:
        records = records_from_log(args.score_only)
        print(f"Chấm lại từ {args.score_only} ({len(records)} câu có trong log). Không gọi LLM.")
        return records, args.score_only, args.model

    records: dict[str, RunRecord] = {}
    if args.resume:
        log_path = args.resume
        # Chỉ bỏ qua câu đã chạy THÀNH CÔNG; câu lỗi (hết hạn mức, mạng...) được chạy lại.
        records = {k: r for k, r in records_from_log(log_path).items() if r.complete and r.success}
        print(f"Chạy tiếp {log_path}: bỏ qua {len(records)} câu đã xong.")
    else:
        log_path = LOGS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    from finco.agent import Agent  # nhập muộn: --score-only không cần MCP/khóa API

    agent = Agent(model=args.model, verbose=False)
    todo = [c for c in cases if c.id not in records]
    print(f"Log: {log_path}")
    print(f"Sẽ chạy {len(todo)}/{len(cases)} câu (model={agent.model}, pace={args.pace}s).")
    infra_fails = 0
    for i, case in enumerate(todo, start=1):
        started = time.time()
        rec, result = run_case(agent, case)
        result.save_log(log_path)  # ghi ngay: lỡ dừng giữa chừng vẫn --resume được
        records[case.id] = rec
        
        print(f"[{i}/{len(todo)}] {case.id:<8} {'ok ' if rec.success else 'LỖI'} "
              f"{len(rec.trace)} tool, {rec.usage.calls} lần gọi LLM, {time.time() - started:.1f}s"
              f"{'' if rec.success else '  -> ' + str(rec.error)}")
        infra_fails = infra_fails + 1 if (not rec.success and is_infra_error(rec.error)) else 0
        if infra_fails >= MAX_INFRA_FAILS:
            print(f"DỪNG SỚM: {infra_fails} câu liên tiếp lỗi do hạn mức/dịch vụ LLM (429/413). "
                  f"Chờ hạn mức hồi rồi chạy tiếp bằng: --resume {log_path}")
            break
        if args.pace and i < len(todo):
            time.sleep(args.pace)
    return records, log_path, agent.model


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    ap = argparse.ArgumentParser(description="Chạy và chấm điểm agent FinCo.")
    ap.add_argument("--testset", type=Path, default=TESTSET_PATH)
    ap.add_argument("--apps", action="store_true", help="thêm toàn bộ hồ sơ trong expected_decisions.json")
    ap.add_argument("--ids", nargs="+", help="chỉ chạy các id này (id câu hỏi hoặc mã hồ sơ APP-xxx)")
    ap.add_argument("--model", help="tên model (mặc định LLM_MODEL trong .env)")
    ap.add_argument("--pace", type=float, default=0.0, help="giây nghỉ giữa hai câu (tránh rate limit)")
    ap.add_argument("--resume", type=Path, help="log .jsonl cũ: bỏ qua câu đã xong, ghi tiếp vào file đó")
    ap.add_argument("--score-only", type=Path, help="chỉ chấm lại từ log này, không gọi LLM")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    expected = load_expected()
    cases = load_cases(args.testset) + (app_cases(expected) if args.apps or args.ids else [])
    if args.ids:
        cases = [c for c in cases if c.id in set(args.ids)]
    if not cases:
        print("Không có câu nào để chạy (kiểm tra --ids / --testset).")
        return 2

    records, log_path, model_name = collect_records(args, cases)

    scores = [
        score_question(c, records.get(c.id) or RunRecord(c.id, [], error="không có trong log"),
                       expected.get(c.target_app_id or ""))
        for c in cases
    ]
    summary = summarize(scores)
    summary.update({"log_file": str(log_path), "model": model_name,
                    "generated_at": datetime.now().isoformat(timespec="seconds")})

    # Bảng riêng của verifier cho các câu thẩm định hồ sơ (định dạng dùng trong báo cáo)
    verified = [
        verify_decision_record(submission_from_trace(c.target_app_id, records[c.id].trace), expected[c.target_app_id])
        for c in cases if c.expected_decision and c.target_app_id in expected and c.id in records
    ]
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
    print(f"Đã ghi kết quả vào {args.out_dir}" + ("" if chart_ok else " (không có biểu đồ: thiếu matplotlib)"))
    if args.score_only:
        return 0
    return 0 if summary["passed"] == summary["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
