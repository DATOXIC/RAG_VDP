"""Điểm vào duy nhất của FinCo. Dùng được cả khi máy không có `make`.

    python run.py setup                 # tạo DB hồ sơ, tải mô hình embedding, kiểm tra khóa API
    python run.py demo [APP-ID ...]     # chạy thẩm định mẫu + một câu hỏi ngoài tài liệu, in bảng chấm
    python run.py all                   # = setup + demo (một lệnh tái lập demo)
    python run.py                       # REPL tương tác (chat)
    python run.py eval [tham số ...]    # = python eval/run_eval.py; không tham số = đủ 37 ca (--apps)
    python run.py eval-quick            # 6 ca đại diện (~150k token), đủ để thấy quy trình chạy thật
    python run.py eval-score [LOG]      # chấm lại một log đã có, KHÔNG gọi LLM (mặc định: log cấu hình chính)
    python run.py compare [--configs A H B]  # đồ thị so sánh các cấu hình từ per_question.csv có sẵn, không gọi LLM
    python run.py retrieval-eval        # đo tầng truy xuất (recall@k, MRR), không cần khóa API
    python run.py interop               # host thứ hai: client JSON-RPC thuần gọi MCP server, không cần khóa API
    python run.py test                  # toàn bộ test offline
    python run.py check                 # test + retrieval-eval + interop + eval-score (không cần khóa API)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

DEFAULT_DEMO_APPS = ["APP-014", "APP-015"]  # cùng số liệu, khác ngày nộp: v1 duyệt, v2 từ chối
# Bộ nhanh: tra cứu, phiên bản, no_answer + 3 hồ sơ (duyệt, v1/v2 theo ngày nộp, DTI). Khoảng 150k token.
QUICK_IDS = ["T01", "V01", "N01", "APP-001", "APP-014", "APP-003"]
DEFAULT_SCORE_LOG = "logs/eval_20261014_202454.jsonl"  # cấu hình chính (A): gpt-oss-120b, không critic
DEMO_REFUSAL_QUESTION = "FinCo có cho vay mua ô tô trả góp không, kỳ hạn tối đa bao nhiêu tháng?"


def _api_key_ready() -> bool:
    from finco.llm import get_client

    try:
        get_client()
        return True
    except ValueError as e:
        print(f"\n[!] {e}\n    Tạo file .env từ .env.example rồi điền GROQ_API_KEY (hoặc OPENAI_API_KEY).")
        return False


def cmd_setup() -> int:
    from finco.store import db

    db.init_db()
    added = db.load_applications()
    print(f"[1/3] DB hồ sơ: {db.get_db_path()} (thêm mới {added}, tổng {len(db.list_applications())} hồ sơ)")

    print("[2/3] Nạp chỉ mục RAG và mô hình embedding (lần đầu có thể tải ~2 GB)...")
    from finco.rag import retriever

    out = retriever.retrieve("tỷ lệ nợ trên thu nhập tối đa", as_of_date="2025-05-01", top_k=1)
    if out.startswith("Error"):
        print(f"[!] Truy xuất lỗi: {out}")
        return 1
    print("      Truy xuất hoạt động.")

    print("[3/3] Khóa API LLM...")
    if _api_key_ready():
        print("      Đã có khóa API.")
    else:
        print("      Chưa có khóa: vẫn chạy được test và chấm lại từ log, nhưng chưa thẩm định được.")
    return 0


def cmd_demo(app_ids: list[str]) -> int:
    from eval.run_eval import load_expected
    from eval.sandbox_db import isolated_db_sandbox
    from eval.verifier import print_summary_table, verify_decision_record
    from finco.agent import Agent

    if not _api_key_ready():
        return 2

    agent = Agent(verbose=True)
    expected = load_expected()
    log_file = ROOT / "logs" / f"demo_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    verdicts = []

    for app_id in app_ids:
        print("\n" + "=" * 78 + f"\n THẨM ĐỊNH {app_id}\n" + "=" * 78)
        with isolated_db_sandbox():  # DB sạch mỗi hồ sơ: chạy demo lại vẫn không đụng "đã có quyết định"
            result = agent.run_sync(app_id, question_id=app_id)
        result.save_log(log_file)
        status = "HOÀN TẤT" if result.success else f"KHÔNG HOÀN TẤT: {result.error_message}"
        print(f"\n-> {status} | {len(result.trace)} lời gọi tool | "
              f"{result.usage.calls} lần gọi LLM, {result.usage.prompt_tokens + result.usage.completion_tokens} token")
        if app_id in expected:
            verdicts.append(verify_decision_record(result.to_submission(), expected[app_id]))

    if not app_ids or app_ids == DEFAULT_DEMO_APPS:
        print("\n" + "=" * 78 + "\n CÂU HỎI NGOÀI TÀI LIỆU (phải từ chối, không bịa)\n" + "=" * 78)
        result = agent.run_sync("Q", question=DEMO_REFUSAL_QUESTION, question_id="demo-no-answer")
        result.save_log(log_file)

    if verdicts:
        print_summary_table(verdicts)
    print(f"Log máy (jsonl, có USAGE): {log_file}")
    return 0 if all(v.is_overall_pass for v in verdicts) else 1


def _run(*args: str) -> int:
    """Chạy một lệnh Python con (cùng thông dịch viên) và trả mã thoát."""
    import subprocess

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # tiến trình con in tiếng Việt được trên Windows
    return subprocess.call([sys.executable, *args], cwd=ROOT, env=env)


def cmd_score(log: str | None = None, out_dir: str | None = None) -> int:
    args = ["eval/run_eval.py", "--score-only", log or DEFAULT_SCORE_LOG, "--apps"]
    return _run(*args, *(["--out-dir", out_dir] if out_dir else []))


def cmd_check() -> int:
    steps = [
        ("test offline", lambda: _run("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")),
        ("đo truy xuất", lambda: _run("eval/retrieval_eval.py")),
        ("host thứ hai (interop)", lambda: _run("examples/client2.py")),
        # mã thoát 1 = có ca không đạt trong log (kết quả thật của cấu hình chính), không phải lỗi chạy
        ("chấm lại log chính", lambda: 0 if cmd_score(out_dir="eval/out/check") in (0, 1) else 2),
    ]
    failed = []
    for name, fn in steps:
        print("\n" + "=" * 78 + f"\n CHECK: {name}\n" + "=" * 78, flush=True)
        if fn() != 0:
            failed.append(name)
    print("\n" + ("CHECK OK" if not failed else "CHECK LỖI: " + ", ".join(failed)))
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "chat"
    rest = argv[1:]
    if cmd == "setup":
        return cmd_setup()
    if cmd == "demo":
        return cmd_demo([a.upper() for a in rest] or DEFAULT_DEMO_APPS)
    if cmd == "all":
        code = cmd_setup()
        return code if code else cmd_demo(DEFAULT_DEMO_APPS)
    if cmd == "eval":
        from eval.run_eval import main as eval_main

        return eval_main(rest or ["--apps"])
    if cmd == "eval-quick":
        from eval.run_eval import main as eval_main

        return eval_main(["--ids", *QUICK_IDS, "--out-dir", "eval/out/quick", *rest])
    if cmd == "eval-score":
        return cmd_score(rest[0] if rest else None)
    if cmd == "compare":
        return _run("eval/compare_chart.py", *rest)
    if cmd == "retrieval-eval":
        return _run("eval/retrieval_eval.py", *rest)
    if cmd == "interop":
        return _run("examples/client2.py", *rest)
    if cmd == "test":
        return _run("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
    if cmd == "check":
        return cmd_check()
    if cmd in ("chat", "run"):
        from finco.chat import main as chat_main

        chat_main()
        return 0
    print(__doc__)
    return 0 if cmd in ("-h", "--help", "help") else 2


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main(sys.argv[1:]))
