"""Đối chứng không dùng RAG: agent vẫn gọi được `search_policy` nhưng tool không trả đoạn chính sách nào.

Mục đích: chứng minh việc có RAG là điều kiện để trả lời đúng (độ đúng phải thấp hơn rõ rệt khi không có RAG).
Cách làm: đặt FINCO_NO_RAG=1 (công tắc trong src/finco/tools.py, mặc định tắt) rồi chạy một tập con của bộ kiểm thử
bằng chính `eval/run_eval.py`, không critic. Kết quả có RAG lấy từ log cấu hình A (cùng model, cùng prompt).

    python eval/no_rag_experiment.py             # chạy thật (gọi LLM)
    python eval/no_rag_experiment.py --score-only logs/eval_XXXX.jsonl   # chỉ tổng hợp lại từ log đã có
Ra: eval/out/no_rag/ (per_question.csv, summary.json, summary.png) và eval/out/no_rag/compare.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "eval" / "out" / "no_rag"
BASELINE = REPO_ROOT / "eval" / "out" / "ab_nocritic" / "per_question.csv"  # cấu hình A: gpt-oss-120b, có RAG, không critic

# Mỗi loại câu vài ca: tra cứu, tổng hợp, phiên bản, tính toán, 4 câu no_answer, và các hồ sơ đại diện.
IDS = ["T01", "T02", "S01", "V01", "V02", "C01", "N01", "N02", "N03", "N04",
       "APP-001", "APP-003", "APP-005", "APP-014", "APP-015", "APP-018"]


def read(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pace", type=float, default=5.0, help="giây nghỉ giữa hai câu")
    ap.add_argument("--score-only", type=Path, help="chỉ chấm lại từ log này, không gọi LLM")
    args = ap.parse_args(argv)

    env = dict(os.environ, FINCO_NO_RAG="1", FINCO_CRITIC="0", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "eval/run_eval.py", "--out-dir", str(OUT)]
    cmd += ["--score-only", str(args.score_only)] if args.score_only else ["--ids", *IDS, "--pace", str(args.pace)]
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    if rc != 0 or not (OUT / "per_question.csv").exists():
        return rc or 1

    base, no_rag = read(BASELINE), read(OUT / "per_question.csv")
    ids = [i for i in IDS if i in no_rag and i in base]
    rows = []
    for i in ids:
        b, n = base[i], no_rag[i]
        rows.append({"id": i, "type": n["type"], "co_rag_dat": b["passed"], "khong_rag_dat": n["passed"],
                     "khong_rag_tu_choi": n["refused"], "co_rag_prompt_token": b["prompt_tokens"],
                     "khong_rag_prompt_token": n["prompt_tokens"], "ghi_chu_khong_rag": n["notes"] or n["error"]})
    
    with open(OUT / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def n_pass(col): return sum(r[col] == "True" for r in rows)
    print("\n" + "=" * 88)
    print(f"{'ID':<9}{'Loại':<14}{'Có RAG':<9}{'Không RAG':<11}Ghi chú (không RAG)")
    for r in rows:
        print(f"{r['id']:<9}{r['type']:<14}{('đạt' if r['co_rag_dat'] == 'True' else 'trượt'):<9}"
              f"{('đạt' if r['khong_rag_dat'] == 'True' else 'trượt'):<11}{r['ghi_chu_khong_rag'][:50]}")
    print(f"\nĐạt toàn diện trên {len(rows)} ca: có RAG {n_pass('co_rag_dat')}/{len(rows)}, không RAG {n_pass('khong_rag_dat')}/{len(rows)}")
    print(f"Đã ghi {OUT / 'compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
