"""Vẽ đồ thị so sánh các cấu hình đã chấm, từ các `per_question.csv` có sẵn (không gọi LLM).

    python eval/compare_chart.py                    # A, H, C
    python eval/compare_chart.py --configs A H B    # chọn cấu hình
Ra: eval/out/compare/compare.png và compare.csv (mã A–J xem docs/ket-qua-danh-gia.md).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "eval" / "out"

# Cấu hình -> (nhãn, các thư mục kết quả; H, I, J chạy theo 4 shard nên gộp lại)
CONFIGS: dict[str, tuple[str, list[str]]] = {
    "A": ("A: gpt-oss-120b", ["ab_nocritic"]),
    "B": ("B: 120b + critic-agent", ["agent_full"]),
    "C": ("C: Qwen2.5-7B", ["local_nocritic"]),
    "D": ("D: Qwen2.5-7B + critic", ["local_critic_v2"]),
    "E": ("E: Qwen2.5-14B", ["qwen"]),
    "G": ("G: gpt-oss:20b (Ollama)", ["vuong20_nocritic"]),
    "H": ("H: gpt-oss-20b (Groq)", [f"g20_none_w{i}" for i in range(1, 5)]),
    "I": ("I: 20b + critic một lần", [f"g20_single_w{i}" for i in range(1, 5)]),
    "J": ("J: 20b + critic-agent", [f"g20_agent_w{i}" for i in range(1, 5)]),
}
COLORS = ["#3b7dd8", "#2e9e5b", "#d8453b", "#e0a526", "#8a5cc2", "#4cb5c4", "#7a7a7a"]


def load(dirs: list[str]) -> list[dict]:
    """Gộp per_question.csv của các thư mục; trùng id thì lấy dòng sau (kết quả chạy lại)."""
    rows: dict[str, dict] = {}
    for d in dirs:
        path = OUT / d / "per_question.csv"
        if not path.exists():
            raise SystemExit(f"Thiếu {path}")
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[r["id"]] = r
    return list(rows.values())


def passed(r: dict) -> bool:
    return r["passed"] == "True"


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "passed": sum(passed(r) for r in rows),
        "llm_calls": sum(int(r["llm_calls"]) for r in rows),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in rows),
        "completion_tokens": sum(int(r["completion_tokens"]) for r in rows),
        "tool_errors": sum(int(r["tool_errors"]) for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", default=["A", "H", "C"], choices=sorted(CONFIGS))
    ap.add_argument("--out-dir", type=Path, default=OUT / "compare")
    args = ap.parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Thiếu matplotlib (pip install -r requirements.txt)", file=sys.stderr)
        return 1

    data = {k: load(CONFIGS[k][1]) for k in args.configs}
    sums = {k: summarize(v) for k, v in data.items()}
    types = sorted({r["type"] for rows in data.values() for r in rows})

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [2.4, 1, 1]})
    width = 0.8 / len(args.configs)
    for i, k in enumerate(args.configs):
        rates = []
        for t in types:
            sub = [r for r in data[k] if r["type"] == t]
            rates.append(100.0 * sum(passed(r) for r in sub) / len(sub) if sub else 0.0)
        xs = [j + i * width for j in range(len(types))]
        ax1.bar(xs, rates, width, label=CONFIGS[k][0], color=COLORS[i % len(COLORS)])
    ax1.set_xticks([j + width * (len(args.configs) - 1) / 2 for j in range(len(types))])
    ax1.set_xticklabels(types, rotation=25)
    ax1.set_ylim(0, 125)
    ax1.set_ylabel("Tỷ lệ đạt (%)")
    ax1.set_title("Tỷ lệ đạt theo loại câu")
    ax1.legend(fontsize=8, loc="upper center", ncol=len(args.configs))

    labels = args.configs
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    ax2.bar(labels, [sums[k]["passed"] for k in labels], color=colors)
    for i, k in enumerate(labels):
        ax2.text(i, sums[k]["passed"] + 0.5, f"{sums[k]['passed']}/{sums[k]['n']}", ha="center", fontsize=9)
    ax2.set_ylim(0, max(s["n"] for s in sums.values()) + 4)
    ax2.set_title("Số ca đạt toàn diện")

    ax3.bar(labels, [sums[k]["prompt_tokens"] / 1e6 for k in labels], color=colors)
    for i, k in enumerate(labels):
        ax3.text(i, sums[k]["prompt_tokens"] / 1e6, f"{sums[k]['prompt_tokens']:,}".replace(",", "."),
                 ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Triệu token")
    ax3.set_title("Tổng prompt token (cả bộ)")

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "compare.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    csv_path = args.out_dir / "compare.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "n", "passed", "llm_calls", "prompt_tokens", "completion_tokens", "tool_errors"])
        for k in args.configs:
            s = sums[k]
            w.writerow([k, CONFIGS[k][0], s["n"], s["passed"], s["llm_calls"], s["prompt_tokens"],
                        s["completion_tokens"], s["tool_errors"]])
    for k in args.configs:
        s = sums[k]
        print(f"{CONFIGS[k][0]:<28} {s['passed']}/{s['n']} đạt, {s['llm_calls']} lượt LLM, {s['prompt_tokens']:,} prompt token")
    print(f"Đã ghi {png} và {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
