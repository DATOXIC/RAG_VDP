"""Đo chất lượng truy xuất: dense, BM25, hybrid trên tests/retrieval_testset.jsonl.

Chạy từ thư mục gốc repo:
    python eval/retrieval_eval.py                    # cả 3 chế độ, có lọc ngày
    python eval/retrieval_eval.py --no-date-filter   # tắt lọc ngày (thí nghiệm bonus #1)
    python eval/retrieval_eval.py --modes dense hybrid --k 5

Cách chấm:
- Một chunk "trúng" khi (doc_id, version, số mục) khớp một mục trong `gold` hoặc `alt`.
  So theo MỤC chứ không theo chunk_id, vì một mục có thể bị tách thành nhiều chunk.
- rank_gold: hạng đầu tiên của chunk thuộc `gold`. rank_any: thuộc `gold` hoặc `alt`.
- Câu no_answer (gold rỗng) không tính recall/MRR; chỉ ghi điểm hạng 1 để tham khảo.
- wrong_version: số chunk trong top-k KHÔNG còn hiệu lực tại as_of_date của câu hỏi.

Ra: bảng tóm tắt trên màn hình và eval/out/retrieval_per_question[_nofilter].csv.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finco.rag.bm25 import BM25Index, is_valid_at  # noqa: E402
from finco.rag.dense import VectorStore  # noqa: E402
from finco.rag.hybrid import HybridSearcher  # noqa: E402
from finco.types import SearchHit  # noqa: E402

TESTSET = REPO_ROOT / "tests" / "retrieval_testset.jsonl"
INDEX_DIR = REPO_ROOT / "data" / "index"
OUT_DIR = REPO_ROOT / "eval" / "out"
MODES = ("dense", "bm25", "hybrid")
KS = (1, 3, 5)


def section_number(section: str) -> Optional[int]:
    """'4. Tỷ lệ nợ trên thu nhập' -> 4."""
    m = re.match(r"\s*(\d+)\.", section)
    return int(m.group(1)) if m else None


def key_of(hit: SearchHit) -> tuple:
    c = hit.chunk
    return (c.doc_id, c.version, section_number(c.section))


def first_rank(hits: list[SearchHit], targets: set) -> Optional[int]:
    """Hạng (bắt đầu từ 1) của chunk đầu tiên thuộc targets, hoặc None."""
    for i, h in enumerate(hits, start=1):
        if key_of(h) in targets:
            return i
    return None


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_keys(items: list[dict]) -> set:
    return {(g["doc_id"], g["version"], g["section"]) for g in items}


def build_searchers(modes) -> dict:
    store = VectorStore()
    store.load(INDEX_DIR)
    bm25 = BM25Index(store.chunks)
    out = {}
    if "dense" in modes:
        out["dense"] = lambda q, k, d: store.search(q, top_k=k, as_of_date=d)
    if "bm25" in modes:
        out["bm25"] = lambda q, k, d: bm25.search(q, top_k=k, as_of_date=d)
    if "hybrid" in modes:
        hybrid = HybridSearcher(store, bm25)
        out["hybrid"] = lambda q, k, d: hybrid.search(q, top_k=k, as_of_date=d)
    return out


def evaluate(questions, searchers, k_max, use_date_filter):
    rows = []
    for q in questions:
        gold = to_keys(q["gold"])
        relevant = gold | to_keys(q.get("alt", []))
        as_of = q.get("as_of_date")
        filter_date = as_of if use_date_filter else None
        for mode, search in searchers.items():
            hits = search(q["query"], k_max, filter_date)
            wrong = (
                sum(1 for h in hits if not is_valid_at(h.chunk, as_of)) if as_of else 0
            )
            rows.append({
                "id": q["id"],
                "type": q["type"],
                "mode": mode,
                "as_of_date": as_of or "",
                "date_filter": "on" if use_date_filter else "off",
                "has_gold": bool(gold),
                "rank_gold": first_rank(hits, gold) if gold else None,
                "rank_any": first_rank(hits, relevant) if gold else None,
                "wrong_version": wrong,
                "top1_chunk": hits[0].chunk.chunk_id if hits else "",
                "top1_score": round(hits[0].score, 4) if hits else "",
                "topk_chunks": " ".join(h.chunk.chunk_id for h in hits),
                "query": q["query"],
            })
    return rows


def summarize(rows, ks):
    """Gom theo (mode, nhóm). Trả list dict để in và ghi."""
    groups = defaultdict(list)
    for r in rows:
        if not r["has_gold"]:
            continue
        groups[(r["mode"], "ALL")].append(r)
        groups[(r["mode"], r["type"])].append(r)

    out = []
    for (mode, grp), rs in groups.items():
        n = len(rs)
        entry = {"mode": mode, "group": grp, "n": n}
        for rank_key, label in (("rank_any", "any"), ("rank_gold", "gold")):
            for k in ks:
                entry[f"recall_{label}@{k}"] = sum(
                    1 for r in rs if r[rank_key] is not None and r[rank_key] <= k
                ) / n
            entry[f"mrr_{label}"] = sum(
                1 / r[rank_key] for r in rs if r[rank_key] is not None
            ) / n
        entry["wrong_version_chunks"] = sum(r["wrong_version"] for r in rs)
        out.append(entry)
    return out


def print_summary(summary, modes, ks):
    types = sorted({s["group"] for s in summary if s["group"] != "ALL"})
    head = f"{'mode':8s}{'n':>3s}" + "".join(f"{f'R@{k}':>7s}" for k in ks) + f"{'MRR':>7s}{'gold@3':>8s}{'wrongV':>8s}"
    print("\n== Tổng (recall tính gold HOẶC alt; gold@3 chỉ tính gold) ==")
    print(head)
    for mode in modes:
        s = next(x for x in summary if x["mode"] == mode and x["group"] == "ALL")
        print(
            f"{mode:8s}{s['n']:>3d}"
            + "".join(f"{s[f'recall_any@{k}']:>7.2f}" for k in ks)
            + f"{s['mrr_any']:>7.2f}{s['recall_gold@3']:>8.2f}{s['wrong_version_chunks']:>8d}"
        )
    print("\n== Theo loại câu (recall@3, gold hoặc alt) ==")
    print(f"{'type':15s}{'n':>3s}" + "".join(f"{m:>9s}" for m in modes))
    for t in types:
        cells = []
        n = 0
        for mode in modes:
            s = next((x for x in summary if x["mode"] == mode and x["group"] == t), None)
            n = s["n"] if s else n
            cells.append(f"{s['recall_any@3']:>9.2f}" if s else f"{'-':>9s}")
        print(f"{t:15s}{n:>3d}" + "".join(cells))


def print_misses(rows, modes, k):
    print(f"\n== Câu KHÔNG có gold/alt trong top-{k} ==")
    any_miss = False
    for r in rows:
        if r["has_gold"] and r["rank_any"] is None and r["mode"] in modes:
            any_miss = True
            print(f"  {r['id']:4s} {r['mode']:7s} [{r['type']}] {r['query'][:70]}")
    if not any_miss:
        print("  (không có)")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--k", type=int, default=max(KS), help="số kết quả lấy về mỗi câu")
    ap.add_argument("--no-date-filter", action="store_true", help="không lọc as_of_date")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    questions = load_testset(TESTSET)
    searchers = build_searchers(args.modes)
    ks = tuple(k for k in KS if k <= args.k)
    rows = evaluate(questions, searchers, args.k, use_date_filter=not args.no_date_filter)

    summary = summarize(rows, ks)
    print(f"Testset: {len(questions)} câu | lọc ngày: {'TẮT' if args.no_date_filter else 'BẬT'}")
    print_summary(summary, args.modes, ks)
    print_misses(rows, args.modes, max(ks))

    na = [r for r in rows if not r["has_gold"] and r["mode"] == "hybrid"]
    for r in na:
        print(f"\n[no_answer] {r['id']}: điểm hạng 1 (hybrid) = {r['top1_score']} -> {r['top1_chunk']}")

    suffix = "_nofilter" if args.no_date_filter else ""
    out = OUT_DIR / f"retrieval_per_question{suffix}.csv"
    write_csv(rows, out)
    print(f"\nĐã ghi {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
