"""Thí nghiệm "đổi một con số trong chính sách thì quyết định phải đổi theo".

Mục đích: chứng minh agent dùng chính sách tra được (RAG) chứ không dựa trí nhớ của model.
Cách làm, trên BẢN SAO TẠM của dự án (dữ liệu thật không bị sửa):
  1. Đổi trần DTI của CP-01 v1 từ 50% xuống một số mới (mặc định 40%), ở mọi chỗ tài liệu nêu con số đó
     (CP-01 v1, CP-01 v2 mục 6.4, bảng so sánh trong CP-15).
  2. Dựng lại chỉ mục vector cho bản sao.
  3. Tính lại đáp án chuẩn: hồ sơ v1 có DTI trong khoảng (trần mới, 50%] đổi từ duyệt/chuyển cấp sang từ chối DTI_EXCEEDED.
  4. Chạy agent (không critic) trên vài hồ sơ bị ảnh hưởng, hồ sơ đối chứng và một câu hỏi về trần DTI;
     câu hỏi này cũng chạy trên chính sách gốc để so sánh trước và sau.

    python eval/policy_change_experiment.py              # chạy thật (gọi LLM)
    python eval/policy_change_experiment.py --new 45     # đổi sang 45%
Ra: eval/out/policy_change/ (before/, after/, summary.csv) và log trong đó. Mã hồ sơ đối chứng nằm ở CONTROL_IDS.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "eval" / "out" / "policy_change"

OLD = 50
CONTROL_IDS = ["APP-001", "APP-012", "APP-019", "APP-015"]  # v1 DTI thấp, v1 chuyển cấp, v1 đã bị từ chối, v2 cùng DTI với APP-014
QA_ID = "X01"
QA_QUESTION = "Trần DTI tối đa theo chính sách CP-01 áp dụng cho hồ sơ nộp ngày 15/03/2025 là bao nhiêu?"
REJECT_CITED = ["CP-01@v1", "CP-02", "CP-03", "CP-05"]


def ignore(_dir, names):
    return [n for n in names if n in {"__pycache__", "out", ".git", ".venv"} or n.endswith(".db") or n.endswith(".pyc")]


def make_copy(dst: Path) -> None:
    for d in ("src", "prompts", "eval", "tests"):
        shutil.copytree(REPO_ROOT / d, dst / d, ignore=ignore)
    (dst / "data").mkdir()
    shutil.copy(REPO_ROOT / "data" / "applications.json", dst / "data" / "applications.json")
    shutil.copytree(REPO_ROOT / "data" / "policies", dst / "data" / "policies")
    shutil.copytree(REPO_ROOT / "data" / "index", dst / "data" / "index")
    for f in ("run.py", ".env"):
        if (REPO_ROOT / f).exists():
            shutil.copy(REPO_ROOT / f, dst / f)


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Không tìm thấy đoạn cần đổi trong {path.name}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def apply_policy_change(root: Path, new: int) -> list[str]:
    pol = root / "data" / "policies"
    v1 = next(pol.glob("CP-01_v1_*.md"))
    v2 = next(pol.glob("CP-01_v2_*.md"))
    cp15 = next(pol.glob("CP-15_*.md"))
    edit(v1, f"là **{OLD}%**. Hồ sơ có DTI lớn hơn {OLD}% bị từ chối.", f"là **{new}%**. Hồ sơ có DTI lớn hơn {new}% bị từ chối.")
    edit(v2, f"DTI tối đa giảm từ {OLD}% xuống 45%.", f"DTI tối đa thay đổi từ {new}% thành 45%.")
    edit(cp15, f"| DTI tối đa | {OLD}% | 45% |", f"| DTI tối đa | {new}% | 45% |")
    return [v1.name, v2.name, cp15.name]


def rebuild_index(root: Path, env: dict) -> None:
    r = subprocess.run([sys.executable, "-m", "finco.rag.build_index"], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("Dựng lại chỉ mục lỗi:\n" + r.stdout[-1500:] + r.stderr[-1500:])


def new_expected(src: Path, dst: Path, new: int) -> dict[str, dict]:
    rows = json.loads(src.read_text(encoding="utf-8"))
    changed = {}
    for r in rows:
        if r["policy_version"] != "v1" or r.get("dti") is None:
            continue
        r["max_dti"] = new / 100
        if new / 100 < r["dti"] <= OLD / 100:
            changed[r["app_id"]] = {"from": r["decision"], "dti": r["dti"]}
            r["decision"], r["reasons"], r["cited"] = "reject", ["DTI_EXCEEDED"], list(REJECT_CITED)
    dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def add_qa(root: Path, expected_pct: int) -> None:
    case = {"id": QA_ID, "question": QA_QUESTION, "type": "tra cứu",
            "expected_answer": f"Trần DTI là **{expected_pct}%**.", "expected_docs": ["CP-01"],
            "is_no_answer": False, "expected_decision": None, "target_app_id": None}
    with open(root / "tests" / "testset.jsonl", "a", encoding="utf-8") as f:
        f.write("\n" + json.dumps(case, ensure_ascii=False) + "\n")


def run_eval(root: Path, ids: list[str], env: dict, pace: float) -> int:
    cmd = [sys.executable, "eval/run_eval.py", "--ids", *ids, "--out-dir", "out", "--pace", str(pace)]
    return subprocess.run(cmd, cwd=root, env=env).returncode


def decisions_from_logs(root: Path) -> dict[str, str]:
    """Quyết định thật sự được ghi (đọc từ log), theo từng mã hồ sơ."""
    out: dict[str, str] = {}
    for f in sorted((root / "logs").glob("eval_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("tool") == "submit_decision" and not str(r.get("result", "")).startswith("Error"):
                args = r.get("args") or {}
                out[r["question_id"]] = str(args.get("decision"))
    return out


def per_question(root: Path) -> dict[str, dict]:
    with open(root / "out" / "per_question.csv", encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", type=int, default=40, help="trần DTI mới của CP-01 v1, đơn vị % (mặc định 40)")
    ap.add_argument("--pace", type=float, default=5.0, help="giây nghỉ giữa hai câu (tránh giới hạn tốc độ)")
    ap.add_argument("--keep", action="store_true", help="giữ lại thư mục tạm")
    args = ap.parse_args(argv)
    if not (0 < args.new < OLD):
        raise SystemExit("--new phải nhỏ hơn 50 để có hồ sơ bị ảnh hưởng")

    tmp = Path(tempfile.mkdtemp(prefix="finco_policy_change_"))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", FINCO_CRITIC="0")
    try:
        # ---- sau khi đổi
        after = tmp / "after"
        make_copy(after)
        env_after = dict(env, PYTHONPATH=f"{after / 'src'}{os.pathsep}{after}")
        edited = apply_policy_change(after, args.new)
        print(f"Đã đổi trần DTI v1 {OLD}% -> {args.new}% trong: {', '.join(edited)}")
        rebuild_index(after, env_after)
        changed = new_expected(REPO_ROOT / "tests" / "expected_decisions.json", after / "tests" / "expected_decisions.json", args.new)
        print("Hồ sơ bị ảnh hưởng:", {k: f"{v['from']} (DTI {v['dti']})" for k, v in changed.items()})
        add_qa(after, args.new)
        ids_after = list(changed) + CONTROL_IDS + [QA_ID]
        run_eval(after, ids_after, env_after, args.pace)

        # ---- trước khi đổi: chỉ câu hỏi về trần DTI (hồ sơ đối chứng đã có trong log cấu hình A)
        before = tmp / "before"
        make_copy(before)
        env_before = dict(env, PYTHONPATH=f"{before / 'src'}{os.pathsep}{before}")
        add_qa(before, OLD)
        run_eval(before, [QA_ID], env_before, 0.0)

        # ---- tổng hợp
        pq_after, dec_after, pq_before = per_question(after), decisions_from_logs(after), per_question(before)
        orig = {r["app_id"]: r for r in json.loads((REPO_ROOT / "tests" / "expected_decisions.json").read_text(encoding="utf-8"))}
        new = {r["app_id"]: r for r in json.loads((after / "tests" / "expected_decisions.json").read_text(encoding="utf-8"))}
        OUT.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in ids_after:
            if i == QA_ID:
                rows.append({"id": i, "loai": "câu hỏi trần DTI", "dap_an_goc": f"{OLD}%", "dap_an_moi": f"{args.new}%",
                             "agent_truoc_khi_doi": "đạt" if pq_before[i]["passed"] == "True" else "trượt",
                             "agent_sau_khi_doi": "đạt" if pq_after[i]["passed"] == "True" else "trượt"})
                continue
            rows.append({"id": i, "loai": "bị ảnh hưởng" if i in changed else "đối chứng",
                         "dap_an_goc": orig[i]["decision"], "dap_an_moi": new[i]["decision"],
                         "agent_truoc_khi_doi": "(log cấu hình A)", "agent_sau_khi_doi": dec_after.get(i, "(không ghi)"),
                         "dat": "đạt" if pq_after[i]["passed"] == "True" else "trượt"})
        fields = ["id", "loai", "dap_an_goc", "dap_an_moi", "agent_truoc_khi_doi", "agent_sau_khi_doi", "dat"]
        with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        for name, root in (("after", after), ("before", before)):
            dst = OUT / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(root / "out", dst)
            logs = root / "logs"
            if logs.exists():
                for lf in logs.glob("eval_*.jsonl"):
                    shutil.copy(lf, dst / lf.name)
        print("\n" + "=" * 90)
        for r in rows:
            print(f"{r['id']:<9}{r['loai']:<18}gốc={r['dap_an_goc']:<10}mới={r['dap_an_moi']:<10}agent sau khi đổi: {r['agent_sau_khi_doi']:<12}{r.get('dat', '')}")
        print(f"Đã ghi {OUT}")
        return 0
    finally:
        if args.keep:
            print("Giữ thư mục tạm:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
