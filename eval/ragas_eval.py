from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Đảm bảo import được các module từ src/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# Thư viện Rich hiển thị giao diện
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TESTSET_PATH = REPO_ROOT / "tests" / "testset.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "out" / "ragas"
DEFAULT_MODEL = os.getenv("RAGAS_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
CONTEXT_TOOLS = ("search_policy", "get_application", "calc_dti")  # nguồn dữ kiện mà câu trả lời được phép dựa vào
NO_ANSWER_PLACEHOLDER = "(Agent không đưa ra câu trả lời)"
DEFAULT_PACE = 8.0  # Nghỉ 8 giây giữa mỗi câu để chống 429 RateLimit (15 RPM)


# =========================================================================
# 1. WRAPPER EMBEDDING BGE-M3 CỤC BỘ (Không tốn token API, tận dụng model có sẵn)
# =========================================================================
def get_local_embeddings():
    """Tận dụng sentence-transformers bge-m3 đã tải sẵn trên máy."""
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class LocalBgeEmbeddings(Embeddings):
        def __init__(self, model_name: str = "BAAI/bge-m3"):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        def embed_query(self, text: str) -> List[float]:
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            return embedding.tolist()

    return LocalBgeEmbeddings()


# =========================================================================
# 2. KHỞI TẠO JUDGE LLM GEMINI
# =========================================================================
def get_gemini_judge(model_name: str):
    """Khởi tạo Gemini làm Giám khảo thông qua endpoint tương thích OpenAI."""
    from langchain_openai import ChatOpenAI

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        console.print("[bold red]Lỗi:[/bold red] Thiếu GEMINI_API_KEY trong file .env.")
        sys.exit(1)

    return ChatOpenAI(
        model=model_name,
        api_key=gemini_key,
        base_url=GEMINI_BASE_URL,
        temperature=0.0,
    )


# =========================================================================
# 3. BÓC TÁCH DỮ LIỆU TỪ LOG OFFLINE
# =========================================================================
def find_latest_log() -> Path:
    """Tự động tìm file log đánh giá phù hợp trong logs/ (ưu tiên log chính có đủ 16 câu)."""
    default_main = REPO_ROOT / "logs" / "eval_20261014_202454.jsonl"
    if default_main.exists():
        return default_main
    log_dir = REPO_ROOT / "logs"
    logs = sorted(log_dir.glob("eval_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        console.print("[bold red]Lỗi:[/bold red] Không tìm thấy file log nào trong logs/.")
        sys.exit(1)
    return logs[0]


def extract_evaluation_dataset(testset_path: Path, log_path: Path) -> list[dict[str, Any]]:
    """Gom 4 trường bắt buộc: user_input, retrieved_contexts, response, reference."""
    from eval.scoring import is_refusal, records_from_log

    # Đọc testset gốc
    testset_cases = {}
    with open(testset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                testset_cases[c["id"]] = c

    # Đọc log thực thi của Agent qua records_from_log chuẩn (gom trace và final_answer từ event stream)
    agent_results = records_from_log(log_path)

    dataset = []
    for qid, case in testset_cases.items():
        if qid not in agent_results:
            continue
        rec = agent_results[qid]

        # Ngữ cảnh: kết quả của các tool tra cứu/tính toán (không tính tool ghi quyết định)
        contexts = []
        for step in rec.trace:
            if step.get("tool") in CONTEXT_TOOLS:
                res = str(step.get("result", "")).strip()
                if res and not res.startswith("Error:"):
                    contexts.append(f"[{step['tool']}]" + chr(10) + res)

        # Nếu Agent không search (hoặc câu từ chối), truyền fallback context
        if not contexts:
            contexts = ["(Không có ngữ cảnh chính sách nào được tra cứu)"]

        dataset.append({
            "id": qid,
            "type": case.get("type", "tra cứu"),
            "is_no_answer": case.get("is_no_answer", False),
            "user_input": case["question"],
            "retrieved_contexts": contexts,
            "response": rec.final_answer or NO_ANSWER_PLACEHOLDER,
            "refused": is_refusal(rec.final_answer or "", lenient=True),
            "reference": case.get("expected_answer") or "",
        })

    return dataset


# =========================================================================
# 4. CHẤM ĐIỂM TUẦN TỰ KÈM CHECKPOINT & TỰ ĐỘNG CHỜ HỒI QUOTA (ANTI-429)
# =========================================================================
def evaluate_sequentially_with_checkpoint(
    items: list[dict[str, Any]],
    judge_llm: Any,
    embeddings: Any,
    out_dir: Path,
    pace: float,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Chấm từng câu một, lưu ngay checkpoint xuống đĩa, tự động retry khi gặp 429."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "ragas_checkpoint.json"

    # Tải checkpoint cũ nếu có
    checkpoint: dict[str, dict[str, Any]] = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            console.print(f"[bold yellow]→ Đã tải {len(checkpoint)} câu từ checkpoint cũ, sẽ tự động bỏ qua các câu này.[/bold yellow]")
        except Exception as e:
            console.print(f"[yellow]Không đọc được checkpoint cũ ({e}), sẽ bắt đầu mới.[/yellow]")

    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ]

    wrapped_llm = LangchainLLMWrapper(judge_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    exhausted_models: set[str] = set()
    used_models: set[str] = {model_name}  # model thực sự đã chấm (có thể đổi giữa chừng khi hết hạn mức)
    total = len(items)
    for idx, item in enumerate(items, start=1):
        qid = item["id"]

        # Đã chấm và khớp cả câu trả lời lẫn ngữ cảnh hiện tại thì bỏ qua (ngữ cảnh đổi thì chấm lại)
        if (
            qid in checkpoint
            and "faithfulness" in checkpoint[qid]
            and checkpoint[qid].get("response") == item.get("response")
            and checkpoint[qid].get("retrieved_contexts") == item.get("retrieved_contexts")
            and checkpoint[qid].get("response") != NO_ANSWER_PLACEHOLDER
        ):
            console.print(f"[{idx}/{total}] Câu [cyan bold]{qid:<4}[/cyan bold] [dim]({item.get('type')})[/dim]: [bold green]ĐÃ CÓ TRONG CHECKPOINT[/bold green] (Bỏ qua)")
            continue

        console.print(f"[{idx}/{total}] Đang chấm câu [cyan bold]{qid:<4}[/cyan bold] [dim]({item.get('type')})[/dim]...")
        single_ds = Dataset.from_list([item])

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                res = evaluate(
                    dataset=single_ds,
                    metrics=metrics,
                    llm=wrapped_llm,
                    embeddings=wrapped_embeddings,
                    raise_exceptions=True,  # Bắt ngoại lệ để xử lý 429
                    show_progress=False,
                )
                df_records = res.to_pandas().to_dict(orient="records")
                scored = df_records[0] if df_records else {}
                merged = {**item, **scored}
                checkpoint[qid] = merged

                # Ghi checkpoint ngay lập tức xuống đĩa
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

                f_score = merged.get("faithfulness", 0.0) or 0.0
                ar_score = merged.get("answer_relevancy", 0.0) or 0.0
                ac_score = merged.get("answer_correctness", 0.0) or 0.0
                console.print(
                    f"      [green]✓ Xong {qid}:[/green] "
                    f"Faithfulness: [bold]{f_score:.1%}[/bold] | "
                    f"Relevancy: [bold]{ar_score:.1%}[/bold] | "
                    f"Correctness: [bold]{ac_score:.1%}[/bold]"
                )
                break

            except Exception as e:
                err_str = str(e)
                is_rate_limit = any(k in err_str.lower() for k in ("429", "quota", "resource_exhausted", "rate_limit"))
                is_server_busy = any(k in err_str.lower() for k in ("503", "unavailable", "high demand", "overload", "server error"))

                if is_rate_limit or is_server_busy:
                    wait_s = 25.0
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE) or re.search(r"retryDelay': '(\d+)s'", err_str)
                    if m:
                        wait_s = float(m.group(1)) + 3.0

                    if wait_s > 60.0:
                        exhausted_models.add(model_name)
                        console.print(
                            f"      [bold red]⚠ Model {model_name} đã hết hạn mức ngày (Google yêu cầu chờ {wait_s:.0f}s ~ {wait_s/3600:.1f}h)![/bold red]"
                        )
                        backup_models = [DEFAULT_MODEL, "gemini-flash-lite-latest", "gemini-3.7-flash"]
                        backup = next((bm for bm in backup_models if bm not in exhausted_models), None)
                        if backup:
                            console.print(f"      [bold green]→ Tự động chuyển Judge sang model dự phòng: {backup}[/bold green]")
                            model_name = backup
                            used_models.add(backup)
                            judge_llm = get_gemini_judge(backup)
                            wrapped_llm = LangchainLLMWrapper(judge_llm)
                            time.sleep(2.0)
                            continue
                        else:
                            console.print("[red]Đã hết model dự phòng khả dụng. Vui lòng chạy lại sau khi Google hồi hạn mức.[/red]")
                            break

                    reason = "503 Máy chủ Google quá tải (High Demand)" if is_server_busy else "429 Chạm giới hạn Quota"
                    console.print(
                        f"      [yellow]⚠ {reason} ở câu {qid}. "
                        f"Tự động ngủ {wait_s:.0f}s để đợi máy chủ hồi (lần thử {attempt}/{max_retries})...[/yellow]"
                    )
                    time.sleep(wait_s)
                else:
                    console.print(f"      [red]⚠ Lỗi không phải quota/server ở câu {qid}: {e}[/red]")
                    if attempt == max_retries:
                        checkpoint[qid] = {**item, "error": str(e)}
                        with open(checkpoint_file, "w", encoding="utf-8") as f:
                            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                    time.sleep(5.0)

        # Nghỉ an toàn pace giây giữa các câu để chống nghẽn RPM
        if pace > 0 and idx < total:
            time.sleep(pace)

    # Gom kết quả theo thứ tự gốc của items
    return [checkpoint[it["id"]] for it in items if "faithfulness" in checkpoint.get(it["id"], {})], sorted(used_models)


# =========================================================================
# 5. HIỂN THỊ BẢNG RICH & XUẤT FILE
# =========================================================================
def display_and_save_results(
    records: list[dict[str, Any]],
    refusal_items: list[dict[str, Any]],
    out_dir: Path,
    judge_models: list[str],
    log_file: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    def _val(v: Any) -> float:
        if v is None:
            return 0.0
        try:
            val = float(v)
            return 0.0 if math.isnan(val) else val
        except (ValueError, TypeError):
            return 0.0

    metric_cols = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("answer_correctness", "Answer Correctness"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
    ]

    table = Table(
        title=f"BẢNG 1: RAGAS TRÊN {len(records)} CÂU HỎI CÓ ĐÁP ÁN",
        header_style="bold magenta", show_lines=True,
    )
    table.add_column("ID", style="cyan bold", justify="center")
    table.add_column("Loại", style="dim", justify="center")
    for _, title in metric_cols:
        table.add_column(title, justify="right")
    for c in records:
        table.add_row(str(c.get("id", "")), str(c.get("type", "")), *[f"{_val(c.get(k)):.2%}" for k, _ in metric_cols])
    averages = {k: (sum(_val(c.get(k)) for c in records) / len(records) if records else None) for k, _ in metric_cols}
    if records:
        table.add_row("[bold green]TRUNG BÌNH[/bold green]", "-", *[f"[bold green]{averages[k]:.2%}[/bold green]" for k, _ in metric_cols])

    console.print("=" * 90)
    console.print(table)

    # Câu no_answer: không chấm bằng RAGAS, chỉ báo từ chối đúng/sai (luật của dự án)
    if refusal_items:
        t2 = Table(title="BẢNG 2: CÂU NGOÀI TÀI LIỆU (không chấm RAGAS, chỉ xét từ chối đúng)", header_style="bold magenta", show_lines=True)
        t2.add_column("ID", style="cyan bold", justify="center")
        t2.add_column("Từ chối đúng", justify="center")
        for it in refusal_items:
            t2.add_row(it["id"], "[green]ĐÚNG[/green]" if it.get("refused") else "[red]SAI[/red]")
        console.print(t2)

    csv_path = out_dir / "ragas_scores.csv"
    if records:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    json_path = out_dir / "ragas_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "judge_models": judge_models,
        "log_evaluated": log_file,
        "ragas_cases_count": len(records),
        "averages": averages,
        "refusal_cases": [{"id": it["id"], "refused": it.get("refused")} for it in refusal_items],
        "scores": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]✓ Đã lưu bảng điểm chi tiết:[/green] {csv_path}")
    console.print(f"[green]✓ Đã lưu tóm tắt JSON:[/green] {json_path}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Đánh giá các câu hỏi đáp trong testset bằng RAGAS (câu no_answer chỉ xét từ chối đúng).")
    parser.add_argument("--log", type=Path, help="Đường dẫn file log (.jsonl), mặc định lấy log chính (cấu hình A).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model Gemini làm Judge.")
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE, help="Số giây nghỉ giữa mỗi câu (mặc định 8.0s để chống 429).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Thư mục xuất báo cáo.")
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    console.print(Panel(
        f"[bold cyan]ĐÁNH GIÁ RAGAS TRÊN TẬP CÂU HỎI FINCO[/bold cyan]\n"
        f"• Log đánh giá : [yellow]{log_path}[/yellow]\n"
        f"• Model Judge  : [green]{args.model}[/green]\n"
        f"• Embedding    : [blue]BAAI/bge-m3 (Cục bộ)[/blue]\n"
        f"• Nghỉ an toàn : [magenta]{args.pace}s / câu (Chống 429 RateLimit)[/magenta]\n"
        f"• Checkpoint   : [cyan]{args.out_dir / 'ragas_checkpoint.json'}[/cyan]",
        border_style="cyan"
    ))

    # 1. Trích xuất Dataset; câu no_answer tách riêng, không chấm RAGAS
    items = extract_evaluation_dataset(TESTSET_PATH, log_path)
    if not items:
        console.print("[red]Không trích xuất được câu hỏi nào từ log![/red]")
        sys.exit(1)
    ragas_items = [it for it in items if not it["is_no_answer"]]
    refusal_items = [it for it in items if it["is_no_answer"]]

    # 2. Khởi tạo Judge & Embeddings
    judge_llm = get_gemini_judge(args.model)
    embeddings = get_local_embeddings()

    # 3. Chạy đánh giá tuần tự kèm checkpoint & auto-retry
    evaluated_records, judge_models = evaluate_sequentially_with_checkpoint(
        ragas_items, judge_llm, embeddings, args.out_dir, args.pace, args.model
    )

    # 4. Hiển thị & Lưu file
    display_and_save_results(evaluated_records, refusal_items, args.out_dir, judge_models, str(log_path))
    if len(evaluated_records) < len(ragas_items):
        console.print(f"[yellow]Mới chấm {len(evaluated_records)}/{len(ragas_items)} câu; chạy lại cùng lệnh để chấm tiếp từ checkpoint.[/yellow]")


if __name__ == "__main__":
    main()
