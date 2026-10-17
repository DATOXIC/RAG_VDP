# TÔ PHƯƠNG ĐẠT

Sử dụng: Claude, Antigravity
Cách sử dụng: Em sử dụng AI để tìm hiểu luồng logic chính để code core xong sau đó thực hiện bổ sung các biện pháp phòng thủ giúp hệ thống chịu lỗi tốt. Ngoài ra những đoạn code nào mang tính trình bày em có để tự AI sinh ra và có đọc lại.

## llm.py

```python
_TRY_AGAIN = re.compile(r"try again in \s+ ((?:\d+h) ? (?:\d+m(?!s)) ? (?:[\d.]+s(?!\w)) ? (?:[\d.]+ms) ?)", re.IGNORECASE)

def _daily_quota_wait(e: Exception) -> Optional[float]:
    """Số giây server yêu cầu chờ nếu lỗi là hết hạn mức token/NGÀY; None nếu không phải lỗi đó."""
    msg = str(e)
    if "per day" not in msg and "(TPD)" not in msg:
        return None

    m = _TRY_AGAIN.search(msg)
    if not m or not m.group(1):
        return float("inf")  # hết hạn mức ngày nhưng không biết bao giờ hồi
        
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total

def critic_endpoint() -> Optional[tuple[str, str]]:
    """(api_key, base_url) riêng cho Critic, hoặc None nếu dùng chung endpoint với Actor.
    """
    if os.getenv("FINCO_CRITIC_PROVIDER", "").strip().lower() == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("FINCO_CRITIC_PROVIDER=gemini nhưng thiếu GEMINI_API_KEY trong .env.")
        return key, os.getenv("FINCO_CRITIC_BASE_URL") or GEMINI_BASE_URL

    base, key = os.getenv("FINCO_CRITIC_BASE_URL"), os.getenv("FINCO_CRITIC_API_KEY")

    if base and key:
        return key, base
    return None

def critic_default_model() -> Optional[str]:
    """Model mặc định của Critic khi gọi với profile='critic' (None = dùng LLM_MODEL như Actor)."""
    m = os.getenv("FINCO_CRITIC_MODEL")
    if m:
        return m

    if os.getenv("FINCO_CRITIC_PROVIDER", "").strip().lower() == "gemini":
        return GEMINI_DEFAULT_MODEL
        
    return None

def get_client(profile: Optional[str] = None) -> OpenAI:
    """profile='critic': client riêng cho Critic nếu có cấu hình (critic_endpoint), không thì dùng chung."""
    global _CLIENT, _CRITIC_CLIENT
    if profile == "critic":
        ep = critic_endpoint()
        if ep is not None:
            if _CRITIC_CLIENT is None:
                _CRITIC_CLIENT = OpenAI(api_key=ep[0], base_url=ep[1], max_retries=0, timeout=DEFAULT_TIMEOUT)
            return _CRITIC_CLIENT

    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not api_key:
            raise ValueError("Thiếu API Key! Vui lòng cấu hình GROQ_API_KEY hoặc OPENAI_API_KEY trong file .env.")

        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=DEFAULT_TIMEOUT
        )
        
    return _CLIENT

```

## agent.py

```python
LlmFn = Callable[[list[dict[str, Any]]], str]

def _default_llm(model: Optional[str], max_tokens: int) -> LlmFn:
    def run(messages: list[dict[str, Any]]) -> str:
        from finco.llm import call_llm 
        return call_llm(messages, tools=None, model=model, temperature=0.0, max_tokens=max_tokens, profile="critic").content
    return run
    
    
def _parse_json(text: str) -> dict[str, Any]:
    """
    Xử lý Văn bản tự do của LLM (có lẫn đoạn văn và JSON)
    """
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start= s.find("{") 
    end = s.rfind("}")
    
    if start < 0 or end <= start:
        raise ValueError("không có JSON object trong phản hồi của critic")

    data = json.loads(s[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON của critic không phải object")
    return data
```

## critic_agent.py

```python
    try:
        data = cm._parse_json(turn.content)
        if self._needs_dti(p) and not any(s["tool"] == "calc_dti" for s in rec.steps):
            
            if step > self.max_steps - 3:  
                raise _NoDti("critic-agent kết luận mà chưa tự tính DTI")
            messages.append({"role": "assistant", "content": turn.content})
            messages.append({"role": "user", "content": (
                "Quyết định này phụ thuộc DTI nên bạn PHẢI tự gọi calc_dti (lãi suất theo CP-03 ứng với "
                "kỳ hạn của hồ sơ) rồi mới kết luận. Gọi calc_dti ngay.")})
            continue
        rec.steps.append({"tool": "__final__", "args": {}, "result": turn.content[:200]})
        return data, own
    except _NoDti:
        raise
    except ValueError:
        if last:
            raise
        # JSON hỏng: nhắc đúng một lần rồi để vòng lặp tiếp tục (vẫn bị chặn bởi max_steps)
        messages.append({"role": "assistant", "content": turn.content or "(rỗng)"})
        messages.append({"role": "user", "content": "Chỉ trả MỘT JSON object đúng định dạng đã nêu, không văn bản khác."})

```

## critic.py

```python
_CITE = re.compile(r"([A-Za-z]+-\d+)(?:[\s@_\-(]*vV)?")

def parse_cite(text: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Tìm trích dẫn.
    """
    m = _CITE.search(text or "")
    if not m:
        return None
    return m.group(1).upper(), (f"v{int(m.group(2))}" if m.group(2) else None)
```

```python
def _parse_json(text: str) -> dict[str, Any]:
    """
    Xử lý Văn bản tự do của LLM (có lẫn đoạn văn và JSON)
    """
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start= s.find("{") 
    end = s.rfind("}")
    
    if start < 0 or end <= start:
        raise ValueError("không có JSON object trong phản hồi của critic")
 
    data = json.loads(s[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON của critic không phải object")
    return data
```

## verifier.py

```python
def calculate_summary_metrics(results: list[VerifyResult]) -> dict[str, float]:
    """Tính toán các chỉ số phần trăm đánh giá tổng thể theo PLAN.md."""
    total = len(results)
    if total == 0:
        return {}
    return {
        "total_cases": total,
        "pass_count": sum(1 for r in results if r.is_overall_pass),
        "overall_pass_rate": round(sum(1 for r in results if r.is_overall_pass) / total * 100, 2),
        "decision_accuracy": round(sum(1 for r in results if r.decision_correct) / total * 100, 2),
        "root_cause_accuracy": round(sum(1 for r in results if r.root_cause_correct) / total * 100, 2),
        "citation_recall_avg": round(sum(r.citation_recall for r in results) / total * 100, 2),
        "trace_valid_rate": round(sum(1 for r in results if r.trace_valid) / total * 100, 2),
        "math_valid_rate": round(sum(1 for r in results if (r.dti_correct and r.rate_correct)) / total * 100, 2),
    }

def print_summary_table(results: list[VerifyResult]) -> None:
    """In bảng tổng kết kết quả chấm thi trực quan ra console."""
    metrics = calculate_summary_metrics(results)
    print("\n" + "=" * 90)
    print(f"{'BẢNG KẾT QUẢ SÁT HẠCH AGENT THẨM ĐỊNH (FINCO VERIFIER)':^90}")
    print("=" * 90)
    print(f"{'Mã HS':<9} | {'Quyết định':<12} | {'Trích dẫn':<10} | {'Trace':<8} | {'Toán (DTI/Lãi)':<15} | {'Kết quả':<8} | {'Lỗi ghi nhận'}")
    print("-" * 90)
    for r in results:
        dec_str = "ĐÚNG" if r.decision_correct else f"SAI ({r.actual_decision})"
        cite_str = f"R={r.citation_recall:.2f}"
        trace_str = "ĐẠT" if r.trace_valid else "LỖI"
        math_str = "ĐẠT" if (r.dti_correct and r.rate_correct) else "SAI"
        status_str = "PASS [V]" if r.is_overall_pass else "FAIL [X]"
        err_str = r.error_message if r.error_message else "-"
        print(f"{r.app_id:<9} | {dec_str:<12} | {cite_str:<10} | {trace_str:<8} | {math_str:<15} | {status_str:<8} | {err_str}")
    print("=" * 90)
    print(f"Tổng số hồ sơ: {metrics.get('total_cases', 0)} | Số ca ĐỖ: {metrics.get('pass_count', 0)}")
    print(f"-> Tỷ lệ Đỗ toàn diện (Overall Pass) : {metrics.get('overall_pass_rate', 0)}%")
    print(f"-> Độ chính xác quyết định (Accuracy): {metrics.get('decision_accuracy', 0)}%")
    print(f"-> Đúng điểm nghẽn (Root cause)      : {metrics.get('root_cause_accuracy', 0)}%")
    print(f"-> Độ phủ trích dẫn TB (Recall)      : {metrics.get('citation_recall_avg', 0)}%")
    print(f"-> Tuân thủ quy trình gọi tool (Trace): {metrics.get('trace_valid_rate', 0)}%")
    print(f"-> Độ chuẩn xác con số (Math)        : {metrics.get('math_valid_rate', 0)}%")
    print("=" * 90 + "\n")
    
def export_results_csv(results: list[VerifyResult], output_path: Path = REPO_ROOT / "eval" / "out" / "agent_eval_results.csv") -> None:
    """Xuất kết quả chi tiết ra file CSV phục vụ báo cáo bảo vệ đồ án."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "app_id", "is_overall_pass", "decision_correct", "actual_decision", "expected_decision",
        "root_cause_correct", "actual_reason_codes", "expected_reason_codes",
        "citation_recall", "citation_precision", "citation_valid",
        "has_search_policy", "has_calc_dti", "read_after_write", "trace_valid",
        "dti_correct", "actual_dti", "expected_dti", "rate_correct", "error_message"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            row = {k: d[k] for k in fieldnames}
            row["actual_reason_codes"] = "|".join(d["actual_reason_codes"])
            row["expected_reason_codes"] = "|".join(d["expected_reason_codes"])
            writer.writerow(row)
    print(f"[Verifier] Đã xuất báo cáo đánh giá chi tiết ra: {output_path}")
```

```python
def submission_from_trace(app_id: str, steps: Iterable[TraceStep]) -> AgentSubmission:
    """Dựng bài nộp từ trace của MỘT hồ sơ.
    - Chỉ tính lời gọi tool THÀNH CÔNG (kết quả không bắt đầu bằng "Error:"): lời gọi lỗi không
      phải bằng chứng agent đã tra cứu / ghi / kiểm chứng.
    - submit_decision, get_decision, calc_dti phải đúng app_id đang chấm.
    - Quyết định, lý do, trích dẫn lấy từ args của lần submit_decision thành công cuối cùng.
    - DTI lấy từ kết quả calc_dti cuối (chuỗi do code sinh, không phải văn bản LLM); lãi suất
      lấy từ args agent truyền vào calc_dti.
    """
    target = app_id.strip().upper()
    tool_calls: list[str] = []
    submit_args: Optional[dict] = None
    dti: Optional[float] = None
    rate: Optional[float] = None

    for step in steps:
        tool, args, result = _step_fields(step)
        # "__answer__" là dòng ghi câu trả lời + usage cuối mỗi câu (không phải lời gọi tool).
        if not tool or tool.startswith("__") or result.lstrip().startswith("Error:"):
            continue
        if tool in ("submit_decision", "get_decision", "calc_dti"):
            if str(args.get("app_id", "")).strip().upper() != target:
                continue
        tool_calls.append(tool)

        if tool == "submit_decision":
            submit_args = args
        elif tool == "calc_dti":
            m = _DTI_LINE.search(result)
            if m:
                dti = float(m.group(1))
                try:
                    rate = _as_fraction(float(str(args.get("annual_rate")).strip().rstrip("%").strip().replace(",", ".")))
                except (TypeError, ValueError):
                    rate = None

    if submit_args is None:
        return AgentSubmission(
            app_id=target, decision="NONE", reason="", tool_calls=tool_calls,
            calculated_dti=dti, used_rate=rate,
        )

    cited = submit_args.get("cited_docs")
    if isinstance(cited, str):
        cited = [cited]
    return AgentSubmission(
        app_id=target,
        decision=str(submit_args.get("decision", "")).strip().lower(),
        reason=str(submit_args.get("reason", "")).strip(),
        cited_docs=[str(c).strip() for c in cited if str(c).strip()] if isinstance(cited, list) else [],
        reason_codes=_clean_codes(submit_args.get("reason_codes")),
        tool_calls=tool_calls,
        calculated_dti=dti,
        used_rate=rate,
    )
```

## retrieval_eval.py

```python
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

```

## ragas_eval.py

Trong file này em nhờ AI bổ trợ cho các đoạn code logic 

```python
# 1. Giám khảo LLM: Gemini qua endpoint OpenAI-compatible
judge_llm = ChatOpenAI(model="gemini-3.5-flash-lite", api_key=gemini_key, base_url=GEMINI_BASE_URL)
# 2. Giám khảo Embeddings: BGE-M3 chạy offline trên máy (0 token)
class LocalBgeEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
embeddings = LocalBgeEmbeddings()
```

```python
dataset = []
for qid, case in testset_cases.items():
    rec = agent_results[qid]

    # Chỉ bóc ngữ cảnh từ 3 tool đọc (search_policy, get_application, calc_dti)
    contexts = [
        f"[{step['tool']}]\n{step['result']}"
        for step in rec.trace 
        if step.get("tool") in ("search_policy", "get_application", "calc_dti")
    ]

    dataset.append({
        "user_input": case["question"],             # Câu hỏi của user
        "retrieved_contexts": contexts,             # Ngữ cảnh Agent tra được
        "response": rec.final_answer,               # Câu trả lời cuối của Agent
        "reference": case.get("expected_answer")    # Đáp án mẫu chuẩn
    })
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Câu trả lời có bám sát ngữ cảnh không
    answer_relevancy,    # Câu trả lời có đúng trọng tâm câu hỏi không
    answer_correctness,  # So với đáp án mẫu
    context_precision,   # Ngữ cảnh tra được có chuẩn không
    context_recall       # Ngữ cảnh tra được có đủ ý không
)

# Chấm điểm từng câu:
res = evaluate(
    dataset=Dataset.from_list([item]),
    metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
    llm=LangchainLLMWrapper(judge_llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)

# Kết quả trả về dict điểm số của câu đó:
score_dict = res.to_pandas().to_dict(orient="records")[0]
# Ví dụ: {'faithfulness': 0.92, 'answer_relevancy': 0.88, ...}
```

```python
# Tính trung bình cộng của từng chỉ số trên toàn bộ các câu hỏi:
averages = {
    metric: sum(row[metric] for row in scored_records) / len(scored_records)
    for metric in ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall")
}
```

## Sử dụng AI để tạo file ***VẼ BIỂU ĐỒ & THÍ NGHIỆM ĐỐI CHỨNG (Chỉ phục vụ viết Report)***

- **`eval/compare_chart.py`:**
- **`eval/no_rag_experiment.py`:**
- **`eval/policy_change_experiment.py`:**
- **`eval/scoring.py`**
- **`eval/run_eval.py`**

# TRƯƠNG KHẮC NHẬT PHI

## File viết mới: retriever.py

### chunker.py

```python
def _split_large_block(block: str, max_body_tokens: int, overlap_tokens: int = 40) -> list[str]:
            

    lines = block.splitlines()
    parts: list[str] = []
    current_lines: list[str] = []
    
    for line in lines:
        test_text = "\n".join(current_lines + [line]).strip()

        if count_tokens(test_text) <= max_body_tokens or not current_lines:
            current_lines.append(line)
        else:
            # Đóng phần hiện tại
            part_str = "\n".join(current_lines).strip()
            if part_str:
                parts.append(part_str)

            # Overlap nội bộ: lấy vài dòng cuối của phần trước
            overlap_lines: list[str] = []
            overlap_count = 0
            for prev_line in reversed(current_lines):
                t_count = count_tokens(prev_line)
                if overlap_count + t_count <= overlap_tokens:
                    overlap_lines.insert(0, prev_line)
                    overlap_count += t_count
                else:
                    break

            current_lines = overlap_lines + [line]

    if current_lines:
        part_str = "\n".join(current_lines).strip()
        if not (parts and part_str == parts[-1]):
            parts.append(part_str)

    return parts if parts else [block]

def _pack_blocks_clean(blocks: list[str], max_body_tokens: int) -> list[str]:
    """Gom các điều khoản sạch sẽ theo ranh giới tự nhiên (KHÔNG OVERLAP giữa các Block).

    Nếu có block nào đơn lẻ quá dài, tách riêng block đó bằng _split_large_block.
    """
    if not blocks:
        return []

    # 1. Nếu toàn bộ section vừa vặn trong ngân sách thân bài -> Giữ nguyên 1 phần
    full_text = "\n\n".join(blocks).strip()
    if count_tokens(full_text) <= max_body_tokens:
        return [full_text]

    # 2. Gom tham lam sạch sẽ theo từng điều khoản
    parts: list[str] = []
    current_blocks: list[str] = []

    for block in blocks:
        block_tokens = count_tokens(block)

        # Trường hợp A: Bản thân block này vượt quá ngân sách thân bài
        if block_tokens > max_body_tokens:
            # Nếu đang có các block gom dở trước đó, đóng phần đó lại
            if current_blocks:
                parts.append("\n\n".join(current_blocks).strip())
                current_blocks = []

            # Chia nhỏ nội bộ chính block khổng lồ này
            sub_parts = _split_large_block(block, max_body_tokens=max_body_tokens)
            parts.extend(sub_parts)
            continue

        # Trường hợp B: Block bình thường -> Thử gom vào current_blocks
        test_text = "\n\n".join(current_blocks + [block]).strip() if current_blocks else block
        if count_tokens(test_text) <= max_body_tokens:
            current_blocks.append(block)
        else:
            # Vượt ngưỡng: Đóng phần hiện tại
            if current_blocks:
                parts.append("\n\n".join(current_blocks).strip())
            # Bắt đầu phần mới sạch sẽ từ block hiện tại (Overlap giữa các block = 0!)
            current_blocks = [block]

    if current_blocks:
        parts.append("\n\n".join(current_blocks).strip())

    return parts
```

### filter.py

```python
def valid_mask(chunks: list[Chunk], as_of_date: Optional[str] = None) -> np.ndarray:
    """Tạo mảng Boolean: True nếu chunk còn hiệu lực tại as_of_date.

    Nếu as_of_date là None: không lọc (trả về toàn True).
    """
    if as_of_date is None:
        return np.ones(len(chunks), dtype=bool)
    return np.array([is_valid_at(c, as_of_date) for c in chunks], dtype=bool)
```

### dense.py

```python
def load(
        self,
        index_dir: str | Path = "data/index",
        expected_chunks: Optional[list[Chunk]] = None,
    ) -> None:
        """Nạp index từ ổ cứng và kiểm tra tính nhất quán.

        Args:
            index_dir: Thư mục chứa vectors.npy, chunks.json, meta.json.
            expected_chunks: Nếu truyền vào (chunk hiện tại từ chunk_all), so mã băm
                để phát hiện index cũ so với dữ liệu/chunker mới.
        Raises:
            FileNotFoundError: Thiếu file index.
            ValueError: Lệch mô hình, lệch số lượng/chiều vector, hoặc index đã cũ.
        """
        in_dir = Path(index_dir)

        # CHECK
        for name in ("vectors.npy", "chunks.json", "meta.json"):
            if not (in_dir / name).exists():
                raise FileNotFoundError(
                    f"Không tìm thấy '{name}' trong: {in_dir}. Hãy build lại index."
                )

        # Nạp METADA
        with open(in_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_id") != MODEL_NAME:
            raise ValueError(
                f"Lệch mô hình: Index được tạo bởi '{meta.get('model_id')}', "
                f"nhưng hệ thống đang chạy '{MODEL_NAME}'!"
            )

        # Nạp Vector
        vectors = np.load(in_dir / "vectors.npy")

        # Nạp Chunks
        with open(in_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks = [Chunk(**item) for item in json.load(f)]

        
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Index hỏng: {vectors.shape[0] if vectors.ndim == 2 else '?'} vector "
                f"nhưng {len(chunks)} chunk. Hãy build lại index."
            )
        
        if meta.get("vector_dim") != vectors.shape[1] or meta.get("num_chunks") != len(chunks):
            raise ValueError("Index hỏng: meta.json không khớp dữ liệu. Hãy build lại index.")

        # 3. Phát hiện index cũ so với dữ liệu hiện tại
        if expected_chunks is not None and meta.get("corpus_hash") != corpus_hash(expected_chunks):
            raise ValueError(
                "Index đã cũ so với dữ liệu/chunker hiện tại. Hãy build lại index."
            )
        
        self.vectors = vectors
        self.chunks = chunks    
        _log(f"Đã nạp thành công {len(self.chunks)} chunks từ index offline.")
```

### hybrid.py

```python
def search(
        self,
        query: str,
        top_k: int = 5,
        as_of_date: Optional[str] = None,
        w_bm25: Optional[float] = None,
        w_dense: Optional[float] = None,
    ) -> list[SearchHit]:
        """Tìm top_k chunk theo điểm trộn.

        Args:
            query: Câu truy vấn.
            top_k: Số kết quả tối đa.
            as_of_date: "YYYY-MM-DD". Có thì chỉ xét chunk hiệu lực tại ngày này.
            w_bm25, w_dense: Ghi đè trọng số (dùng cho thí nghiệm).
        Returns:
            Danh sách SearchHit giảm dần theo điểm trộn; rỗng nếu truy vấn rỗng
            hoặc không chunk nào hiệu lực. score là điểm trộn, dense_score và
            bm25_score là điểm thô.
        """
        if top_k <= 0 or not query.strip():
            return []
        wb = self.w_bm25 if w_bm25 is None else w_bm25
        wd = self.w_dense if w_dense is None else w_dense

        candidates = np.flatnonzero(self.bm25.valid_mask(as_of_date))
        if candidates.size == 0:
            return []

        # Điểm thô 
        raw_bm25 = self.bm25.get_scores(query)[candidates]
        raw_dense = (self.store.vectors @ embed([query])[0])[candidates]

        # Chuẩn hóa -> Fusion 
        fused = wb * min_max(raw_bm25) + wd * min_max(raw_dense)
        if self.derived_penalty:
            derived = np.array(
                [self.store.chunks[i].doc_id in DERIVED_DOCS for i in candidates]
            )
            fused = fused - self.derived_penalty * derived

        order = np.argsort(fused)[::-1][:top_k]
        return [
            SearchHit(
                chunk=self.store.chunks[candidates[j]],
                score=float(fused[j]),
                dense_score=float(raw_dense[j]),
                bm25_score=float(raw_bm25[j]),
            )
            for j in order
        ]
```

# NGUYỄN ĐĂNG VƯƠNG

### mcp_client.py — to_openai_tools

```python
schema = dict(getattr(tool, "input_schema", getattr(tool, "inputSchema", {})))
properties = dict(schema.get("properties", {}))
required = list(schema.get("required", []))

if require_thought:
    new_properties = {
        "thought": {
            "type": "string",
            "description": (
                "Suy nghĩ ngắn gọn (1-2 câu): Tại sao gọi công cụ này? "
                "Cần dữ liệu gì và dự định làm gì tiếp theo với kết quả?"
            ),
        }
    }
    new_properties.update(properties)
    properties = new_properties
    if "thought" not in required:
        required.insert(0, "thought")
```

### tools.py — Thu nhập đủ điều kiện, PMT và DTI

```python
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
```

### tools.py — Kiểm tra tập đóng reason_codes

```python
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
```

### store/db.py — Ghi quyết định và xử lý trùng

```python
with _connect() as conn:
    exists = conn.execute("SELECT 1 FROM applications WHERE app_id = ?", (decision.app_id,)).fetchone()
    if exists is None:
        raise AppNotFound(decision.app_id)
    try:
        conn.execute(
            "INSERT INTO decisions (app_id, decision, reason, cited_docs, decided_at, reason_codes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (decision.app_id, decision.decision, decision.reason,
             json.dumps(decision.cited_docs), decision.decided_at,
             json.dumps(decision.reason_codes)),
        )
    except sqlite3.IntegrityError:  # trùng khóa chính app_id
        raise DecisionExists(decision.app_id)
return decision
```

Phạm vi Người 2: types.py, tools.py, store/db.py, mcp_server.py, mcp_client.py, eval/sandbox_db.py và tests/test_submit_decision.py. Công cụ hỗ trợ: Claude Code, Codex.