# Khai báo sử dụng AI

> Bắt buộc theo đề: công cụ nào, phần nào, prompt chính, đoạn code do AI sinh. Mỗi thành viên **tự điền và xác nhận phần của mình**; mục chưa có người điền ghi rõ "CHƯA KHAI BÁO" thay vì để trống.
> Ai viết phần nào phải giải thích được từng dòng phần đó (PLAN.md mục 3).

## 1. Công cụ

| Công cụ | Dùng để | Người dùng |
|---|---|---|
| Claude Code (Anthropic), phiên làm việc trong repo | thiết kế và viết code `eval/`, bắt cầu trace → verifier, `reason_codes`, viết lại vòng lặp agent, `call_llm_tools`, `eval/run_eval.py`, testset, Makefile/run.py/README | thành viên chạy phiên này (**điền họ tên**; phần `eval/` thuộc TÔ PHƯƠNG ĐẠT, agent/MCP/chat thuộc NGUYỄN ĐĂNG VƯƠNG) |
| *(CHƯA KHAI BÁO)* | phần `src/finco/rag/` (hybrid, filter, chunker), `llm.py` | TRƯƠNG KHẮC NHẬT PHI |
| *(CHƯA KHAI BÁO)* | phần MCP server/client, `chat.py`, `audit.py` | NGUYỄN ĐĂNG VƯƠNG |

Mô hình LLM chạy trong sản phẩm (không phải công cụ viết code): Groq, `openai/gpt-oss-20b` (cấu hình bằng `LLM_MODEL`). Thư viện client: SDK `openai` (đề không liệt kê, cần xác nhận với giảng viên, xem PLAN.md mục 20).

## 2. Phần nào do AI viết hoặc sửa

| File / phần | AI làm gì | Người kiểm chứng bằng cách |
|---|---|---|
| `eval/verifier.py`: `submission_from_trace`, `verify_root_cause`, so khớp trích dẫn theo phiên bản | viết mới | `tests/test_verifier.py`, chạy trên trace giả lập |
| `eval/judge.py`: `_require_bool`, `judge_error`, bỏ `identifies_root_cause` | sửa | đọc diff; test `bool("false")` |
| `src/finco/types.py`, `store/db.py`, `tools.py`: `reason_codes` (tập đóng, khớp quyết định, migration DB) | viết | `tests/test_submit_decision.py`, migration chạy trên bản sao `finco.db` |
| `src/finco/llm.py`: `_create_with_retry` dùng chung, `call_llm_tools`, `MalformedToolCallError`, timeout mặc định | sửa | `tests/test_llm.py` (43 test, client giả) |
| `src/finco/agent.py`: bọc lỗi, ép ghi + đọc lại, log theo bước, `run_id` duy nhất | viết lại | `tests/test_agent.py` (LLM giả, MCP server thật, DB sandbox) |
| `prompts/agent_system.txt` | viết lại cho function calling, thứ tự CP-04, hỏi đáp/từ chối | `tests/test_prompt.py` |
| `eval/run_eval.py`, `eval/scoring.py`, `tests/testset.jsonl` | viết mới | `tests/test_run_eval.py` |
| `Makefile`, `run.py`, `README.md`, `.env.example`, cập nhật `PLAN.md` mục 4b/5/8/9 | viết | chạy `python run.py setup`, đọc lại |

## 3. Prompt chính (tóm tắt, không dán nguyên văn)

- "Tôi đang làm phần đánh giá kết quả của LLM ... nhận ra tôi đang giả định xử lý trên các đối tượng, nhưng LLM không trả về đối tượng" → hướng dựng bài nộp từ trace tool call thay vì từ văn bản.
- "Bạn hãy đọc lại hai file yêu cầu và xác nhận thầy có yêu cầu ReAct hay không" → xác nhận đề chỉ yêu cầu parse action an toàn; nhóm giữ function calling.
- "Giúp tôi đánh giá hệ thống từ commit bc3c727 đến hiện tại" → danh sách lỗi và rủi ro; sau đó "sửa hết các nhóm vấn đề, đừng gọi API LLM thật" → các thay đổi ở mục 2.

## 4. Giới hạn và trung thực

- Các test của agent dùng **LLM giả** (kịch bản định sẵn); chúng chứng minh vòng lặp, bọc lỗi và ép quy trình đúng, **không** chứng minh model thật quyết định đúng. Số liệu chất lượng trong báo cáo chỉ lấy từ `make eval` chạy thật (log trong `logs/`).
- Code AI sinh được thành viên đọc, chạy test và chịu trách nhiệm giải thích khi vấn đáp.

## 5. Đoạn code do AI sinh (khai báo của thành viên RAG)

```
def corpus_hash(chunks: list[Chunk]) -> str:
    """Mã băm nội dung toàn bộ chunk, dùng để phát hiện index cũ so với dữ liệu mới."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
```

```
    def _valid_mask(self, as_of_date: str) -> np.ndarray:
        """Tạo một mảng Boolean dựa trên một thời điểm xác định

        TRUE nếu còn hiệu lực
        FALSE ngược lại
        """
        return np.array(
            [
                c.effective_date <= as_of_date
                and (c.expiry_date is None or as_of_date <= c.expiry_date)
                for c in self.chunks
            ],
            dtype=bool,
        )
```
