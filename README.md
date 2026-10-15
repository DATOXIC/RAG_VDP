# FinCo: Trợ lý thẩm định hồ sơ vay

Agent thẩm định hồ sơ vay tiêu dùng cho công ty tài chính **giả định** FinCo: đọc hồ sơ, tra chính sách đúng phiên bản theo ngày nộp, tính DTI bằng code, ra quyết định, **ghi vào DB rồi đọc lại để xác nhận**. Toàn bộ chính sách và hồ sơ là dữ liệu giả lập (xem `data/README.md`).

Thành phần: RAG (dense + hybrid BM25, lọc theo ngày hiệu lực) · vòng lặp agent viết tay · 5 tool qua MCP server (stdio) · bộ chấm điểm bằng code + giám khảo LLM.

## Chạy nhanh

Yêu cầu: Python 3.11+, có mạng ở lần đầu (tải mô hình embedding `BAAI/bge-m3`, ~2 GB), khóa API LLM của một endpoint tương thích OpenAI (mặc định Groq; xem mục "Mô hình và endpoint").

```bash
cp .env.example .env        # điền GROQ_API_KEY (hoặc OPENAI_API_KEY + OPENAI_BASE_URL)
make setup && make demo
```

Không có `make` (Windows): dùng thẳng `python run.py`.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python run.py setup
.venv/Scripts/python run.py demo
```

| Lệnh | Việc làm |
|---|---|
| `make setup` | tạo venv, cài thư viện, tạo DB hồ sơ, tải/nạp mô hình embedding, kiểm tra khóa API |
| `make demo` | thẩm định APP-014 và APP-015 (cùng số liệu, khác ngày nộp → v1 duyệt, v2 từ chối) + một câu hỏi ngoài tài liệu; in bảng chấm. Đổi hồ sơ: `make demo APPS="APP-003 APP-018"` |
| `make run` | REPL tương tác: `list`, `APP-xxx`, `ask <câu hỏi>`, `history`, `audit <mã>` |
| `make eval` | chạy `tests/testset.jsonl` (16 câu, 4 `no_answer`), ghi `logs/eval_*.jsonl` và `eval/out/*` |
| `make eval ARGS="--apps"` | thêm cả 21 hồ sơ, so với `tests/expected_decisions.json` |
| `make eval-score LOG=logs/eval_X.jsonl` | chấm lại một log đã có, **không gọi LLM** |
| `make retrieval-eval` | đo riêng tầng truy xuất (recall@k, MRR), không cần khóa API |
| `make test` | toàn bộ test offline (agent giả, chấm điểm, MCP, truy xuất): không tốn lượt gọi LLM |

Giới hạn tốc độ của API: `make eval ARGS="--pace 8"` nghỉ 8 giây giữa các câu; log được ghi ngay sau mỗi câu nên bị ngắt giữa chừng vẫn chạy tiếp được bằng `ARGS="--resume logs/eval_X.jsonl"`.

## Mô hình và endpoint

Mọi lời gọi LLM đi qua SDK `openai` tới một endpoint tương thích OpenAI, cấu hình bằng `.env` (xem `.env.example`):

| Vai trò | Biến | Ghi chú |
|---|---|---|
| Actor (agent) | `GROQ_API_KEY` (+ `GROQ_BASE_URL`) hoặc `OPENAI_API_KEY` + `OPENAI_BASE_URL`; `LLM_MODEL` | mặc định `openai/gpt-oss-20b` trên Groq; `gpt-oss-120b` là cấu hình cho số liệu chính; Qwen2.5 7B/14B chạy trên vLLM (Colab T4) qua `OPENAI_BASE_URL` |
| Critic (tùy chọn) | `FINCO_CRITIC=0` tắt; `FINCO_CRITIC_MODE=single\|agent`; `FINCO_CRITIC_PROVIDER=gemini` + `GEMINI_API_KEY`, hoặc `FINCO_CRITIC_BASE_URL` + `FINCO_CRITIC_API_KEY` | code mặc định **bật** chế độ `single`; kết quả A/B chưa thấy critic cải thiện nên các số liệu "không critic" chạy với `FINCO_CRITIC=0` |
| Giám khảo LLM | `JUDGE_MODEL` | chỉ `eval/judge.py`, nên khác Actor |

Các model đã đánh giá và log tương ứng: `docs/ket-qua-danh-gia.md` mục 1. Chỉ cần một khóa Groq là chạy được `make demo`; Gemini và vLLM chỉ cần cho thí nghiệm critic và so sánh model.

## Kiến trúc

```
run.py / make ─► finco/agent.py (vòng lặp tự viết, MAX_STEPS)
                   │   ├─ finco/llm.py: call_llm_tools (retry, backoff, timeout, usage) ◄─ prompts/agent_system.txt
                   │   └─ finco/mcp_client.py ── list_tools() ĐỘNG ──► finco/mcp_server.py (stdio)
                   │                                                    ├─ search_policy   [đọc, RAG]
                   │                                                    ├─ get_application [đọc]
                   │                                                    ├─ calc_dti        [đọc, tính bằng code]
                   │                                                    ├─ submit_decision [GHI]
                   │                                                    └─ get_decision    [đọc → xác nhận]
                   ▼
              trace (LogLine) ─► eval/verifier.py (quyết định, mã lý do, trích dẫn, DTI, ghi + đọc lại)
                              └► eval/judge.py    (giám khảo LLM, tùy chọn)
```

Quyết định thiết kế chính:

- **Native function calling + `thought`.** Schema tool lấy từ `list_tools()` của MCP server rồi đổi sang định dạng OpenAI; mỗi tool được tiêm thêm tham số `thought` bắt model nói lý do trước khi gọi (giữ được "Thought" để audit mà không phải parse văn bản). Agent không có registry tool viết tay.
- **Không raise ra ngoài.** Lỗi tool, lỗi MCP, tham số JSON hỏng, tên tool lạ, hết retry LLM đều thành `Error: ...` hoặc kết thúc gọn với `success=False`; trace và usage vẫn được giữ.
- **Ghi xong tự đọc lại.** `submit_decision` thành công thì code tự gọi `get_decision` và đính kèm kết quả vào phản hồi cho model; verifier kiểm tra cặp ghi + đọc lại trong trace.
- **Đo bằng trace, không bằng văn bản tự do.** Verifier dựng bài nộp từ chính các lời gọi tool (`submission_from_trace`), nên không phụ thuộc cách model diễn đạt.
- **Critic chốt chặn trong `submit_decision`** (`src/finco/critic.py`, `critic_agent.py`): kiểm tra bằng chứng bằng code rồi nhờ một LLM tự suy ra quyết định để so sánh; chỉ chặn khi phản đối trích được nguyên văn bằng chứng, tối đa 2 lần chặn. Là thí nghiệm có đo (A/B trong `docs/ket-qua-danh-gia.md`), tắt bằng `FINCO_CRITIC=0`.
- **Thứ tự xử lý theo CP-04** (loại trừ → chứng từ → DTI → thẩm quyền, dừng ở bước đầu tiên không đạt): agent chỉ tra cứu những gì bước hiện tại cần, tiết kiệm lượt gọi LLM.

RAG: chunk theo mục `##` của Markdown (một mục = một điều kiện hoàn chỉnh; chunk dài hơn 430 token mới tách theo điều khoản), mỗi chunk mang dòng đầu `[doc_id version | hiệu lực từ ...]` để chunk giữa tài liệu vẫn biết thuộc đâu. Hybrid = 0,3 BM25 + 0,7 cosine, chuẩn hóa min-max sau khi **lọc theo ngày hiệu lực** (`as_of_date`), vì hai phiên bản CP-01 gần như trùng chữ, chỉ khác con số nên tìm kiếm ngữ nghĩa thuần không phân biệt được.

## Mã lý do

`submit_decision` nhận thêm `reason_codes` (tập đóng, phải khớp quyết định) để chấm "đúng điểm nghẽn" bằng so sánh tập hợp, không cần LLM đọc lý do:

| Quyết định | Mã |
|---|---|
| `approve` | `APPROVED` |
| `reject` | `AGE`, `AMOUNT_RANGE`, `TERM_RANGE`, `OVERDUE`, `DTI_EXCEEDED` |
| `need_info` | `MISSING_DOCS`, `MISSING_DEBT_DECLARATION` |
| `escalate` | `ESCALATE_MANAGER`, `ESCALATE_COUNCIL`, `ESCALATE_EMPLOYEE` |

## Log và kết quả

- `logs/run_<run_id>.jsonl`: mỗi lượt REPL một file. `logs/eval_<thời gian>.jsonl` và `logs/demo_<thời gian>.jsonl`: gộp nhiều câu.
- Mỗi dòng: `ts, run_id, question_id, step, tool, args, result, usage{calls, prompt_tokens, completion_tokens, max_prompt}, cache_hit`. `usage` là mức dùng **tích lũy tại bước đó**. Dòng cuối mỗi câu có `tool = "__answer__"` chứa câu trả lời và USAGE của cả câu (kể cả câu không gọi tool nào).
- `eval/out/per_question.csv`, `eval/out/summary.json`, `eval/out/summary.png`: kết quả từng câu và tổng hợp (độ đúng quyết định, đúng mã lý do, tỷ lệ từ chối đúng/nhầm, faithfulness theo luật, token).
- `.env` không bao giờ được commit; `logs/sessions/` (nhật ký REPL cá nhân) bị bỏ qua bởi git.

## Cấu trúc thư mục

```
data/policies/*.md        tài liệu chính sách (có front matter, 2 phiên bản CP-01)
data/applications.json    21 hồ sơ vay giả lập      data/index/  vector đã tính sẵn (commit)
prompts/agent_system.txt  prompt hệ thống (tách khỏi code)
src/finco/                llm, agent, mcp_server, mcp_client, tools, chat, audit, store/, rag/
eval/                     run_eval.py (chạy), scoring.py (chấm), verifier.py, judge.py, sandbox_db.py, retrieval_eval.py
tests/                    testset.jsonl, expected_decisions.json (chỉ cho eval), test_*.py
logs/  report/            log chạy thật, báo cáo
```

## Giới hạn đã biết

- Mô hình nhỏ có thể gọi tool sai hoặc dừng sớm; khi model dừng mà chưa ghi quyết định, agent nhắc gọi `submit_decision` (mỗi lần model dừng sớm nhắc một lần, trong giới hạn `MAX_STEPS`); hết bước mà vẫn chưa ghi thì đánh dấu thất bại, không tự sửa tiếp.
- Faithfulness theo luật chỉ kiểm tra tài liệu được trích có thật trong kết quả tool, chưa kiểm từng khẳng định (`eval/judge.py` là giám khảo LLM tùy chọn, chạy riêng, và chính nó là LLM nên cũng có thể sai).
- Tầng truy xuất còn nhạy với cách đặt truy vấn: truy vấn kiểu "CP-01 ..." có thể trả bản diễn giải (CP-15/16) trước chính sách gốc, nên prompt dùng truy vấn bám tên tài liệu (có test giữ ràng buộc này: `tests/test_prompt.py`).
- Quyết định `need_info` chưa thay được bằng quyết định mới khi hồ sơ bổ sung đủ (CP-07 mục 3.3).
