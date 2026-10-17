# FinCo: Trợ lý thẩm định tín dụng tự động (Agentic RAG & MCP)

> **FinCo** là hệ thống AI Agent hỗ trợ thẩm định hồ sơ vay tiêu dùng tự động: đối chiếu dữ liệu khách hàng với 17 văn bản chính sách theo đúng ngày hiệu lực (`as_of_date`), tính toán DTI bằng code deterministic, ra quyết định tín dụng và tự động ghi/đọc đối soát qua giao thức chuẩn **Model Context Protocol (MCP)**.
> 
> *Toàn bộ dữ liệu hồ sơ và văn bản chính sách là dữ liệu giả lập phục vụ nghiên cứu (chi tiết xem `data/README.md`).*

---

## 1. Khởi động nhanh (Quickstart)

Hệ thống hỗ trợ chạy bằng `make` (Linux/macOS) hoặc trực tiếp bằng `python run.py` (Windows đa nền tảng).

### Bước 1: Cấu hình khóa API (`.env`)
Sao chép file mẫu:
```bash
cp .env.example .env
```
Mở `.env` và điền khóa API:
* **Khóa chính (Bắt buộc để chạy Agent):**
  * `GROQ_API_KEY`: Dùng cho Agent thẩm định chính (mặc định mô hình `openai/gpt-oss-120b` trên hạ tầng Groq siêu tốc).
* **Khóa nghiên cứu (Tùy chọn - Dành cho đánh giá & phản biện):**
  * `GEMINI_API_KEY`: Dùng khi chạy bộ chỉ số RAGAS (`make ragas`) hoặc bật chốt chặn phản biện Multi-Agent Critic (`FINCO_CRITIC_PROVIDER=gemini`).

### Bước 2: Chạy hệ thống với 1 lệnh duy nhất
Tự động tạo môi trường ảo `.venv`, cài đặt thư viện, khởi tạo CSDL SQLite và chạy kịch bản thẩm định mẫu:

* **Trên Linux / macOS:**
  ```bash
  make all    # Tương đương: make setup && make demo
  ```
* **Trên Windows (PowerShell / CMD không có Make):**
  ```powershell
  python run.py all    # Tương đương: python run.py setup && python run.py demo
  ```

*(Lần đầu chạy sẽ tải mô hình embedding `BAAI/bge-m3` ~2GB về máy; các lần sau chạy hoàn toàn cục bộ).*

---

## 2. Lộ trình kiểm thử cho Người chấm (Tester / Grader Roadmap)

Để thuận tiện cho Giảng viên và người chấm kiểm chứng từ mức độ nhẹ đến toàn diện, nhóm khuyến nghị quy trình 4 cấp độ:

| Cấp độ kiểm thử | Lệnh chạy (Make / Windows) | Mục đích & Đặc điểm kiểm chứng |
| :--- | :--- | :--- |
| **1. Kiểm thử máy sạch (Offline)** | `make check`<br>`python run.py check` | **Không cần API key.** Chạy toàn bộ 220 bài unit test, đo tầng truy xuất RAG, kiểm thử giao thức MCP stdio và chấm lại log nộp bài. |
| **2. Xem Demo nghiệp vụ** | `make demo`<br>`python run.py demo` | Thẩm định cặp hồ sơ đối chứng đổi chính sách (`APP-014` vs `APP-015`) + 1 câu hỏi bẫy ngoài tài liệu (chứng minh Agent không bịa đặt). |
| **3. Đánh giá nhanh (Quick Eval)** | `make eval-quick`<br>`python run.py eval-quick` | Chạy 6 ca tiêu biểu (~150k tokens, tiêu tốn ít quota, phù hợp gói API miễn phí). |
| **4. Tái lập toàn bộ Benchmark** | `make eval`<br>`python run.py eval` | Chạy đầy đủ 37 ca (16 câu hỏi + 21 hồ sơ). Nên dùng kèm `ARGS="--pace 8"` để tránh chạm Rate Limit của Groq. |

<details>
<summary><b>🛠️ Danh mục toàn bộ lệnh tra cứu nâng cao (Bấm vào để mở rộng)</b></summary>

| Lệnh Make | Lệnh Python tương đương | Chức năng cụ thể |
| :--- | :--- | :--- |
| `make setup` | `python run.py setup` | Tạo `.venv`, cài `requirements.txt`, khởi tạo DB hồ sơ, tải embedding model. |
| `make run` | `python run.py` | Bật giao diện hội thoại dòng lệnh tương tác (CLI REPL). |
| `make test` | `python run.py test` | Chạy 220 bài unit test offline trong `tests/` bằng unittest. |
| `make retrieval-eval` | `python run.py retrieval-eval` | Đo Recall@k và MRR của tầng RAG (không tốn token). |
| `make eval-score` | `python run.py eval-score [LOG]` | Chấm điểm và vẽ biểu đồ lại từ 1 file log có sẵn mà **không gọi lại LLM**. |
| `make compare` | `python run.py compare` | Vẽ đồ thị so sánh hiệu năng các cấu hình Agent (A, H, C) vào `eval/out/compare/`. |
| `make policy-change` | `python run.py policy-change` | Thí nghiệm đổi số chính sách (DTI 50% -> 40%) trên sandbox để kiểm chứng tính linh hoạt. |
| `make ragas` | `python eval/ragas_eval.py` | Đo 5 chỉ số RAGAS (yêu cầu `GEMINI_API_KEY` và `requirements-eval.txt`). |
| `make interop` | `python examples/client2.py` | Client thứ 2 viết bằng Python thuần gọi MCP Server qua JSON-RPC stdio. |
| `make clean` | *Lệnh dọn dẹp* | Xóa toàn bộ cache `__pycache__` và file tạm. |

</details>

---

## 3. Kiến trúc Hệ thống (System Architecture)

```
run.py / Makefile ──► src/finco/agent.py (Vòng lặp Agent tự viết, MAX_STEPS=15)
                       │   ├─ src/finco/llm.py: Native Tool Calling + Retry Backoff
                       │   │                   (Cấu hình Prompt: prompts/agent_system.txt)
                       │   └─ src/finco/mcp_client.py ── JSON-RPC stdio ──► src/finco/mcp_server.py
                       │                                                     ├─ search_policy   [RAG Hybrid + Time Filter]
                       │                                                     ├─ get_application [Đọc hồ sơ vay CSDL]
                       │                                                     ├─ calc_dti        [Tính toán tài chính chuẩn]
                       │                                                     ├─ submit_decision [Ghi quyết định vào DB]
                       │                                                     └─ get_decision    [Đọc lại xác thực ghi thành công]
                       ▼
                  Audit Trace (LogLine JSONL)
                       ├─► eval/verifier.py (Thẩm tra tính đúng đắn: DTI, Mã lý do, Trích dẫn, Ghi + Đọc lại)
                       └─► eval/judge.py    (Giám khảo LLM độc lập)
```

### Các quyết định thiết kế cốt lõi:
1. **Giao thức MCP chuẩn hóa (Model Context Protocol):** Agent không tự tích hợp công cụ lỏng lẻo trong code mà tương tác với 5 công cụ qua MCP Server độc lập bằng `stdio transport`. Cho phép bất kỳ MCP Client nào (`client2.py`) cũng dùng chung được logic nghiệp vụ.
2. **Native Tool Calling kết hợp tham số `thought`:** Mọi tool trong schema đều được tiêm thêm thuộc tính `thought`, bắt buộc mô hình phải trình bày lý do suy luận trước khi phát sinh hành động, phục vụ nhật ký kiểm toán mà không cần parse regex thủ công.
3. **Cơ chế Ghi xong Đọc lại (Write-then-Read Verification):** Khi gọi `submit_decision`, hệ thống tự động gọi ngầm `get_decision` để lấy bản ghi thực tế vừa lưu trong SQLite trả về cho Agent. Trình `verifier.py` bắt buộc phải thấy cặp hành động này trong trace thì mới tính điểm hợp lệ.
4. **Tầng RAG Hybrid kết hợp Lọc theo thời gian (`as_of_date`):**
   * Kết hợp **BGE-M3 Dense Embedding (70%)** và **BM25 Sparse (30%)** kèm thuật toán chuẩn hóa Min-Max và Reciprocal Rank Fusion.
   * Lọc siêu dữ liệu theo ngày hiệu lực: Đảm bảo khi thẩm định hồ sơ cũ (nộp tháng 05/2025), RAG chỉ trích xuất chính sách CP-01 v1; khi thẩm định hồ sơ mới (tháng 07/2025), RAG tự động chọn CP-01 v2.
5. **Chốt chặn phản biện Multi-Agent Critic (Tùy chọn):** Cơ chế chốt chặn độc lập tại bước `submit_decision` để rà soát lỗi tính toán DTI và mã lý do trước khi lưu vĩnh viễn vào CSDL (`FINCO_CRITIC=1`).

---

## 4. Bảng Quy Chuẩn Mã Lý Do (Reason Codes)

Khi gọi `submit_decision`, Agent bắt buộc phải truyền mã lý do chuẩn thuộc tập đóng khớp với quyết định:

| Quyết định (`decision`) | Mã lý do chuẩn (`reason_codes`) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- |
| **`approve`** | `APPROVED` | Đủ điều kiện phê duyệt khoản vay. |
| **`reject`** | `AGE`, `AMOUNT_RANGE`, `TERM_RANGE`, `OVERDUE`, `DTI_EXCEEDED` | Vi phạm tuổi, số tiền, kỳ hạn, lịch sử nợ xấu, hoặc DTI vượt trần. |
| **`need_info`** | `MISSING_DOCS`, `MISSING_DEBT_DECLARATION` | Thiếu chứng từ bắt buộc (sao kê/HĐLĐ) hoặc chưa kê khai nghĩa vụ nợ. |
| **`escalate`** | `ESCALATE_MANAGER`, `ESCALATE_COUNCIL`, `ESCALATE_EMPLOYEE` | Vượt thẩm quyền chi nhánh, khách hàng VIP hoặc là nhân viên FinCo. |

---

## 5. Đánh giá Chất lượng chuyên sâu bằng RAGAS

Ngoài 37 ca kiểm thử nghiệp vụ, hệ thống tích hợp sẵn script đánh giá theo tiêu chuẩn học thuật **RAGAS** ([`eval/ragas_eval.py`](eval/ragas_eval.py)):

* **5 Chỉ số đo lường:** *Faithfulness* (Độ trung thực, 96.2%), *Answer Relevancy* (Độ liên quan, 91.5%), *Context Precision*, *Context Recall*, và *Answer Correctness*.
* **Cách thực hiện:** `make ragas` (Chạy trên log có sẵn, dùng `GEMINI_API_KEY` làm giám khảo khách quan độc lập với Actor Groq, tự lưu checkpoint nếu bị ngắt quãng).
* Chi tiết số liệu đối chứng lưu tại `eval/out/ragas/` và báo cáo phân tích tại [`docs/ket-qua-danh-gia.md`](docs/ket-qua-danh-gia.md).

---

## 6. Cấu trúc Thư mục Dự án

```text
├── data/
│   ├── policies/*.md          # 17 văn bản chính sách tín dụng nội bộ (có hiệu lực thời gian)
│   ├── applications.json      # 21 hồ sơ vay tiêu dùng giả lập
│   └── index/                 # Chỉ mục vector Dense và từ khóa BM25 đã tính toán sẵn
├── prompts/
│   ├── agent_system.txt       # Prompt chỉ dẫn nghiệp vụ chính của Agent
│   └── critic_system.txt      # Prompt chốt chặn phản biện rủi ro tín dụng
├── src/finco/
│   ├── agent.py               # Vòng lặp Agent thực thi đa bước
│   ├── llm.py                 # Lớp tích hợp LLM (cơ chế retry, exponential backoff, circuit breaker)
│   ├── mcp_server.py          # MCP Server cung cấp 5 công cụ thẩm định qua stdio
│   ├── mcp_client.py          # MCP Client kết nối và định dạng schema công cụ
│   ├── tools.py               # Triển khai chi tiết 5 công cụ nghiệp vụ
│   ├── store/db.py            # Quản lý CSDL SQLite lưu trữ hồ sơ và quyết định
│   └── rag/                   # Bộ máy RAG (Loader, Chunker, BM25, Dense, Hybrid, Filter)
├── eval/                      # Toàn bộ mã nguồn đánh giá (run_eval, verifier, judge, ragas_eval)
├── tests/                     # 220 bài unit test tự động (testset.jsonl, expected_decisions.json)
├── logs/                      # Nhật ký kiểm toán JSONL từng bước chạy thực tế (chỉ mục: logs/README.md)
├── Makefile                   # Tập lệnh chuẩn hóa cho môi trường Linux / macOS
├── run.py                     # Task Runner tương đương thuần Python cho môi trường Windows
└── report/report.md           # Toàn văn báo cáo khoa học 9 mục chuẩn mực
```

---

## 7. Các Giới hạn Kỹ thuật Đã biết (Known Limitations)

1. **Ngưỡng giới hạn Token của gói API miễn phí:** Khi chạy đồng thời nhiều vòng lặp phản biện phức tạp, kích thước ngữ cảnh có thể chạm giới hạn HTTP 413 hoặc Rate Limit của Groq (khắc phục bằng cờ `--pace 8` và cơ chế ngắt mạch an toàn).
2. **Độ ổn định của các Model nhỏ (SLM < 14B):** Thử nghiệm với `Qwen2.5-7B` hoặc `gpt-oss-20b` cục bộ cho thấy mô hình nhỏ dễ gặp lỗi quên gọi tool xác nhận; hệ thống đã tích hợp cơ chế cảnh báo Nudge để khắc phục.
3. **Độ nhạy từ khóa RAG:** Truy vấn chứa tên mã văn bản viết tắt (như "CP-01") có thể trả về các tài liệu sổ tay hướng dẫn (CP-15/CP-16) trước văn bản gốc; prompt hệ thống đã được tối ưu để truy xuất theo tiêu đề nghiệp vụ cụ thể.
