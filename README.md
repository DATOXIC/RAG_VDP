# FinCo — Trợ lý thẩm định tín dụng tự động sử dụng Agentic RAG và MCP

FinCo hỗ trợ thẩm định hồ sơ vay tiêu dùng: tra chính sách theo ngày nộp, tính tỷ lệ nợ trên thu nhập (DTI), đề xuất quyết định, ghi vào SQLite và đọc lại để xác nhận. Hệ thống cũng trả lời câu hỏi về chính sách, kèm nguồn và phiên bản tài liệu.

**Repo:** [github.com/DATOXIC/RAG_VDP](https://github.com/DATOXIC/RAG_VDP)

Đây là đồ án môn AI Programming with Python. Công ty FinCo, chính sách và hồ sơ vay đều là dữ liệu giả lập do nhóm xây dựng cho đề tài; hệ thống chỉ phục vụ học tập và đánh giá.

## 1. Cài đặt và chạy demo

Chạy các lệnh từ thư mục gốc của repo. Khuyến nghị **Python 3.13** (phiên bản nhóm đã thử); bộ thư viện hiện tại yêu cầu ít nhất Python 3.12 do ghim `numpy==2.5.3`. Máy cần Internet để cài thư viện, tải mô hình/tokenizer lần đầu và gọi API LLM.

### Cấu hình API

Sao chép `.env.example` thành `.env`:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# Linux / macOS
cp .env.example .env
```

Điền khóa Groq và đặt cấu hình sau trong `.env` để chạy cấu hình chính của báo cáo:

```dotenv
GROQ_API_KEY=<khoa_Groq_cua_ban>
LLM_MODEL=openai/gpt-oss-120b
RETRIEVAL_MODE=hybrid
FINCO_CRITIC=0
```

`FINCO_CRITIC=0` cần được đặt rõ: code server hiện mặc định bật critic nếu biến này bị bỏ trống. Khóa Gemini chỉ cần cho thí nghiệm critic dùng Gemini hoặc đánh giá RAGAS. Giữ `.env` trên máy cá nhân, không đưa vào Git hay gói nộp.

### Có GNU Make

Sau khi cấu hình `.env`, chạy một lệnh để tạo môi trường, cài thư viện, khởi tạo dữ liệu và chạy demo:

```bash
make all
```

Makefile dùng lệnh `python` trên PATH. Nếu máy dùng tên khác, chỉ định thông dịch viên, ví dụ `make all PYTHON=python3.13`.

### Windows PowerShell không có Make

Cài môi trường và thư viện một lần:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sau đó khởi tạo dữ liệu và chạy demo:

```powershell
.\.venv\Scripts\python.exe run.py all
```

`run.py all` thực hiện `setup` rồi `demo`; phần cài thư viện nằm ở các lệnh pip phía trên. Lần đầu, hệ thống có thể tải mô hình embedding `BAAI/bge-m3` khoảng 2 GB cùng dữ liệu tokenizer. Khi đã có cache, embedding chạy cục bộ; agent vẫn gọi API LLM.

### Demo gồm những gì?

Demo mặc định chạy hai hồ sơ có cùng DTI khoảng **47,09%** và một câu hỏi ngoài tài liệu:

| Tình huống | Chính sách áp dụng | Kết quả kỳ vọng |
|---|---|---|
| `APP-014` | CP-01 v1, trần DTI 50% | `approve` |
| `APP-015` | CP-01 v2, trần DTI 45% | `reject` |
| Hỏi về vay mua ô tô | Corpus không có thông tin | Từ chối trả lời |

CP-01 v1 áp dụng đến hết **31/03/2025**; v2 có hiệu lực từ **01/04/2025**. Hai hồ sơ được chạy trên DB sandbox riêng để có thể lặp lại demo. Trace và mức dùng token được ghi vào `logs/demo_<timestamp>.jsonl`.

## 2. Các lệnh kiểm tra và đánh giá

Bảng dưới giả định đã cài thư viện và chạy `setup`. Cột Python dùng thông dịch viên trong `.venv`: Windows là `.\.venv\Scripts\python.exe`, Linux/macOS là `.venv/bin/python`.

| Mục đích | Make | Python trong `.venv` |
|---|---|---|
| Khởi tạo DB, kiểm tra RAG và cấu hình API | `make setup` | `python run.py setup` |
| Chạy lại demo | `make demo` | `python run.py demo` |
| Hội thoại CLI | `make run` | `python run.py` |
| Unit test, không gọi LLM | `make test` | `python run.py test` |
| Đo riêng truy xuất dense/BM25/hybrid | `make retrieval-eval` | `python run.py retrieval-eval` |
| Kiểm chứng MCP bằng client thứ hai | `make interop` | `python run.py interop` |
| Chấm lại log chính, không gọi LLM | `make eval-score` | `python run.py eval-score` |
| Chạy mới 6 ca tiêu biểu | `make eval-quick` | `python run.py eval-quick` |
| Chạy mới đủ 37 ca | `make eval` | `python run.py eval --apps` |
| Vẽ so sánh A, H, C từ CSV có sẵn | `make compare` | `python run.py compare` |

`make setup` còn tạo `.venv` và cài thư viện; `python run.py setup` chỉ thực hiện phần khởi tạo ứng dụng. Các lệnh không gọi LLM không cần khóa API, nhưng tầng truy xuất và một số test vẫn cần mô hình/tokenizer đã tải sẵn nếu chạy không có Internet.

Chạy đủ bộ đánh giá với thời gian nghỉ giữa các ca:

```bash
make eval ARGS="--apps --pace 8"
```

```powershell
.\.venv\Scripts\python.exe run.py eval --apps --pace 8
```

Khi truyền thêm tùy chọn cho `run.py eval`, giữ `--apps` để bao gồm 21 hồ sơ; thiếu cờ này sẽ chỉ chạy 16 câu trong testset. Thời gian nghỉ giúp giảm lỗi giới hạn tốc độ, nhưng không loại bỏ được lỗi vượt kích thước ngữ cảnh hoặc hết quota.

Chấm lại log chính và lưu vào một thư mục riêng, không cần API:

```powershell
.\.venv\Scripts\python.exe eval/run_eval.py --score-only logs/eval_20261014_202454.jsonl --apps --out-dir scratch/rescore
```

Script đánh giá trực tiếp trả mã thoát `1` nếu có ca trượt; đọc bảng điểm và thông báo lỗi để phân biệt kết quả benchmark với lỗi thực thi.

Mỗi lần đánh giá xuất `summary.json`, `per_question.csv` và biểu đồ `summary.png` khi có matplotlib. CSV ghi kết quả từng câu cùng số lần gọi LLM, prompt token, completion token và prompt lớn nhất. Chỉ mục bằng chứng: [logs/README.md](logs/README.md) và [eval/out/README.md](eval/out/README.md).

## 3. Dữ liệu và kiến trúc

Corpus gồm **18 tệp chính sách Markdown**, tương ứng **17 mã CP-01 đến CP-17**; riêng CP-01 có hai phiên bản. Dữ liệu còn có **21 hồ sơ vay giả lập**. Tài liệu nguồn được nộp cùng chỉ mục vector; chi tiết metadata và các tình huống nằm trong [data/README.md](data/README.md).

```text
Người dùng / run.py
        |
        v
Agent loop tự viết (MAX_STEPS mặc định 15)
        |-- LLM: native tool calling, timeout, retry
        |-- Prompt: agent_underwrite.txt / agent_qa.txt
        |
        v
MCP client -- JSON-RPC qua stdio --> MCP server
                                      |-- search_policy   --> RAG
                                      |-- get_application --> SQLite
                                      |-- calc_dti        --> Tính PMT và DTI
                                      |-- submit_decision --> Ghi SQLite
                                      `-- get_decision    --> Đọc lại quyết định
        |
        v
Trace JSONL + USAGE --> Bộ chấm bằng code --> CSV / biểu đồ
```

Agent discover danh sách tool từ MCP server. Sau khi `submit_decision` thành công, code agent gọi `get_decision` ngay, ghi bước đọc lại vào trace và đưa kết quả về cho model.

### Thiết kế RAG

- **Chunking:** chia theo tiêu đề mục Markdown; mục dài được tách/gom theo điều khoản với ngân sách mục tiêu 430 token. Header giữ mã tài liệu, phiên bản và ngày hiệu lực. Chỉ mục nộp kèm có 112 chunk.
- **Dense:** embedding `BAAI/bge-m3`, xếp hạng bằng độ tương đồng cosine.
- **Hybrid:** chuẩn hóa min–max điểm BM25 và dense, trộn theo trọng số **0,3 BM25 + 0,7 dense**. CP-15/CP-16 bị trừ nhẹ điểm để ưu tiên chính sách gốc khi điểm gần nhau.
- **Lọc thời gian:** dùng `as_of_date` để chọn chunk theo ngày hiệu lực và ngày hết hạn trước khi xếp hạng. Khi đã đọc hồ sơ, tool có thể lấy ngày nộp làm mặc định.

Đổi chế độ bằng `RETRIEVAL_MODE=dense` hoặc `hybrid` trong `.env`. Sau khi sửa nội dung **hoặc metadata** nguồn, dựng lại index và chạy lại đánh giá trước khi cập nhật số liệu báo cáo:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m finco.rag.build_index
```

```bash
PYTHONPATH=src:. .venv/bin/python -m finco.rag.build_index
```

### Hợp đồng tool

| Tool | Đầu vào chính | Kết quả / tác động |
|---|---|---|
| `search_policy` | `query`, `as_of_date`, `top_k` | Đoạn chính sách kèm mã, phiên bản và ngày hiệu lực |
| `get_application` | `app_id` | Thông tin hồ sơ vay |
| `calc_dti` | `app_id`, `annual_rate` | Thu nhập đủ điều kiện, khoản trả hàng tháng và DTI |
| `submit_decision` | `app_id`, `decision`, `reason`, `cited_docs`, `reason_codes` | Ghi quyết định vào SQLite |
| `get_decision` | `app_id` | Đọc quyết định đã lưu để kiểm chứng |

Bốn loại quyết định là `approve`, `reject`, `need_info`, `escalate`. Mã lý do có cấu trúc được định nghĩa trong [src/finco/types.py](src/finco/types.py); phần thực thi ở [src/finco/tools.py](src/finco/tools.py).

## 4. Kết quả đã lưu và giới hạn

Bộ đánh giá gồm **16 câu hỏi**, trong đó **4 câu `no_answer`**, cộng **21 hồ sơ vay**. Bảng dưới là kết quả các lần chạy đã lưu, không phải cam kết cho mọi lần chạy mới:

| Cấu hình | Model / môi trường | Critic | Đạt toàn diện |
|---|---|---|---|
| A — cấu hình chính | gpt-oss-120b / Groq | Tắt | **34/37** |
| H | gpt-oss-20b / Groq | Tắt | **33/37** |
| C | Qwen2.5-7B-Instruct-AWQ / vLLM | Tắt | **10/37** |

“Đạt toàn diện” đòi hỏi ca chạy thành công và thỏa các tiêu chí áp dụng: đáp án, mã lý do, trích dẫn, phép tính, ghi và đọc lại. Cấu hình A còn trượt `N03` (suy diễn câu trả lời ngoài corpus), `A02` và `APP-018` (đúng quyết định chuyển cấp nhưng sai mã lý do).

Đánh giá RAGAS bổ sung trên 12 câu có đáp án của cấu hình A đã lưu: Faithfulness **0,944**, Answer Relevancy **0,810**, Answer Correctness **0,598**, Context Precision **0,801**, Context Recall **1,000**. Nguồn: [ragas_summary.json](eval/out/ragas/ragas_summary.json). Chỉ số kiểm tra mã tài liệu bằng luật trong bộ chấm chính chỉ kiểm tra một phần grounding, không thay thế việc đánh giá nội dung được nguồn hỗ trợ.

Các giới hạn cần lưu ý:

- Mẫu đánh giá nhỏ; kết quả còn phụ thuộc model, hạ tầng API và lần chạy. Từ chối ngoài tài liệu và chọn đúng mã lý do vẫn có ca sai.
- Truy xuất có thể lấy thêm tài liệu diễn giải hoặc tài liệu nhiễu. Khi cập nhật chính sách, cần dựng lại index, rà đáp án chuẩn và đánh giá lại.
- Critic là cấu hình thí nghiệm; kết quả đã lưu chưa cho thấy cải thiện rõ so với cấu hình chính. Log có chạy lại một câu cần được đối chiếu theo từng `run_id` khi phân tích.

Phân tích đầy đủ: [report/ket-qua-danh-gia.md](report/ket-qua-danh-gia.md).

## 5. Thí nghiệm bổ sung

- **Đổi chính sách:** `make policy-change` hoặc `python run.py policy-change` thay trần DTI của CP-01 v1 từ 50% xuống 40% trên bản sao tạm, dựng lại index và chạy các ca bị ảnh hưởng cùng ca đối chứng. Thí nghiệm gọi API LLM.
- **Critic:** đặt `FINCO_CRITIC=1`; chọn `FINCO_CRITIC_MODE=single` hoặc `agent`. Nếu dùng Gemini, đặt thêm `FINCO_CRITIC_PROVIDER=gemini` và `GEMINI_API_KEY` theo [.env.example](.env.example).
- **RAGAS:** cài thêm `requirements-eval.txt`, cấu hình `GEMINI_API_KEY`, rồi chạy `make ragas`. Trên Windows không có Make:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe eval/ragas_eval.py
```

RAGAS chấm log đã lưu, không chạy lại actor, nhưng vẫn gọi API giám khảo. Script lưu checkpoint để tiếp tục lần chạy bị gián đoạn.

## 6. Tệp cần đọc và thành viên

| Đường dẫn | Nội dung |
|---|---|
| [report/report.docx](report/report.docx) | Báo cáo Word |
| [ai-usage.md](ai-usage.md) | Khai báo sử dụng AI theo thành viên |
| [data/](data/) | Chính sách nguồn, hồ sơ mẫu và chỉ mục vector |
| [src/finco/agent.py](src/finco/agent.py) | Agent loop và bước đọc lại sau khi ghi |
| [src/finco/llm.py](src/finco/llm.py) | Gọi LLM, parse tool call, timeout và retry |
| [src/finco/mcp_server.py](src/finco/mcp_server.py) | MCP server qua stdio |
| [src/finco/mcp_client.py](src/finco/mcp_client.py) | MCP client và chuyển schema tool |
| [src/finco/rag/](src/finco/rag/) | Loader, chunker, dense, BM25, hybrid và bộ lọc |
| [prompts/](prompts/) | Prompt thẩm định, hỏi đáp và critic |
| [tests/](tests/) | Unit test, testset và đáp án chuẩn hồ sơ |
| [eval/](eval/) | Bộ chấm, đo truy xuất và thí nghiệm |
| [logs/README.md](logs/README.md) | Chỉ mục log, cấu hình và nguồn bằng chứng |

| Thành viên | Phần phụ trách chính |
|---|---|
| Tô Phương Đạt | Agent, tích hợp LLM, critic và đánh giá |
| Nguyễn Đăng Vương | MCP, tool nghiệp vụ, SQLite và lệnh chạy |
| Trương Khắc Nhật Phi | Dữ liệu chính sách, chunking và truy xuất RAG |

Lịch sử đóng góp nằm trong Git; các commit tiêu biểu được liệt kê ở báo cáo. Khi bàn giao mã nguồn để chấm, giữ lịch sử `.git`, dữ liệu nguồn và log minh chứng; loại `.venv`, `.env`, cache và tệp tạm khỏi gói nộp.
