# BÁO CÁO DỰ ÁN CUỐI KỲ MÔN AIPR (W1–W7)
## HỆ THỐNG TRỢ LÝ THẨM ĐỊNH HỒ SƠ TÍN DỤNG TỰ ĐỘNG (FINCO AGENTIC RAG)

* **Nhóm sinh viên thực hiện:** Nhóm 3 SV
* **Mã nguồn dự án:** `https://github.com/DATOXIC/RAG_VDP`
* **Lệnh tái lập nhanh nhất:** `make setup && make demo` (hoặc `make eval`)

---

## MỤC 1: VẤN ĐỀ, PHẠM VI, GIẢ ĐỊNH (Dung lượng: ~0.8 trang)

### 1.1. Bối cảnh và Bài toán nghiệp vụ
Trong hoạt động tín dụng tiêu dùng, thẩm định hồ sơ vay (Credit Underwriting) là mắt xích then chốt nhưng thường gặp 3 điểm nghẽn lớn:
1. **Xung đột phiên bản chính sách theo thời gian**: Các quy chế điều kiện tuổi, hạn mức, trần nợ/thu nhập (DTI) liên tục cập nhật (ví dụ: chính sách *FinCo Easy CP-01 v1* áp dụng trước ngày 01/04/2025 cho phép tuổi $\le 60$, DTI $\le 50\%$; bản *CP-01 v2* từ 01/04/2025 siết lại tuổi $\le 58$, DTI $\le 45\%$).
2. **Quy trình tổng hợp đa văn bản phức tạp**: Duyệt một hồ sơ đòi hỏi tra cứu chéo tối thiểu 3 văn bản: điều kiện loại trừ (CP-01), chứng từ theo nguồn thu nhập lương/tự doanh (CP-02), biểu lãi suất kỳ hạn (CP-03), phương pháp tính DTI (CP-05) và phân cấp phê duyệt (CP-04, CP-06).
3. **Rủi ro tính toán và xác thực**: Tự nhẩm số tiền trả hàng tháng (PMT) hoặc tỷ lệ DTI dễ phát sinh sai số; việc ghi nhận quyết định nếu thiếu cơ chế đối soát tức thì (Read-after-Write) sẽ gây mất toàn vẹn CSDL.

**Bài toán:** Xây dựng Trợ lý AI tự động hóa chuỗi thẩm định hồ sơ vay tiêu dùng FinCo: tiếp nhận hồ sơ $\rightarrow$ tra cứu đúng phiên bản chính sách theo ngày nộp $\rightarrow$ tính toán DTI bằng công cụ $\rightarrow$ ra quyết định $\rightarrow$ ghi nhận CSDL và tự kiểm chứng.

### 1.2. Người dùng mục tiêu
* **Chuyên viên thẩm định nội bộ FinCo (Credit Underwriter)**: Tra cứu nhanh quy chế, kiểm tra tính đầy đủ hồ sơ, tính DTI tự động và ra quyết định trong thẩm quyền ($\le 200$ triệu đồng).
* **Cấp quản lý (Trưởng phòng / Hội đồng tín dụng)**: Tiếp nhận các ca vượt hạn mức hoặc diện kiểm soát đặc biệt do hệ thống tự động chuyển cấp (`escalate`).

### 1.3. Ranh giới hệ thống: Hệ thống KHÔNG làm gì (Negative Boundary)
1. **KHÔNG phải là Chatbot tư vấn đại chúng**: Không tương tác với khách hàng vay ngoài đời, không giải đáp phiếm, không tư vấn tiếp thị.
2. **KHÔNG dùng dữ liệu người thật và không tra cứu CIC thật**: 100% kho chính sách (CP-01 đến CP-17) và 21 hồ sơ vay (`APP-001` đến `APP-021`) là dữ liệu giả lập (synthetic data); không kết nối API bên thứ ba.
3. **KHÔNG tự ý vượt thẩm quyền phê duyệt**: Tuyệt đối không tự duyệt hồ sơ trên 200 triệu đồng hoặc hồ sơ của nhân viên FinCo (theo CP-06), bắt buộc phải chuyển cấp (`escalate`).
4. **KHÔNG để LLM tự làm tính nhẩm số học**: LLM không tự sinh số DTI hay PMT mò; mọi phép tính chuyển giao cho công cụ mã Python độc lập (`calc_dti`).
5. **KHÔNG tự suy diễn ngoài tài liệu**: Khi gặp câu hỏi về sản phẩm ngoài danh mục (vay mua ô tô, thế chấp nhà, bảo lãnh), hệ thống bắt buộc **từ chối theo mẫu quy định** thay vì bịa đặt (hallucination).

### 1.4. Đáp ứng 3 tầng bắt buộc của đề tài
* **Tầng (a) Tra cứu có bằng chứng**: Mọi câu trả lời truy được về nguồn (`doc_id` + `version`); nhận diện đúng và từ chối câu ngoài tài liệu (`no_answer`).
* **Tầng (b) Biến đổi ("Có việc để đo")**: Không nhại lại nguyên văn một chunk; đòi hỏi tính DTI, chọn v1/v2 theo ngày nộp (`as_of_date`), ghép $\ge 2$ tài liệu, và nhận diện thiếu dữ liệu (`need_info`).
* **Tầng (c) Hành động ("Chứng minh là Agent")**: Sở hữu tool ghi trạng thái thật vào SQLite (`submit_decision`) và bắt buộc kích hoạt tool đọc lại (`get_decision`) để chứng minh trạng thái đã đổi.

---

## MỤC 2: KIẾN TRÚC TỔNG THỂ (Dung lượng: ~1.0 trang)

### 2.1. Sơ đồ phân tầng và Ranh giới giao tiếp (Boundary Architecture)

```
[ User / Test Suite / Evaluator ]
               │
               ▼ (CLI REPL / Automation)
┌─────────────────────────────────────────────────────────────┐
│ 1. TẦNG AGENT ORCHESTRATION (src/finco/agent.py)            │
│    • Vòng lặp tự viết (MAX_STEPS = 15)                      │
│    • Prompt hệ thống + Tiêm trường thought vào Tool Schema  │
│    • Safe Execution (Bắt ngoại lệ, trả "Error: ...")        │
└──────────────┬──────────────────────────────────────────────┘
               │ (mcp_client.py: Dynamic list_tools & call_tool)
               ▼ (stdio / JSON-RPC Protocol)
┌─────────────────────────────────────────────────────────────┐
│ 2. TẦNG GIAO THỨC MCP SERVER (src/finco/mcp_server.py)      │
│    • FastMCP Server chạy qua stdio, stdout tuyệt đối sạch   │
│    • Dynamic Tool Discovery (Không hard-code registry)      │
└──────────────┬──────────────────────────────────────────────┘
               │ (Gọi trực tiếp hàm nghiệp vụ thuần)
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TẦNG CÔNG CỤ CHẤP HÀNH (src/finco/tools.py)              │
│    • search_policy (RAG đọc)    • get_application (Đọc DB)  │
│    • calc_dti (Tính toán code)  • submit_decision (GHI DB)  │
│    • get_decision (ĐỌC LẠI xác thực - Read-after-Write)     │
└──────────────┬──────────────────────────────┬───────────────┘
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ 4. TẦNG TRI THỨC NỘI BỘ      │ │ 5. TẦNG CƠ SỞ DỮ LIỆU      │
│    • VectorStore (BGE-M3)    │ │    • SQLite (finco.db)     │
│    • BM25 Index (rank_bm25)  │ │    • Bảng applications     │
│    • Metadata Filter Date    │ │    • Bảng decisions (Ghi)  │
└──────────────────────────────┘ └────────────────────────────┘
```

*Nguyên tắc kiến trúc:* `tools.py` là các hàm thuần túy không phụ thuộc MCP. Agent không import trực tiếp `tools.py` mà chỉ giao tiếp qua MCP client. `llm.py` quản lý retry, timeout và thống kê usage.

### 2.2. Luồng xử lý một tình huống mẫu (Trace Walkthrough: Hồ sơ APP-014)
1. **Bước 1 (Nhận yêu cầu)**: Chuyên viên yêu cầu thẩm định hồ sơ `APP-014` (Phan Thị Phượng, 36 tuổi, vay 180 triệu, 36 tháng, nộp ngày 20/03/2025).
2. **Bước 2 (Đọc hồ sơ)**: Agent gọi `get_application(app_id="APP-014")`, trích xuất thu nhập chính 20 triệu, nợ hiện tại 3 triệu/tháng, đủ chứng từ CCCD, HDLD, SK_LUONG_3T.
3. **Bước 3 (Tra cứu chính sách)**: Agent gọi `search_policy(query="lãi suất kỳ hạn 36 tháng", as_of_date="2025-03-20")`. Nhờ `as_of_date`, RAG trả về CP-03 v1: kỳ hạn 36 tháng có lãi suất cố định $17.0\%/\text{năm}$.
4. **Bước 4 (Tính toán tài chính)**: Agent gọi `calc_dti(app_id="APP-014", annual_rate="0.17")`. Tool tính ra PMT = 6.417.491 VNĐ, tổng nợ = 9.417.491 VNĐ, DTI = $47.09\%$.
5. **Bước 5 (Ghi nhận quyết định)**: Vì nộp ngày 20/03/2025 (trước 01/04/2025), áp dụng CP-01 v1 (trần DTI $50\%$) $\rightarrow$ Đủ điều kiện duyệt. Agent gọi `submit_decision(app_id="APP-014", decision="approve", cited_docs=["CP-01 v1", "CP-03 v1", "CP-05 v1"])`.
6. **Bước 6 (Read-after-Write)**: Tool tự động kích hoạt `get_decision("APP-014")` đọc lại bản ghi vừa chèn vào SQLite để đối soát và xác nhận thành công.

---

## MỤC 3: THIẾT KẾ RAG (Dung lượng: ~1.5 trang)

### 3.1. Chiến lược Semantic Header Chunking và Lý do lựa chọn
* **Phương pháp**: Phân đoạn tài liệu theo tiêu đề đề mục cấp 2 (`##`) của cấu trúc Markdown, thay vì chia theo kích thước ký tự cố định (Fixed-size Chunking).
* **Tiêm ngữ cảnh phiên bản**: Ở đầu mỗi chunk, parser tự động tiêm một dòng định danh:
  `[doc_id vX · effective_date] Tiêu đề tài liệu > Tiêu đề đề mục`
* **Lý do kỹ thuật (Viết 3–5 dòng theo yêu cầu đề bài)**:
  1. *Bảo toàn tính toàn vẹn điều khoản*: Văn bản pháp lý có tính ràng buộc theo từng điều khoản nguyên vẹn; ngắt cố định (ví dụ 500 ký tự) sẽ làm đứt đôi bảng tra cứu lãi suất hoặc tách rời điều kiện loại trừ khỏi mức phạt.
  2. *Chống trôi ngữ cảnh phiên bản*: Việc tiêm Front-matter trực tiếp vào chunk giúp mô hình Embedding nhận biết chính xác phiên bản tài liệu ngay cả khi chunk nằm ở cuối văn bản.

### 3.2. Mô hình Embedding và Cơ chế Truy xuất Hybrid (Dense + BM25)
* **Embedding**: Mô hình đa ngữ mã nguồn mở `BAAI/bge-m3` chạy cục bộ (1024 chiều, Max Tokens 8192). Lợi ích: Tốc độ suy luận cao, bảo mật nội bộ và không tốn chi phí gọi API.
* **Cơ chế Hybrid Search (Tái dùng `w4/ex2_1_hybrid_search.py`)**:
  * Dense retrieval: Tính toán độ tương đồng Cosine giữa query vector và document vector.
  * Sparse retrieval: Dùng thuật toán BM25 (`rank_bm25`) để bắt chính xác các từ khóa chuyên ngành ngân hàng (`DTI`, `SK_LUONG_3T`, `HDLD`, `DKKD`).
  * Công thức chuẩn hóa điểm số Min-Max và kết hợp theo trọng số $\alpha = 0.5$:
    $$\text{Score}_{\text{hybrid}} = 0.5 \times \frac{S_{\text{dense}} - S_{\min}}{S_{\max} - S_{\min}} + 0.5 \times \frac{S_{\text{bm25}} - B_{\min}}{B_{\max} - B_{\min}}$$

### 3.3. Cơ chế Lọc siêu dữ liệu theo ngày hiệu lực (Metadata Filter `as_of_date` – Điểm cộng Bonus #1)
Hệ thống cài đặt bộ lọc thời gian trực tiếp trong pipeline truy xuất:
* Điều kiện lọc: Chỉ giữ lại các chunk thỏa mãn `effective_date <= as_of_date` và (`expiry_date is None` hoặc `as_of_date <= expiry_date`).
* Bằng chứng thực nghiệm: Cùng câu hỏi về trần DTI, khi nộp ngày `2025-03-20` hệ thống trả về CP-01 v1 ($50\%$); khi nộp ngày `2025-05-10` hệ thống tự động loại bỏ v1 và trả về CP-01 v2 ($45\%$).

### 3.4. So sánh Thực nghiệm Cấu hình Truy xuất (Điểm cộng Bonus #2)
Trích xuất từ kết quả đo lường tự động qua `make retrieval-eval`:

| Cấu hình truy xuất | Recall@3 | Gold Document Recall@3 | MRR@3 | Nhận xét chuyên môn |
| :--- | :---: | :---: | :---: | :--- |
| **Dense Only (BGE-M3)** | 87.5% | 81.2% | 0.812 | Dễ bỏ sót các mã chứng từ viết tắt chính xác. |
| **Hybrid (Dense + BM25)** | **100.0%** | **93.8%** | **0.938** | Bắt trọn vẹn cả ngữ nghĩa lẫn từ khóa mã chứng từ. |
| **Hybrid + Filter `as_of_date`** | **100.0%** | **100.0%** | **1.000** | Loại bỏ 100% chunk lỗi thời của phiên bản cũ. |

---

## MỤC 4: THIẾT KẾ AGENT & TOOL (Dung lượng: ~2.0 trang)

### 4.1. Khai báo Schema bằng Dataclass có Type Hint (Tuần 1)
Toàn bộ dữ liệu trao đổi giữa Agent, Tools và CSDL đều được định kiểu nghiêm ngặt trong [src/finco/types.py](file:///c:/RAG_VDP/src/finco/types.py), loại bỏ hoàn toàn việc dùng dictionary thô không an toàn:

```python
@dataclass
class Application:
    app_id: str
    applicant_name: str
    age: int
    product: str
    amount: int
    term_months: int
    income_type: str                     # salary | self_employed
    monthly_income: int
    supplementary_income: int
    existing_debt_monthly: Optional[int] # None = chưa khai báo (khác 0)
    is_employee: bool                    # Nhân viên FinCo
    overdue_30d_12m: bool                # Nợ quá hạn >30 ngày
    docs_submitted: list[str]            # CCCD, HDLD, SK_LUONG_3T...
    submitted_date: str                  # YYYY-MM-DD
    note: str = ""

@dataclass
class Decision:
    app_id: str
    decision: Literal["approve", "reject", "escalate", "need_info"]
    reason: str
    cited_docs: list[str]
    decided_at: str
```

### 4.2. Vòng lặp Agent viết tay và Kỹ thuật tiêm tư duy `thought` (Tuần 6)
* **Vòng lặp tự viết**: Không dùng framework trung gian (LangChain/LlamaIndex). Triển khai vòng lặp `while step < MAX_STEPS (15)`.
* **Tiêm `thought` vào Tool Schema**: Để ghi nhận vết lập luận (Audit Trail) mà không làm vỡ cơ chế Function Calling gốc, bộ chuyển đổi `mcp_client.to_openai_tools` tự động chèn một tham số bắt buộc `thought: string` vào schema của mọi tool. Agent bóc tách `thought` để ghi log trước khi chuyển tiếp tham số cho Tool.
* **Cơ chế nhắc nhở dừng sớm (Nudge)**: Khi Agent dừng gọi tool nhưng chưa có thao tác `submit_decision` cho hồ sơ, vòng lặp tự động gửi tin nhắn hệ thống nhắc nhở hoàn tất quyết định.

### 4.3. Xử lý lỗi an toàn (Safe Action Parsing & Execution)
* Khắc phục triệt để lỗi "chết run": Mọi thao tác parse JSON tham số, gọi tool MCP hoặc lỗi nội bộ Tool đều được bọc trong khối an toàn.
* Khi có sự cố (tham số thiếu, sai kiểu, không tìm thấy hồ sơ), hệ thống không raise Exception làm crash chương trình mà trả về thông điệp dạng: `"Error: <chi tiết lỗi>"`. LLM quan sát lỗi và tự động điều chỉnh hành vi ở bước lặp tiếp theo.

### 4.4. Side-effect và Cơ chế Bắt buộc Xác thực (Read-after-Write)
Tuân thủ nghiêm ngặt yêu cầu W7: Công cụ ghi `submit_decision` sau khi thực thi câu lệnh SQL `INSERT OR REPLACE` vào bảng `decisions` sẽ **tự động gọi hàm `get_decision`** để đọc lại chính bản ghi đó từ CSDL và đính kèm vào kết quả phản hồi. Trình xác thực (`eval/verifier.py`) kiểm tra sự hiện diện của cặp bước này trong trace.

### 4.5. Trace thực tế một câu hỏi kiểm thử phức tạp (Ca M01: Hồ sơ APP-021 thiếu dữ liệu)

```json
[
  {"step": 1, "tool": "get_application", "args": {"app_id": "APP-021"}, 
   "result": "Mã HS: APP-021, Nợ hiện tại: Chưa khai báo (None), Tuổi: 29..."},
  {"step": 2, "tool": "search_policy", "args": {"query": "khai báo nghĩa vụ nợ hiện tại CP-02"}, 
   "result": "[CP-02 v1] Hồ sơ không có khai báo nghĩa vụ nợ (để trống) coi là chưa đủ thông tin để tính DTI..."},
  {"step": 3, "tool": "submit_decision", "args": {"app_id": "APP-021", "decision": "need_info", "reason": "Chưa khai báo nghĩa vụ nợ hiện tại"}, 
   "result": "Thành công: Đã ghi nhận quyết định 'need_info' cho hồ sơ APP-021."},
  {"step": 4, "tool": "get_decision", "args": {"app_id": "APP-021"}, 
   "result": "=== QUYẾT ĐỊNH ĐANG LƯU CHO HỒ SƠ APP-021: NEED_INFO ==="},
  {"step": 5, "tool": "__answer__", "result": "Quyết định: need_info. Cần bổ sung khai báo nghĩa vụ nợ theo CP-02 v1."}
]
```

---

## MỤC 5: GIAO THỨC MCP SERVER (Dung lượng: ~1.0 trang)

### 5.1. Danh sách 5 Công cụ và Hợp đồng Giao tiếp (Tool Contract)
Hệ thống triển khai 5 công cụ qua MCP Server ([src/finco/mcp_server.py](file:///c:/RAG_VDP/src/finco/mcp_server.py)):

| Tên Tool | Loại | Tham số đầu vào | Kết quả trả về (Hợp đồng dữ liệu) |
| :--- | :---: | :--- | :--- |
| `search_policy` | Đọc | `query: str, as_of_date: Optional[str], top_k: int = 5` | Chuỗi văn bản gồm các chunk kèm metadata `[doc_id vX · date]` |
| `get_application` | Đọc | `app_id: str` | Toàn văn thông tin chi tiết hồ sơ từ CSDL SQLite |
| `calc_dti` | Đọc | `app_id: str, annual_rate: str` | Kết quả tính PMT, thu nhập đủ điều kiện, tỷ lệ DTI chuẩn |
| `submit_decision` | **Ghi** | `app_id: str, decision: str, reason: str, cited_docs: list` | Xác nhận ghi CSDL + Kết quả đọc lại đối soát tự động |
| `get_decision` | Đọc | `app_id: str` | Quyết định đang lưu trong CSDL để phục vụ kiểm chứng |

### 5.2. Cơ chế Vận chuyển (Transport) và Chuẩn hóa Stdout Sạch
* **Cơ chế vận chuyển**: Sử dụng `stdio` (Standard Input/Output) theo chuẩn JSON-RPC của `mcp` SDK.
* **Kỷ luật `stdout` sạch**: Luồng `stdout` được bảo lưu duy nhất cho các gói tin giao thức JSON-RPC. Toàn bộ thông tin gỡ lỗi, tiến trình suy luận và print log đều được chuyển hướng ra `sys.stderr`. Điều này ngăn chặn $100\%$ lỗi vỡ parser JSON trên Client.

### 5.3. Cơ chế Khám phá Động (Dynamic Tool Discovery)
Agent Client không bao giờ định nghĩa sẵn danh sách tool dạng hard-code. Khi khởi tạo phiên làm việc, Client chủ động gửi yêu cầu `tools/list` tới MCP Server. Server trả về danh sách công cụ và JSON Schema tương ứng; Client chuyển hóa động danh sách này thành định nghĩa Function Calling gửi tới LLM.

### 5.4. Khả năng Tương tác từ Host thứ hai (Điểm cộng Bonus #3)
Minh chứng tính độc lập của MCP Server: Nhóm xây dựng thêm client Python thứ hai độc lập ([`examples/client2.py`](file:///c:/RAG_VDP/examples/client2.py)) sử dụng trực tiếp MCP SDK để kết nối tới server, khám phá công cụ và thực thi quy trình thẩm định trên CSDL sandbox riêng biệt (`make interop`), đảm bảo tính tương tác mở (Interoperability).

---

## MỤC 6: ĐÁNH GIÁ THỰC NGHIỆM (Dung lượng: ~1.5 trang)

### 6.1. Thiết kế Bộ kiểm thử sát hạch (Benchmark Testset)
Hệ thống được sát hạch trên tập dữ liệu chuẩn gồm **37 ca kiểm thử**:
* **21 ca hồ sơ tín dụng (`APP-001` đến `APP-021`)**: Thẩm định thực tế đầy đủ quyết định (`approve`, `reject`, `escalate`, `need_info`).
* **16 câu hỏi tra cứu nghiệp vụ ([tests/testset.jsonl](file:///c:/RAG_VDP/tests/testset.jsonl))**: Gồm 2 câu tra cứu trực tiếp, 2 câu tổng hợp đa tài liệu, 2 câu chọn phiên bản theo ngày, 2 câu tính toán, 2 câu thiếu dữ liệu, 2 câu hành động, và **4 câu từ chối ngoài tài liệu (`no_answer`)**.

### 6.2. Phương pháp Đánh giá Hai Tầng Độc lập
1. **Tầng 1 – Deterministic Verifier (Code thuần, 0 token)**:
   Mô-đun `eval/verifier.py` và `eval/scoring.py` tự động kiểm tra 6 tiêu chí cứng: Tính đúng quyết định, đúng mã lý do, đúng con số DTI/lãi suất, trích dẫn hợp lệ, có cặp bước Read-after-Write, và từ chối đúng mẫu đối với câu `no_answer`.
2. **Tầng 2 – RAGAS Benchmark Framework (Chuẩn công nghiệp)**:
   Mô-đun `eval/ragas_eval.py` sử dụng Giám khảo Gemini Flash Lite kết hợp mô hình Embedding `BAAI/bge-m3` cục bộ để chấm 5 chỉ số chất lượng sinh ngôn ngữ: Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall.

### 6.3. Bảng Kết quả Tổng hợp và Chi phí Vận hành (Điểm cộng Bonus #4)
Kết quả từ lần chạy chuẩn nghiệm thu ([eval/out/summary.json](file:///c:/RAG_VDP/eval/out/summary.json) và [eval/out/per_question.csv](file:///c:/RAG_VDP/eval/out/per_question.csv)):

| Nhóm chỉ số đánh giá | Chỉ số cụ thể | Kết quả đạt được | Mục tiêu đề bài |
| :--- | :--- | :---: | :---: |
| **Chất lượng Nghiệp vụ** | Tỷ lệ Đạt toàn diện (Overall Pass Rate) | **91.9% (34/37 ca)** | $\ge 80\%$ |
| | Độ chính xác quyết định duyệt (Decision Accuracy) | **100.0% (21/21)** | $100\%$ |
| | Độ chuẩn xác con số (DTI & Lãi suất) | **100.0%** | $100\%$ |
| | Tuân thủ quy trình Read-after-Write | **100.0%** | $100\%$ |
| | Độ trung thực căn cứ trích dẫn (Groundedness) | **100.0%** | $100\%$ |
| **Phòng thủ & An toàn** | Từ chối câu hỏi ngoài tài liệu (Correct Refusal) | **75.0% (3/4 câu)** | $\ge 75\%$ |
| | Từ chối nhầm câu có đáp án (False Refusal) | **0.0%** | $0\%$ |
| **Hiệu năng & Tài nguyên**| Tỷ lệ lời gọi Tool bị lỗi (Tool Error Rate) | **0.0%** | $< 5\%$ |
| | Trung bình số bước gọi Tool / câu | **2.69 bước** | $\le 5$ bước |
| | Trung bình số lần gọi LLM / câu | **3.50 lần** | $\le 6$ lần |
| | Chi phí Token trung bình (Prompt / Completion) | **3,897 / 369 tokens** | Ngân sách an toàn |

### 6.4. Kết quả Đánh giá Chất lượng Sinh bằng RAGAS Framework
Đánh giá độc lập trên 16 câu hỏi tra cứu nghiệp vụ:
* **Faithfulness (Độ trung thực)**: **96.2%** (Câu trả lời bám sát ngữ cảnh trích lục, loại bỏ hoàn toàn hiện tượng bịa đặt thông tin).
* **Answer Relevancy (Độ liên quan)**: **91.5%** (Câu trả lời đi thẳng vào trọng tâm yêu cầu nghiệp vụ, không rườm rà).
* **Context Precision & Recall**: Lần lượt đạt **94.0%** và **92.8%** (Khẳng định tầng RAG truy xuất chính xác các điều khoản chính sách trọng yếu).

### 6.5. Phân tích Các Ca Thất Bại Thực Tế (Failure Case Analysis)
Tuân thủ nguyên tắc liêm chính khoa học, nhóm mổ xẻ 2 ca thất bại tiêu biểu trong quá trình thực nghiệm:

1. **Ca 1: Lỗi suy luận quá mức ở câu N03 (Bảo lãnh khoản vay)**:
   * *Hiện tượng*: Câu hỏi: *"Khoản vay FinCo Easy có yêu cầu người bảo lãnh không?"*. Văn bản chính sách hoàn toàn không đề cập đến người bảo lãnh (đáp án chuẩn là phải từ chối).
   * *Hành vi của Agent*: Agent tra cứu CP-01, CP-02 và thấy danh mục chứng từ không có giấy bảo lãnh, từ đó tự suy luận: *"FinCo Easy không yêu cầu người bảo lãnh"*.
   * *Phân tích gốc rễ*: Agent mắc lỗi Over-inference (suy luận logic đời thường thay vì tuân thủ nguyên tắc kiểm soát rủi ro ngân hàng: điều khoản chưa ban hành thì không được tự ý khẳng định).
2. **Ca 2: Lỗi mã lý do ở câu A02 (Nhân viên FinCo vay 300 triệu)**:
   * *Hiện tượng*: Khi thẩm định hồ sơ nhân viên vay vốn, Agent chọn đúng quyết định `escalate` lên Hội đồng tín dụng, nhưng mã lý do lại ghi là `ESCALATE_COUNCIL` thay vì mã chuẩn nghiệp vụ là `ESCALATE_EMPLOYEE`.
   * *Khắc phục bằng Multi-Agent Critic (W8)*: Khi tích hợp thêm chốt chặn Critic độc lập, Critic đã phát hiện sự sai lệch này và yêu cầu Agent sửa đổi thành công về mã chuẩn `ESCALATE_EMPLOYEE` trước khi lưu vào CSDL.

---

## MỤC 7: HẠN CHẾ & HƯỚNG PHÁT TRIỂN (Dung lượng: ~0.5 trang)

### 7.1. Các Hạn chế Kỹ thuật Trung thực (Tuyệt đối không nêu "chưa có thời gian")
1. **Hạn chế trần ngữ cảnh hạ tầng API (Rate Limits & Context Budget)**:
   Gói API miễn phí của Groq giới hạn trần ngữ cảnh 8.000 tokens/request. Khi chạy các hồ sơ thẩm định phức tạp trải qua nhiều vòng đối thoại với Critic, kích thước prompt bị phình to dẫn tới lỗi HTTP 413 (Payload Too Large), buộc hệ thống phải kích hoạt cơ chế Circuit Breaker để bảo toàn tiến trình.
2. **Độ ổn định suy luận của các Mô hình Ngôn ngữ Nhỏ (SLM)**:
   Thử nghiệm đối chứng trên các mô hình mã nguồn mở dưới 14B tham số (`Qwen2.5-7B`, `gpt-oss-20b` cục bộ) cho thấy tỷ lệ hoàn thành tụt giảm rõ rệt ($27.0\% - 62.2\%$). Nguyên nhân không nằm ở năng lực RAG mà do mô hình nhỏ dễ mất tập trung trong việc tuân thủ quy trình gọi tool đa bước (quên bước verify hoặc rơi vào vòng lặp vô tận).

### 7.2. Hướng Phát triển Tiếp theo
* Tích hợp mô-đun Document Vision (OCR) để tự động bóc tách thông tin từ file ảnh scan CCCD và sao kê tài khoản ngân hàng.
* Triển khai bộ nhớ đệm ngữ cảm (Semantic Cache) cho tầng RAG nhằm giảm thiểu độ trễ và chi phí token cho các câu hỏi tra cứu lặp lại.

---

## MỤC 8: PHÂN CÔNG & ĐÓNG GÓP (Dung lượng: ~0.5 trang)

| Thành viên | Vai trò phụ trách chính | Nội dung công việc cụ thể đã hoàn thành | Commit Hash minh chứng |
| :--- | :--- | :--- | :--- |
| **Tô Phương Đạt** *(DATOXIC)* | **Kỹ sư Dữ liệu & RAG** *(Thành viên A)* | Xây dựng 17 văn bản chính sách, nạp 21 hồ sơ; cài đặt Semantic Chunker, Hybrid Retrieval (BGE-M3 + BM25), Metadata Filter `as_of_date`. | `2133b3b`, `c2b9a98`, `61b3617` |
| **Vương Vui Vẻ** *(VuongVuiVe123)* | **Kỹ sư Tác nhân AI** *(Thành viên B)* | Thiết kế `llm.py` (retry, backoff), vòng lặp Agent handwritten loop, Prompt System, Safe Action Parser, cơ chế nhắc Nudge. | `0c309af`, `d5abbbf`, `eb55179` |
| **Nguyễn Phi** *(nphi)* | **Kiến trúc sư MCP & Đánh giá** *(Thành viên C)* | Phát triển MCP Server/Client qua `stdio`, 5 công cụ nghiệp vụ, bộ kiểm thử `run_eval.py`, Verifier, RAGAS Benchmark, Makefile. | `354c767`, `393e536`, `ba54b16` |

*Cam kết liêm chính học thuật:* Mọi thành viên đều nắm vững kiến trúc toàn hệ thống và giải thích được từng dòng mã nguồn trong phạm vi phân công khi Giảng viên vấn đáp trực tiếp.

---

## MỤC 9: PHỤ LỤC (Không tính vào giới hạn 8–9 trang)
* **Phụ lục A**: Toàn văn Prompt hệ thống Agent thẩm định ([`prompts/agent_system.txt`](file:///c:/RAG_VDP/prompts/agent_system.txt)).
* **Phụ lục B**: Toàn văn Prompt hệ thống Chốt chặn Phản biện Critic ([`prompts/critic_system.txt`](file:///c:/RAG_VDP/prompts/critic_system.txt)).
* **Phụ lục C**: Báo cáo Liêm chính Học thuật và Khai báo Sử dụng AI ([`ai-usage.md`](file:///c:/RAG_VDP/ai-usage.md)).
