# PLAN – Trợ lý thẩm định hồ sơ vay (FinCo)

> Tài liệu duy nhất để cả nhóm hiểu **làm gì, ai làm, làm theo hợp đồng nào, khi nào xong**.
> Đọc mục 1–3 trước (5 phút). Mục còn lại là tra cứu.
> Đổi hợp đồng (mục 6) phải báo cả nhóm và sửa file này trong cùng commit.

---

## 1. Dự án là gì (30 giây)

**FinCo** là một công ty tài chính **hư cấu**. Mỗi ngày chuyên viên thẩm định phải xét hồ sơ vay: tra đúng phiên bản chính sách, tính tỷ lệ nợ trên thu nhập (DTI), ra quyết định và ghi lại.

Hệ thống của nhóm là **trợ lý làm thay chuỗi việc đó**:

```
"Thẩm định hồ sơ APP-014 và ghi quyết định"
   → xem hồ sơ → tra chính sách (đúng phiên bản theo ngày nộp)
   → tra lãi suất → tính DTI → so sánh → quyết định
   → GHI quyết định vào DB → ĐỌC LẠI để xác nhận → trả lời kèm nguồn
```

**Hệ thống KHÔNG làm:** tư vấn vay cho người thật; dùng dữ liệu thật của ai; tra điểm tín dụng thật; thay con người quyết định ca lớn (chỉ "leo thang").
Toàn bộ chính sách và hồ sơ là **dữ liệu giả lập do nhóm tự viết**.

**Đã được GV xác nhận:** đề tự đề xuất được chấp nhận; được dùng `matplotlib`.
*(Lưu tin nhắn xác nhận làm bằng chứng.)*

---

## 2. Vì sao đề này hợp lệ (đọc để biết mình đang bảo vệ điều gì)

| Tầng bắt buộc | Nghĩa | Trong dự án này |
|---|---|---|
| (a) Tra cứu | Câu trả lời truy được về `doc_id` + đúng version; câu không có đáp án phải bị từ chối | Mọi quyết định kèm tài liệu trích; hỏi thứ chính sách không có (vd. vay mua ô tô) → từ chối |
| (b) Biến đổi | Không thể copy nguyên một chunk: phải tính / tổng hợp / so sánh / từ chối-vì-thiếu | Tính DTI; chọn v1 hay v2 theo ngày nộp; ghép nhiều tài liệu; thiếu sao kê → "chưa đủ dữ liệu" |
| (c) Hành động | Có tool GHI đổi trạng thái thật + bước ĐỌC LẠI để verify | `submit_decision` → `get_decision` |

**Không dính loại bị cấm:** không chỉ tra cứu · không phải tool chỉ-đọc · dữ liệu tự bịa nên không tra Google được và LLM không biết trước.

**Ví dụ minh họa tầng (b)** – cùng một hồ sơ, DTI = 47%:
- Nộp 20/3/2025 → áp chính sách v1 (tối đa 50%) → **duyệt**.
- Nộp 10/5/2025 → áp chính sách v2 (tối đa 45%) → **từ chối**.

Đây là chỗ hệ thống *có thể sai*, và chỗ có thể sai chính là chỗ chấm được điểm.

---

## 3. Việc của từng người (tóm tắt)

| | **A – Dữ liệu & RAG** | **B – Agent** | **C – MCP, đánh giá, đóng gói** |
|---|---|---|---|
| Sở hữu | `data/`, `rag/`, nạp dữ liệu | `llm.py`, `agent.py`, `verifier.py`, `prompts/` | `tools.py`, `mcp_*.py`, `eval/`, `tests/`, `Makefile`, README |
| Làm gì | Viết chính sách + hồ sơ; chunking; dense, BM25, hybrid; lọc theo ngày hiệu lực | Gọi LLM (retry/timeout/usage); vòng lặp `MAX_STEPS`; parse an toàn; verifier; trace | 5 tool; MCP server + client; testset; `make eval`; log USAGE; CSV + đồ thị; client MCP thứ hai |
| Bonus phụ trách | #1 Metadata filter, #2 So sánh cấu hình | (hỗ trợ ca thất bại) | #3 Host thứ hai, #4 Đồ thị + CSV |
| Vấn đáp sẽ bị hỏi | Chunking vì sao? Dense vs hybrid? Lọc ngày làm sao? | JSON hỏng thì sao? MAX_STEPS để làm gì? Verify để làm gì? | Vì sao stdout phải sạch? Client discover thế nào? Faithfulness đo ra sao? |

**Quy tắc:** ai viết phần nào phải **giải thích được từng dòng** phần đó. Dùng AI để viết thì được, nhưng không giải thích được = điểm cá nhân 0 (xem mục 12).

---

## 4. Kiến trúc

```
make demo / make eval
   │
   ▼
 agent.py  (vòng lặp tự viết, MAX_STEPS)
   │  ┌───────────── llm.py (retry, timeout, usage) ◄── prompts/*.txt
   │  ▼
   │ mcp_client.py ── list_tools() ĐỘNG ──► mcp_server.py (stdio)
   │                                          ├─ search_policy   [đọc, RAG]
   │                                          ├─ get_application [đọc]
   │                                          ├─ calc_dti        [đọc, tính bằng code]
   │                                          ├─ submit_decision [GHI]
   │                                          └─ get_decision    [đọc → VERIFY]
   ▼
 verifier.py (kiểm tra bằng code: doc_id có thật? số khớp? có bước verify?)
```

Quy tắc kiến trúc:
- `tools.py` là **hàm thuần**, không biết MCP. `mcp_server.py` chỉ bọc lại. (Dễ test.)
- Agent **không import `tools.py`**; chỉ gọi tool qua MCP client.
- LLM **không tự nhẩm số**: mọi phép tính đi qua `calc_dti`.
- Mọi quyết định phải trích `doc_id`; thiếu → tool trả `Error`.

---

## 4b. Quyết định kỹ thuật bám giáo trình (W1–W8)

| Tuần | Giáo trình dạy | Áp dụng trong dự án |
|---|---|---|
| W1 | `dataclass`, `TypedDict`, JSON encoder tùy biến | Mọi tham số/kết quả tool là dataclass/TypedDict; log `.jsonl` dùng encoder cho `datetime` |
| W2 | retry + backoff + jitter; lỗi nào nên/không nên retry; `string.Template`; log JSONL; không hard-code key | `llm.py`: retry `RateLimit`/`Connection`/5xx, **không** retry `Authentication`/`BadRequest`; prompt trong `prompts/*.txt` |
| W3 | `Document`/`DocumentChunk`, `VectorStore`, cosine bằng `numpy`, chunk + overlap | Store in-memory tự viết, **bỏ Chroma**; chunk theo mục Markdown |
| W4 | grounding contract, pipeline 5 bước, hybrid BM25 + vector (chuẩn hóa), pre-filter metadata, ngân sách token | `search_policy` + prompt agent "chỉ dựa vào ngữ cảnh; không đủ thì nói không đủ" |
| W5 | faithfulness / relevance / precision, tự viết LLM-judge, chẩn đoán theo mẫu, đổi một biến mỗi lần | `eval/`, xem mục 11 |
| W6 | function calling vs ReAct; `max_iterations`; không `eval()`; cửa sổ trượt | Function calling gốc + trường `thought` tiêm vào mỗi tool (ReAct lai); `MAX_STEPS`; parse tham số an toàn, xem mục 9 |
| W7 | MCP host/client/server; `list_tools`/`call_tool`; thiết kế tool như API công khai; stdout là giao thức; probe thô; agent không có registry viết tay | Xem mục 6.4 và 10 |
| W8 | 4 mẫu đa agent (pipeline, orchestrator–worker, router, critic); handoff contract | **Không bắt buộc**, xem bên dưới |

**Quyết định về đa agent:** dùng **một agent + verifier bằng code** (không dùng 5 agent như repo cũ). Giáo trình W8 nói: đo agent đơn trước; nếu tách ra không tốt hơn ở điểm nào chỉ ra được thì việc tách chỉ là trang trí. Verifier bằng code đã là "người phản biện" độc lập (không dùng chung ngữ cảnh với agent soạn, nên không phải con dấu cao su). Nếu còn thời gian, thử **mẫu critic** (một LLM phản biện riêng, nhận bản nháp + bằng chứng, ngân sách sửa = 1) như một thí nghiệm A/B; chỉ giữ nếu số liệu tốt hơn.

**Tái dùng bài lab của chính các bạn** (Lab 3 `VectorStore`, Lab 4 `RAGSystem`, Lab 5 evaluator, Lab 6 `Agent`, Lab 7 `rag_server`): đây là code các bạn đã viết và hiểu, tiết kiệm nhiều ngày nhất. Phải sửa theo hợp đồng ở mục 6.

## 5. Cấu trúc thư mục

```
RAG_VDP/
├── PLAN.md                  # file này
├── Makefile                 # setup | demo | run | eval | eval-score | retrieval-eval | test
├── run.py                   # điểm vào: setup | demo | chat | eval (dùng khi máy không có make)
├── README.md                # CHỈ lệnh chạy + cách tái lập
├── ai-usage.md              # khai báo dùng AI (bắt buộc)
├── .env.example             # không bao giờ commit .env
├── requirements.txt
├── data/
│   ├── policies/*.md        # tài liệu chính sách (có front matter)
│   ├── README.md            # từ điển dữ liệu + bảng tình huống cài sẵn
│   ├── applications.json    # hồ sơ vay giả lập (21 hồ sơ)
│   ├── *.db                 # SQLite sinh khi chạy – KHÔNG commit (.gitignore)
│   └── index/               # vector đã tính sẵn (.npy + chunks + model_id) – commit
├── prompts/                 # *.txt – prompt tách khỏi code
├── src/finco/
│   ├── types.py             # dataclass dùng chung (HỢP ĐỒNG)
│   ├── llm.py
│   ├── rag/                 # loader, chunker, dense, bm25, hybrid, filter
│   ├── store/               # SQLite: applications, decisions
│   ├── tools.py
│   ├── mcp_server.py
│   ├── mcp_client.py
│   ├── agent.py
│   ├── chat.py              # REPL (rich)
│   └── audit.py             # nhật ký phiên FINCO-xxxx
├── examples/client2.py      # bonus #3: host MCP thứ hai
├── eval/                    # run_eval.py (chạy), scoring.py (chấm), verifier.py, judge.py, sandbox_db.py, retrieval_eval.py, out/ (CSV, hình)
├── tests/                   # testset.jsonl + expected_decisions.json (chỉ cho eval)
├── scratch/                 # test cá nhân của từng người – .gitignore, không nộp
├── logs/                    # >= 3 file .jsonl chạy thật
└── report/                  # report.docx + sơ đồ
```

---

## 6. HỢP ĐỒNG chung (chốt ngày 1, đổi phải báo cả nhóm)

### 6.1 Front matter của mọi tài liệu chính sách
Một dòng một khóa `khóa: giá trị`, **không dùng danh sách hay YAML lồng nhau** (parser tự viết chỉ cần tách theo dấu `:` đầu tiên).
```
doc_id: CP-01
title: Chính sách cho vay tiêu dùng FinCo Easy
version: v2
effective_date: 2025-04-01
owner: Phòng Quản lý rủi ro
supersedes: CP-01 v1        # tùy chọn
status: active              # tùy chọn: active | superseded | expired
language: vi                # tùy chọn
expiry_date: 2025-03-31     # tùy chọn; chỉ có ở bản đã bị thay thế (CP-01 v1)
```
Bắt buộc: `doc_id`, `title`, `version`, `effective_date`, `owner`. Cặp (`doc_id`, `version`) là duy nhất. **Không có `expiry_date` nghĩa là còn hiệu lực.** Chi tiết: `data/README.md` mục 2.

### 6.2 Bảng `applications` (nguồn: `data/applications.json`, đơn vị tiền: đồng, số nguyên)
| Trường | Kiểu | Ghi chú |
|---|---|---|
| `app_id` | str | `APP-001`..`APP-021` |
| `applicant_name` | str | tên giả lập |
| `age` | int | tuổi tại ngày nộp |
| `product` | str | `FinCo Easy` |
| `amount` | int | số tiền vay |
| `term_months` | int | kỳ hạn |
| `income_type` | str | `salary` / `self_employed` |
| `monthly_income` | int | thu nhập chính bình quân tháng |
| `supplementary_income` | int | thu nhập phụ (chỉ tính một phần, xem CP-05) |
| `existing_debt_monthly` | int \| None | **`None` = chưa khai báo, khác với `0`** (xem CP-02) |
| `is_employee` | bool | là nhân viên FinCo (xem CP-06) |
| `overdue_30d_12m` | bool | nợ quá hạn >30 ngày trong 12 tháng gần nhất |
| `docs_submitted` | list[str] | mã chứng từ: `CCCD`, `HDLD`, `SK_LUONG_3T`, `DKKD`, `SK_TK_6T` |
| `submitted_date` | str | `YYYY-MM-DD`, quyết định phiên bản chính sách |
| `note` | str | ghi chú tự do, không chứa đáp án |

Lưu ý khi cài đặt SQLite: `docs_submitted` lưu chuỗi JSON; `is_employee`/`overdue_30d_12m` lưu 0/1 nhưng đọc ra phải là `bool`; `existing_debt_monthly` cho phép `NULL`.

### 6.3 Bảng `decisions` (bảng mà tool GHI thay đổi; bắt đầu **rỗng**)
| Trường | Kiểu | Ghi chú |
|---|---|---|
| `app_id` | str | **khóa chính** → mỗi hồ sơ chỉ một quyết định (ép ở tầng DB) |
| `decision` | str | `approve` \| `reject` \| `escalate` \| `need_info` (có thể dùng `Literal`) |
| `reason` | str | lý do, nêu điều kiện làm căn cứ |
| `cited_docs` | list[str] | không được rỗng; ghi `"CP-01 v2"`, `"CP-05"` (CP-07 mục 2.2); lưu chuỗi JSON |
| `decided_at` | str | thời điểm ghi |

Ngoại lệ duy nhất cho "một quyết định": quyết định `need_info` được thay bằng quyết định khác khi hồ sơ đã bổ sung đủ (CP-07 mục 3.3). Nếu chưa làm phần này, ghi rõ là giới hạn.

### 6.3b Kiểu dùng chung trong `src/finco/types.py`
Chỉ đặt vào đây dữ liệu **được truyền giữa các module** (hai người khác nhau cùng cần hiểu). Class có logic (`VectorStore`) và hàm tính (`pmt`, `dti`) thì để ở module riêng.
`Application`, `Decision` (cần ngay cho `db.py`); `Chunk`, `SearchHit` (ingestion, truy xuất); kết quả `calc_dti`; dòng log (khi eval và agent cùng đọc). Import theo gói: `from finco.types import Application` (tên `types` trùng module chuẩn nên **không đặt ở thư mục gốc**).

### 6.4 Chữ ký 5 tool
| Tool | Loại | Tham số | Trả về |
|---|---|---|---|
| `search_policy` | đọc | `query`, `as_of_date` (tùy chọn, `YYYY-MM-DD`), `top_k` (tối đa 5) | văn bản có nguồn, mỗi đoạn dạng `[doc_id vX · effective_date] nội dung`, đã cắt theo ngân sách token |
| `get_application` | đọc | `app_id` | hồ sơ |
| `calc_dti` | đọc | `app_id`, `annual_rate` | `monthly_payment`, `qualified_income`, `dti`, `formula` |
| `submit_decision` | **GHI** | `app_id`, `decision`, `reason`, `cited_docs` | xác nhận đã ghi |
| `get_decision` | đọc (**verify**) | `app_id` | quyết định đang lưu, hoặc `Error` nếu chưa có |

Ràng buộc:
- `calc_dti` **bắt buộc nhận `annual_rate`** → agent phải tự tra lãi suất từ chính sách, không bỏ qua RAG.
- `submit_decision` trả `Error` nếu `cited_docs` rỗng, `app_id` không tồn tại, hoặc hồ sơ đã có quyết định.
- **Mọi tool không bao giờ `raise`**; lỗi trả chuỗi bắt đầu bằng `"Error: ..."`.
- **Chế độ truy vấn (`dense`/`hybrid`) KHÔNG phải tham số của tool** (giáo trình W7: mỗi tham số lộ ra là một lời hứa mà model phải đặt đúng, mà model không có cơ sở để chọn). Chế độ đặt bằng biến môi trường `RETRIEVAL_MODE` truyền cho server; `make eval-compare` chạy lại với từng giá trị.
- Không lộ ra MCP: tham số đường dẫn, `threshold`, `batch_search`, hàm ghi index. Kết quả luôn có nguồn (`doc_id`) và bị chặn độ dài.
- `as_of_date` sai định dạng → trả `"Error: ..."` (không đoán).
- Tool ghi/đọc dùng dataclass/TypedDict trong `types.py`, **không dùng dict thô**.
- Mô tả tool viết như prompt: nói rõ *khi nào dùng, khi nào KHÔNG dùng, tham số lấy từ đâu*.

### 6.5 Một dòng log (mỗi bước, file `.jsonl`)
`ts`, `run_id`, `question_id`, `step`, `tool`, `args`, `result`, `usage` {`calls`,`prompt_tokens`,`completion_tokens`,`max_prompt`}, `cache_hit`

### 6.6 Một dòng testset
`id`, `question`, `type`, `expected_answer`, `expected_docs`, `is_no_answer`, `expected_decision`, `target_app_id` (nếu có)

---

## 7. Dữ liệu (A dẫn dắt, cả nhóm góp)

Dùng **con số riêng của FinCo**, không bắt chước chính sách ngân hàng thật. Kiểm tra mỗi câu hỏi bằng cách hỏi LLM *không kèm tài liệu*: nếu nó đoán đúng → đổi con số.

**Trạng thái: ĐÃ VIẾT XONG.** Chi tiết từng trường, cách ingestion đọc, và bảng tình huống cài sẵn nằm ở `data/README.md` (đọc file đó trước khi viết code đọc dữ liệu). Toàn bộ là dữ liệu giả lập.

15 file trong `data/policies/` (14 mã, riêng `CP-01` có 2 phiên bản; tách chunk theo `## ` cho **68 chunk**). `CP-01..CP-07` là lõi quyết định đáp án; `CP-08..CP-14` là tài liệu nhiễu của sản phẩm khác (thẻ, tiền gửi, bảo mật, khiếu nại, ưu đãi đã hết hạn, định danh), cố ý gần từ vựng với FinCo Easy nhưng không làm đổi đáp án nào (xem `data/README.md` mục 2). Phần lõi:

| doc_id | Nội dung | Số liệu / bẫy cài vào |
|---|---|---|
| CP-01 v1 | Điều kiện cho vay (hiệu lực 1/1/2024, hết hạn 31/3/2025) | tuổi 22–60, vay 10–800 triệu, kỳ hạn 6–60 tháng, DTI tối đa **50%** |
| CP-01 v2 | Như trên (hiệu lực 1/4/2025, thay v1) | tuổi 22–58, vay 10–700 triệu, kỳ hạn 6–48 tháng, DTI tối đa **45%**; các mục gần giống chữ v1, chỉ khác số → bắt buộc lọc theo ngày |
| CP-02 | Chứng từ hồ sơ | Lương: `CCCD`, `HDLD`, `SK_LUONG_3T`; tự kinh doanh: `CCCD`, `DKKD`, `SK_TK_6T`; để trống nợ hiện tại ≠ 0; thiếu → `need_info`, không từ chối |
| CP-03 | Lãi suất, phí | 14,0% (6–12 tháng), 15,5% (13–24), 17,0% (25–36), 18,5% (37–60); trả góp đều (annuity) |
| CP-04 | Phân cấp phê duyệt, thứ tự xử lý | ≤200 triệu chuyên viên; >200–500 Trưởng phòng (escalate); >500 Hội đồng (escalate). Thứ tự: loại trừ → chứng từ → DTI → thẩm quyền |
| CP-05 | Thu nhập đủ điều kiện, DTI | Lương = chính + 70% phụ; tự kinh doanh = 80% chính + 50% phụ. `DTI = (PMT + nợ hiện tại) / thu nhập đủ điều kiện`; **bằng ngưỡng vẫn qua**, chỉ lớn hơn mới rớt |
| CP-06 | Ngoại lệ, nhóm hạn chế | Nợ quá hạn >30 ngày/12 tháng → từ chối; nhân viên FinCo luôn escalate lên Hội đồng sau khi đạt mọi điều kiện; không có ngoại lệ DTI |
| CP-07 | Ghi nhận quyết định | 4 trạng thái; phải trích tài liệu; một quyết định/hồ sơ; đọc lại sau khi ghi |
| CP-08 | Giải ngân, chăm sóc khách hàng | **Tài liệu nhiễu** cố ý (giờ làm việc, khiếu nại), kiểm tra truy xuất bỏ qua đúng chỗ |

`data/applications.json`: **21 hồ sơ** (`APP-001`..`APP-021`) gồm các ca: sạch · vượt DTI · thiếu chứng từ · nộp đúng ranh giới 31/3 và 1/4 · vay đúng 200 triệu · cặp hồ sơ giống nhau khác phiên bản (008/009, 012/013, 014/015, 016/017) · bẫy thu nhập phụ (019) · nhân viên FinCo (018) · nợ chưa khai (`null`, 021).

`tests/expected_decisions.json`: đáp án chuẩn từng hồ sơ (quyết định, phiên bản, DTI, lý do, tài liệu cần trích). **Chỉ dùng cho eval, agent không được đọc.** Đáp án do một bộ quy tắc tham chiếu viết theo văn bản chính sách tính ra: **sửa số trong chính sách thì phải tính lại file này.** Phân bố: 6 approve, 9 reject, 3 need_info, 3 escalate.

Chỉ dùng văn bản tự viết, **một định dạng duy nhất** (`.md` + `applications.json`) → không cần xử lý dữ liệu phức tạp. Nhóm nên **tự tính tay** vài hồ sơ (APP-014, APP-019) để kiểm tra chéo đáp án chuẩn.

---

## 8. RAG (A)

- **Chunking:** theo mục `##` của Markdown; thêm dòng đầu chunk gồm tiêu đề tài liệu + version. *Viết 3–5 dòng giải thích lý do vào report.*
- **Embedding chạy local (thực tế dùng `BAAI/bge-m3`, đa ngôn ngữ, ~2 GB; bản kế hoạch ban đầu ghi họ Jina):** đặt sau một hàm `embed(texts)`. Backend mặc định là **mô hình Jina mã nguồn mở chạy local** qua `sentence-transformers` (không cần key); `make setup` tải mô hình về cache một lần. Backend `JINA_API_KEY` chỉ là tùy chọn khi có key. **Tài liệu và truy vấn phải dùng cùng một mô hình**: file vector `.npy` lưu kèm `model_id`, lúc chạy so khớp, lệch thì báo lỗi rõ ràng (không âm thầm trộn hai không gian vector). Vector tài liệu tính sẵn và commit. Cần kiểm tra: mô hình Jina bạn chọn hỗ trợ tiếng Việt không, giấy phép (một số bản là phi thương mại, ổn cho bài tập), và có cần `trust_remote_code` không (ghim `revision` cụ thể).
- **Không dùng vector DB** (giáo trình W3 dạy `VectorStore` tự viết, cosine bằng `numpy`; Chroma không nằm trong giáo trình): vài chục chunk, `numpy` là đủ.
- **Hybrid theo W4:** điểm BM25 và điểm cosine **phải chuẩn hóa min–max trước khi trộn** (trộn điểm thô là lỗi kinh điển); trọng số khởi điểm 0.3 BM25 + 0.7 vector, chỉnh bằng thí nghiệm.
- **Lọc metadata trước khi chấm điểm** (pre-filter theo `effective_date`, như W4), không lọc sau.
- **Re-ranking (CrossEncoder) là tùy chọn:** mô hình `ms-marco` của giáo trình huấn luyện cho tiếng Anh nên tiếng Việt sẽ kém; chỉ thử nếu dữ liệu tiếng Anh và còn thời gian.
- **Ngân sách ngữ cảnh:** đếm token bằng `tiktoken`, cắt kết quả theo ngân sách; xử lý 3 ca biên của W4: không có chunk phù hợp, truy vấn rỗng, ngữ cảnh quá dài.
- Có thể tái dùng `VectorStore`/`RAGSystem` từ Lab 3–4 của chính các bạn (kiểu trả về `{query, answer, sources, num_chunks, context_tokens}`).
- **Truy vấn:** `dense` và `hybrid` (dense + BM25). Tái dùng `w4/ex2_1_hybrid_search.py` của lớp, **không viết lại**.
- **Lọc metadata:** chỉ trả chunk của tài liệu có `effective_date <= as_of_date` và (không có `expiry_date` hoặc `as_of_date <= expiry_date`). Ví dụ: `as_of_date=2025-03-31` → CP-01 v1; `2025-04-01` → CP-01 v2. Không truyền `as_of_date` thì dùng bản đang hiệu lực. Đây là phép thử quan trọng nhất của tầng truy xuất.
- **Gom theo tài liệu** khi trả kết quả để không lặp cùng một tài liệu.

## 9. Agent (B)

- **Dùng function calling gốc của API (đã đổi so với bản đầu của kế hoạch, theo cách `src/finco/agent.py` được triển khai; cần nhóm xác nhận).** Đề (mục 2.2, 2.6) chỉ yêu cầu vòng lặp tự viết và parse action an toàn, không bắt ReAct văn bản. Để vẫn có "Thought" cho audit, `mcp_client.to_openai_tools` tiêm thêm tham số bắt buộc `thought` vào schema mỗi tool; agent tách `thought` ra khỏi tham số trước khi gọi tool. "Parse an toàn" nay nằm ở biên tool: JSON tham số hỏng/không phải đối tượng, tên tool lạ, tool ném lỗi → trả `Error: ...` cho model, không raise.
- Vòng lặp: gọi LLM → có `tool_calls` thì chạy từng tool qua MCP và đưa kết quả lại (role `tool`) → không còn `tool_calls` thì dừng; hết `MAX_STEPS` (mặc định 12: quy trình dài nhất cần ~9 lần gọi) thì trả kết quả dừng gọn kèm `success=False`, không treo. Lượt thẩm định chưa ghi + đọc lại thành công bị nhắc tối đa 2 lần (`MAX_NUDGES`) rồi đánh dấu thất bại.
- **Parse an toàn:** JSON hỏng (vá một lần thiếu `}`) / không phải đối tượng / tên tool lạ / tool hoặc MCP ném lỗi / model viết `Action:` thành văn bản / server từ chối `tool_use_failed` → KHÔNG raise; trả `Error: ...` hoặc nhắc model, tối đa `MAX_NUDGES` lần rồi dừng gọn. Mỗi kiểu hỏng có test riêng trong `tests/test_agent.py` (LLM giả, không tốn API).
- **Không bao giờ `eval()`** đầu vào hay đầu ra của tool (W6); phép tính dùng hàm có kiểu hoặc bảng tra.
- **Bộ nhớ hội thoại:** cửa sổ trượt nhưng **luôn giữ câu hỏi/nhiệm vụ gốc** (lỗi kinh điển: cửa sổ quá nhỏ làm agent quên nhiệm vụ).
- **Agent không có `TOOL_REGISTRY` hay schema viết tay** (W7): danh sách và schema tool sinh từ `list_tools()`. Ngoại lệ có chủ đích: `agent.py` biết tên cặp ghi + kiểm chứng (`WRITE_TOOL`, `VERIFY_TOOL`) để ép mục 2.7 của đề; hai hằng số này nằm ở đầu file.
- Prompt nằm trong `prompts/*.txt`, không nhúng trong code.
- Sau `submit_decision` **bắt buộc** gọi `get_decision`; verifier kiểm tra điều này trong trace.
- **Verifier (bằng code):** `doc_id` trích có nằm trong tập chunk đã truy xuất? Con số trong câu trả lời khớp kết quả tool? Có cặp ghi → đọc lại?
- **Calibration:** không tìm thấy căn cứ → nói rõ "không có trong tài liệu"; thiếu dữ liệu để tính → `need_info`, không đoán.

## 10. MCP (C)

- Dùng `mcp` SDK, chạy qua **stdio**. **Log chỉ ra stderr** để stdout sạch (một `print()` trong tool làm hỏng luồng JSON-RPC).
- **Ghim phiên bản `mcp` trong `requirements.txt`.** Giáo trình W7 nêu 2 thế hệ: v1 (`from mcp.server.fastmcp import FastMCP`) và v2 (`from mcp.server import MCPServer`, `from mcp import Client`); decorator `@mcp.tool()` giữ nguyên. Phần lớn hướng dẫn trên mạng là v1. Repo cũ import theo kiểu v1 nhưng ghi `mcp>=2`: **phải chọn một bản và chạy thử import** trước khi viết tiếp.
- Client spawn server bằng `sys.executable` + **đường dẫn tuyệt đối** (spawn `python` với đường dẫn tương đối gây lỗi "Connection closed during initialize" trên máy GV).
- Nạp `VectorStore` **một lần ở mức module**, không dựng lại index trong một lần gọi tool.
- Có script **probe JSON-RPC thô** (chỉ thư viện chuẩn: `initialize` → `tools/list` → `tools/call`) để tách lỗi server khỏi lỗi agent.
- Client gọi `list_tools()` rồi biến kết quả thành định nghĩa tool cho LLM — **không hard-code tên tool**.
- Server bọc đúng 5 hàm trong `tools.py`.

## 11. Đánh giá (C dẫn dắt)

**Testset 16 câu** (≥12, trong đó ≥4 `no_answer`):

| Loại | Số câu |
|---|---|
| Tra 1 tài liệu | 2 |
| Tổng hợp ≥2 tài liệu | 3 |
| Phiên bản (chọn v1/v2) | 2 |
| Tính toán | 3 |
| Thiếu dữ liệu để suy luận | 2 |
| **`no_answer`** (gần giống câu có đáp án, không lạc đề) | **4** |
| Hành động + verify | 2 |

**Cam kết:** ít nhất 6/16 câu cần tính ngày / tổng hợp / so sánh (để không bị coi là tìm kiếm thuần).

**Số liệu:** độ đúng quyết định · recall@k truy xuất · tỷ lệ trích dẫn hợp lệ · faithfulness (quy tắc + LLM-judge một phần) · tỷ lệ từ chối đúng · token từng câu.

**Cách đo bám W5 (tự viết; `ragas` chưa dùng, xem mục 20):** faithfulness = tách câu trả lời thành các claim nguyên tử, LLM-judge (`temperature=0`) xét từng claim có được ngữ cảnh hỗ trợ không; thêm answer relevance và context precision@k; **chẩn đoán theo mẫu số liệu** (faithfulness thấp → prompt/ngữ cảnh; precision thấp → truy xuất) và chỉ đổi **một biến mỗi lần** khi cải tiến. Phải kiểm tra thủ công giám khảo LLM trên một mẫu. Testset theo phổ W4: câu trực tiếp / tổng hợp / **ngoài tài liệu**.

**Quy tắc của `make eval`:**
1. **Sao chép DB sạch trước mỗi lần chạy** (tool ghi sẽ làm lần sau đẹp giả).
2. Ghi rõ có dùng cache hay không; 3 lần chạy trong `logs/` phải là **chạy thật**.
3. Xuất bảng, `eval/out/per_question.csv` và đồ thị.

**Phải có ≥1 ca thất bại thật** được phân tích (lỗi ở truy xuất hay suy luận?).

**Thí nghiệm chứng minh "dùng đúng chính sách":**
- Chạy **không có RAG** → độ đúng phải thấp hơn rõ rệt.
- **Đổi một con số** trong tài liệu → đáp án phải đổi theo.

---

## 12. Điểm cộng (file ghi tối đa +6, nhưng chỉ liệt kê 4 mục × +1)

| # | Việc làm thêm | Người | Bằng chứng nộp |
|---|---|---|---|
| 1 | **Metadata filter dùng thật** (`as_of_date`) | A | Trace cùng hồ sơ đổi ngày → đổi phiên bản; thí nghiệm bật/tắt filter |
| 2 | **So sánh thêm 1 cấu hình** (dense vs hybrid; nếu còn thời gian thêm chunking) | A viết cờ, C chạy | `make eval-compare` + bảng trong report |
| 3 | **MCP dùng được từ host thứ hai** | C | `examples/client2.py` (client dùng `mcp` SDK) **và** script probe JSON-RPC thô không dùng SDK, cùng gọi một server; `make interop` (chạy trên **bản sao DB**) |
| 4 | **Đồ thị + CSV theo từng câu** | C | `eval/out/per_question.csv` + hình trong report |

2 điểm còn lại GV "có thể cân nhắc" → xem như phần nâng chất lượng: đối chứng có/không RAG, đổi số, so sánh 2 mô hình embedding, so sánh cả chunking lẫn dense-vs-hybrid, và thí nghiệm **critic** của W8 (xem mục 4b).

**Cổng quyết định:** cuối ngày 6 nếu phần lõi chưa chạy xanh → **cắt bonus #3 và #4**. Bonus #1 và #2 gần như là một phần của lõi nên giữ.

---

## 13. Lịch 10 ngày

| Ngày | A | B | C | Mốc chung |
|---|---|---|---|---|
| 1 | CP-01 v1; chốt danh sách tài liệu | `llm.py` retry/timeout/usage | Khung repo, `Makefile`, `.env.example`, hợp đồng; khai báo sẵn cờ eval | Hợp đồng ký; có commit của **cả 3 người** |
| 2 | CP-01 v2, CP-02..04 | Khung vòng lặp, tool giả | Bảng SQLite, nạp `applications.json` | |
| 3 | Chunker, dense | Prompt; parse an toàn | `get_application`, `calc_dti`, `submit_decision`, `get_decision` | Mỗi tool chạy riêng |
| 4 | BM25, hybrid | Ráp agent + tool thật | MCP server + client | **End-to-end lần 1** |
| 5 | **Filter `as_of_date` (#1)** | Verify, xử lý lỗi, calibration | Testset 16 câu + `run_eval.py` có `--retrieval` | Chạy thử toàn bộ |
| 6 | Sửa truy xuất theo lỗi | Sửa agent theo trace | `make eval` đầy đủ + CSV (#4) | **Cổng: lõi xanh?** |
| 7 | Dense vs hybrid (#2); filter on/off | Verifier hoàn chỉnh | 3 lần chạy thật → `logs/`; chọn ca thất bại | **Đóng băng tính năng** |
| 8 | Đối chứng không-RAG; đổi số | Phân tích ca thất bại | `client2.py` + `make interop` (#3); đồ thị | Số liệu cuối |
| 9 | Viết report phần mình | Viết report phần mình | Report, `ai-usage.md`, README | Bản nháp |
| 10 | Cold start trên máy sạch | Hỏi vấn đáp chéo | Điền commit hash vào report | **Nộp** |

**Không thêm tính năng sau ngày 7.** Ba ngày cuối dành cho độ tin cậy, report và luyện bảo vệ.

---

## 14. Quy trình git

- `main` luôn chạy được. Mỗi người làm nhánh riêng (`a/...`, `b/...`, `c/...`), tạo PR, **người khác review** rồi mới merge.
- Commit nhỏ, thường xuyên, rõ ràng: `rag: add BM25 index`, `agent: parse tool call safely`.
- **Cả ba phải có commit từ ngày 1.** Không dồn một commit cuối kỳ. Không squash toàn bộ lịch sử.
- Không bao giờ commit `.env`. Trước khi nộp: quét repo và `git log -p` tìm khóa API.
- Ghi commit hash tương ứng từng phần vào mục 8 của report (GV đối chiếu khi vấn đáp).

---

## 15. Checklist nộp bài

- [ ] `report.docx` 8–9 trang, đủ 9 mục (có ≥1 ca thất bại; ≥2 hạn chế trung thực; commit hash)
- [ ] Repo kèm `.git`, lịch sử thể hiện cả 3 người
- [ ] `logs/`: ≥3 file `.jsonl` chạy thật, có timestamp, bước, args/kết quả tool, USAGE từng câu
- [ ] `tests/testset.jsonl`: ≥12 câu (kế hoạch 16), ≥4 `no_answer`
- [ ] `ai-usage.md`: công cụ nào, phần nào, prompt chính, đoạn code do AI sinh
- [ ] `README.md`: `make setup && make demo`, `make eval`; **chạy được trên máy sạch**
- [ ] `.env.example` đầy đủ; **không lộ key**
- [ ] Dữ liệu có sẵn trong repo (không phụ thuộc tải từ web)
- [ ] Mọi số trong report tái lập được bằng `make eval`

## 16. Yêu cầu kỹ thuật bắt buộc (tick khi xong)

- [ ] Python 3.11+, chạy bằng đúng 1 lệnh
- [ ] Vòng agent **viết tay** (chỉ dùng: mcp SDK, thư viện embedding, `rank_bm25`, numpy, pandas, python-docx; thêm `matplotlib` đã được GV đồng ý; `openai` và `python-dotenv` ngoài danh sách nên ghi trong `ai-usage.md`/README và xác nhận với GV)
- [ ] Tool schema bằng dataclass/TypedDict có type hint
- [ ] `call_llm` có retry + timeout + log lỗi HTTP; prompt tách khỏi code
- [ ] ≥2 chế độ truy vấn (dense + hybrid); chunking có lý do
- [ ] `MAX_STEPS`; tool lỗi trả `"Error: ..."`; parse action an toàn
- [ ] ≥1 tool ghi + bước verify
- [ ] MCP: tự viết, ≥2 tool (≥1 ghi), stdio, stdout sạch, client discover động

---

## 17. Bảng điểm (để biết ưu tiên)

| Khối | Tiêu chí | Điểm |
|---|---|---|
| Nhóm (60) | Phạm vi & tính thực tiễn | 5 |
| | Chất lượng RAG | 13 |
| | Agent loop & thiết kế tool | 12 |
| | Tái lập & kỹ thuật (1 lệnh, README, không lộ key, git cả nhóm, đúng checkpoint) | 12 |
| | MCP server | 8 |
| | Đánh giá & bằng chứng | 10 |
| Cá nhân (40) | Bảo vệ phần mình | 15 |
| | Trả lời câu hỏi kỹ thuật | 15 |
| | Sửa code tại chỗ | 10 |
| Cộng | Việc làm thêm | tối đa +6 |

**Không giải thích được code mình nộp, hoặc không khai báo AI → điểm cá nhân = 0.**

---

## 18. Buổi bảo vệ (20 phút/nhóm)

1. **Cold start:** chạy lệnh trong README trên máy GV.
2. **Demo:** mỗi người 3 phút, trả lời **1 câu GV đưa tại chỗ** (không có trong testset). Chuẩn bị sẵn danh sách chủ đề của kho dữ liệu để GV hỏi sát.
3. **Kiến trúc:** không đọc slide; sơ đồ in sẵn; mỗi người ~1.5 phút.
4. **Vấn đáp riêng:** 1 câu chính + 1 câu đào sâu mỗi người.
5. **Sửa code trực tiếp:** 1 yêu cầu mỗi người → biết file/dòng mình viết, quen IDE.

---

## 19. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Giới hạn tốc độ của LLM khi chạy eval nhiều lần | Chạy thử sớm, ước lượng token, giảm số câu khi thử |
| Embedding tải nặng làm cold start chậm | Ghi rõ trong README; commit sẵn vector tài liệu; chọn mô hình nhỏ |
| Dữ liệu quá dễ → số liệu đẹp giả | Cài bẫy; chạy đối chứng không-RAG; kiểm tra bằng thí nghiệm đổi số |
| `make eval` lần sau hưởng lợi từ lần trước | Luôn sao chép DB sạch trước mỗi lần chạy |
| Bonus làm hỏng lõi | Cổng cuối ngày 6; bonus #3, #4 làm sau lõi |
| Một người không hiểu phần mình | Đọc code chéo ngày 9–10; ai cũng bị hỏi phần của mình |
| Lộ key | `.env` trong `.gitignore`; quét trước khi nộp |
| Thành viên chậm làm kẹt cả nhóm | Hợp đồng chốt sớm; tool giả để làm song song; họp 10 phút mỗi ngày |

---

## 20. Đã chốt và còn mở

**Đã chốt:**
- [x] **Ngôn ngữ dữ liệu: tiếng Việt** (chính sách, hồ sơ, câu hỏi). Cần embedding hỗ trợ tiếng Việt.
- [x] **Dữ liệu đầu vào đã viết xong** (mục 7): 9 chính sách, 21 hồ sơ, đáp án chuẩn.
- [x] **Client LLM: SDK `openai`** (nhóm chọn; vẫn nên xác nhận với GV, xem dưới).
- [x] **`requirements.txt` đã có** và ghim phiên bản; có thêm `python-dotenv` (đọc `.env`, đề có nhắc `.env.example`).
- [x] **Test cá nhân** để trong `scratch/` (không commit). `tests/testset.jsonl` vẫn là sản phẩm nộp và **phải commit**.

**Còn mở:**
- [ ] **`ragas`: chưa dùng.** Có trong giáo trình/GV có dùng, nhưng danh sách thư viện được dùng ở đề (mục 2.2) không nhắc, và nó kéo `langchain`/`datasets` có thể xung đột phiên bản. Hỏi GV; nếu được phép, tách ra `requirements-eval.txt` riêng để không ảnh hưởng `make setup`. Mặc định: tự viết `eval/judge.py` (mục 11).
- [ ] **Mô hình LLM** dùng là gì (quyết định mức sai của agent và giới hạn eval)? Giáo trình dùng SDK `openai` (và nói Groq tương thích, mã giống hệt). **Xác nhận với GV** SDK `openai` có được dùng làm client gọi API không (danh sách thư viện của đề không nhắc đến client LLM).
- [ ] GV: 2 điểm cộng còn lại (+6 vs 4 mục) tính cho việc gì?
- [ ] **Mô hình LLM** dùng là gì (quyết định mức sai của agent và giới hạn eval)? Giáo trình dùng SDK `openai` (và nói Groq tương thích, mã giống hệt). **Xác nhận với GV** SDK `openai` có được dùng làm client gọi API không (danh sách thư viện của đề không nhắc đến client LLM).
- [ ] Bản `mcp`: v1 hay v2 (xem mục 10)? `requirements.txt` đang ghim `mcp==2.2.0` (bản máy nhóm đang cài); nếu giáo trình dùng `FastMCP` v1 thì đổi sang `mcp==1.30.0` và báo người làm MCP. Chạy thử import trước ngày 3.
- [ ] Còn giữ bài lab W3–W7 của mình không? Đó là điểm xuất phát nhanh nhất.
- [ ] Đã lấy `w4/ex2_1_hybrid_search.py` và DOC0001–0006 của lớp chưa? (xem file hybrid để biết GV dùng embedding nào)
- [ ] Chọn mô hình embedding local: thử 2 mô hình trên 10 câu rồi chọn.
- [ ] **Bảo vệ "không cần cấu hình":** embedding local là làm được, nhưng **LLM vẫn cần API key** (xem phương án bên dưới). Chốt trước ngày 8: chạy trên laptop của nhóm với `.env` có sẵn, hay máy GV với key nhập tại chỗ, và có chế độ replay dự phòng không.
- [ ] Máy GV có mạng lúc `make setup` không? Nếu không, kèm thư mục mô hình trong gói nộp (ngoài git) và chạy offline (`HF_HUB_OFFLINE=1`).

---

## 21. Trải nghiệm một lệnh cho giảng viên

**Mục tiêu:** GV gõ đúng một lệnh là tương tác được ngay.

```
make run          # hoặc, nếu máy không có make:  python run.py
```

Các bước (mỗi bước chạy lại vẫn an toàn): tạo venv và cài thư viện nếu thiếu → tải mô hình Jina nếu thiếu → kiểm tra vector (`model_id` khớp) → **sao chép DB gốc sang DB chạy** → xử lý API key → mở REPL.

**API key:** dùng `OPENAI_API_KEY` từ môi trường hoặc `.env` nếu có; nếu không, hỏi một lần bằng `getpass`, chỉ giữ trong bộ nhớ, không ghi ra đâu. Nếu bỏ qua thì chuyển sang **chế độ replay**, ghi rõ trên màn hình, chỉ trả lời các câu testset đã ghi sẵn, câu khác thì nói thẳng cần key.

**REPL (terminal, không thêm thư viện):** gõ câu hỏi tự do để agent chạy và in từng bước kèm nguồn. Lệnh phụ:

| Lệnh | Việc làm |
|---|---|
| `/apps` | liệt kê hồ sơ vay mẫu |
| `/policies` | liệt kê tài liệu chính sách, version, ngày hiệu lực |
| `/decision <APP-ID>` | đọc trạng thái quyết định trực tiếp từ DB (GV tự kiểm chứng) |
| `/trace` | chi tiết lượt vừa chạy, kể cả USAGE |
| `/mode dense\|hybrid` | đổi chế độ truy vấn |
| `/reset` | khôi phục DB ban đầu |
| `/help`, `/quit` | trợ giúp, thoát |

**Độ bền trên máy lạ:** lỗi in ngắn kèm cách sửa, không in traceback dài; spawn MCP server bằng `sys.executable` + đường dẫn tuyệt đối; mỗi phiên ghi log `.jsonl` vào `logs/`.

**Việc phải thử:** chạy `make run` trên một máy sạch (hoặc thư mục + venv mới) trước ngày nộp.
