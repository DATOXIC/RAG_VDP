# BẢN NHÁP REPORT (chép sang report.docx)

> Cách dùng: mỗi mục bên dưới đúng theo khung **mục 4 của `02-Yeu-cau-Thuc-hien.pdf`** (9 mục, 8–9 trang). Số trang ghi trong ngoặc là ngân sách của đề. Chỗ `[HÌNH]`, `[BẢNG]`, `[ĐIỀN]` là việc còn lại của bạn.
> Mọi con số lấy từ repo (đã đối chiếu). Đừng thêm số nào không có trong `docs/ket-qua-danh-gia.md` hoặc `eval/out/`.
> Bìa của report.docx hiện đã có; nhớ thêm **tên nhóm, họ tên, MSSV** vào bìa.

---

## 1. Vấn đề, phạm vi, giả định (0,5–1 trang)

**Bài toán.** FinCo là một công ty tài chính **hư cấu**. Chuyên viên thẩm định phải xét từng hồ sơ vay tiêu dùng: tra đúng phiên bản chính sách theo ngày nộp, tra lãi suất theo kỳ hạn, tính tỷ lệ nợ trên thu nhập (DTI), ra quyết định (`approve`, `reject`, `need_info`, `escalate`) và ghi lại. Hệ thống của nhóm là một agent làm thay chuỗi việc đó và trả lời kèm nguồn.

**Người dùng.** Chuyên viên thẩm định (xét hồ sơ, hỏi chính sách) và người kiểm toán (đọc lại quyết định, nhật ký phiên `FINCO-xxxx`).

**Vì sao đủ 3 tầng của đề tài** (chép bảng này, ngắn):

| Tầng | Trong dự án |
|---|---|
| (a) Tra cứu | Mọi quyết định trích `doc_id` + phiên bản; câu không có trong tài liệu (ví dụ vay mua ô tô) phải bị từ chối |
| (b) Biến đổi | Tính DTI bằng code; chọn v1 hay v2 theo ngày nộp; ghép nhiều tài liệu; thiếu sao kê thì `need_info` |
| (c) Hành động | `submit_decision` ghi vào SQLite, ngay sau đó `get_decision` đọc lại để chứng minh |

**Ví dụ minh họa tầng (b):** cùng một hồ sơ DTI 47%. Nộp 20/3/2025 áp CP-01 v1 (tối đa 50%) thì duyệt; nộp 10/5/2025 áp v2 (tối đa 45%) thì từ chối. Đây là chỗ hệ thống có thể sai, nên chấm được.

**Hệ thống KHÔNG làm:** tư vấn vay cho người thật; dùng dữ liệu thật; tra điểm tín dụng thật; thay con người quyết định ca lớn (chỉ `escalate`).

**Giả định:** toàn bộ 18 tài liệu chính sách (17 mã CP-01..CP-17) và 21 hồ sơ do nhóm tự viết; con số riêng của FinCo nên LLM không thể biết trước. Đơn vị tiền là đồng; một hồ sơ chỉ có một quyết định.

> Lưu ý: đề tài là **tự đề xuất**, GV đã xác nhận (PLAN.md mục 1). Hãy nói một câu về điều này.

---

## 2. Kiến trúc tổng thể (1 trang)

**[HÌNH 1 – sơ đồ kiến trúc]** vẽ từ PLAN.md mục 4 (dùng draw.io/PowerPoint). Phải có: `agent.py` (vòng lặp), `llm.py`, `prompts/*.txt`, **ranh giới MCP** (đường nét đứt giữa `mcp_client.py` và `mcp_server.py`, nhãn "stdio"), 5 tool, SQLite, chỉ mục vector, verifier (chỉ dùng khi chấm).

```
make demo / make eval
   │
   ▼
 agent.py (vòng lặp tự viết, MAX_STEPS)
   │  llm.py (retry, timeout, usage) ◄── prompts/*.txt
   ▼
 mcp_client.py ── list_tools() ĐỘNG ──► mcp_server.py (stdio)
                                         ├─ search_policy   [đọc, RAG]
                                         ├─ get_application [đọc]
                                         ├─ calc_dti        [đọc, tính bằng code]
                                         ├─ submit_decision [GHI] ◄── critic (tùy chọn)
                                         └─ get_decision    [đọc → VERIFY]
```

**Ba nguyên tắc kiến trúc** (viết thành 3 gạch đầu dòng):
1. `tools.py` là hàm thuần, không biết MCP; `mcp_server.py` chỉ bọc lại. Agent **không import** `tools.py`, chỉ gọi qua MCP client.
2. LLM không tự nhẩm số: mọi phép tính qua `calc_dti`.
3. Mọi quyết định phải trích `doc_id`; thiếu thì tool trả `Error`.

**Luồng xử lý một câu hỏi mẫu (APP-014, nộp 20/3/2025).** Dùng trace thật ở mục 4 (đừng viết lại hai lần; ở đây chỉ nêu 1–2 câu và dẫn "xem mục 4").

---

## 3. Thiết kế RAG (1,5 trang)

**Chunking và lý do** (đề đòi 3–5 dòng giải thích; chép nguyên văn ý này):
- Cắt theo mục `##` của Markdown, tối đa 430 token (`tiktoken`, `cl100k_base`), chia nhỏ thêm theo điều khoản `1.1.`, `3.4.` với mục dài. Mỗi chunk có dòng đầu gồm tiêu đề tài liệu + phiên bản.
- Lý do: chính sách viết theo điều khoản; một điều khoản là một đơn vị ý nghĩa nên không bị cắt giữa con số và điều kiện của nó. Dòng đầu chứa phiên bản để hai bản CP-01 gần giống chữ vẫn phân biệt được.
- Kết quả: 18 file, 88 mục, **112 chunk** (lõi CP-01..07: 35; nhiễu CP-08..14: 33; tài liệu dài CP-15..17: 44).

**Embedding.** `BAAI/bge-m3` chạy cục bộ (đa ngôn ngữ, hợp tiếng Việt, không cần key). Vector tính sẵn, lưu kèm `model_id` và `corpus_hash`; lệch thì báo lỗi rõ.

**Hai chế độ truy vấn.**
- `dense`: cosine bằng `numpy` trên vector bge-m3.
- `hybrid`: 0,3 BM25 + 0,7 dense, **chuẩn hóa min–max trước khi trộn** (tái dùng `w4/ex2_1_hybrid_search.py`, cải tiến ở lọc ngày và gom theo tài liệu).
- Chế độ **không** phải tham số của tool (mỗi tham số lộ ra là một lời hứa model phải đặt đúng); đặt bằng `RETRIEVAL_MODE`, mặc định `hybrid`. Hãy nói rõ "dùng truy vấn nào" như đề yêu cầu.

**Lọc metadata theo ngày hiệu lực (bonus +1).** Lọc **trước** khi chấm điểm: chỉ giữ chunk có `effective_date <= as_of_date` và chưa hết hạn. `as_of_date=2025-03-31` trả CP-01 v1, `2025-04-01` trả CP-01 v2.

**Số liệu truy xuất (bonus +1; `make retrieval-eval`, 22 truy vấn, 20 truy vấn có đáp án vàng):**

| Chế độ | Có tài liệu đúng trong top-3 | Đúng ở hạng 1 | Lấy nhầm phiên bản |
|---|---|---|---|
| BM25 | 12/20 (60%) | 6/20 | 0 |
| Dense | 15/20 (75%) | 9/20 | 0 |
| **Hybrid** | **18/20 (90%)** | **10/20** | 0 |
| Hybrid, **tắt** lọc ngày | 18/20 | 10/20 | **2** |

Đọc: hybrid hơn dense 3 truy vấn trong top-3; tắt lọc ngày làm lấy nhầm phiên bản 2 lần. Đây là bằng chứng cho hai bonus cùng lúc.

**Hạn chế đã biết (dùng lại ở mục 7):** truy vấn kéo thêm chunk nhiễu (CP-09 thẻ tín dụng) nên Context Precision của RAGAS thấp ở S01 (0,00) và C01 (0,33).

---

## 4. Thiết kế Agent & tool (2 trang)

**4.1 Schema.** Mọi tham số và kết quả là `dataclass` có type hint trong `src/finco/types.py` (`Application`, `Decision`, `Document`, `Chunk`, `SearchHit`, `Usage`, `LogLine`, `TestCase`). Không dùng dict thô. Chứng minh: `grep -n "@dataclass" src/finco/types.py`.

**4.2 Năm tool, mô tả viết như prompt** (bảng; mỗi mô tả nói *khi nào dùng, khi nào KHÔNG, tham số lấy từ đâu*):

| Tool | Loại | Tham số | Ràng buộc |
|---|---|---|---|
| `search_policy` | đọc | `query`, `as_of_date`, `top_k` ≤ 5 | trả `[doc_id vX · ngày] nội dung`, cắt theo ngân sách token |
| `get_application` | đọc | `app_id` | |
| `calc_dti` | đọc | `app_id`, `annual_rate` | **bắt buộc** nhận lãi suất → agent phải tự tra CP-03 |
| `submit_decision` | **GHI** | `app_id`, `decision`, `reason`, `cited_docs` | `Error` nếu thiếu trích dẫn, sai mã hồ sơ, hoặc đã có quyết định |
| `get_decision` | đọc (**verify**) | `app_id` | `Error` nếu chưa có |

**4.3 Vòng lặp** (`src/finco/agent.py`, viết tay): gọi LLM, có `tool_calls` thì chạy từng tool qua MCP rồi đưa kết quả lại, hết `tool_calls` thì dừng.
- **`MAX_STEPS`** mặc định 15 (`FINCO_MAX_STEPS`; quy trình dài nhất ~9 lần gọi). Hết bước thì trả `success=False` gọn, không treo.
- **Xử lý lỗi:** tool lỗi trả chuỗi `"Error: ..."`, **không raise**; JSON tham số hỏng, không phải đối tượng, tên tool lạ → trả `Error` để model tự sửa. LLM lỗi hoặc trả rỗng thì lượt thất bại gọn, trace và USAGE vẫn được ghi. Mỗi kiểu có test trong `tests/test_agent.py` (LLM giả).
- **Function calling gốc + trường `thought`:** `mcp_client.to_openai_tools` tiêm tham số bắt buộc `thought` vào schema mỗi tool để có "suy nghĩ" cho audit; agent tách nó ra trước khi gọi tool. *(Nếu giám khảo hỏi "sao không ReAct văn bản": đề chỉ yêu cầu vòng lặp tự viết và parse an toàn; parse an toàn nằm ở biên tool.)*
- **Nudge:** nếu model dừng mà chưa ghi quyết định thành công, agent nhắc gọi `submit_decision` (chỉ nói "chưa ghi", không nói ghi gì).
- **Không `eval()`** bất cứ thứ gì; không có `TOOL_REGISTRY` viết tay (danh sách tool đến từ `list_tools()`).
- **Prompt tách khỏi code:** `prompts/agent_system.txt`, `agent_underwrite.txt`, `agent_qa.txt`.
- **Gọi API:** `llm.py` dùng SDK `openai`, retry + backoff + jitter cho `RateLimit`/`Connection`/5xx, **không** retry `Authentication`/`BadRequest`; timeout đặt tường minh; log lỗi HTTP ra stderr.

**4.4 Side effect + verify (đề 2.7).** `submit_decision` ghi vào bảng `decisions` (bắt đầu rỗng). Sau khi ghi thành công, **code tự gọi** `get_decision` (cùng bước, một dòng log riêng) và đính kết quả vào phản hồi. Verifier (`eval/verifier.py`) kiểm tra cặp này trong trace, nên model không thể "tự diễn".

**4.5 Trace thật một câu hỏi** (lấy từ `logs/eval_20261014_202454.jsonl`, cấu hình A; đây là trace chuẩn để dán, rút gọn kết quả):

| Bước | Tool | Tham số chính | Kết quả (rút gọn) | Prompt token lũy kế |
|---|---|---|---|---|
| 1 | `get_application` | `APP-014` | Phan Thị Phượng, 36 tuổi, vay 180 triệu, 36 tháng, lương 20 triệu | 2.258 |
| 2 | `search_policy` | `as_of_date=2025-03-20`, điều kiện/DTI | `[CP-01 v1]` DTI tối đa **50%** | 4.788 |
| 3 | `search_policy` | chứng từ nhóm hưởng lương | `[CP-02 v1]` | 7.944 |
| 4 | `search_policy` | lãi suất theo kỳ hạn | `[CP-03 v1]` 25–36 tháng: **17%** | 11.790 |
| 5 | `search_policy` | phân cấp phê duyệt | `[CP-04 v1]` ≤200 triệu: chuyên viên | 16.152 |
| 6 | `search_policy` | ngoại lệ, nợ quá hạn | `[CP-06 v1]` | 21.430 |
| 7 | `calc_dti` | `annual_rate=17%` | PMT **6.417.491** đ, nợ cũ 3.000.000 đ; DTI = (6.417.491 + 3.000.000) / 20.000.000 = **47,09%** | 27.273 |
| 8 | `submit_decision` | `approve`, trích CP-01..05 v1 | "Đã ghi nhận quyết định" | 33.458 |
| 8 | `get_decision` (tự động) | `APP-014` | `APPROVE`, mã `APPROVED`, đã lưu lúc 2026-10-15T10:50:58 | 33.458 |
| — | câu trả lời cuối | | quyết định + lý do, mỗi ý kèm `[CP-xx v1]` | **40.467** (9 lần gọi LLM) |

Nhận xét: `as_of_date=2025-03-20` ở bước 2–6 làm hệ thống chọn đúng v1; cùng hồ sơ nếu nộp sau 1/4/2025 sẽ ra CP-01 v2 (DTI ≤ 45%). Bước 8 chứng minh ghi + đọc lại.

Nhận xét thêm: DTI 47,09% nằm giữa hai ngưỡng (v1 tối đa 50% nên **duyệt**; v2 tối đa 45% sẽ **từ chối**). Chính hồ sơ này minh họa tầng (b) ở mục 1. Số liệu trên lấy thẳng từ kết quả `calc_dti` trong log.

**4.6 Critic (thí nghiệm, 3–4 dòng).** Chốt chặn nhúng trong `submit_decision` ở phía server: Tầng 1 kiểm bằng chứng bằng code (0 token), Tầng 2 là LLM tự suy ra quyết định rồi code so sánh; chặn tối đa 2 lần. Hai chế độ `single` và `agent`. Kết quả ở mục 6.

---

## 5. MCP server (1 trang)

- **Server** `src/finco/mcp_server.py`, dùng `mcp` SDK (`MCPServer`, `@mcp.tool()`), **5 tool** (4 đọc + 1 ghi `submit_decision`); đề đòi ≥2 tool, ≥1 ghi.
- **Transport:** stdio. **stdout sạch:** log cấu hình `stream=sys.stderr`; không có `print()` trong tool (một `print` làm hỏng luồng JSON-RPC).
- **Client discover động:** `mcp_client.py` gọi `list_tools()` rồi biến kết quả thành định nghĩa tool cho LLM; **không hard-code tên tool**. (Ngoại lệ có chủ đích: `agent.py` biết tên cặp ghi/verify.)
- **Spawn server** bằng `sys.executable` + đường dẫn tuyệt đối (tránh lỗi "Connection closed during initialize" trên máy khác).
- **Hợp đồng tool:** bảng ở mục 4.2; mọi tool trả `Error: ...` thay vì raise; chế độ truy vấn không lộ ra làm tham số.
- **Host thứ hai (bonus +1):** `examples/client2.py` (client JSON-RPC thuần `initialize` → `tools/list` → `tools/call`, không dùng SDK), chạy `make interop` trên bản sao DB. Chứng minh interop: cùng server, hai client khác nhau.

**[HÌNH 2 – tùy chọn]** ảnh chụp `make interop` ra danh sách 5 tool.

---

## 6. Đánh giá (1,5 trang)

**6.1 Testset và cách đo.** 37 tình huống: 16 câu `tests/testset.jsonl` (2 tra cứu, 2 tổng hợp, 2 phiên bản, 2 tính toán, 2 thiếu dữ liệu, **4 `no_answer`**, 2 hành động) + 21 hồ sơ (`tests/expected_decisions.json`: 6 approve, 9 reject, 3 need_info, 3 escalate). Chấm **bằng code**: quyết định, mã lý do, trích dẫn đúng phiên bản (recall ≥ 0,8), DTI và lãi suất, cặp ghi + đọc lại; `no_answer` đạt khi từ chối bằng câu chuẩn và không ghi quyết định. Mỗi ca chạy trên DB sandbox sạch, temperature 0, một lần.

**6.2 Kết quả chính** (chép bảng 2.1 của `docs/ket-qua-danh-gia.md`, chỉ 3 cột A, H, C):

| Chỉ số | gpt-oss-120b (A) | gpt-oss-20b (H) | Qwen2.5-7B (C) |
|---|---|---|---|
| Đạt toàn diện | **34/37 (92%)** | 33/37 (89%) | 10/37 (27%) |
| Chạy xong không lỗi | 37/37 | 37/37 | 30/37 |
| Đúng quyết định (21 hồ sơ) | 21/21 | 20/21 | 6/21 |
| Lượt gọi LLM (cả 37 ca) | 198 | 209 | 353 |
| Prompt token (cả 37 ca) | 689.453 | 655.390 | 1.501.556 |
| Lỗi gọi tool | 0 | 0 | 16 |

**Chi phí token từng câu (đề đòi bắt buộc).** Ví dụ APP-014 (cấu hình A): 9 lần gọi LLM, 40.467 prompt token tích lũy, 2.698 completion token, `max_prompt` 7.009. Trung bình toàn bộ: 689.453 / 37 ≈ 18.600 prompt token mỗi ca. **Chèn thêm bảng chi phí 8–10 dòng từ `eval/out/ab_nocritic/per_question.csv`** (cột calls, prompt, completion). `[BẢNG]`

**6.3 Đồ thị + CSV (bonus +1).** Chèn `eval/out/compare/compare.png` (so sánh A, H, C; `make compare`) `[HÌNH 3]` và, nếu còn chỗ, `eval/out/ab_nocritic/summary.png` (từng câu của A) `[HÌNH 4]`; nói một câu: `make eval` tự sinh `per_question.csv`, `agent_eval_results.csv`, `summary.json`, `summary.png`.

**6.4 Ca thất bại + phân tích (bắt buộc ≥1).** Dùng **APP-018** (cấu hình A): quyết định `escalate` đúng nhưng mã lý do `ESCALATE_COUNCIL`, chuẩn là `ESCALATE_EMPLOYEE`. Nguyên nhân: model nhầm thẩm quyền của **nhân viên FinCo** (ngoại lệ CP-06) sang Hội đồng; lỗi ở **suy luận**, không phải truy xuất (đo riêng: recall@3 hybrid đạt 0,90 và tài liệu đúng đã nằm trong ngữ cảnh). Cùng lỗi ở A02. Câu **N03** trả lời bịa thay vì từ chối là lỗi calibration. Nhóm lỗi của Qwen 7B (bỏ qua tra cứu, ghi xong không đọc lại, chọn sai lãi suất theo kỳ hạn) thuộc tuân thủ quy trình.

**6.5 So sánh cấu hình + critic (ngắn).** Critic-agent Gemini trên 120b: 32/37 so với 34/37 của A, tốn thêm 61% prompt token (1.107.864 so với 689.453). Trên 20b (Groq): H 33, I 32, J 32. Chênh lệch nằm trong nhiễu của 37 mẫu (±13 điểm phần trăm); kết luận trung thực là **chưa thấy cải thiện**, và bằng chứng mới phủ các Actor ít lỗi. Mặc định của code là critic **bật** (quyết định của nhóm) nhưng số liệu chính (A) chạy không critic; ghi rõ điều này, đừng để người đọc hiểu nhầm.

**6.6 RAGAS (bổ sung, 2–3 dòng).** Trên 12 câu có đáp án: Faithfulness 0,94, Answer Relevancy 0,81, Answer Correctness 0,60, Context Precision 0,80, Context Recall 1,00; giám khảo `gemini-3.5-flash-lite`, một lần chạy. Chỉ là số liệu bổ sung vì giám khảo nhiễu (V02 từ 0,0 lên 1,00 giữa hai lần chạy).

**6.7 Tái lập.** Mọi số trong mục này chạy lại được bằng `make eval-score LOG=logs/<file>.jsonl` mà không cần khóa API (nêu 3 log chính: A, H, B, xem `logs/README.md`).

---

## 7. Hạn chế & hướng phát triển (0,5 trang)

Đề yêu cầu ≥2 hạn chế **trung thực**, không viết "chưa có thời gian". Chọn 3–4 trong số này (đều có bằng chứng trong repo):

1. **Một lần chạy mỗi cấu hình, 37 mẫu:** sai số ±13 điểm phần trăm nên chênh 34 so với 32 không có ý nghĩa thống kê. Hướng: chạy lặp nhiều lần để đo độ biến thiên.
2. **Critic chưa chứng minh được lợi ích:** thử trên Actor ít lỗi nên không có dư địa; chưa thử critic mạnh trên Actor sai nhiều quyết định (gpt-oss:20b chạy Ollama, đúng 17/21). Hướng: thí nghiệm đó.
3. **Lỗi mã lý do nhánh nhân viên FinCo** (APP-018, A02) và 1 ca calibration (N03) chưa sửa gốc. Hướng: quy tắc rõ trong prompt rồi đo lại.
4. **Truy xuất còn nhiễu:** top-3 hybrid chỉ 90%, Context Precision thấp khi truy vấn kéo chunk nhiễu; chưa có re-ranking (CrossEncoder `ms-marco` huấn luyện cho tiếng Anh).
5. **Nền tảng khác nhau làm khó so sánh:** cùng gpt-oss-20b nhưng Ollama đạt 23/37, Groq đạt 33/37; giới hạn 8.000 token/phút của gói miễn phí gây HTTP 413 ở ca dài (APP-011, APP-018) làm lệch số liệu.
6. **Chưa có đối chứng không-RAG** và thí nghiệm "đổi một con số trong tài liệu thì đáp án phải đổi theo" (đã lên kế hoạch ở PLAN.md mục 11, chưa chạy). Nên chạy nếu còn thời gian, hoặc nói thẳng là chưa.

---

## 8. Phân công & đóng góp (0,5 trang)

Đề yêu cầu "ai làm gì + commit hash tương ứng" để GV đối chiếu. Dưới đây **tôi suy ra từ `git log`**, bạn phải **tự xác nhận với từng bạn** trước khi nộp (lưu ý: git có 3 tên tác giả; "TÔ PHƯƠNG ĐẠT" là một tài khoản gộp PR, 3 commit).

| Thành viên (tên git) | Phần phụ trách | Commit tiêu biểu |
|---|---|---|
| `[ĐIỀN tên]` (**DATOXIC**, 63 commit) | Dữ liệu giả lập (`data/`, 21 hồ sơ, đáp án chuẩn), loader + chunking + embedding bge-m3, `filter.py`, 5 tool trong `tools.py`, `verifier.py`, `judge.py`, `run_eval.py`, MCP server/client đầu tiên, `client2.py` + Makefile, RAGAS | `4faf405` hồ sơ vay; `848b2b5` dữ liệu văn bản; `dc6cf11` chunking; `d9b96a8` VectorStore + bge-m3; `1ba0773` search_policy; `0e368c5` submit_decision; `4f5e85e` verifier; `e6d8a4e` MCP server + client; `610553e` client2 + Makefile; `354c767` RAGAS |
| `[ĐIỀN tên]` (**nphi**, 24 commit) | BM25 + hybrid + retriever, `retrieve_eval`, `build_index.py`, viết lại chunker và `filter.py`, `llm.py` (retry/timeout), `system_prompt`, demo | `1d6adf8` BM25; `093537b` hybrid + test; `6603a4a` retrieve_eval; `f465fc8` build_index; `84181fc` rewrite chunker; `15870d4` create llm.py; `31456cc` system_prompt; `9d8537b` demo |
| `[ĐIỀN tên]` (**VuongVuiVe123**, 46 commit) | Class `Agent` và vòng lặp, trường `thought` trong schema, RateLimit, `chat.py` (REPL), `audit.py`, critic + critic-agent, thí nghiệm đa model, tài liệu critic | `13a56d6` thêm `thought`; `73320c9` Class Agent; `b6d6919` bắt RateLimitError; `074dad2` chat.py; `b427fef` audit.py; `5e55fc0` critic.py; `6fc2bc4` multi-agent critic; `d9b3d49` evaluate; `0c309af` document critic |

> ⚠ Hai việc cần làm trước khi nộp mục này:
> 1. Hỏi từng bạn xác nhận phần của mình và sửa lại bảng cho đúng (git chỉ cho biết ai commit, không cho biết ai thiết kế). PLAN.md ghi vai A/B/C nhưng thực tế phân bổ đã lệch (ví dụ `llm.py` do nphi viết, không phải người làm agent).
> 2. Đề nói "git của cả 3 người" nhưng một file đề ghi "nhóm 2 SV" và một file ghi "3 SV". Xác nhận số thành viên thật với thầy rồi sửa bảng cho khớp.

---

## 9. Phụ lục

- **A. Prompt gốc:** `prompts/agent_system.txt`, `agent_underwrite.txt`, `agent_qa.txt`, `critic.txt`, `critic_agent.txt` (dán nguyên văn; đây là việc cơ học).
- **B. Trace dài:** trace đầy đủ APP-014 (bước 1–8 với kết quả đủ) và, nếu muốn, trace APP-015 (cùng số liệu, nộp sau 1/4/2025 nên chọn v2 và từ chối). APP-014/015 là cặp trong `make demo`.
- **C. `ai-usage.md`:** bạn tự viết; dán vào phụ lục. Đề đòi: công cụ nào, cho phần nào, prompt chính, đoạn code nào do AI sinh. Không có file này hoặc không giải thích được code thì điểm cá nhân = 0.
- **D. (tùy chọn)** bảng cấu hình A–J (từ `docs/ket-qua-danh-gia.md` mục 1) nếu còn trang.

---

## Danh sách việc còn lại cho bạn (theo thứ tự)

1. Vẽ **[HÌNH 1]** sơ đồ kiến trúc (cần in sẵn cả cho buổi bảo vệ: "sơ đồ in sẵn").
2. Chèn **[HÌNH 3]** `summary.png` và bảng chi phí token từng câu **[BẢNG]** từ `per_question.csv`.
3. Điền tên thành viên và xác nhận mục 8.
4. Viết `ai-usage.md` rồi dán vào phụ lục.
5. Chép nội dung sang `report.docx`, giữ 8–9 trang: mục 4 (2 trang) và mục 6 (1,5 trang) là chỗ dễ vượt; nếu dư, cắt 6.5, 6.6 và bớt dòng trong 4.3.
6. Tự tính tay APP-014 một lần (PMT 6.417.491 + nợ cũ 3.000.000, chia 20.000.000 = 47,09%) để trả lời được khi vấn đáp.
7. Lấy lại tin nhắn xác nhận của thầy về đề tự đề xuất và `matplotlib`, `openai` SDK (PLAN.md mục 20 còn mở).
