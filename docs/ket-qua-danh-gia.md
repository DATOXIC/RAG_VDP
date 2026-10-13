# Kết quả đánh giá (đoạn dùng cho báo cáo)

## 1. Thiết lập đánh giá

Hệ thống được đánh giá trên 37 tình huống: 16 câu trong `tests/testset.jsonl` (12 câu hỏi đáp chính sách gồm tra cứu, tổng hợp, phiên bản, tính toán, thiếu dữ liệu, cộng 4 câu `no_answer` và 2 câu hành động) và 21 hồ sơ vay trong `tests/expected_decisions.json`. Mỗi tình huống chạy trên một cơ sở dữ liệu sandbox sạch, không dùng bộ nhớ đệm, nhiệt độ 0, chạy một lần.

Việc chấm điểm do code thực hiện, không dùng LLM:
- **Hồ sơ và câu hành động:** `eval/verifier.py` dựng bài nộp từ trace các lời gọi tool rồi so với đáp án chuẩn ở năm điểm: quyết định, mã lý do (tập đóng), trích dẫn (đúng phiên bản, recall ≥ 0,8), DTI và lãi suất, cặp ghi + đọc lại.
- **Câu `no_answer`:** đạt khi agent từ chối bằng câu chuẩn và không ghi quyết định.
- **Câu hỏi đáp có đáp án:** đạt khi có đủ các ý chính đánh dấu trong đáp án chuẩn, trích đúng tài liệu và không từ chối nhầm.

Hai model được dùng, cùng prompt, cùng RAG, cùng bộ công cụ MCP:

| Model | Vai trò | Nơi chạy |
|---|---|---|
| `openai/gpt-oss-20b` | model chính của sản phẩm | Groq API (gói miễn phí) |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | model đối chứng, nhỏ hơn và tự lưu trữ | vLLM trên Colab T4, `max-model-len 8192` |

## 2. Kết quả

**Qwen2.5-14B-Instruct-AWQ, đủ 37 tình huống** (`logs/eval_qwen2.5-14b_full.jsonl`):

| Chỉ số | Kết quả |
|---|---|
| Đạt toàn diện | 21,6% (8/37) |
| Lượt chạy hoàn tất không lỗi | 81,1% |
| Đúng quyết định, đúng mã lý do (23 hồ sơ và câu hành động) | 43,5% |
| Có ghi + đọc lại + tra cứu | 69,6% |
| Đúng DTI và lãi suất | 60,9% |
| Từ chối đúng (câu `no_answer`) | 25,0% |
| Từ chối nhầm (câu có đáp án) | 0,0% |
| Faithfulness theo luật | 100% |
| Tỷ lệ lời gọi tool lỗi | 18,9% |
| Trung bình lời gọi tool, lượt gọi LLM mỗi câu | 6,2 và 6,7 |

**So sánh trên 16 câu testset** (hồ sơ chưa có số liệu cho `gpt-oss`, xem mục 4):

| Loại câu | gpt-oss-20b | Qwen2.5-14B |
|---|---|---|
| Tra cứu (2) | 2 | 1 |
| Tổng hợp (2) | 1 | 0 |
| Phiên bản chính sách (2) | 2 | 1 |
| Tính toán (2) | 0 | 0 |
| Thiếu dữ liệu (2) | 2 | 1 |
| `no_answer` (4) | 4 | 1 |
| Hành động (2) | 0 | 0 |
| **Tổng (16)** | **11** | **4** |

Faithfulness theo luật bằng 100% ở cả hai model: không câu nào trích tài liệu chưa từng xuất hiện trong kết quả công cụ. Cả hai model không từ chối nhầm câu nào có đáp án.

## 3. Phân tích lỗi

Lỗi của model nhỏ tập trung ở **tuân thủ quy trình**, ít ở kiến thức chính sách:

| Nhóm lỗi | Biểu hiện (Qwen) | Số ca |
|---|---|---|
| Trả lời không tra cứu | 0 lời gọi tool, trả lời từ trí nhớ | 3 câu (T02, N02, N04) |
| Ghi xong không đọc lại | có `submit_decision` nhưng không gọi `get_decision` | 3 hồ sơ (APP-007, 010, 019) |
| Hết bước hoặc xoay vòng tra cứu | vượt `MAX_STEPS`, hoặc prompt vượt giới hạn ngữ cảnh | 2 hồ sơ (APP-018, 011) |
| Duyệt nhầm hồ sơ thiếu thông tin | `approve` khi thiếu chứng từ hoặc chưa khai báo nợ | APP-004, 005 |
| Không nhận ra sản phẩm khác | dùng biểu lãi suất vay tiêu dùng để trả lời câu hỏi về vay thế chấp | N01, N02, N04 |
| Thiếu trích dẫn trong câu trả lời | nội dung đúng nhưng không nêu tài liệu | nhiều câu hỏi đáp |

Với `gpt-oss-20b`, hai nhóm lỗi còn lại ở testset là thiếu trích dẫn (S02, C01, C02) và một ca hành động sai (A01).

Tầng truy xuất không phải nguyên nhân chính: đo riêng, recall@3 đạt 1,00 và gold@3 đạt 0,90 sau khi giảm điểm tài liệu diễn giải (CP-15, CP-16).

## 4. Hạn chế

- **Chưa có kết quả hồ sơ cho `gpt-oss-20b`.** Lần chạy 21 hồ sơ bị giới hạn token của gói miễn phí (HTTP 429), nên chúng không phản ánh năng lực model; số liệu hồ sơ trong báo cáo hiện chỉ của Qwen. Cần chạy lại khi có hạn mức.
- **Một lần chạy mỗi model** nên chưa đo được độ biến thiên; với 37 mẫu, chênh lệch nhỏ giữa các nhóm chưa có ý nghĩa thống kê.
- **Hai model chạy trên hai nền tảng khác nhau** (Groq và Colab T4), nên so sánh chỉ mang tính định hướng.
- Bộ chấm yêu cầu trích dẫn xuất hiện trong văn bản trả lời của câu hỏi đáp; một câu trả lời đúng nhưng không nêu mã tài liệu bị tính là trượt.

## 5. Kết luận và hướng tiếp theo

Hệ thống vận hành đúng từ đầu đến cuối trên model thật (MCP, tra cứu, ghi và đọc lại, log có USAGE, chấm điểm tự động). Chất lượng phụ thuộc mạnh vào model, và phần lớn lỗi của model nhỏ là những bước mà code có thể ép buộc. Đây là cơ sở cho giai đoạn đa agent: để code thực hiện bước ghi và đọc lại, và thêm một agent phản biện kiểm tra trích dẫn trước khi ghi, rồi đo A/B bằng chính bộ đánh giá này.
