# Kết quả đánh giá (đoạn dùng cho báo cáo)

## 1. Thiết lập đánh giá

Hệ thống được đánh giá trên 37 tình huống: 16 câu trong `tests/testset.jsonl` (12 câu hỏi đáp chính sách gồm tra cứu, tổng hợp, phiên bản, tính toán, thiếu dữ liệu, cộng 4 câu `no_answer` và 2 câu hành động) và 21 hồ sơ vay trong `tests/expected_decisions.json`. Mỗi tình huống chạy trên một cơ sở dữ liệu sandbox sạch, không dùng bộ nhớ đệm, nhiệt độ 0, chạy một lần.

Việc chấm điểm do code thực hiện, không dùng LLM:
- **Hồ sơ và câu hành động:** `eval/verifier.py` dựng bài nộp từ trace các lời gọi tool rồi so với đáp án chuẩn ở năm điểm: quyết định, mã lý do (tập đóng), trích dẫn (đúng phiên bản, recall ≥ 0,8), DTI và lãi suất, cặp ghi + đọc lại.
- **Câu `no_answer`:** đạt khi agent từ chối bằng câu chuẩn và không ghi quyết định.
- **Câu hỏi đáp có đáp án:** đạt khi có đủ các ý chính đánh dấu trong đáp án chuẩn, trích đúng tài liệu và không từ chối nhầm.

Các model và cấu hình đã chạy, cùng prompt, cùng RAG, cùng bộ công cụ MCP:

| Cấu hình | Actor | Critic | Nơi chạy | Log |
|---|---|---|---|---|
| A | `openai/gpt-oss-120b` | không | Groq (gói miễn phí) | `logs/eval_20261014_202454.jsonl` |
| B | `openai/gpt-oss-120b` | critic-agent `gemini-3.1-flash-lite` | Groq + Google AI Studio | `logs/eval_agent_full_merged.jsonl` |
| C | `Qwen/Qwen2.5-7B-Instruct-AWQ` | không | vLLM trên Colab T4, `max-model-len 8192` | `logs/eval_20261015_133814.jsonl` |
| D | `Qwen/Qwen2.5-7B-Instruct-AWQ` | critic một lần gọi (chính nó), bản đã sửa | như C | `logs/eval_local_critic_v2_merged.jsonl` |
| E | `Qwen/Qwen2.5-14B-Instruct-AWQ` | không | như C | `logs/eval_qwen2.5-14b_full.jsonl` |
| G | `gpt-oss:20b` | không | Ollama (máy riêng của đồng đội, 16/10) | `logs/eval_20261016_011437.jsonl` |
| H | `openai/gpt-oss-20b` | không | Groq | `logs/eval_20261016_0215{34,39,44,49}.jsonl` (4 shard, đã `--resume`) |
| I | `openai/gpt-oss-20b` | critic một lần gọi `gemini-3.1-flash-lite` | Groq + Google AI Studio | `logs/eval_20261016_0148{56,01,06,11}.jsonl` (lẫn với J) |
| J | `openai/gpt-oss-20b` | critic-agent `gemini-3.1-flash-lite` | như I | như I |

Critic là một chốt chặn nhúng trong tool `submit_decision` (xem `src/finco/critic.py`, `src/finco/critic_agent.py`): Tầng 1 kiểm tra bằng chứng bằng code (không tốn token); Tầng 2 là LLM tự suy ra quyết định đúng rồi code so sánh, và mọi phản đối phải trích dẫn nguyên văn từ bằng chứng thì mới được chặn. Critic-agent là bản nâng cấp: một agent thứ hai có vòng lặp riêng và ba tool chỉ-đọc (`get_application`, `search_policy`, `calc_dti`), tự tra cứu độc lập chứ không đọc bằng chứng của Actor, và bị ép tự tính DTI trước khi kết luận nếu quyết định phụ thuộc DTI. Sau 2 lần bị chặn, Actor chỉ còn được ghi `escalate`.

## 2. Kết quả

### 2.1. Ảnh hưởng của model (không critic)

| Chỉ số | gpt-oss-120b (A) | gpt-oss-20b (H) | Qwen2.5-7B (C) |
|---|---|---|---|
| Đạt toàn diện | **34/37 (92%)** | 33/37 (89%) | 10/37 (27%) |
| Chạy xong không lỗi | 37/37 | 37/37 | 30/37 |
| Đúng quyết định, 21 hồ sơ | **21/21** | 20/21 | 6/21 |
| Lượt gọi LLM (cả 37 ca) | 198 | 209 | 353 |
| Prompt token (cả 37 ca) | 689.453 | 655.390 | 1.501.556 |
| Lỗi gọi tool | 0 | 0 | 16 |

Qwen2.5-14B (E) đạt 8/37 (21,6%), trong đó từ chối đúng `no_answer` 25%, tỉ lệ lời gọi tool lỗi 18,9%. `gpt-oss-20b` trên Groq (H) đạt **33/37 (89,2%)**, đúng quyết định 20/21 hồ sơ (sai APP-018), chỉ kém 120b một ca (mục 2.3).

Theo loại câu (số câu đạt):

| Loại câu | gpt-oss-120b | gpt-oss-20b (H) | Qwen2.5-7B |
|---|---|---|---|
| Tra cứu (2) | 2 | 2 | 0 |
| Tổng hợp (2) | 2 | 2 | 0 |
| Phiên bản chính sách (2) | 2 | 2 | 1 |
| Tính toán (2) | 2 | 0 | 0 |
| Thiếu dữ liệu (2) | 2 | 2 | 2 |
| `no_answer` (4) | 3 | 4 | 4 |
| Hành động (2) | 1 | 1 | 0 |
| Hồ sơ (21) | 20 | 20 | 3 |

Cột H cho thấy gpt-oss-20b gần như ngang 120b (33 so với 34 ca); khác biệt nằm ở hai câu tính toán C01, C02 (0/2 so với 2/2), còn 20b lại từ chối đúng đủ 4/4 câu `no_answer`. Về quyết định hồ sơ, 120b không sai ca nào còn 20b sai một (APP-018), nên Actor 20b cũng có rất ít dư địa để critic cải thiện.

Kết luận về kích thước: chênh lệch giữa họ model là rất lớn (92% so với 22 đến 27%), nhưng **kích thước đơn thuần không giải thích được**: Qwen 14B không tốt hơn Qwen 7B (8 và 10 ca). Hai họ model còn khác nền tảng phục vụ, độ dài ngữ cảnh và cách gọi tool, và gpt-oss là MoE (chỉ khoảng 5B tham số kích hoạt mỗi token).

### 2.2. Thí nghiệm A/B critic

**Actor gpt-oss-120b: không critic (A) so với critic-agent Gemini (B).** Cả hai chạy đủ 37 ca.

| Chỉ số | Không critic (A) | Critic-agent (B) |
|---|---|---|
| Đạt toàn diện | **34/37** | 32/37 |
| Chạy xong không lỗi | 37/37 | 35/37 |
| Đúng quyết định, 21 hồ sơ | 21/21 | 21/21 |
| Lượt gọi LLM của Actor | 198 | 204 |
| Prompt token của Actor | 689.453 | 732.316 |
| Thêm cho critic | 0 | 171 lượt, 375.548 prompt token, 5.414 completion token |
| **Tổng prompt token** | 689.453 | **1.107.864 (+61%)** |
| Lỗi gọi tool (gồm các lần bị chặn) | 0 | 5 |

Bốn ca khác nhau giữa hai nhánh:

| Ca | A | B | Ghi chú |
|---|---|---|---|
| A02 | trượt | đạt | không tách được tác động của critic khỏi nhiễu |
| APP-011 | đạt | trượt | lỗi hạ tầng Groq HTTP 413 (request vượt 8.000 token) |
| APP-019 | đạt | trượt | thiếu trích dẫn, recall 0,75 so với ngưỡng 0,8 |
| C02 | đạt | trượt | câu hỏi đáp, **không đi qua critic**: nhiễu thuần túy |

Hoạt động của critic-agent trong 32 lần thẩm tra:

| Kết quả | Số lần | Ghi chú |
|---|---|---|
| Đồng ý (pass) | 25 | |
| Chặn có trích dẫn | 3 | cả ba ở APP-018 |
| Lỗi: không trả JSON | 2 | APP-006, `gemini-3.1-flash-lite` |
| Lỗi: kết luận khi chưa tự tính DTI | 2 | APP-011, dù đã bị nhắc |

**Actor Qwen2.5-7B: không critic (C), critic một lần gọi bản đầu, và bản đã sửa (D).**

| Chỉ số | Không critic (C) | Critic bản đầu | Critic bản sửa (D) |
|---|---|---|---|
| Đạt toàn diện | 10/37 | 8/37 | 9/37 |
| Đúng quyết định, 21 hồ sơ | 6 | 2 | 5 |
| Chạy xong không lỗi | 30 | 24 | 25 |
| Lỗi gọi tool | 16 | 41 | 35 |
| Lượt gọi LLM | 353 | 379 | 398 |

Ba chỉnh sửa (không chặn vì quên trích dẫn mà tự bổ sung, giới hạn nội dung phản hồi, và không phạt Actor khi critic chỉ lỗi hạ tầng) đưa critic từ gây hại rõ ràng về gần mức không critic, nhưng chưa vượt.

### 2.3. Actor yếu hơn: gpt-oss-20b (G, H, I, J)

Mục đích: kiểm tra critic trên một Actor yếu hơn 120b (120b đã đúng quyết định 21/21 nên critic không có chỗ để cải thiện). Lưu ý về dư địa: trên Groq, 20b (H) chỉ sai 1 quyết định hồ sơ (APP-018), nên dư địa để critic cải thiện cũng rất nhỏ; chỉ môi trường Ollama (G, đúng 17/21 hồ sơ) có dư địa đáng kể. Vì vậy kết quả I, J cho thấy critic không gây hại và không giúp trong điều kiện ít lỗi, chưa chứng minh critic vô dụng trên một Actor sai nhiều. Cả bốn cấu hình chạy đủ 37 ca.

| Cấu hình | Nơi chạy | Critic | Đạt toàn diện | Chạy xong | Prompt + completion token của Actor |
|---|---|---|---|---|---|
| G | Ollama | không | 23/37 (62,2%) | 31/37 | 930.617 |
| **H** | Groq | không | **33/37** | 37/37 | 704.410 |
| I | Groq | một lần gọi (Gemini) | 32/37 | 37/37 | 750.222 |
| J | Groq | critic-agent (Gemini) | 32/37 | 36/37 | 759.339 |

Đọc kết quả:

- **Critic không cải thiện.** So với H (cùng model, cùng nơi chạy, chỉ khác critic): I và J mỗi cấu hình hơn **0** ca và làm hỏng **1** ca (S01). S01 là câu hỏi đáp **không đi qua critic**, nên chênh lệch này là nhiễu chứ không phải tác động của critic. Chênh lệch tổng 33 so với 32 nằm trong sai số (±3 ca).
- **Thứ tạo khác biệt lớn là môi trường chạy.** Cùng một model 20b, G (Ollama) chỉ đạt 23/37, còn H (Groq) đạt 33/37, chênh 10 ca; G trượt 14 ca, H trượt 4 ca (A02, APP-018, C01, C02). Nguyên nhân chưa được tách rõ (lượng tử hóa, mẫu chat, ngữ cảnh), nên **không nên so G với I, J để kết luận về critic**: chênh lệch 23 so với 32 gần như hoàn toàn do nơi chạy. Đây chính là lý do phải có cấu hình H làm mốc đối chứng cùng nền tảng.
- Critic chỉ làm tăng nhẹ token của Actor (+6,5% cho một lần gọi, +7,8% cho critic-agent); chưa tính token của Gemini cho critic.
- Critic-agent trên 20b hoạt động đúng thiết kế (25 lần đồng ý, 3 lần chặn) nhưng ba lần chặn vẫn không biến ca sai thành đúng.
- Kết quả của H có hai ca đầu bị lỗi hạ tầng "hết hạn mức token/ngày" (APP-016, APP-019), đã chạy lại riêng hai ca đó bằng `--resume` trên key khác; các ca đã xong không bị chạy lại. Các key Groq dùng chung hạn mức token của tổ chức, nên thêm key không tăng thông lượng.
- Log của I và J **ghi chung vào các file** `eval_20261016_0148xx.jsonl` do hai lần chạy khởi động cùng giây và trùng tên file. Điểm số không bị ảnh hưởng (mỗi tiến trình tự chấm và ghi CSV riêng ở `eval/out/g20_single_w*`, `eval/out/g20_agent_w*`), nhưng `make eval-score` trên các file này sẽ lẫn hai cấu hình.

### 2.4. Đánh giá bổ sung bằng RAGAS

Ngoài chấm bằng code (mục 1), nhóm chạy thư viện RAGAS (`eval/ragas_eval.py`, `make ragas`) trên **log có sẵn của cấu hình A**, không chạy lại Actor. Giám khảo là `gemini-3.5-flash-lite`, embedding là `BAAI/bge-m3` cục bộ. Ngữ cảnh của mỗi câu gồm kết quả của `search_policy`, `get_application` và `calc_dti` (số liệu tính toán đến từ hai tool sau; nếu chỉ lấy `search_policy`, câu trả lời đúng vẫn bị coi là thiếu căn cứ). Câu `no_answer` **không** chấm RAGAS vì Answer Relevancy phạt câu từ chối đúng; nhóm này chỉ xét từ chối đúng bằng code (N01, N02, N04 đúng, N03 sai, khớp mục 2.1).

Trung bình 12 câu có đáp án (một lần chạy, kết quả ở `eval/out/ragas/`):

| Faithfulness | Answer Relevancy | Answer Correctness | Context Precision | Context Recall |
|---|---|---|---|---|
| 0,94 | 0,81 | 0,60 | 0,80 | 1,00 |

Điểm từng câu:

| Câu | Loại | Faith. | Relev. | Correct. | Ctx Prec. | Ctx Rec. |
|---|---|---|---|---|---|---|
| T01 | tra cứu | 1,00 | 0,98 | 0,94 | 1,00 | 1,00 |
| T02 | tra cứu | 1,00 | 0,77 | 0,95 | 1,00 | 1,00 |
| S01 | tổng hợp | 1,00 | 0,91 | 0,92 | 0,00 | 1,00 |
| S02 | tổng hợp | 0,75 | 0,79 | 0,54 | 1,00 | 1,00 |
| V01 | phiên bản | 1,00 | 1,00 | 0,37 | 1,00 | 1,00 |
| V02 | phiên bản | 1,00 | 0,76 | 0,66 | 1,00 | 1,00 |
| C01 | tính toán | 1,00 | 0,81 | 0,49 | 0,33 | 1,00 |
| C02 | tính toán | 1,00 | 0,87 | 0,41 | 0,58 | 1,00 |
| M01 | thiếu dữ liệu | 0,78 | 0,90 | 0,33 | 0,83 | 1,00 |
| M02 | thiếu dữ liệu | 1,00 | 0,87 | 0,57 | 1,00 | 1,00 |
| A01 | hành động | 0,92 | 0,49 | 0,47 | 0,87 | 1,00 |
| A02 | hành động | 0,88 | 0,58 | 0,51 | 1,00 | 1,00 |

Đọc kết quả:
- **Câu trả lời bám sát căn cứ:** Faithfulness 0,94 và Context Recall 1,00, tức mọi ý cần có đều nằm trong ngữ cảnh mà Agent đã tra cứu; phần còn thiếu của hệ thống không nằm ở truy xuất.
- **Context Precision thấp ở vài câu** (S01 0,00; C01 0,33) vì truy vấn kéo thêm đoạn nhiễu (ví dụ CP-09 thẻ tín dụng), khớp hạn chế về nhạy cảm của truy vấn ở mục 4.
- **Answer Correctness chỉ khoảng 0,6 không có nghĩa là sai 40%.** Chỉ số so khớp ngữ nghĩa với đáp án chuẩn dài và có định dạng, nên câu đúng hẳn như V01 (tuổi tối đa 60) vẫn chỉ đạt 0,37. Answer Relevancy của câu hành động (A01, A02) thấp vì câu trả lời là bản ghi quyết định, không phải một câu trả lời cho câu hỏi.
- **Giám khảo nhiễu:** lần chạy đầu (khi ngữ cảnh mới chỉ gồm `search_policy`) cho V02 Faithfulness 0,0 dù câu trả lời trích đúng "45%", lần chạy sau cho 1,00. Vì vậy RAGAS chỉ là **số liệu bổ sung**: 12 câu, một giám khảo bản nhẹ, một lần chạy; kết luận về chất lượng dựa vào chấm bằng code ở mục 2.1.

### 2.5. Thí nghiệm đổi một con số trong chính sách

**Mục đích:** kiểm tra agent có dùng chính sách tra được (RAG) hay dựa vào trí nhớ của model. Nếu đổi một con số trong tài liệu mà quyết định đổi theo, thì quyết định đó đến từ tài liệu.

**Cách làm** (`make policy-change`, `eval/policy_change_experiment.py`): trên **bản sao tạm** của dự án (dữ liệu thật không bị sửa), đổi trần DTI của CP-01 v1 từ 50% xuống 40% ở mọi chỗ tài liệu nêu con số này (CP-01 v1, CP-01 v2 mục 6.4, bảng so sánh trong CP-15), dựng lại chỉ mục vector, tính lại đáp án chuẩn (hồ sơ v1 có DTI trong khoảng 40% đến 50% đổi từ duyệt sang từ chối `DTI_EXCEEDED`), rồi chạy agent gpt-oss-120b không critic (cấu hình A) trên 7 ca. Kết quả ở `eval/out/policy_change/`.

| Ca | Vai trò | Đáp án gốc | Đáp án sau khi đổi | Agent sau khi đổi |
|---|---|---|---|---|
| APP-014 (DTI 47,09%) | bị ảnh hưởng | duyệt | từ chối | **từ chối**, lý do ghi "47.09% > ngưỡng tối đa 40% theo CP-01 v1" |
| APP-016 (DTI 48,48%) | bị ảnh hưởng | duyệt | từ chối | **từ chối** |
| APP-001 (DTI 24,62%) | đối chứng, v1 DTI thấp | duyệt | duyệt | duyệt |
| APP-012 (chuyển cấp) | đối chứng, v1 | chuyển cấp | chuyển cấp | chuyển cấp (quyết định, mã lý do, trích dẫn và số liệu đều đúng); **trượt vì lỗi hạ tầng HTTP 413** của Groq (giới hạn 8.000 token mỗi phút) ở bước trả lời cuối |
| APP-019 (DTI 51,53%) | đối chứng, v1 đã bị từ chối | từ chối | từ chối | từ chối |
| APP-015 (DTI 47,09%, v2) | đối chứng, cùng DTI với APP-014 nhưng áp dụng v2 | từ chối | từ chối | từ chối |
| X01 (câu hỏi về trần DTI ngày 15/03/2025) | câu hỏi đáp | 50% | 40% | trước khi đổi trả lời **50%**, sau khi đổi trả lời **40%** |

**Kết luận:** cả hai hồ sơ bị ảnh hưởng đổi từ duyệt sang từ chối đúng như đáp án mới, câu hỏi về trần DTI đổi từ 50% sang 40%, còn các hồ sơ đối chứng giữ nguyên. Như vậy quyết định của agent theo nội dung chính sách tra được, không theo trí nhớ của model. Giới hạn: chỉ 7 ca, một lần chạy, một model; APP-012 trượt do hạ tầng chứ không do thay đổi chính sách.

## 3. Phân tích lỗi

### 3.1. Model nhỏ: lỗi tuân thủ quy trình

Lỗi của Qwen tập trung ở **tuân thủ quy trình**, ít ở kiến thức chính sách:

| Nhóm lỗi | Biểu hiện (Qwen) | Số ca |
|---|---|---|
| Trả lời không tra cứu | 0 lời gọi tool, trả lời từ trí nhớ | 3 câu (T02, N02, N04) |
| Ghi xong không đọc lại | có `submit_decision` nhưng không gọi `get_decision` | 3 hồ sơ (APP-007, 010, 019) |
| Hết bước hoặc xoay vòng tra cứu | vượt `MAX_STEPS`, hoặc prompt vượt giới hạn ngữ cảnh | 2 hồ sơ (APP-018, 011) |
| Duyệt nhầm hồ sơ thiếu thông tin | `approve` khi thiếu chứng từ hoặc chưa khai báo nợ | APP-004, 005 |
| Chọn sai lãi suất theo kỳ hạn | dùng 15,5% thay cho 17% hoặc 18,5% theo biểu CP-03, kéo theo sai DTI | nhiều hồ sơ |
| Không nhận ra sản phẩm khác | dùng biểu lãi suất vay tiêu dùng để trả lời câu hỏi về vay thế chấp | N01, N02, N04 |

Tầng truy xuất không phải nguyên nhân chính: đo riêng, recall@3 đạt 1,00 và gold@3 đạt 0,90 sau khi giảm điểm tài liệu diễn giải (CP-15, CP-16).

### 3.2. Ca thất bại của model mạnh (gpt-oss-120b, không critic)

| Ca | Biểu hiện | Phân tích |
|---|---|---|
| APP-018 | `escalate` đúng nhưng mã lý do `ESCALATE_COUNCIL`, chuẩn là `ESCALATE_EMPLOYEE` | Model nhầm thẩm quyền của **nhân viên FinCo** (ngoại lệ CP-06) sang Hội đồng |
| A02 | cùng lỗi mã lý do `ESCALATE_COUNCIL` thay vì `ESCALATE_EMPLOYEE` | cùng một lỗi có hệ thống ở nhánh nhân viên |
| N03 | không từ chối dù tài liệu không chứa thông tin | lỗi calibration: trả lời bịa thay vì nói không đủ dữ liệu |

Đây là ca thất bại tiêu biểu để phân tích: quyết định tổng thể đúng (`escalate`) nên chỉ so khớp mã lý do mới phát hiện được lỗi, và lỗi nằm ở một quy tắc cụ thể (nhân viên FinCo đi theo nhánh riêng, không theo số tiền vay).

### 3.3. Critic: vì sao không cải thiện

1. **Không có dư địa với Actor mạnh.** gpt-oss-120b đã đúng quyết định 21/21; chỗ còn sai là mã lý do (APP-018, A02) và calibration (N03), mà critic không sửa được.
2. **Critic bất đồng nhưng cũng sai.** Cả 3 lần critic chặn ở APP-018 đều đề xuất `approve` / `APPROVED`, trong khi đáp án chuẩn là `escalate` / `ESCALATE_EMPLOYEE`; vì cùng một hạn chế về quy tắc nhân viên nên critic không phải một góc nhìn độc lập thực sự ở điểm này.
3. **Critic yếu hơn mức cần.** Với model nhỏ làm critic (Qwen 7B hoặc Gemini Flash Lite), khoảng 12,5% lần thẩm tra thất bại về định dạng JSON hoặc bỏ qua bước tính DTI. Khi critic lỗi hạ tầng, quyết định của Actor vẫn được ghi kèm cờ chưa xác minh nên không bị phạt oan, nhưng cũng không có giá trị bổ sung.
4. **Phản hồi chặn làm ngữ cảnh Actor phình.** Ở gói miễn phí (giới hạn 8.000 token mỗi request), điều này dẫn tới lỗi HTTP 413 ở các ca dài (APP-011, APP-018).

Quan sát có kiểm chứng: critic-agent thật sự độc lập (tự đọc hồ sơ, tự tra CP-01, CP-03, CP-04, tự gọi `calc_dti`) và hoàn tất thẩm tra trong 5 đến 6 lượt gọi, khoảng 12.000 đến 14.000 prompt token mỗi lần.

## 4. Hạn chế

- **Một lần chạy mỗi cấu hình**, nên chưa đo được độ biến thiên. Với 37 mẫu, khoảng sai số 95% của một tỉ lệ khoảng ±13 điểm phần trăm; các chênh lệch 1 đến 3 ca (như 34 so với 32) nằm trong nhiễu và không có ý nghĩa thống kê. Kết luận A/B chỉ nên đọc là "không thấy cải thiện", không phải "critic có hại".
- **Kết quả gpt-oss-20b phụ thuộc môi trường chạy:** cùng model nhưng Ollama (G) đạt 23/37 còn Groq (H) đạt 33/37 (xem mục 2.3). Cấu hình critic của 20b chỉ so sánh được với H, không so với G.
- **Giới hạn hạ tầng ảnh hưởng kết quả:** HTTP 413 do request vượt 8.000 token mỗi phút (gói `on_demand`) làm hỏng ca dài như APP-011, APP-016, APP-018; HTTP 429 và hết hạn mức token/ngày làm kéo dài thời gian. Đây là lỗi hạ tầng, không phải lỗi model, nhưng có thể làm lệch số liệu giữa các nhánh.
- **Các model chạy trên các nền tảng khác nhau** (Groq, Colab T4), nên so sánh theo kích thước chỉ mang tính định hướng; chưa có "thang kích thước" cùng họ trên cùng nền tảng.
- **Critic và Actor dùng model khác họ ở cấu hình B** nên chỉ giảm được một phần lỗi tương quan; ở cấu hình D thì dùng chung model nên lỗi có thể tương quan.
- Bộ chấm yêu cầu trích dẫn xuất hiện trong văn bản trả lời của câu hỏi đáp; một câu trả lời đúng nhưng không nêu mã tài liệu bị tính là trượt (ví dụ C02).
- Đã làm thí nghiệm đổi một con số (mục 2.5) nhưng chỉ trên 7 ca và một model. **Chưa có** đối chứng không dùng RAG.

## 5. Kết luận và hướng tiếp theo

Hệ thống vận hành đúng từ đầu đến cuối trên model thật (MCP, tra cứu, ghi và đọc lại, log có USAGE, chấm điểm tự động). Kết luận chính:

1. **Chất lượng model quyết định kết quả.** gpt-oss-120b đạt 92%, Qwen nhỏ đạt 22 đến 27%; lỗi của model nhỏ là lỗi tuân thủ quy trình, không phải thiếu kiến thức chính sách.
2. **Critic chưa cho thấy lợi ích đo được, nhưng bằng chứng mới chỉ phủ các Actor ít lỗi.** Trên gpt-oss-120b (A so với B) và gpt-oss-20b trên Groq (H so với I, J: 33/37 so với 32/37 và 32/37, mục 2.3), critic không vượt nhánh không critic, và critic-agent tốn thêm khoảng 61% prompt token (trên 120b). Cả hai Actor này gần như không còn quyết định sai (120b: 0/21, 20b: 1/21) nên critic không có chỗ để cải thiện; với Qwen 7B, critic tự chấm cũng không giúp (mục 2.2). **Chưa thử** critic mạnh (Gemini) trên một Actor sai nhiều quyết định, ví dụ gpt-oss:20b chạy Ollama (G, đúng 17/21 hồ sơ); đây là thí nghiệm còn thiếu và nên ghi trong hướng tiếp theo.
3. **Cấu hình mặc định nên là không critic** (`FINCO_CRITIC=0`) cho đến khi có bằng chứng ngược lại; critic-agent giữ lại như một cấu hình thí nghiệm có tài liệu (`FINCO_CRITIC_MODE=agent`, `FINCO_CRITIC_PROVIDER=gemini`).

Hướng tiếp theo, theo thứ tự đáng làm: (a) chạy lặp lại cấu hình A và B đủ nhiều lần để đo độ biến thiên thật; (b) sửa gốc lỗi mã lý do của nhánh nhân viên (APP-018, A02) bằng quy tắc rõ trong prompt và đo lại; (c) chạy critic Gemini trên Actor sai nhiều quyết định (gpt-oss:20b qua Ollama, G), rồi thử critic mạnh hơn hoặc cùng họ với Actor, và cho critic xác minh riêng mã lý do thay vì chỉ quyết định; (d) làm thang kích thước cùng họ trên cùng nền tảng (ví dụ `gpt-oss-20b` và `120b`, hoặc Llama 8B và 70B) để kết luận đúng về ảnh hưởng của số tham số.
