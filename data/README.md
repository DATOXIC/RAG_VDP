# Dữ liệu đầu vào – FinCo Easy

> **Toàn bộ dữ liệu là giả lập** do nhóm tự viết cho môn học: công ty FinCo, người vay, con số và quy tắc đều hư cấu. Không dùng để tư vấn tài chính thật.

## 1. Có gì trong thư mục

| Đường dẫn | Là gì | Đi vào đâu khi ingestion |
|---|---|---|
| `data/policies/*.md` | 18 tài liệu chính sách (17 mã `CP-01..CP-17`, riêng `CP-01` có 2 phiên bản). `CP-01..CP-07` là lõi của FinCo Easy, quyết định đáp án chuẩn. `CP-08..CP-17` là **tài liệu bổ trợ và nhiễu** (xem dưới) | Tách chunk → embedding → chỉ mục RAG |
| `data/applications.json` | 21 hồ sơ vay giả lập | Nạp vào bảng SQLite `applications` |
| `tests/expected_decisions.json` | Đáp án chuẩn từng hồ sơ (quyết định, phiên bản, DTI, lý do, tài liệu cần trích) | **Chỉ dùng cho eval**, agent không được đọc |

## 2. Định dạng tài liệu chính sách

Mỗi file là Markdown, phần đầu là front matter dạng `khóa: giá trị` (một dòng một khóa, **không dùng danh sách hay YAML lồng nhau**, để parser tự viết chỉ cần tách theo dấu `:`).

| Khóa | Bắt buộc | Ý nghĩa |
|---|---|---|
| `doc_id` | có | Mã tài liệu, ví dụ `CP-01` |
| `title` | có | Tên tài liệu |
| `version` | có | `v1`, `v2`... Cặp (`doc_id`, `version`) là duy nhất |
| `effective_date` | có | Ngày bắt đầu hiệu lực (`YYYY-MM-DD`) |
| `owner` | có | Đơn vị sở hữu |
| `expiry_date` | không | Ngày cuối cùng còn hiệu lực. Không có nghĩa là còn hiệu lực |
| `supersedes` | không | Phiên bản bị thay thế |
| `status` | không | `active`, `superseded` (bị bản mới thay) hoặc `expired` (chương trình đã hết hạn) |
| `language` | không | `vi` |

**Chunk theo mục `## `.** Mỗi mục là một chunk. Khi nhúng, thêm một dòng đầu `[doc_id version | hiệu lực từ ...] tiêu đề – mục` để chunk giữa tài liệu vẫn biết nó thuộc đâu. Cả 18 tài liệu cho **88 chunk** (lõi 38 + nhiễu 30 + tài liệu dài 20). Các chunk của `CP-01..CP-14` dài 35–170 từ; `CP-15..CP-17` có chunk tới khoảng 630 từ để thử nhánh chia nhỏ theo điều khoản của chunker.

**Tài liệu dài (`CP-15..CP-17`)** viết như văn bản thực tế: `CP-15` sổ tay thẩm định (thuật ngữ, quy trình từng bước, bảng ngưỡng theo phiên bản, 3 ví dụ giả định, sai sót thường gặp), `CP-16` hỏi đáp khách vay (nhắc lại số liệu theo hai phiên bản, nên có chunk trùng ý với `CP-01`), `CP-17` xử lý nợ quá hạn sau giải ngân (nhóm nợ, lãi phạt, nhắc nợ). Chúng chỉ nhắc lại hoặc bổ sung, không đổi số nào của `CP-01..CP-07`; `CP-15/16` là bản diễn giải nên khi mâu thuẫn thì chính sách gốc được ưu tiên.

**Tài liệu nhiễu (`CP-08..CP-14`)** thuộc sản phẩm khác của FinCo, cố ý dùng từ vựng gần giống FinCo Easy để truy xuất phải phân biệt: `CP-09` thẻ tín dụng (có DTI 40%, lãi suất 28%, kỳ miễn lãi, đều **không** áp dụng cho FinCo Easy); `CP-10` tiền gửi tiết kiệm (có bảng lãi **tiền gửi** theo kỳ hạn, dễ nhầm với lãi suất vay ở `CP-03`); `CP-11` bảo mật dữ liệu; `CP-12` khiếu nại thẻ và tiền gửi (dễ nhầm với khiếu nại thẩm định ở `CP-08`); `CP-13` ưu đãi mùa hè 2025 (đã hết hạn 31/08/2025, để kiểm tra lọc theo `expiry_date`); `CP-14` định danh khách hàng. **Không tài liệu nhiễu nào làm đổi đáp án của 21 hồ sơ**, và không tài liệu nào nói về vay thế chấp, vay mua ô tô hay vay doanh nghiệp, để các câu hỏi `no_answer` về những chủ đề đó vẫn đúng. Khi viết thêm tài liệu, giữ hai nguyên tắc này.

**Chọn đúng phiên bản.** Hồ sơ nộp trước `2025-04-01` áp `CP-01 v1`, từ ngày đó áp `CP-01 v2`. Hai phiên bản có các mục gần như giống hệt nhau chỉ khác con số (ví dụ mục "4. Tỷ lệ nợ trên thu nhập (DTI) tối đa": 50% so với 45%). Vì vậy tìm kiếm ngữ nghĩa thuần sẽ **không phân biệt được**; phải lọc theo `effective_date` (bonus metadata filter).

## 3. Bảng `applications` (từ `applications.json`)

Tiền tệ là **đồng (VND)**, số nguyên.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `app_id` | chuỗi | `APP-001` ... `APP-021` |
| `applicant_name` | chuỗi | Tên giả lập |
| `age` | số nguyên | Tuổi tại ngày nộp |
| `product` | chuỗi | `FinCo Easy` |
| `amount` | số nguyên | Số tiền vay |
| `term_months` | số nguyên | Kỳ hạn (tháng) |
| `income_type` | chuỗi | `salary` hoặc `self_employed` |
| `monthly_income` | số nguyên | Thu nhập chính bình quân tháng |
| `supplementary_income` | số nguyên | Thu nhập phụ bình quân tháng |
| `existing_debt_monthly` | số nguyên hoặc `null` | Tiền trả nợ hiện tại mỗi tháng. `null` = chưa khai báo (khác với `0`) |
| `is_employee` | bool | Là nhân viên FinCo |
| `overdue_30d_12m` | bool | Có nợ quá hạn trên 30 ngày trong 12 tháng gần nhất |
| `docs_submitted` | danh sách chuỗi | Mã chứng từ đã nộp: `CCCD`, `HDLD`, `SK_LUONG_3T`, `DKKD`, `SK_TK_6T` |
| `submitted_date` | chuỗi | Ngày nộp (`YYYY-MM-DD`), quyết định phiên bản chính sách |
| `note` | chuỗi | Ghi chú tự do, **không chứa đáp án** |

Bảng `decisions` (do tool ghi tạo ra) bắt đầu **rỗng**, xem hợp đồng ở `PLAN.md` mục 6.3.

## 4. Các tình huống đã cài sẵn (để làm testset)

| Hồ sơ | Kiểm tra điều gì | Kết quả chuẩn |
|---|---|---|
| APP-001 | Trường hợp sạch, v1 | approve |
| APP-002 | Trường hợp sạch, v2; kỳ hạn 25 tháng rơi vào mức lãi 17,0% (không phải 15,5%) | approve |
| APP-003 | DTI vượt cả hai phiên bản (63%) | reject |
| APP-004 | Thiếu `SK_LUONG_3T` | need_info |
| APP-005 | Tự kinh doanh thiếu `SK_TK_6T` | need_info |
| APP-006 | Tự kinh doanh (hệ số 80% và 50%); vay đúng 200 triệu = ngưỡng chuyên viên | approve |
| APP-007 | Tuổi 21, dưới 22 | reject |
| APP-008 / APP-009 | Cùng tuổi 59, khác ngày nộp: v2 tối đa 58 tuổi, v1 tối đa 60 | reject / approve |
| APP-010 | Nợ quá hạn | reject |
| APP-011 | 350 triệu, trong khoảng 200–500 triệu | escalate (Trưởng phòng) |
| APP-012 / APP-013 | Cùng 750 triệu: v1 cho tối đa 800, v2 chỉ 700 | escalate (Hội đồng) / reject |
| APP-014 / APP-015 | Cùng DTI 47,09%: v1 cho tối đa 50%, v2 chỉ 45% | approve / reject |
| APP-016 / APP-017 | Cùng DTI 48,48%, nộp 31/3 và 1/4 (đúng ranh giới ngày) | approve / reject |
| APP-018 | Nhân viên FinCo, vay nhỏ | escalate (Hội đồng) |
| APP-019 | Bẫy thu nhập phụ: tính 70% thì DTI 51,53% (reject), tính nhầm 100% thì 46,38% (qua) | reject |
| APP-020 | Kỳ hạn 54 tháng, v2 chỉ cho tối đa 48 | reject |
| APP-021 | Chưa khai nghĩa vụ nợ | need_info |

Phân bố quyết định chuẩn: 6 approve, 9 reject, 3 need_info, 3 escalate.

## 5. Quy tắc đã dùng để tính đáp án chuẩn

Đáp án trong `tests/expected_decisions.json` được tính bằng một bộ quy tắc tham chiếu viết đúng theo văn bản chính sách:

1. Chọn phiên bản `CP-01` theo `submitted_date`.
2. **Loại trừ:** tuổi (22 đến mức tối đa của phiên bản), nợ quá hạn, số tiền (10 triệu đến mức tối đa), kỳ hạn (6 tháng đến mức tối đa). Không đạt thì `reject`.
3. **Chứng từ:** thiếu mã chứng từ hoặc `existing_debt_monthly = null` thì `need_info`.
4. **DTI** = (khoản trả hàng tháng của khoản vay mới + nợ hiện tại) / thu nhập đủ điều kiện. Vượt ngưỡng (lớn hơn, không phải bằng) thì `reject`.
5. **Thẩm quyền:** nhân viên FinCo thì `escalate`; số tiền ≤ 200 triệu thì `approve`; ≤ 500 triệu thì `escalate` (Trưởng phòng); lớn hơn thì `escalate` (Hội đồng).

Nếu sửa số trong bất kỳ chính sách nào, **phải tính lại** `expected_decisions.json`, nếu không số liệu eval sẽ sai.
