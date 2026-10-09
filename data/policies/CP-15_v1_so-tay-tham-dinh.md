---
doc_id: CP-15
title: Sổ tay thẩm định hồ sơ vay FinCo Easy
version: v1
effective_date: 2024-01-01
owner: Phòng Vận hành tín dụng
status: active
language: vi
---

# Sổ tay thẩm định hồ sơ vay FinCo Easy

## 1. Mục đích và cách dùng sổ tay

1.1. Sổ tay này hướng dẫn chuyên viên thẩm định thực hiện từng thao tác khi xử lý một hồ sơ vay FinCo Easy, từ lúc nhận hồ sơ đến lúc ghi nhận quyết định. Sổ tay **không đặt ra quy định mới**. Mọi ngưỡng, tỷ lệ và thẩm quyền đều lấy từ các chính sách gốc: CP-01 (điều kiện cho vay), CP-02 (chứng từ), CP-03 (lãi suất và phí), CP-04 (phân cấp phê duyệt), CP-05 (cách tính DTI), CP-06 (ngoại lệ) và CP-07 (ghi nhận quyết định).

1.2. Khi nội dung sổ tay khác với chính sách gốc, chính sách gốc được ưu tiên. Chuyên viên phát hiện điểm khác biệt phải báo cho Phòng Vận hành tín dụng để sửa sổ tay, không tự áp dụng theo cách hiểu riêng.

1.3. Sổ tay dành cho chuyên viên thẩm định nội bộ. Khách hàng muốn tìm hiểu điều kiện vay nên xem tài liệu hỏi đáp dành cho khách vay (CP-16).

## 2. Thuật ngữ thường dùng

2.1. **Ngày nộp**: ngày hồ sơ được ghi nhận trong hệ thống. Đây là mốc duy nhất để chọn phiên bản chính sách. Ngày thẩm định, ngày bổ sung chứng từ và ngày giải ngân không được dùng để chọn phiên bản.

2.2. **Thu nhập chính**: thu nhập bình quân hàng tháng của người vay, xác minh được từ chứng từ. Với người hưởng lương đó là lương cơ bản; với người tự kinh doanh đó là doanh thu ròng bình quân từ sao kê.

2.3. **Thu nhập phụ**: các khoản như phụ cấp, thưởng định kỳ hoặc thu nhập ngoài hoạt động chính, có bằng chứng định kỳ. Thu nhập phụ chỉ được tính một phần vào thu nhập đủ điều kiện (xem CP-05).

2.4. **Thu nhập đủ điều kiện**: con số dùng làm mẫu số của DTI sau khi áp hệ số theo nhóm thu nhập. Không dùng thu nhập chính cộng thu nhập phụ trực tiếp.

2.5. **Nghĩa vụ nợ hiện tại**: tổng số tiền người vay phải trả hàng tháng cho các khoản vay đang có, theo khai báo của chính người vay. Khai 0 đồng nghĩa là không có khoản nợ nào; để trống là chưa khai.

2.6. **PMT**: khoản trả hàng tháng của khoản vay mới theo phương thức niên kim, tính bằng công thức ở CP-05 mục 2.

2.7. **DTI**: tỷ lệ giữa tổng nghĩa vụ trả nợ hàng tháng (PMT cộng nghĩa vụ nợ hiện tại) và thu nhập đủ điều kiện. DTI càng cao thì người vay càng khó cân đối chi tiêu.

2.8. **Chuyển cấp trên** (`escalate`): hồ sơ đã đạt mọi điều kiện nhưng không thuộc thẩm quyền quyết định của chuyên viên, nên chuyển lên Trưởng phòng hoặc Hội đồng tín dụng.

## 3. Quy trình xử lý từng bước

3.1. **Bước 0 – Xác định phiên bản chính sách.** Đọc ngày nộp của hồ sơ. Ngày nộp trước 01/04/2025 thì áp dụng CP-01 phiên bản v1. Ngày nộp từ 01/04/2025 trở đi thì áp dụng CP-01 phiên bản v2. Hồ sơ nộp ngày 31/03/2025 vẫn xét theo v1 dù việc thẩm định diễn ra sau ngày 01/04/2025. Ghi lại phiên bản đã chọn ngay từ đầu vì mọi bước sau đều dựa vào đó.

3.2. **Bước 1 – Điều kiện loại trừ.** Kiểm tra lần lượt bốn nội dung. Thứ nhất, tuổi tại ngày nộp phải nằm trong khoảng cho phép của phiên bản (bảng ở mục 4). Thứ hai, số tiền vay phải nằm trong khoảng cho phép. Thứ ba, kỳ hạn phải nằm trong khoảng cho phép. Thứ tư, người vay không có nợ quá hạn trên 30 ngày trong 12 tháng gần nhất (CP-06 mục 1). Chỉ cần một nội dung không đạt thì quyết định là `reject`, và không cần xem tiếp các bước sau. Lý do từ chối nêu đúng nội dung không đạt cùng ngưỡng của phiên bản đã áp dụng.

3.3. **Bước 2 – Chứng từ và thông tin.** Đối chiếu mã chứng từ trong hồ sơ với nhóm thu nhập của người vay. Nhóm hưởng lương cần `CCCD`, `HDLD`, `SK_LUONG_3T`. Nhóm tự kinh doanh cần `CCCD`, `DKKD`, `SK_TK_6T`. Kiểm tra thêm phần khai báo nghĩa vụ nợ: để trống thì hồ sơ chưa đủ thông tin. Hồ sơ thiếu bất kỳ mục nào được ghi nhận `need_info` và liệt kê rõ từng mục còn thiếu, không được ghi chung chung là "thiếu hồ sơ". Không tính DTI khi hồ sơ chưa đủ thông tin, vì một con số tính trên dữ liệu thiếu sẽ dễ bị hiểu nhầm là kết quả cuối.

3.4. **Bước 3 – Tính DTI.** Thực hiện theo thứ tự sau để tránh sai sót. Chọn lãi suất theo kỳ hạn tại CP-03, chú ý ranh giới giữa các mức, ví dụ 24 tháng thuộc mức 15,5% còn 25 tháng thuộc mức 17,0%. Tính PMT bằng công thức niên kim. Tính thu nhập đủ điều kiện theo nhóm thu nhập. Cộng PMT với nghĩa vụ nợ hiện tại rồi chia cho thu nhập đủ điều kiện. So sánh với ngưỡng DTI của phiên bản đã chọn ở Bước 0. DTI bằng ngưỡng thì đạt, chỉ DTI lớn hơn ngưỡng mới bị `reject`. Phải dùng giá trị chưa làm tròn khi so sánh. Không có ngoại lệ về DTI ở bất kỳ cấp nào (CP-06 mục 3).

3.5. **Bước 4 – Thẩm quyền.** Chỉ hồ sơ đã qua ba bước trên mới đến bước này. Nếu người vay là nhân viên FinCo thì hồ sơ luôn được chuyển Hội đồng tín dụng, bất kể số tiền. Nếu không, xét số tiền vay: không quá 200 triệu đồng thì chuyên viên `approve`; trên 200 triệu đến 500 triệu đồng thì `escalate` lên Trưởng phòng; trên 500 triệu đồng thì `escalate` lên Hội đồng tín dụng. Số tiền vay đúng bằng 200 triệu hay 500 triệu thuộc cấp thấp hơn.

3.6. **Bước 5 – Ghi nhận và kiểm tra lại.** Ghi quyết định vào hệ thống kèm lý do và danh sách tài liệu căn cứ theo mã tài liệu và phiên bản. Ngay sau đó đọc lại quyết định đã lưu để xác nhận trạng thái và nội dung đúng. Chỉ khi đã đọc lại và khớp mới được báo kết quả cho khách hàng. Nếu hệ thống báo hồ sơ đã có quyết định thì không ghi đè (CP-07 mục 3).

## 4. Bảng ngưỡng theo phiên bản chính sách

4.1. Bảng dưới đây tổng hợp các ngưỡng của CP-01 để tra nhanh. Chọn cột theo ngày nộp, không theo ngày thẩm định.

| Nội dung | CP-01 v1 (nộp trước 01/04/2025) | CP-01 v2 (nộp từ 01/04/2025) |
|---|---|---|
| Tuổi | 22 đến 60 | 22 đến 58 |
| Số tiền vay | 10 triệu đến 800 triệu đồng | 10 triệu đến 700 triệu đồng |
| Kỳ hạn | 6 đến 60 tháng | 6 đến 48 tháng |
| DTI tối đa | 50% | 45% |

4.2. Các nội dung không đổi giữa hai phiên bản gồm: nhóm người vay, chứng từ (CP-02), biểu lãi suất (CP-03), phân cấp phê duyệt (CP-04), cách tính DTI (CP-05), ngoại lệ (CP-06) và ghi nhận quyết định (CP-07).

4.3. Hệ số thu nhập theo nhóm, dùng chung cho cả hai phiên bản:

| Nhóm thu nhập | Hệ số thu nhập chính | Hệ số thu nhập phụ |
|---|---|---|
| Hưởng lương (`salary`) | 100% | 70% |
| Tự kinh doanh (`self_employed`) | 80% | 50% |

## 5. Ví dụ xử lý minh họa

Các ví dụ dưới đây là tình huống giả định để tập thao tác, không phải hồ sơ thật.

5.1. **Ví dụ A – hồ sơ đạt, chuyên viên duyệt.** Người vay hưởng lương, nộp ngày 10/06/2025 nên áp dụng v2. Vay 150 triệu đồng trong 24 tháng, thu nhập chính 22 triệu, thu nhập phụ 4 triệu, nghĩa vụ nợ hiện tại 1,5 triệu đồng mỗi tháng. Tuổi, số tiền, kỳ hạn đều trong khoảng cho phép và chứng từ đầy đủ. Kỳ hạn 24 tháng thuộc mức lãi 15,5% nên PMT khoảng 7,31 triệu đồng. Thu nhập đủ điều kiện là 22 triệu cộng 70% của 4 triệu, bằng 24,8 triệu đồng. DTI khoảng (7,31 + 1,5) / 24,8 ≈ 35,5%, thấp hơn ngưỡng 45%. Số tiền 150 triệu không vượt 200 triệu, nên chuyên viên `approve`.

5.2. **Ví dụ B – hồ sơ cần chuyển Trưởng phòng.** Người vay tự kinh doanh, nộp ngày 20/02/2025 nên áp dụng v1. Vay 300 triệu đồng trong 36 tháng, thu nhập chính 40 triệu, thu nhập phụ 10 triệu, nghĩa vụ nợ hiện tại 3 triệu đồng mỗi tháng. Kỳ hạn 36 tháng thuộc mức lãi 17,0% nên PMT khoảng 10,70 triệu đồng. Thu nhập đủ điều kiện là 80% của 40 triệu cộng 50% của 10 triệu, bằng 37 triệu đồng. DTI khoảng (10,70 + 3) / 37 ≈ 37,0%, thấp hơn ngưỡng 50% của v1. Số tiền 300 triệu nằm trong khoảng trên 200 đến 500 triệu, nên chuyên viên `escalate` lên Trưởng phòng.

5.3. **Ví dụ C – DTI vượt ngưỡng, không chuyển cấp trên.** Người vay hưởng lương, nộp ngày 05/05/2025 nên áp dụng v2. Vay 80 triệu đồng trong 12 tháng với lãi suất 14,0%, thu nhập chính 15 triệu, không có thu nhập phụ, nghĩa vụ nợ hiện tại 3 triệu đồng. PMT khoảng 7,18 triệu đồng nên DTI khoảng (7,18 + 3) / 15 ≈ 67,9%, lớn hơn ngưỡng 45%. Quyết định là `reject`. Dù số tiền vay nhỏ, chuyên viên không được chuyển hồ sơ lên cấp trên để xin ngoại lệ, vì CP-04 mục 3.1 và CP-06 mục 3.1 không cho phép.

## 6. Sai sót thường gặp khi thẩm định

6.1. **Chọn phiên bản theo ngày thẩm định.** Đây là lỗi phổ biến nhất. Hồ sơ nộp cuối tháng 03/2025 nhưng được thẩm định đầu tháng 04/2025 vẫn áp dụng v1.

6.2. **Cộng đủ 100% thu nhập phụ.** Thu nhập phụ chỉ được tính theo hệ số của nhóm thu nhập (70% với người hưởng lương, 50% với người tự kinh doanh). Tính đủ 100% làm DTI thấp hơn thực tế và có thể đẩy một hồ sơ đáng lẽ bị từ chối thành đạt.

6.3. **Nhầm ranh giới kỳ hạn.** Khoản vay 24 tháng thuộc mức 15,5% còn 25 tháng thuộc mức 17,0%. Lãi suất sai làm PMT và DTI sai theo.

6.4. **Làm tròn trước khi so sánh ngưỡng.** DTI 45,04% làm tròn thành 45,0% trông như bằng ngưỡng nhưng thực tế vẫn lớn hơn 45%. Luôn so sánh giá trị chưa làm tròn.

6.5. **Tính phí thẩm định vào khoản trả hàng tháng.** Phí thẩm định thu một lần khi giải ngân và không nằm trong PMT khi xét DTI.

6.6. **Coi ô nợ hiện tại để trống là 0 đồng.** Để trống là chưa khai báo và hồ sơ phải ghi `need_info`. Chỉ khi người vay khai 0 đồng mới được hiểu là không có nợ.

6.7. **Chuyển hồ sơ nhân viên FinCo lên cấp chuyên viên.** Hồ sơ nhân viên FinCo đã đạt mọi điều kiện luôn lên Hội đồng tín dụng. Ngược lại, nhân viên FinCo không đạt điều kiện vẫn bị `reject` hoặc `need_info` như khách hàng khác.

6.8. **Ghi quyết định không có tài liệu căn cứ.** Hệ thống từ chối ghi quyết định thiếu danh sách tài liệu. Hãy ghi đúng mã tài liệu và phiên bản của những chính sách thực sự đã dùng.

## 7. Lưu trữ và đối soát

7.1. Mỗi quyết định phải truy được về: hồ sơ gốc, phiên bản chính sách đã áp dụng, các con số trung gian (PMT, thu nhập đủ điều kiện, DTI) và người ghi nhận. Chuyên viên nên ghi các con số trung gian vào phần lý do để người kiểm tra sau này tái hiện được kết quả.

7.2. Khi chính sách có phiên bản mới, hồ sơ đã có quyết định không bị xét lại. Chỉ hồ sơ chưa có quyết định và có ngày nộp từ ngày hiệu lực mới áp dụng phiên bản mới.

7.3. Cuối mỗi tháng, Phòng Vận hành tín dụng chọn ngẫu nhiên một số hồ sơ đã quyết định để đối soát. Các điểm kiểm tra gồm phiên bản chính sách, cách tính DTI, thẩm quyền và việc đọc lại quyết định sau khi ghi.
