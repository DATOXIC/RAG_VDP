---
doc_id: CP-04
title: Phân cấp phê duyệt và thứ tự xử lý hồ sơ
version: v1
effective_date: 2024-01-01
owner: Ban Tín dụng
status: active
language: vi
---

# Phân cấp phê duyệt và thứ tự xử lý hồ sơ

## 1. Thẩm quyền theo số tiền vay

1.1. Số tiền vay từ 200 triệu đồng trở xuống (tính cả 200 triệu): chuyên viên thẩm định có thẩm quyền quyết định duyệt.

1.2. Số tiền vay trên 200 triệu đến 500 triệu đồng (tính cả 500 triệu): chuyên viên không được tự duyệt, phải chuyển lên **Trưởng phòng** phê duyệt.

1.3. Số tiền vay trên 500 triệu đồng: phải chuyển lên **Hội đồng tín dụng** phê duyệt.

## 2. Thứ tự xử lý hồ sơ

Chuyên viên xử lý hồ sơ theo đúng thứ tự dưới đây, dừng ngay ở bước đầu tiên không đạt:

2.1. **Bước 1 – Điều kiện loại trừ:** kiểm tra tuổi, số tiền vay, kỳ hạn theo chính sách CP-01 đúng phiên bản, và tình trạng nợ quá hạn theo CP-06. Không đạt thì **từ chối** (`reject`).

2.2. **Bước 2 – Chứng từ và thông tin:** kiểm tra theo CP-02. Thiếu thì ghi nhận **cần bổ sung** (`need_info`).

2.3. **Bước 3 – Tỷ lệ nợ trên thu nhập:** tính DTI theo CP-05. DTI vượt ngưỡng của CP-01 thì **từ chối** (`reject`), ở mọi mức số tiền vay.

2.4. **Bước 4 – Thẩm quyền:** nếu hồ sơ đạt các bước trên, xét thẩm quyền theo mục 1. Trong thẩm quyền của chuyên viên thì **duyệt** (`approve`); vượt thẩm quyền thì **chuyển cấp trên** (`escalate`).

## 3. Nguyên tắc

3.1. Hồ sơ không đạt DTI không được chuyển cấp trên để xin ngoại lệ. Cấp trên chỉ xét các hồ sơ đã đạt mọi điều kiện.

3.2. Mỗi hồ sơ chỉ có một quyết định tại một thời điểm. Cách ghi nhận quyết định xem CP-07.
