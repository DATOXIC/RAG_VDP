---
doc_id: CP-07
title: Quy trình ghi nhận quyết định thẩm định
version: v1
effective_date: 2024-01-01
owner: Phòng Vận hành tín dụng
status: active
language: vi
---

# Quy trình ghi nhận quyết định thẩm định

## 1. Bốn trạng thái quyết định

1.1. `approve` – duyệt: hồ sơ đạt mọi điều kiện và nằm trong thẩm quyền của chuyên viên.

1.2. `reject` – từ chối: hồ sơ không đạt điều kiện loại trừ hoặc DTI vượt ngưỡng.

1.3. `escalate` – chuyển cấp trên: hồ sơ đạt mọi điều kiện nhưng vượt thẩm quyền chuyên viên, hoặc thuộc nhóm bắt buộc chuyển cấp trên theo CP-06.

1.4. `need_info` – cần bổ sung: hồ sơ thiếu chứng từ hoặc thiếu thông tin theo CP-02.

## 2. Nội dung bắt buộc của mỗi quyết định

2.1. Lý do quyết định, nêu rõ điều kiện làm căn cứ.

2.2. Danh sách tài liệu chính sách đã căn cứ, ghi theo mã tài liệu và phiên bản (ví dụ CP-01 v2, CP-05). Quyết định không có tài liệu căn cứ không được ghi nhận.

2.3. Thời điểm ghi nhận quyết định.

## 3. Nguyên tắc ghi nhận

3.1. Mỗi hồ sơ chỉ có một quyết định. Hồ sơ đã có quyết định thì không ghi thêm quyết định mới; muốn sửa phải có yêu cầu của cấp phê duyệt cao hơn.

3.2. Ngay sau khi ghi nhận, chuyên viên phải đọc lại quyết định từ hệ thống để xác nhận đã lưu đúng trạng thái và nội dung.

3.3. Quyết định `need_info` có thể được thay bằng quyết định khác khi hồ sơ đã bổ sung đầy đủ.
