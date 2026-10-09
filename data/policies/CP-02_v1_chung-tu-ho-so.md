---
doc_id: CP-02
title: Quy định chứng từ hồ sơ vay
version: v1
effective_date: 2024-01-01
owner: Phòng Vận hành tín dụng
status: active
language: vi
---

# Quy định chứng từ hồ sơ vay

## 1. Chứng từ chung cho mọi hồ sơ

1.1. Căn cước công dân còn hiệu lực (mã chứng từ `CCCD`).

## 2. Nhóm người hưởng lương (salary)

2.1. Hợp đồng lao động còn hiệu lực (mã `HDLD`).

2.2. Sao kê tài khoản nhận lương của **3 tháng gần nhất** (mã `SK_LUONG_3T`).

2.3. Hồ sơ nhóm này đủ chứng từ khi có cả ba mã: `CCCD`, `HDLD`, `SK_LUONG_3T`.

## 3. Nhóm người tự kinh doanh (self_employed)

3.1. Giấy đăng ký kinh doanh còn hiệu lực (mã `DKKD`).

3.2. Sao kê tài khoản kinh doanh của **6 tháng gần nhất** (mã `SK_TK_6T`).

3.3. Hồ sơ nhóm này đủ chứng từ khi có cả ba mã: `CCCD`, `DKKD`, `SK_TK_6T`.

## 4. Khai báo nghĩa vụ nợ hiện tại

4.1. Người vay phải khai báo tổng số tiền trả nợ hàng tháng của các khoản vay hiện có. Nếu không có khoản vay nào thì khai là 0 đồng.

4.2. Hồ sơ không có khai báo nghĩa vụ nợ (để trống, khác với khai 0 đồng) được coi là chưa đủ thông tin để tính DTI.

## 5. Xử lý hồ sơ thiếu chứng từ hoặc thiếu thông tin

5.1. Hồ sơ thiếu một hoặc nhiều chứng từ, hoặc thiếu khai báo nghĩa vụ nợ, **không bị từ chối** mà được ghi nhận trạng thái `need_info` (cần bổ sung), kèm danh sách mục còn thiếu.

5.2. Việc xét DTI và thẩm quyền chỉ tiến hành sau khi hồ sơ đã đủ chứng từ và thông tin.
