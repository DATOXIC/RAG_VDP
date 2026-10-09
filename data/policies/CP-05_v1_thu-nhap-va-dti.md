---
doc_id: CP-05
title: Cách tính thu nhập đủ điều kiện và tỷ lệ nợ trên thu nhập (DTI)
version: v1
effective_date: 2024-01-01
owner: Phòng Quản lý rủi ro tín dụng
status: active
language: vi
---

# Cách tính thu nhập đủ điều kiện và tỷ lệ nợ trên thu nhập (DTI)

## 1. Thu nhập đủ điều kiện

1.1. Nhóm hưởng lương (`salary`): thu nhập đủ điều kiện hàng tháng bằng **thu nhập chính** cộng **70%** thu nhập phụ (phụ cấp, thưởng định kỳ).

1.2. Nhóm tự kinh doanh (`self_employed`): thu nhập đủ điều kiện hàng tháng bằng **80%** thu nhập chính cộng **50%** thu nhập phụ.

1.3. Thu nhập chính và thu nhập phụ là số bình quân hàng tháng đã được xác minh từ chứng từ theo CP-02.

## 2. Khoản trả nợ hàng tháng của khoản vay mới

2.1. Khoản trả hàng tháng theo phương thức niên kim:

`PMT = P × r / (1 − (1 + r)^(−n))`

trong đó P là số tiền vay, r là lãi suất năm chia 12 (lãi suất theo kỳ hạn tại CP-03), n là số tháng của kỳ hạn.

2.2. Phí thẩm định và phí khác không tính vào khoản trả hàng tháng khi xét DTI.

## 3. Công thức DTI

3.1. `DTI = (PMT của khoản vay mới + nghĩa vụ nợ hiện tại hàng tháng) / thu nhập đủ điều kiện hàng tháng`

3.2. Nghĩa vụ nợ hiện tại hàng tháng là số tiền người vay đã khai báo theo CP-02 mục 4.

3.3. DTI được so sánh với ngưỡng tối đa của CP-01 theo đúng phiên bản áp dụng. DTI **bằng** ngưỡng thì đạt; chỉ DTI **lớn hơn** ngưỡng mới bị từ chối. Khi so sánh dùng giá trị chưa làm tròn; khi báo cáo có thể làm tròn đến một chữ số thập phân của phần trăm.

## 4. Ví dụ minh họa (không phải hồ sơ thật)

4.1. Vay 100 triệu đồng, 12 tháng, lãi suất 14,0%/năm thì PMT khoảng 8,98 triệu đồng mỗi tháng. Với thu nhập đủ điều kiện 30 triệu đồng và nợ hiện tại 2 triệu đồng thì DTI = (8,98 + 2) / 30 ≈ 36,6%.
