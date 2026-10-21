import sys

# Đảm bảo in tiếng Việt chuẩn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_diff(file1: str, file2: str) -> str:
    """So sánh sự khác biệt giữa hai tài liệu."""
    with open(file1, "r", encoding="utf-8") as f1:
        lines1 = [line.strip() for line in f1 if line.strip()]

    with open(file2, "r", encoding="utf-8") as f2:
        lines2 = [line.strip() for line in f2 if line.strip()]

    diff = []

    # 1. Những dòng bị xóa hoặc thay đổi (chỉ có ở file 1)
    for line in lines1:
        if line not in lines2:
            diff.append(f"- {line}")

    # 2. Những dòng được thêm mới (chỉ có ở file 2)
    for line in lines2:
        if line not in lines1:
            diff.append(f"+ {line}")

    return "\n".join(diff)


if __name__ == "__main__":
    f1 = "data/policies/CP-01_v1_chinh-sach-cho-vay.md"
    f2 = "data/policies/CP-01_v2_chinh-sach-cho-vay.md"

    print("=" * 60)
    print("SO SÁNH THAY ĐỔI: CP-01 v1 vs CP-01 v2")
    print("=" * 60)
    print(get_diff(f1, f2))
