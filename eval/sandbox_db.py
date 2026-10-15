import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator
from finco.store import db

ROOT = Path(__file__).parents[1]
# FINCO_SANDBOX_SUFFIX: mỗi worker song song một file sandbox riêng (vd. "_w1"), tránh dẫm lên nhau.
SANDBOX_PATH = ROOT / "data" / f"finco_sandbox{os.getenv('FINCO_SANDBOX_SUFFIX', '')}.db"

@contextmanager
def isolated_db_sandbox(sandbox_path: Path = SANDBOX_PATH):
    """
    Tạo Database giả để Test
    """
    old = os.environ.get("FINCO_DB")

    # Xóa dữ liệu cũ nếu còn
    if sandbox_path.exists():
        sandbox_path.unlink()

    try:
        os.environ["FINCO_DB"] = str(sandbox_path)
        db.init_db()
        db.load_applications()
        yield sandbox_path
    finally:
        if old is None:
            os.environ.pop("FINCO_DB", None)
        else:
            os.environ["FINCO_DB"] = old

        if sandbox_path.exists():
            sandbox_path.unlink()


# if __name__ == "__main__":
#     from finco.types import Decision
#     print("\n" + "=" * 70)
#     print("KIỂM THỬ CƠ CHẾ DATABASE SANDBOX (EVAL/SANDBOX.PY)")
#     print("=" * 70)
#     print(f"[*] Trước khi vào sandbox: FINCO_DB = {os.environ.get('FINCO_DB', 'Mặc định (finco.db)')}")
#     with isolated_db_sandbox() as s_path:
#         print(f"[+] ĐÃ VÀO SANDBOX! FINCO_DB = {os.environ.get('FINCO_DB')}")
#         print(f"[+] File sandbox tồn tại: {s_path.exists()} (Đường dẫn: {s_path.name})")
#         # 1. Kiểm tra số lượng hồ sơ
#         apps = db.list_applications()
#         print(f"[+] Số hồ sơ trong bảng applications: {len(apps)} (Kỳ vọng: 21)")
#         # 2. Kiểm tra bảng decisions có đúng là trống 100% không
#         try:
#             db.get_decision("APP-001")
#             print("[-] LỖI: Bảng decisions không trống!")
#         except db.DecisionNotFound:
#             print("[+] Bảng decisions HOÀN TOÀN TRỐNG (Đạt chuẩn 100%!)")
#         # 3. Thử nghiệm ghi 1 quyết định mẫu vào sandbox
#         dummy_decision = Decision(
#             app_id="APP-001",
#             decision="approve",
#             reason="Thử nghiệm ghi vào sandbox",
#             decided_at="",
#             cited_docs=["CP-01 v1"],
#         )
#         db.insert_decision(dummy_decision)
#         saved = db.get_decision("APP-001")
#         print(f"[+] Ghi quyết định mẫu vào sandbox THÀNH CÔNG: {saved.app_id} -> {saved.decision}")
#     print("\n[-] ĐÃ THOÁT KHỎI SANDBOX!")
#     print(f"[*] Biến FINCO_DB sau khi thoát: {os.environ.get('FINCO_DB', 'Đã phục hồi (Không có biến)')}")
#     print(f"[*] File sandbox đã được xóa dọn dẹp: {not SANDBOX_PATH.exists()}")
#     print("=" * 70 + "\n")