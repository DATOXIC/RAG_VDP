"""Lớp duy nhất nói chuyện với SQLite (bảng applications và decisions).

Quy tắc dùng chung:
- Không dùng print() trong file này: MCP server dùng stdout để truyền dữ liệu, một dòng print sẽ làm hỏng nó.
- Hàm lỗi thì ném ngoại lệ có tên ở dưới. Việc đổi thành chuỗi "Error: ..." cho agent do tầng tool làm.
- Mỗi hàm tự mở và tự đóng kết nối (REPL và MCP server là hai tiến trình cùng dùng một file DB).
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from finco.types import Application, Decision

# --------------------------------------------------------------------------- CẤU HÌNH
# Tính đường dẫn từ vị trí file này, không phụ thuộc thư mục đang chạy lệnh.
# Gốc dự án = src/finco/store/db.py đi lên 3 cấp.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
APPLICATIONS_JSON = PROJECT_ROOT / "data" / "applications.json"
VALID_DECISIONS = ("approve", "reject", "escalate", "need_info")


def get_db_path() -> Path:
    """Đường dẫn file DB. Đặt biến môi trường FINCO_DB để dùng DB khác (ví dụ bản sao sạch khi chạy eval)."""
    return Path(os.environ.get("FINCO_DB", PROJECT_ROOT / "data" / "finco.db"))


# --------------------------------------------------------------------------- NGOẠI LỆ
class AppNotFound(Exception):
    """Không có hồ sơ với app_id này."""
class DecisionExists(Exception):
    """Hồ sơ này đã có quyết định, không ghi thêm."""
class DecisionNotFound(Exception):
    """Hồ sơ này chưa có quyết định."""
class InvalidDecision(Exception):
    """Quyết định không hợp lệ (trạng thái lạ, hoặc không trích tài liệu)."""


# --------------------------------------------------------------------------- KẾT NỐI
@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Mở kết nối, tự commit nếu thành công, tự rollback nếu lỗi, luôn đóng khi xong."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # truy cập cột theo tên: row["app_id"]
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------- TẠO BẢNG
def init_db() -> None:
    """Tạo 2 bảng nếu chưa có. Gọi lại nhiều lần vẫn an toàn và KHÔNG xóa dữ liệu cũ."""
    get_db_path().parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            app_id TEXT PRIMARY KEY,               -- Mã hồ sơ (vd: 'APP-001')
            applicant_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            product TEXT NOT NULL,
            amount INTEGER NOT NULL,               -- VNĐ
            term_months INTEGER NOT NULL,
            income_type TEXT NOT NULL,             -- salary | self_employed
            monthly_income INTEGER NOT NULL,
            supplementary_income INTEGER NOT NULL,
            existing_debt_monthly INTEGER,         -- NULL = chưa khai báo (khác với 0)
            is_employee INTEGER NOT NULL,          -- 0 | 1
            overdue_30d_12m INTEGER NOT NULL,      -- 0 | 1
            docs_submitted TEXT NOT NULL,          -- danh sách, lưu dạng chuỗi JSON
            submitted_date TEXT NOT NULL,          -- 'YYYY-MM-DD'
            note TEXT NOT NULL DEFAULT ''
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            app_id TEXT PRIMARY KEY,               -- khóa chính: mỗi hồ sơ chỉ 1 quyết định
            decision TEXT NOT NULL CHECK (decision IN ('approve','reject','escalate','need_info')),
            reason TEXT NOT NULL,
            cited_docs TEXT NOT NULL,              -- danh sách, lưu dạng chuỗi JSON
            decided_at TEXT NOT NULL
        );
        """)


# --------------------------------------------------------------------------- HỒ SƠ VAY
def _row_to_application(row: sqlite3.Row) -> Application:
    """Đổi một dòng DB thành Application: khôi phục list và bool. existing_debt_monthly giữ nguyên None."""
    data = dict(row)
    data["docs_submitted"] = json.loads(data["docs_submitted"])
    data["is_employee"] = bool(data["is_employee"])
    data["overdue_30d_12m"] = bool(data["overdue_30d_12m"])
    return Application(**data)


def load_applications(json_path: str | Path = APPLICATIONS_JSON) -> int:
    """Nạp applications.json vào bảng. Hồ sơ đã có thì bỏ qua (chạy lại không bị nhân đôi).

    Returns:
        Số hồ sơ mới được thêm.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Qua Application(**a) để chắc chắn dữ liệu khớp dataclass (thiếu/thừa trường sẽ báo lỗi ngay).
    apps = [Application(**a) for a in raw]

    sql = """
    INSERT OR IGNORE INTO applications (
        app_id, applicant_name, age, product, amount, term_months,
        income_type, monthly_income, supplementary_income,
        existing_debt_monthly, is_employee, overdue_30d_12m,
        docs_submitted, submitted_date, note
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _connect() as conn:
        before = conn.total_changes
        for a in apps:
            conn.execute(sql, (
                a.app_id, a.applicant_name, a.age, a.product, a.amount, a.term_months,
                a.income_type, a.monthly_income, a.supplementary_income,
                a.existing_debt_monthly, int(a.is_employee), int(a.overdue_30d_12m),
                json.dumps(a.docs_submitted), a.submitted_date, a.note,
            ))
        return conn.total_changes - before


def get_application(app_id: str) -> Application:
    """Lấy một hồ sơ. Không có thì ném AppNotFound."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM applications WHERE app_id = ?", (app_id,)).fetchone()
    if row is None:
        raise AppNotFound(app_id)
    return _row_to_application(row)


def list_applications(limit: int | None = None) -> list[Application]:
    """Lấy danh sách hồ sơ, sắp theo app_id. limit=None là lấy tất cả."""
    sql = "SELECT * FROM applications ORDER BY app_id ASC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [_row_to_application(r) for r in rows]


# --------------------------------------------------------------------------- QUYẾT ĐỊNH
def insert_decision(decision: Decision) -> Decision:
    """Ghi quyết định. Trả lại đúng quyết định đã ghi (decided_at được điền nếu để trống).

    Ném:
        InvalidDecision: trạng thái không hợp lệ, hoặc cited_docs rỗng.
        AppNotFound: hồ sơ không tồn tại.
        DecisionExists: hồ sơ đã có quyết định.
    """
    if decision.decision not in VALID_DECISIONS:
        raise InvalidDecision(f"decision phải là một trong {VALID_DECISIONS}, nhận được '{decision.decision}'")
    
    if not decision.cited_docs:
        raise InvalidDecision("cited_docs rỗng: quyết định phải trích ít nhất một tài liệu chính sách")

    if not decision.decided_at:
        decision.decided_at = datetime.now().isoformat(timespec="seconds")

    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM applications WHERE app_id = ?", (decision.app_id,)).fetchone()
        if exists is None:
            raise AppNotFound(decision.app_id)
        try:
            conn.execute(
                "INSERT INTO decisions (app_id, decision, reason, cited_docs, decided_at) VALUES (?, ?, ?, ?, ?)",
                (decision.app_id, decision.decision, decision.reason,
                 json.dumps(decision.cited_docs), decision.decided_at),
            )
        except sqlite3.IntegrityError:  # trùng khóa chính app_id
            raise DecisionExists(decision.app_id)
    return decision


def get_decision(app_id: str) -> Decision:
    """Đọc quyết định đang lưu (dùng cho bước verify sau khi ghi). Chưa có thì ném DecisionNotFound."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM decisions WHERE app_id = ?", (app_id,)).fetchone()
    if row is None:
        raise DecisionNotFound(app_id)
    return Decision(
        app_id=row["app_id"],
        decision=row["decision"],
        reason=row["reason"],
        decided_at=row["decided_at"],
        cited_docs=json.loads(row["cited_docs"]),
    )


def reset_decisions() -> None:
    """Xóa toàn bộ quyết định (lệnh /reset). Hồ sơ vay giữ nguyên."""
    with _connect() as conn:
        conn.execute("DELETE FROM decisions")

if __name__ == "__main__":
    init_db()
    added = load_applications()
    print(f"DB: {get_db_path()} | thêm mới {added} hồ sơ | tổng {len(list_applications())}")
