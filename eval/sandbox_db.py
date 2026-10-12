import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator
from finco.store import db

ROOT = Path(__file__).parents[1]
SANDBOX_PATH = ROOT / "data" / "finco_sandbox.db"

@contextmanager
def isolated_db_sandbox(sandbox_path: Path = SANDBOX_PATH):
    """
    Tạo Database giả để Test
    """
    old = os.getevn.get("FINCO_DB")

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