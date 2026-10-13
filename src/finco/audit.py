"""Nhật ký kiểm toán cho REPL chat: mỗi lượt chạy một file logs/sessions/FINCO-XXXXXXXX.json.

Nội dung lấy từ trace của lượt chạy; log máy (jsonl, có USAGE) nằm riêng ở logs/run_<run_id>.jsonl.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from finco.agent import AgentRunResult

SESSIONS_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "sessions"


def generate_audit_id() -> str:
    """Mã duy nhất cho mỗi phiên, ví dụ FINCO-8A1C2F9B."""
    return f"FINCO-{uuid.uuid4().hex[:8].upper()}"


def save_audit_session(
    result: AgentRunResult,
    query: str,
    audit_id: Optional[str] = None,
    log_dir: Path | str = SESSIONS_LOG_DIR,
) -> tuple[str, Path]:
    """Lưu một lượt chạy ra file JSON. Trả (audit_id, đường dẫn file)."""
    audit_id = audit_id or generate_audit_id()
    path = Path(log_dir) / f"{audit_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "audit_id": audit_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_id": result.app_id,
        "run_id": result.run_id,
        "query": query,
        "status": "SUCCESS" if result.success else "FAILED",
        "error_message": result.error_message,
        "usage": asdict(result.usage),
        "tools_called": result.tool_calls,
        "steps_detail": [
            {"step_number": line.step, "tool_name": line.tool, "tool_args": line.args, "observation": line.result}
            for line in result.trace
        ],
        "final_answer": result.final_answer,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit_id, path


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_audit_session(audit_id: str, log_dir: Path | str = SESSIONS_LOG_DIR) -> Optional[dict[str, Any]]:
    """Đọc phiên theo mã; chấp nhận cả 'FINCO-XXXX' lẫn 'XXXX'."""
    code = audit_id.strip().upper()
    if not code.startswith("FINCO-"):
        code = f"FINCO-{code}"
    return _read(Path(log_dir) / f"{code}.json")


def list_recent_audits(limit: int = 10, log_dir: Path | str = SESSIONS_LOG_DIR) -> list[dict[str, Any]]:
    """Các phiên gần nhất (mới nhất trước), mỗi phiên một dòng tóm tắt."""
    files = sorted(Path(log_dir).glob("FINCO-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in files[:limit]:
        data = _read(path)
        if data:
            rows.append({
                "audit_id": data.get("audit_id", path.stem),
                "created_at": data.get("created_at", ""),
                "app_id": data.get("app_id", ""),
                "query": data.get("query", "")[:50],
                "status": data.get("status", ""),
                "tools_count": len(data.get("tools_called", [])),
            })
    return rows
