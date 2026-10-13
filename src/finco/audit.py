from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from finco.agent import AgentRunResult
from finco.types import Usage

# Thư mục lưu trữ các phiên kiểm toán: logs/sessions/
REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_LOG_DIR = REPO_ROOT / "logs" / "sessions"


def generate_audit_id() -> str:
    """Sinh mã duy nhất cho mỗi phiên (vd: FINCO-8A1C2F9B)."""
    uid = uuid.uuid4().hex[:8].upper()
    return f"FINCO-{uid}"


def save_audit_session(
    result: AgentRunResult,
    query: str,
    audit_id: Optional[str] = None,
    log_dir: Path | str = SESSIONS_LOG_DIR,
) -> tuple[str, Path]:
    """Lưu toàn bộ chi tiết phiên làm việc của Agent ra file JSON 2 tầng.
    Tầng 1 (Summary): Thông tin tổng quan để tìm kiếm nhanh và vẽ timeline.
    Tầng 2 (Detail): Từng bước ReAct chi tiết gồm Thought, Action, Observation.
    Trả về:
        (audit_id, path_to_file)
    """
    audit_id = audit_id or generate_audit_id()
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{audit_id}.json"

    # Trích xuất timeline tóm tắt các tool đã gọi
    timeline = []
    for step in result.steps:
        if step.tool_name:
            obs_preview = step.observation.replace("\n", " ").strip()
            if len(obs_preview) > 150:
                obs_preview = obs_preview[:150] + "..."
            timeline.append({
                "step": step.step_number,
                "tool": step.tool_name,
                "args": step.tool_args,
                "observation_preview": obs_preview,
            })

    # Đóng gói dữ liệu kiểm toán đầy đủ 2 tầng
    data = {
        "audit_id": audit_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_id": result.app_id,
        "run_id": result.run_id,  # khóa sang logs/run_<run_id>.jsonl
        "query": query,
        "status": "SUCCESS" if result.success else "FAILED",
        "error_message": result.error_message,
        "usage": asdict(result.usage) if result.usage else {},
        "tools_called": result.tool_calls,
        "timeline": timeline,
        "steps_detail": [asdict(s) for s in result.steps],
        "final_answer": result.final_answer,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return audit_id, file_path


def load_audit_session(audit_id: str, log_dir: Path | str = SESSIONS_LOG_DIR) -> Optional[dict[str, Any]]:
    """Đọc file kiểm toán theo mã audit_id (hỗ trợ nhập 'FINCO-XXXX' hoặc chỉ 'XXXX')."""
    raw_id = audit_id.strip().upper()
    if not raw_id.startswith("FINCO-"):
        raw_id = f"FINCO-{raw_id}"

    file_path = Path(log_dir) / f"{raw_id}.json"
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_recent_audits(limit: int = 10, log_dir: Path | str = SESSIONS_LOG_DIR) -> list[dict[str, Any]]:
    """Liệt kê các phiên kiểm toán gần nhất sắp xếp theo thời gian mới nhất."""
    folder = Path(log_dir)
    if not folder.exists():
        return []

    files = sorted(folder.glob("FINCO-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries = []
    for fp in files[:limit]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append({
                    "audit_id": data.get("audit_id", fp.stem),
                    "created_at": data.get("created_at", ""),
                    "app_id": data.get("app_id", ""),
                    "query": data.get("query", "")[:50],
                    "status": data.get("status", ""),
                    "tools_count": len(data.get("tools_called", [])),
                })
        except Exception:
            continue
    return summaries