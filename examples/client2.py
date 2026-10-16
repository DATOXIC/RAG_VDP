"""
HOST CLIENT THỨ 2
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = sys.argv[1] if len(sys.argv) > 1 else "APP-001"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="finco_interop_"))
    db = tmp / "finco.db"
    shutil.copy(ROOT / "data" / "finco.db", db)
    env = dict(os.environ, FINCO_DB=str(db), FINCO_CRITIC="0", FINCO_LOG_LEVEL="WARNING",
               PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "finco.mcp_server"], cwd=ROOT, env=env, text=True, encoding="utf-8",
        bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    nid = 0

    def send(msg):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def rpc(method, params=None):
        nonlocal nid
        nid += 1
        send({"jsonrpc": "2.0", "id": nid, "method": method, "params": params or {}})
        while True:  # bỏ qua thông báo/log không phải response của id này
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("server đóng stdout")
            msg = json.loads(line)
            if msg.get("id") == nid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    def call(name, **args):
        res = rpc("tools/call", {"name": name, "arguments": args})
        return res["content"][0]["text"]

    try:
        info = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "client2-raw-jsonrpc", "version": "0"}})
        print("server:", info["serverInfo"])
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        names = []
        for t in rpc("tools/list")["tools"]:
            names.append(t["name"])
            print(f"tool: {t['name']} | required: {t['inputSchema'].get('required')}")
        assert len(names) >= 2, "cần >= 2 tool"
        assert {"get_application", "submit_decision", "get_decision"} <= set(names)

        print(f"\nget_application({APP}) ->", call("get_application", app_id=APP)[:200].replace("\n", " | "))
        print("search_policy ->", call("search_policy", query="DTI tối đa", top_k=1)[:160].replace("\n", " | "))
        w = call("submit_decision", app_id=APP, decision="need_info", reason="interop test (bản sao DB)",
                 cited_docs=["CP-03"], reason_codes=["MISSING_DOCS"])
        print("submit_decision ->", w[:160].replace("\n", " | "))
        r = call("get_decision", app_id=APP)
        print("get_decision (read-back) ->", r[:160].replace("\n", " | "))
        assert "need_info" in r.lower(), "đọc lại không khớp"
        print("\nINTEROP OK: host thứ hai (JSON-RPC thô) dùng được server FinCo.")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
