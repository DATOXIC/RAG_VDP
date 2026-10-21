"""Vẽ sơ đồ kiến trúc FinCo, khổ A4 ngang để in: python docs/figures/make_architecture.py
Ra: architecture_print.png (300 dpi) và architecture_print.pdf (vector) cùng thư mục."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11.69, 6.9))
ax.set_xlim(0, 100)
ax.set_ylim(-3.5, 63)
ax.axis("off")

BLUE, RED, GREY, GREEN, AMBER = "#dce8fa", "#f8d9d5", "#eeeeee", "#dff1e4", "#fbefcf"


def box(x, y, w, h, text, fc, fs=10.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec="#444444", lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(p, q, text=None, both=False, dx=0, dy=1.1, fs=9.5):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<->" if both else "->", lw=1.2, color="#333333"))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, fontsize=fs, ha="center", color="#333333",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)


ax.text(50, 61.2, "FinCo – Agent thẩm định hồ sơ vay: kiến trúc tổng thể", ha="center", fontsize=16, fontweight="bold")

# cột trái
box(2, 48, 28, 9.5, "Người dùng\nmake demo / make eval\nREPL chat", GREY)
box(2, 18, 28, 26, "AGENT  (agent.py)\nvòng lặp tự viết\nMAX_STEPS = 15\ntool lỗi trả chuỗi \"Error: ...\"\nnhắc ghi quyết định nếu quên\ntự gọi get_decision sau khi ghi\nprompt: prompts/*.txt", GREEN)
box(2, 3, 28, 9, "mcp_client.py\ntự khám phá tool\nbằng list_tools()", BLUE)
arrow((16, 48), (16, 44.4), "①  câu hỏi / mã hồ sơ", dx=9, dy=-0.2)
arrow((16, 18), (16, 12.4), "③", dx=2, dy=0)

# LLM
box(40, 50, 36, 8, "LLM  (API tương thích OpenAI)\nllm.py: retry, timeout, đếm token", AMBER)
arrow((30, 38), (40, 52.5), both=True)
ax.text(60, 48.1, "②  messages / tool_calls", fontsize=9.5, ha="center", color="#333333")

# ranh giới MCP
ax.plot([35.5, 35.5], [1.0, 46.5], ls="--", color="#b3261e", lw=1.6)
ax.text(34.6, 27, "RANH GIỚI MCP  ·  stdio, JSON-RPC", rotation=90, fontsize=10, color="#b3261e", ha="center", va="center", fontweight="bold")
arrow((30, 7.5), (41, 7.5), "④ gọi tool", both=True, dx=0, dy=-2.3)

# MCP server
ax.add_patch(FancyBboxPatch((41, 1.5), 58, 45, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec="#444444", lw=1.2))
ax.text(70, 43.2, "MCP SERVER  (mcp_server.py, bọc tools.py)", ha="center", fontsize=12.5, fontweight="bold")
box(43, 34, 29, 6, "search_policy   [đọc]", BLUE)
box(43, 26, 29, 6, "get_application   [đọc]", BLUE)
box(43, 18, 29, 6, "calc_dti   [đọc, tính bằng code]", BLUE)
box(43, 9.5, 29, 6.5, "submit_decision   [GHI]\nqua Critic (tùy chọn)\ntrước khi ghi", RED, bold=True, fs=10)
box(43, 3, 29, 5, "get_decision   [đọc → VERIFY]", BLUE)

box(76, 33, 21, 8, "RAG\nBM25 + bge-m3 (hybrid)\n112 chunk, lọc as_of_date", GREEN, fs=10)
box(76, 3, 21, 25, "SQLite\n\nbảng applications\n(21 hồ sơ)\n\nbảng decisions\n(quyết định đã ghi)", GREY)
arrow((72, 37), (76, 37))
arrow((72, 29), (76, 25))
arrow((72, 21), (76, 20))
arrow((72, 12.8), (76, 12.8))
arrow((72, 5.5), (76, 7))
ax.text(74, 13.6, "ghi", fontsize=9, ha="center", color="#b3261e", fontweight="bold")
ax.text(74, 8.0, "⑤", fontsize=9, ha="center", color="#333333")

ax.text(50, -2.8, "⑤ = đọc lại bằng get_decision sau khi ghi.  Nét đứt đỏ: ranh giới MCP. Xanh dương: tool chỉ đọc. Đỏ: tool ghi (duy nhất). Vàng: LLM / critic.", ha="center", fontsize=9.5, color="#555555")
out = Path(__file__).parent
fig.savefig(out / "architecture_print.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out / "architecture_print.pdf", bbox_inches="tight", pad_inches=0.15)
