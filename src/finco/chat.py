"""
Giao diện dòng lệnh tương tác (REPL Chat) cho Trợ lý Thẩm định FinCo.
Hỗ trợ:
- Xem bảng hồ sơ: 'list'
- Thẩm định hồ sơ: 'APP-xxx' hoặc 'eval APP-xxx'
- Hỏi đáp chính sách: 'ask <câu hỏi>' hoặc câu hỏi tự do
- Tra cứu nhật ký kiểm toán: 'audit <mã_audit>'
- Xem lịch sử các phiên: 'history'
- Thoát: 'exit' / 'q'
"""
import sys
from pathlib import Path

# =========================================================================
# 1. BẢO VỆ ĐƯỜNG DẪN SYS.PATH (Tránh lỗi shadowing module 'types' của Python)
# =========================================================================
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

# Nếu chạy trực tiếp 'python src/finco/chat.py', Python sẽ tự chèn CURRENT_DIR vào sys.path[0].
# Loại bỏ nó để tránh rich/typing import nhầm file finco/types.py thay vì thư viện chuẩn.
if str(CURRENT_DIR) in sys.path:
    sys.path.remove(str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

# =========================================================================
# 2. CÁC THƯ VIỆN CẦN DÙNG
# =========================================================================
import json
from dataclasses import asdict

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from finco.agent import Agent
from finco.audit import (
    generate_audit_id,
    list_recent_audits,
    load_audit_session,
    save_audit_session,
)
from finco.store.db import list_applications

console = Console()


# =========================================================================
# 3. BANNER NHÓM SINH VIÊN
# =========================================================================
def show_banner():
    banner_text = """[bold cyan]HỆ THỐNG TRỢ LÝ THẨM ĐỊNH TÍN DỤNG FINCO (MCP + RAG)[/bold cyan]
[dim]Dự án môn học: Hệ thống Trợ lý Thông minh & Đánh giá Tự động[/dim]

[bold yellow]Nhóm sinh viên thực hiện:[/bold yellow]
  • [bold white]TÔ PHƯƠNG ĐẠT[/bold white]        - MSSV: [green]24110086[/green]
  • [bold white]TRƯƠNG KHẮC NHẬT PHI[/bold white] - MSSV: [green]24110120[/green]
  • [bold white]NGUYỄN ĐĂNG VƯƠNG[/bold white]    - MSSV: [green]24110086[/green]
"""
    console.print(Panel(banner_text, border_style="cyan", title="[bold]FINCO ASSISTANT[/bold]"))
    print("[dim]Gõ '[bold green]help[/bold green]' để xem hướng dẫn các câu lệnh hỗ trợ.[/dim]\n")


# =========================================================================
# 4. BẢNG HIỂN THỊ HỒ SƠ VAY (RICH TABLE)
# =========================================================================
def show_applications_table(limit: int | None = 15):
    """Hiển thị danh sách hồ sơ vay trong database dạng bảng trực quan."""
    apps = list_applications(limit=limit)
    if not apps:
        print("[yellow]Hiện chưa có hồ sơ nào trong cơ sở dữ liệu.[/yellow]")
        return

    table = Table(
        title=f"DANH SÁCH {len(apps)} HỒ SƠ VAY TRONG HỆ THỐNG",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Mã HS", style="cyan bold", justify="center")
    table.add_column("Họ và tên", style="green")
    table.add_column("Tuổi", justify="center")
    table.add_column("Số tiền vay", justify="right", style="yellow")
    table.add_column("Kỳ hạn", justify="center")
    table.add_column("Thu nhập chính", justify="right")
    table.add_column("Ngày nộp", justify="center", style="blue")
    table.add_column("Ghi chú", style="dim")

    for app in apps:
        table.add_row(
            app.app_id,
            app.applicant_name,
            str(app.age),
            f"{app.amount:,} đ",
            f"{app.term_months} tháng",
            f"{app.monthly_income:,} đ",
            app.submitted_date,
            app.note or "Không có",
        )
    console.print(table)


# =========================================================================
# 5. HIỂN THỊ CHI TIẾT NHẬT KÝ KIỂM TOÁN (TREE VIEW CHO AUDIT ID)
# =========================================================================
def show_audit_view(audit_id: str):
    """Đọc và vẽ sơ đồ cây suy nghĩ / hành động của một phiên kiểm toán."""
    data = load_audit_session(audit_id)
    if not data:
        print(f"[red]✗ Không tìm thấy nhật ký kiểm toán cho mã: [bold]{audit_id}[/bold][/red]")
        print("[dim]Gõ 'history' để xem các mã kiểm toán gần đây.[/dim]")
        return

    # Panel tóm tắt
    status_color = "green" if data["status"] == "SUCCESS" else "red"
    summary_text = (
        f"[bold]Mã kiểm toán :[/bold] [cyan]{data['audit_id']}[/cyan]\n"
        f"[bold]Thời gian    :[/bold] {data['created_at']}\n"
        f"[bold]Đối tượng    :[/bold] [yellow]{data['app_id']}[/yellow]\n"
        f"[bold]Yêu cầu gốc  :[/bold] {data['query']}\n"
        f"[bold]Trạng thái   :[/bold] [{status_color}]{data['status']}[/{status_color}]\n"
        f"[bold]Công cụ dùng :[/bold] {', '.join(data['tools_called']) or 'Không gọi tool'}"
    )
    console.print(Panel(summary_text, title=f"TÓM TẮT PHIÊN {data['audit_id']}", border_style="blue"))

    # Cây diễn giải từng bước ReAct (Thought -> Action -> Observation)
    steps = data.get("steps_detail", [])
    if steps:
        tree = Tree(f"[bold magenta]Hành trình thẩm định ({len(steps)} bước)[/bold magenta]")
        for s in steps:
            step_no = s.get("step_number", 1)
            tool = s.get("tool_name", "Không có tool")
            step_node = tree.add(f"[bold yellow]Bước {step_no}: Gọi {tool}[/bold yellow]")
            
            # Thought
            thought = s.get("thought", "").strip()
            if thought:
                step_node.add(f"[italic cyan]Suy nghĩ (Thought):[/italic cyan] {thought}")
            
            # Action args
            args = s.get("tool_args", {})
            step_node.add(f"[green]Tham số (Args):[/green] {json.dumps(args, ensure_ascii=False)}")
            
            # Observation
            obs = s.get("observation", "").strip()
            obs_preview = obs if len(obs) <= 300 else obs[:300] + "..."
            step_node.add(f"[dim]Kết quả tool (Observation):[/dim] {obs_preview}")

        console.print(tree)

    # Kết luận cuối
    if data.get("final_answer"):
        console.print(
            Panel(
                data["final_answer"],
                title="[bold green]KẾT LUẬN CUỐI CÙNG TỪ AI[/bold green]",
                border_style="green",
            )
        )


def show_recent_history():
    """Hiển thị bảng các mã audit gần nhất để tiện tra cứu."""
    sessions = list_recent_audits(limit=10)
    if not sessions:
        print("[yellow]Chưa có phiên làm việc nào được lưu.[/yellow]")
        return

    table = Table(title="LỊCH SỬ CÁC PHIÊN KIỂM TOÁN GẦN NHẤT", show_lines=True, header_style="bold cyan")
    table.add_column("Mã Audit ID", style="cyan bold", justify="center")
    table.add_column("Thời gian", justify="center")
    table.add_column("Hồ sơ", justify="center", style="yellow")
    table.add_column("Trạng thái", justify="center")
    table.add_column("Số tool gọi", justify="right")
    table.add_column("Yêu cầu", style="dim")

    for s in sessions:
        st_style = "[green]SUCCESS[/green]" if s["status"] == "SUCCESS" else "[red]FAILED[/red]"
        table.add_row(
            s["audit_id"],
            s["created_at"],
            s["app_id"],
            st_style,
            str(s["tools_count"]),
            s["query"],
        )
    console.print(table)
    print("[dim]Dùng lệnh '[bold cyan]audit <MÃ_AUDIT>[/bold cyan]' để xem chi tiết từng bước.[/dim]\n")


def show_help():
    """Hướng dẫn cú pháp lệnh."""
    help_text = """[bold cyan]CÁC CÂU LỆNH HỖ TRỢ:[/bold cyan]
  • [bold green]list[/bold green]                 : Xem danh sách các hồ sơ vay hiện có trong DB
  • [bold green]APP-xxx[/bold green]              : Thẩm định hồ sơ vay theo mã (vd: [cyan]APP-003[/cyan])
  • [bold green]eval APP-xxx[/bold green]         : Cú pháp tương đương để thẩm định hồ sơ
  • [bold green]ask <câu hỏi>[/bold green]        : Hỏi đáp về chính sách tín dụng FinCo
  • [bold green]audit <MÃ_AUDIT>[/bold green]    : Xem lại nhật ký suy nghĩ & tool call của một phiên
  • [bold green]history[/bold green]              : Xem danh sách các mã Audit ID gần đây
  • [bold green]help[/bold green]                 : Hiển thị bảng trợ giúp này
  • [bold green]exit[/bold green] / [bold green]quit[/bold green] / [bold green]q[/bold green]    : Thoát chương trình
"""
    console.print(Panel(help_text, border_style="yellow", title="HƯỚNG DẪN SỬ DỤNG"))


# =========================================================================
# 6. VÒNG LẶP CHÍNH (INTERACTIVE REPL)
# =========================================================================
def main():
    show_banner()
    agent = Agent(verbose=True)

    while True:
        try:
            raw_input = Prompt.ask("\n[bold green]finco>[/bold green]")
        except (KeyboardInterrupt, EOFError):
            print("\n[yellow]Đã thoát chương trình. Tạm biệt bạn![/yellow]")
            break

        query = raw_input.strip()
        if not query:
            continue

        cmd_lower = query.lower()

        # 1. Các lệnh điều khiển hệ thống
        if cmd_lower in ("exit", "quit", "q"):
            print("[yellow]Đã thoát chương trình. Tạm biệt bạn![/yellow]")
            break

        if cmd_lower == "help":
            show_help()
            continue

        if cmd_lower == "list":
            show_applications_table()
            continue

        if cmd_lower == "history":
            show_recent_history()
            continue

        if cmd_lower.startswith("audit"):
            parts = query.split(maxsplit=1)
            if len(parts) < 2:
                print("[yellow]Vui lòng nhập mã kiểm toán. Ví dụ: audit FINCO-1A2B3C4D[/yellow]")
            else:
                show_audit_view(parts[1])
            continue

        # 2. Xử lý yêu cầu chạy Agent (Thẩm định hoặc Hỏi đáp)
        audit_id = generate_audit_id()
        app_id_target = "Q"
        actual_question = query

        # Phân loại lệnh eval / APP-xxx
        if query.upper().startswith("APP-"):
            app_id_target = query.upper()
            actual_question = f"Hãy thẩm định hồ sơ {app_id_target} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
        elif cmd_lower.startswith("eval"):
            parts = query.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip().upper().startswith("APP-"):
                app_id_target = parts[1].strip().upper()
                actual_question = f"Hãy thẩm định hồ sơ {app_id_target} theo quy trình chính sách FinCo và ghi nhận quyết định vào hệ thống."
            else:
                print("[yellow]Cú pháp: eval <MÃ_HỒ_SƠ>, ví dụ: eval APP-003[/yellow]")
                continue
        elif cmd_lower.startswith("ask"):
            parts = query.split(maxsplit=1)
            actual_question = parts[1] if len(parts) > 1 else query

        console.print(f"\n[cyan]⚙ Đang thực thi phiên kiểm toán [bold]{audit_id}[/bold]...[/cyan]\n")

        # Chạy Agent
        result = agent.run_sync(app_id=app_id_target, question=actual_question)


        result.save_log() # eval/verifier.py, eval/run_eval.py

        _, log_path = save_audit_session(result, query=query, audit_id=audit_id) # Người dùng

        # In kết quả cuối cùng
        console.print(
            Panel(
                result.final_answer,
                title="[bold green]=== KẾT QUẢ THẨM ĐỊNH TỪ AI ===[/bold green]",
                border_style="green",
            )
        )

        # In thông tin kiểm toán và mã tra cứu
        audit_box = (
            f"[bold green]✓ Đã hoàn tất và lưu nhật ký kiểm toán:[/bold green]\n"
            f"  • Mã tra cứu : [bold cyan]{audit_id}[/bold cyan]\n"
            f"  • File lưu   : [dim]{log_path}[/dim]\n"
            f"  • Token Dùng : {result.usage.calls} lượt gọi | {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out\n"
            f"[italic yellow]Bạn có thể gõ '[bold cyan]audit {audit_id}[/bold cyan]' bất kỳ lúc nào để xem lại cây suy nghĩ và tool call![/italic yellow]"
        )
        console.print(Panel(audit_box, border_style="cyan"))


if __name__ == "__main__":
    main()