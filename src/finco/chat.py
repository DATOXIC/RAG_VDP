from dataclasses import asdict

from rich import print
from rich.table import Table
from rich.prompt import Prompt
from finco.agent import Agent
from finco.store.db import list_applications
import json

print("=========================================================")
print("[bold blue] TÔ PHƯƠNG ĐẠT - 24110086 [/bold blue]")
print("[bold blue] TRƯƠNG KHẮC NHẬT PHI - 24110120 [/bold blue]")
print("[bold blue] NGUYỄN ĐĂNG VƯƠNG - 24110086 [/bold blue]")
print("=========================================================")

def show_applications_table(limit: int | None = 10):

    apps = list_applications(limit=limit)
    
    if not apps:
        print("[yellow]Hiện chưa có hồ sơ nào trong cơ sở dữ liệu.[/yellow]")
        return

    title = f"DANH SÁCH {len(apps)} HỒ SƠ VAY TRONG HỆ THỐNG"
    table = Table(title=title, show_lines=True, header_style="bold magenta")


    table.add_column("Mã HS", style="cyan bold", justify="center")
    table.add_column("Họ và tên", style="green")
    table.add_column("Tuổi", justify="center")
    table.add_column("Số tiền vay", justify="right", style="yellow")
    table.add_column("Kỳ hạn", justify="center")
    table.add_column("Thu nhập/tháng", justify="right")
    table.add_column("Ngày nộp", justify="center", style="blue")
    table.add_column("Ghi chú", style="dim")


    for app in apps:
        table.add_row(
            app.app_id,
            app.applicant_name,
            str(app.age),
            f"{app.amount:,} đ",          # Định dạng 150,000,000 đ
            f"{app.term_months} tháng",
            f"{app.monthly_income:,} đ",
            app.submitted_date,
            app.note or "Không có",
        )
    # 5. In bảng ra màn hình
    print(table)

agent = Agent(verbose=True)

while True:
    user_query = Prompt.ask("\n[bold green]Bạn cần hỏi gì? (gõ 'exit' để thoát)[/bold green]")
    
    # Điều kiện thoát
    if user_query.strip().lower() in ("exit", "quit", "q"):
        print("[yellow]Tạm biệt bạn![/yellow]")
        break
    
    # Bỏ qua nếu người dùng bấm Enter rỗng
    if not user_query.strip():
        continue
    print(f"\n[cyan]Đang xử lý yêu cầu:[/cyan] {user_query}...\n")
    
    # 3. Phân biệt người dùng nhập mã hồ sơ hay câu hỏi tự do
    if user_query.strip().upper().startswith("APP-"):
        # Nhập mã hồ sơ (ví dụ APP-003)
        app_id = user_query.strip().upper()
        result = agent.run_sync(app_id=app_id)
    else:
        # Câu hỏi chính sách tự do
        result = agent.run_sync(app_id="Q", question=user_query)
        
    # 4. In câu trả lời cuối cùng
    print("\n[bold blue]=== KẾT QUẢ TỪ AI ===[/bold blue]")
    print(result.final_answer)
    print("\n[bold blue]=== USAGE ===[/bold blue]")
    print((asdict(result.usage)))


