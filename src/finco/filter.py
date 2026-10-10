from typing import Optional
from finco.types import Chunk
import datetime

def parse_time(date_time: Optional[str]):
    if not date_time or not isinstance(date_time,str):
        raise ValueError("Ngày không được để trống và phải là chuỗi string")
    clean_date = date_time.strip()

    try:
        parsed = datetime.strptime(date_time,"%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Ngày '{date_time}' không đúng định dạng chuẩn YYYY-MM-DD.")



def is_active_chunk(chunk: Chunk, as_of_date: str):
    if chunk.effective_date > as_of_date:
        return False
    if chunk.expiry_date and as_of_date > chunk.expiry_date:
        return False
    return True
    
