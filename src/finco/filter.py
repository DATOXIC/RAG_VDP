from typing import Optional
from finco.types import Chunk
import numpy as np
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



def is_active_chunk(chunks: list[Chunk], as_of_date: str):
    return np.array(
                [
                    c.effective_date <= as_of_date
                    and (c.expiry_date is None or as_of_date <= c.expiry_date)
                    for c in chunks
                ],
                dtype=bool,
            )


    
