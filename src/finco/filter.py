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

def _validate_date(date_str: str) -> bool:
    """Kiểm tra chuỗi ngày có đúng định dạng YYYY-MM-DD hay không."""
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except (ValueError, AttributeError):
        return False


def bit_mask(chunks: list[Chunk], as_of_date: str):
    """
        Returns:
            Mảng array True False 
    """
    return np.array(
                [
                    c.effective_date <= as_of_date
                    and (c.expiry_date is None or as_of_date <= c.expiry_date)
                    for c in chunks
                ],
                dtype=bool,
            )

def get_top_valid_indices(scores: np.ndarray, chunks: list[Chunk], as_of_date: str, top_k: int = 5) -> np.array:
    """
    Nếu mảng array đang chứa các con số mà truyền vào mảng nhị phân thỉ nó chỉ giữ lại các con số ứng với trị trí nó là số 1. Nếu một mảng chứa con số mà truyền vào mảng chỉ số index 1,2,3,4 thì nó cũng giữ các con số giá trị tương ứng với index. Nếu một mảng chứa các index mà truyền vào một mảng index nữa thì nó trở thành phép ánh xạ
        Returns:
            np.ndarray: Mảng chứa các index gốc của Top-K chunk hợp lệ (ví dụ: [0, 4]).
    """
    if top_k <= 0 or scores.size == 0:
        return np.empty(0, dtype=int)
    
    # 1. Khởi tạo danh sách ứng viên ban đầu [0, 1, ..., N-1]
    candidates = np.arange(len(chunks))

    # 2. Lọc theo ngày hiệu lực (nếu có truyền as_of_date)
    if as_of_date is not None:
        mask = bit_mask(chunks, as_of_date)
        candidates = candidates[mask]

    # 3. Nếu không có chunk nào còn hiệu lực -> trả về rỗng ngay
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    
    # 4. Lấy điểm số của riêng các ứng viên hợp lệ
    candidate_scores = scores[candidates]

    # 5. Sắp xếp điểm giảm dần và lấy Top-K
    sorted_local_indices = np.argsort(candidate_scores)[::-1][:top_k]

    # 6. Ánh xạ ngược lại index gốc trong kho chunks
    return candidates[sorted_local_indices]

    
