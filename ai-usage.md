```
def corpus_hash(chunks: list[Chunk]) -> str:
    """Mã băm nội dung toàn bộ chunk, dùng để phát hiện index cũ so với dữ liệu mới."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
```

```
    def _valid_mask(self, as_of_date: str) -> np.ndarray:
        """Tạo một mảng Boolean dựa trên một thời điểm xác định
        
        TRUE nếu còn hiệu lực
        FALSE ngược lại 
        """
        return np.array(
            [
                c.effective_date <= as_of_date
                and (c.expiry_date is None or as_of_date <= c.expiry_date)
                for c in self.chunks
            ],
            dtype=bool,
        )
```