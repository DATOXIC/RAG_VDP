# Chỉ mục log

Mọi file trong thư mục này là log thật của một lần chạy (`ts`, bước, tool, tham số, kết quả, `usage`). Không file nào bị xóa: chúng ghi lại quá trình lặp từ prompt đầu tiên đến cấu hình cuối. Chấm lại bất kỳ file nào không cần khóa API:

```bash
make eval-score LOG=logs/<tên file>.jsonl
```

Thời gian trong tên file là giờ bắt đầu chạy (`YYYYMMDD_HHMMSS`). Log gộp (`*_merged`) do ghép các shard chạy song song bằng nhiều khóa API (để tránh giới hạn token ngày của Groq) nên không phải một tiến trình duy nhất.

## Tóm tắt: 5 nhóm kết quả (đều 37 ca, một lần chạy mỗi cấu hình)

| Nhóm | Câu hỏi | Cấu hình (đạt/37) | Kết luận |
|---|---|---|---|
| 1. Model | Kích thước model | **A** 120b: **34** · H 20b (Groq): 33 · C Qwen 7B: 10 · E Qwen 14B: 8 | gpt-oss tốt hơn hẳn Qwen; 14B không hơn 7B |
| 2. Critic, Actor mạnh | Thêm critic vào 120b | A (không): 34 · B (critic-agent): 32 | không cải thiện, thêm ~61% token |
| 3. Critic, Actor yếu | Thêm critic vào 20b | H (không): 33 · I (single): 32 · J (agent): 32 | không cải thiện |
| 4. Critic tự chấm | Qwen 7B tự làm critic | C (không): 10 · D (bản sửa): 9 · bản đầu: 8 | không cải thiện |
| 5. Môi trường | Cùng 20b, khác nơi chạy | G Ollama: 23 · H Groq: 33 | chênh 10 ca do nền tảng |

Số liệu chính của báo cáo là **A**. Ba log đề xuất cho yêu cầu "≥3 lần chạy thật": **A**, **H**, **B** (khác cấu hình, đủ 37 ca, có USAGE). Thư mục kết quả chấm tương ứng: `eval/out/README.md`.

Phân loại file: *chính thức* (mục 1), *lịch sử đủ 37 ca* (mục 2), *shard / smoke / chạy dở* (mục 3, chỉ để truy vết), *log critic* (mục 4). Hai script `run_batch_*.sh` và `run_*.jsonl` là công cụ và phiên REPL, không phải kết quả.

## 1. Các cấu hình đã đánh giá đủ 37 ca

Chi tiết cấu hình và số liệu: `docs/ket-qua-danh-gia.md`.

| Mã | Actor | Critic | Log | Ghi chú |
|---|---|---|---|---|
| A | gpt-oss-120b (Groq) | không | `eval_20261014_202454.jsonl` | cấu hình chính, 1 file liền |
| B | gpt-oss-120b (Groq) | critic-agent, Gemini 3.1 flash-lite | `eval_agent_full_merged.jsonl` | gộp shard (21:06 ngày 15/10 và các lần chạy lại ca lỗi) |
| C | Qwen2.5-7B (vLLM, Colab) | không | `eval_20261015_133814.jsonl` | 1 file liền |
| D | Qwen2.5-7B | critic một lần gọi, bản đã sửa (v2) | `eval_local_critic_v2_merged.jsonl` | gộp 3 shard 18:05 ngày 15/10 và lần chạy lại |
| E | Qwen2.5-14B (vLLM, Colab) | không | `eval_qwen2.5-14b_full.jsonl` | gộp từ 3 file 20:45–20:52 ngày 13/10 |
| G | gpt-oss:20b (Ollama) | không | `eval_20261016_011437.jsonl` | 23/37 |
| H | gpt-oss-20b (Groq) | không | `eval_20261016_021534`, `_021539`, `_021544`, `_021549` | 4 shard đã `--resume`; 33/37 |
| I, J | gpt-oss-20b (Groq) | single (I) / agent (J), Gemini | `eval_20261016_014856`, `_014901`, `_014906`, `_014911` | **hai cấu hình ghi chung** vào các file này (trùng tên); điểm tính từ `eval/out/g20_single_w*` và `g20_agent_w*`: 32/37 mỗi cấu hình |

## 2. Các lần chạy lịch sử đủ 37 ca (không dùng làm số liệu chính)

| Log | Vai trò |
|---|---|
| `eval_20261013_102004.jsonl` | lần chạy đầu tiên của toàn bộ bộ test, trước khi sửa prompt và truy vấn |
| `eval_20261015_144257.jsonl` | Qwen 7B + critic bản đầu (v1): 8/37, bằng chứng về lỗi critic cũ phải sửa |

## 3. Shard, lần chạy dở và thử nghiệm

Giữ lại để truy vết; số liệu của chúng đã nằm trong các log gộp ở mục 1 hoặc là thử nghiệm riêng.

| Nhóm | Log (theo thời gian) | Mục đích |
|---|---|---|
| Chạy thử gpt-oss-20b | `eval_gptoss_final.jsonl` | chỉ 2 ca (A01, C01), chạy bằng `run_batch_groq.sh`; **không** phải một cấu hình đánh giá (đã bỏ cấu hình F cũ) |
| Demo | `demo_20261013_235154.jsonl` | `make demo`: APP-014, APP-015 và một câu ngoài tài liệu |
| Chạy tay một ca | `eval_20261014_173509`, `_202036`, `20261015_125316`, `_131147`, `_133205` | kiểm tra từng ca (A01, A02) trong lúc sửa prompt hoặc tool |
| Lặp sửa lỗi | `eval_20261014_131515`, `_153148`, `_170836`, `_210104`, `20261015_160543` | các lần chạy một phần khi sửa agent, prompt và critic |
| Shard của E | `eval_20261013_204348`, `_205240`, `_205242` | 16 câu + 2 shard hồ sơ của Qwen 14B |
| Shard của D | `eval_20261015_180544`, `_180549`, `_180554` | 3 worker của critic v2 (Qwen 7B) |
| Thử critic-agent | `eval_20261015_190246`, `_193943`, `_200912`, `_205004`, `_233155` | smoke test 2–3 hồ sơ (APP-003, APP-014, APP-018…) khi xây critic-agent và sửa thought_signature |
| Shard của B | `eval_20261015_210650`, `_210655`, `_210701` | 3 worker chạy song song của critic-agent |
| 20b (16/10) | `eval_20261016_*` | thí nghiệm gpt-oss-20b: Ollama (không critic) và Groq (single/agent), đang chạy |

## 4. Log của Critic

`critic_*.jsonl`: mỗi dòng là một lần Critic xem xét (`FINCO_CRITIC_LOG`): các bước tự tra cứu, kết luận, số token. Đi kèm các lần chạy tương ứng ở trên.

| Log | Thuộc |
|---|---|
| `critic_agent_agent_full_*.jsonl` | B (critic-agent trên 120b) |
| `critic_agent_smoke*.jsonl` | smoke test critic-agent |
| `critic_v20_*.jsonl`, `critic_g20_*.jsonl` | thí nghiệm 20b ngày 16/10 |
