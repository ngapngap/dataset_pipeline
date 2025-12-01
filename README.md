# 📚 Dataset Pipeline - Văn bản Pháp luật BHXH Việt Nam

> ⚠️ **QUAN TRỌNG**: Pipeline này được thiết kế **CHUYÊN BIỆT** cho việc tạo dataset Q&A từ **văn bản pháp luật cho ngành BHXH Việt Nam** (Luật, Nghị định, Thông tư, Quyết định...). 
>
> **KHÔNG SỬ DỤNG** với các loại tài liệu khác (sách, báo, blog, truyện...) vì hệ thống sẽ kiểm tra và **TRỪ ĐIỂM NẶNG** nếu câu trả lời thiếu trích dẫn văn bản pháp luật (số hiệu văn bản, Điều, Khoản, Nghị định...).

## 🎯 Mục đích

Tự động tạo dataset Q&A chất lượng cao từ các văn bản pháp luật để fine-tune mô hình AI tư vấn pháp luật, với các yêu cầu nghiêm ngặt:

- ✅ Câu hỏi phải **thực tế** như người dân hỏi
- ✅ Câu trả lời **BẮT BUỘC** có căn cứ pháp lý (Điều X, Khoản Y, Luật Z...)
- ✅ Có số liệu cụ thể (%, năm, VNĐ...)
- ❌ **LOẠI BỎ** câu trả lời thiếu trích dẫn văn bản pháp luật

## 📋 Yêu cầu đầu vào

### ✅ Văn bản được hỗ trợ:
- **Luật**: Luật BHXH, Luật BHYT, Bộ luật Lao động...
- **Nghị định**: 143/2018/NĐ-CP, 146/2018/NĐ-CP...
- **Thông tư**: Các thông tư hướng dẫn
- **Quyết định**: 595/QĐ-BHXH, 948/QĐ-BHXH...

### ❌ KHÔNG hỗ trợ:
- Sách, giáo trình
- Bài báo, blog
- Tài liệu nội bộ không có số hiệu
- Truyện, văn học
- Bất kỳ tài liệu nào KHÔNG phải văn bản pháp luật

## Tính năng

- ✅ **Multi-provider**: Hỗ trợ Gemini, OpenAI, Anthropic, Ollama, vLLM, và 12+ providers khác
- ✅ **Multi-threading**: Mỗi API key chạy 1 thread riêng
- ✅ **YAML Config**: Cấu hình linh hoạt qua file YAML
- ✅ **Auto-save**: Tự động lưu kết quả trung gian
- ✅ **Quality Evaluation**: Đánh giá chất lượng với **kiểm tra căn cứ pháp lý bắt buộc**
- ✅ **Resume**: Có thể resume nếu bị gián đoạn

## Cấu trúc thư mục

```
dataset_pipeline/
├── config.yaml          # File cấu hình chính
├── run.py               # CLI entry point
├── pipeline.py          # Pipeline orchestrator
├── core/
│   ├── config.py        # Config loader
│   ├── logger.py        # Logger
│   └── utils.py         # Utilities
├── providers/
│   ├── base.py          # Base provider class
│   ├── gemini.py        # Google Gemini
│   ├── openai.py        # OpenAI
│   ├── anthropic.py     # Anthropic Claude
│   └── factory.py       # Provider factory
└── steps/
    ├── extractor.py     # Text extraction
    ├── generator.py     # Q&A generation
    └── evaluator.py     # Quality evaluation
```

## Cài đặt

```bash
# Cài đặt dependencies
pip install PyYAML tqdm

# Provider APIs (chọn theo nhu cầu)
pip install google-generativeai  # Gemini
pip install openai               # OpenAI
pip install anthropic            # Claude

# Text extraction (tùy loại file)
pip install python-docx          # .docx
pip install PyPDF2               # .pdf
```

## Sử dụng

### 1. Cấu hình

Chỉnh sửa file `config.yaml`:

```yaml
project:
  name: "my_project"
  description: "Mô tả dự án"

input:
  documents_dir: "data/raw"

api_keys:
  gemini_keys_file: "gemini_keys.txt"

generation:
  provider: "gemini"
  model: "gemini-2.0-flash"
  num_qa_per_chunk: 5

evaluation:
  mode: "hybrid"
  min_score: 7.0

output:
  base_dir: "output"
  format: "chat"
```

### 2. Chuẩn bị API keys

Tạo file chứa API keys (mỗi key 1 dòng):

```
# gemini_keys.txt
AIzaSyxxxxxxxxxxxxxxxxxxxxxxx
AIzaSyyyyyyyyyyyyyyyyyyyyyyyy
```

### 3. Chạy pipeline

```bash
# Chạy toàn bộ pipeline
python run.py

# Chỉ chạy 1 bước
python run.py --steps extract
python run.py --steps generate
python run.py --steps evaluate
python run.py --steps export

# Chạy nhiều bước
python run.py --steps generate evaluate

# Resume từ checkpoint
python run.py --resume

# Retry failed chunks
python run.py --retry-failed

# Dùng config khác
python run.py -c my_config.yaml
```

## Các bước trong pipeline

### 1. Extract
Trích xuất text từ các file văn bản pháp luật (.txt, .docx, .pdf)

### 2. Generate
Sinh cặp câu hỏi-trả lời từ text sử dụng LLM:
- Chia text thành chunks
- Mỗi chunk sinh N cặp Q&A
- Chạy đa luồng (1 thread / 1 API key)
- **Prompt yêu cầu trích dẫn văn bản pháp luật trong mọi câu trả lời**

### 3. Evaluate
Đánh giá chất lượng Q&A với **quy tắc đặc biệt cho văn bản pháp luật**:
- **Kiểm tra căn cứ pháp lý**: Câu trả lời PHẢI có trích dẫn (Điều, Khoản, Luật, Nghị định...)
- **TRỪ 5 ĐIỂM** nếu thiếu căn cứ pháp lý → gần như tự động bị loại
- **+1 điểm** nếu trích dẫn cụ thể Điều + Khoản
- Rule-based + LLM scoring

### 4. Export
Xuất dataset theo format:
- `instruction`: Format instruction-following (mặc định)
- `chat`: Format chat messages (cho fine-tuning)
- `simple`: Format Q&A đơn giản

Tự động split thành train/validation/test.

## Output

```
output/
├── extracted/
│   └── extracted_documents.json
├── generated/
│   ├── qa_generated.json
│   ├── qa_generated.jsonl
│   └── failed_chunks.json
├── evaluated/
│   ├── qa_good.json          # Q&A đạt chất lượng (có căn cứ pháp lý)
│   ├── qa_bad.json           # Q&A bị loại (thiếu căn cứ pháp lý hoặc chất lượng thấp)
│   └── evaluation_stats.json
├── final/
│   ├── dataset_final.json
│   ├── dataset_final.jsonl
│   ├── train.json / train.jsonl
│   ├── validation.json / validation.jsonl
│   └── test.json / test.jsonl
├── pipeline_state.json
└── logs/
    └── pipeline.log
```

## ⚠️ Quy tắc đánh giá chất lượng

### Bắt buộc có căn cứ pháp lý
Mọi câu trả lời **PHẢI** chứa ít nhất một trong các pattern sau:
- `Theo Điều X` / `Khoản Y`
- `Luật số XX/YYYY/QH`
- `Nghị định số XX/YYYY/NĐ-CP`
- `Thông tư số XX/YYYY/TT-XXX`
- `Quyết định số XX/QĐ-XXX`

### Điểm trừ/cộng
| Tiêu chí | Điểm |
|----------|------|
| Thiếu căn cứ pháp lý | **-5 điểm** |
| Có Điều + Khoản cụ thể | +1 điểm |
| Có số liệu (%, VNĐ, năm) | +1 điểm |
| Độ dài < 100 ký tự | -2 điểm |

### Ngưỡng đạt
- Điểm tối thiểu: **6/10** (tương đương 3/5 trong config)
- Câu trả lời thiếu căn cứ pháp lý gần như chắc chắn bị loại (10 - 5 = 5 < 6)

## Tùy chỉnh

### Custom Prompt
Thay đổi prompt sinh Q&A trong `config.yaml`:

```yaml
generation:
  prompt_template: |
    Dựa trên văn bản sau, tạo {num_qa} cặp Q&A...
    
    VĂN BẢN:
    {content}
    
    OUTPUT FORMAT (JSON):
    [{"question": "...", "answer": "..."}]
```

### Thêm Provider mới
1. Tạo file trong `providers/`
2. Kế thừa `BaseLLMProvider`
3. Implement `_initialize()` và `_call_api()`
4. Đăng ký trong `factory.py`

## Troubleshooting

### Rate limit
- Tăng `rate_limit_per_minute` trong config
- Thêm nhiều API keys vào file

### Out of Memory
- Giảm `chunk_size` trong config
- Giảm `max_threads`

### Quality thấp / Nhiều Q&A bị loại
- Kiểm tra văn bản đầu vào có phải **văn bản pháp luật** không
- Điều chỉnh prompt template để nhấn mạnh yêu cầu trích dẫn
- Dùng model tốt hơn (gemini-1.5-pro, gpt-4)

### Lỗi "Thiếu căn cứ pháp lý"
- **Nguyên nhân**: Văn bản đầu vào không phải văn bản pháp luật
- **Giải pháp**: Chỉ sử dụng Luật, Nghị định, Thông tư, Quyết định có số hiệu rõ ràng

## 📝 Ví dụ Q&A đạt chuẩn

```json
{
  "instruction": "Tôi đóng BHXH được 25 năm, năm nay tôi 55 tuổi. Tôi có đủ điều kiện hưởng lương hưu không?",
  "input": "",
  "output": "Theo Điều 54, Khoản 1 của Luật Bảo hiểm xã hội số 58/2014/QH13, người lao động nam có đủ 20 năm đóng BHXH trở lên được hưởng lương hưu khi đủ 60 tuổi (từ 2028 là 62 tuổi). Với trường hợp của bạn: đã đóng 25 năm BHXH (đủ điều kiện về thời gian), nhưng mới 55 tuổi (chưa đủ tuổi nghỉ hưu). Bạn cần đợi đến khi đủ tuổi theo quy định hoặc xem xét các trường hợp nghỉ hưu sớm tại Điều 55 nếu làm công việc nặng nhọc, độc hại.",
  "source": "Luật BHXH số 58/2014/QH13"
}
```

## License

MIT
