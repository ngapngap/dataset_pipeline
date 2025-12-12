# Dataset Pipeline V2 - BHXH Viet Nam

> **Hệ thống tự động tạo dataset Q&A chất lượng cao cho Fine-tuning LLM từ văn bản pháp luật**

Hệ thống được thiết kế chuyên biệt để xử lý các văn bản pháp luật Việt Nam (Luật, Nghị định, Thông tư) và tạo ra các cặp Hỏi-Đáp (Q&A) theo chuẩn ChatML, phục vụ việc huấn luyện các mô hình ngôn ngữ lớn (LLM) chuyên ngành Bảo hiểm xã hội.

## 🚀 Tính năng nổi bật

- **Tự động hóa hoàn toàn (Auto Mode)**: Chỉ cần 1 lệnh `python run.py`, hệ thống tự động kiểm tra tiến độ và chạy tiếp các bước chưa hoàn thành.
- **Đa nền tảng (Multi-provider)**: Hỗ trợ đồng thời **OpenAI, Google Gemini, Anthropic (Claude)** và các mô hình Local/Custom thông qua API chuẩn OpenAI (Ollama, vLLM, DeepSeek...).
- **Giám sát trực quan (Dashboard)**: Giao diện web real-time để theo dõi tiến độ, log và thống kê chất lượng dataset.
- **Chia dữ liệu thông minh (Document-based Split)**: Chia tập Train/Test/Val dựa trên **văn bản gốc** thay vì ngẫu nhiên từng câu, ngăn chặn triệt để vấn đề **Data Leakage**.
- **Cơ chế đánh giá & "Cứu hộ" (Evaluate & Rescue)**:
    - Tự động lọc trùng lặp (Deduplication).
    - Đánh giá chất lượng Q&A dựa trên tiêu chí pháp lý (số hiệu, điều khoản).
    - Tự động sửa chữa (Rescue) các câu hỏi thiếu trích dẫn nếu có thể.
    - Tự động sinh lại (Regenerate) các đoạn văn bản có chất lượng Q&A thấp.
- **Smart Caching**: Lưu cache API response để tiết kiệm chi phí và thời gian khi chạy lại.
- **Hiệu năng cao**: Hỗ trợ đa luồng (Multi-threading) và xử lý song song nhiều API key.

## 🛠 Yêu cầu hệ thống

- Python 3.9+
- RAM: 8GB+ (Khuyến nghị 16GB nếu chạy local models)
- API Keys (nếu dùng cloud providers: Gemini, OpenAI, Claude...)

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone <repo-url>
cd dataset_pipeline_v2
```

### 2. Tạo môi trường ảo (Khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## ⚙️ Cấu hình

File cấu hình chính: `config.yaml`. Sao chép từ `config.example.yaml` nếu cần.

### Cấu hình Provider
Bạn có thể cấu hình nhiều provider khác nhau. Ví dụ với **Gemini** và **Ollama**:

```yaml
llm:
  provider: "gemini"  # Provider mặc định sẽ chạy
  providers:
    gemini:
      model: "gemini-2.0-flash"
      api_keys_file: "../gemini_keys.txt" # File chứa danh sách key, mỗi key 1 dòng
      threads_per_key: 5 # Số luồng chạy song song cho mỗi key
    
    ollama: # Chạy model local
      base_url: "http://localhost:11434/v1"
      model: "vistral"
      api_key: "ollama"
```

### Cấu hình Xử lý (Processing)
```yaml
processing:
  chunk_size: 4000      # Kích thước đoạn văn bản (ký tự)
  qa_per_chunk: 5       # Số lượng câu hỏi sinh ra từ mỗi đoạn
  threads_per_key: 50   # Tăng tốc độ xử lý (lưu ý rate limit của API)
  cache:
    enabled: true       # Bật cache để resume được khi lỗi
```

## ▶️ Hướng dẫn sử dụng

### 1. Chạy Pipeline (Khuyến nghị)
Lệnh này sẽ tự động kiểm tra các bước đã chạy và chỉ chạy tiếp các bước còn thiếu.
```bash
python run.py
```

### 2. Các tùy chọn khác
```bash
# Bắt buộc chạy lại từ đầu (xóa hết dữ liệu cũ)
python run.py --force

# Chỉ chạy một bước cụ thể (extract, generate, evaluate, split, export)
python run.py --step evaluate

# Retry các chunks bị lỗi trong quá trình generate
python run.py --retry-failed

# Sử dụng file config khác
python run.py -c custom_config.yaml
```

### 3. Giám sát qua Dashboard (Mới)
Mở terminal mới và chạy:
```bash
python run_dashboard.py
```
Truy cập **http://localhost:8000** để xem:
- Tiến độ real-time của các bước.
- Log chi tiết.
- Biểu đồ thống kê chất lượng dataset.

### 4. Kiểm tra chất lượng Dataset
Sau khi pipeline chạy xong, chạy script kiểm tra:
```bash
python check_dataset_quality.py
```
Script này sẽ báo cáo chi tiết về:
- Số lượng câu trùng lặp.
- Độ dài trung bình câu hỏi/câu trả lời.
- Tỷ lệ câu trả lời có trích dẫn pháp lý (Điều/Khoản/Số hiệu văn bản).
- Phân phối câu hỏi theo văn bản nguồn.

## 🔄 Quy trình hoạt động (5 Steps)

```
+-------------------------------------------------------------+
|                    PIPELINE V2 (5 STEPS)                    |
+-------------------------------------------------------------+
|                                                             |
|  1. EXTRACT      Trích xuất text từ PDF/Word (Docx)        |
|       |          Input: data/raw -> Output: data/evaluated  |
|       v                                                     |
|  2. GENERATE     Sinh Q&A pairs bằng LLM                   |
|       |          (Multi-thread, Cache, Retry)               |
|       v                                                     |
|  3. EVALUATE     Quy trình Đánh giá & Sửa chữa:            |
|       |          +-- Deduplicate (Loại bỏ trùng lặp)       |
|       |          +-- Evaluate (Chấm điểm pháp lý)          |
|       |          +-- Rescue (Thêm trích dẫn còn thiếu)     |
|       |          +-- Regenerate (Sinh lại chunks kém)      |
|       v                                                     |
|  4. SPLIT        Chia tập Train/Val/Test theo DOCUMENT     |
|       |          (Tránh Data Leakage tuyệt đối)            |
|       v                                                     |
|  5. EXPORT       Xuất Dataset chuẩn ChatML                 |
|                  Output: output/final/*.jsonl              |
+-------------------------------------------------------------+
```

## 📂 Cấu trúc thư mục

```
dataset_pipeline_v2/
├── config.yaml               # File cấu hình chính
├── run.py                    # Script chạy Pipeline (CLI)
├── run_dashboard.py          # Script chạy Dashboard web
├── requirements.txt          # Danh sách thư viện
├── pipeline.py               # Class chính điều phối pipeline
├── pipeline_v2.py            # (Legacy) Version cũ
│
├── core/                     # Các module cốt lõi (Config, Logger, Utils)
├── dashboard/                # Mã nguồn Web Dashboard
├── providers/                # Modules kết nối LLM (OpenAI, Gemini, Custom...)
├── steps/                    # Logic từng bước (Extractor, Generator, Evaluator...)
├── legal_knowledge/          # Dữ liệu tri thức pháp luật bổ trợ
│
├── data/                     # Dữ liệu trung gian
│   ├── generated/            # Kết quả thô từ LLM
│   └── evaluated/            # Kết quả sau khi đánh giá
│
├── output/                   # Kết quả đầu ra
│   ├── extracted/            # Text đã trích xuất
│   ├── generated/            # File Q&A thô
│   ├── evaluated/            # File Q&A đã lọc (good/bad/duplicates)
│   ├── split/                # File đã chia tập train/test/val
│   └── final/                # Dataset cuối cùng (ChatML format)
│
└── logs/                     # File log hệ thống
```

## ❓ FAQ (Câu hỏi thường gặp)

**Q: Pipeline bị dừng giữa chừng do mất mạng/hết tiền API?**
A: Đừng lo, chỉ cần chạy lại `python run.py`. Hệ thống có cơ chế **Checkpoint** và **Cache**, sẽ tự động tiếp tục từ điểm dừng gần nhất mà không mất tiền chạy lại các phần đã xong.

**Q: Tại sao không có bước Tokenize?**
A: Việc tokenize phụ thuộc vào model cụ thể bạn định fine-tune. Pipeline này xuất ra định dạng text (JSONL/ChatML) phổ quát nhất để bạn có thể dùng với bất kỳ tokenizer nào sau đó.

**Q: Làm sao để thêm một provider mới (ví dụ DeepSeek)?**
A: Khai báo trong `config.yaml` phần `providers` với loại `custom` hoặc `openai`. Chỉ cần điền đúng `base_url` và `api_key`.

**Q: Tôi muốn kiểm tra các câu hỏi bị loại bỏ (Bad Q&A)?**
A: Kiểm tra file `output/evaluated/qa_bad.json`. Bạn cũng có thể dùng script `check_bad_samples.py` để xem nhanh 20 mẫu đầu tiên.

## 📜 License

Project được phát hành dưới giấy phép MIT.
