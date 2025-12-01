# 📚 Dataset Pipeline V2 - BHXH Việt Nam

> **Pipeline tạo dataset Q&A theo triết lý: Dạy STYLE, không dạy CONTENT**

## 🎯 Triết lý V2

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   FINE-TUNE ≠ Nhớ nội dung luật                                │
│   FINE-TUNE = Học CÁCH TRẢ LỜI như chuyên gia BHXH             │
│                                                                 │
│   ✅ Model học: Format, thuật ngữ, cấu trúc câu trả lời        │
│   ✅ RAG lo: Cung cấp nội dung chính xác, cập nhật             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ❓ Tại sao cách tiếp cận này?

| Cách cũ (Content-focused) | Cách mới (Style-focused) |
|---------------------------|--------------------------|
| ❌ Model cố nhớ nội dung luật | ✅ Model học CÁCH trả lời |
| ❌ Khi luật đổi → phải retrain | ✅ Khi luật đổi → update RAG |
| ❌ Dễ hallucinate số liệu | ✅ Số liệu từ RAG context |
| ❌ Cần model lớn (70B+) | ✅ Model 7B đủ dùng |

### 📋 Output Format: ChatML

**Tại sao chọn ChatML?**
- ✅ Chuẩn công nghiệp (OpenAI, Claude, Gemini)
- ✅ Dễ tích hợp RAG
- ✅ Multi-turn conversation ready
- ✅ Nhiều model đã pre-train với format này

---

## 📊 Tổng quan tính năng

| Vấn đề | Giải pháp |
|--------|-----------|
| Văn bản BHXH có nhiều luật chồng chéo | ✅ **Temporal Validator** - Lọc QA theo luật còn hiệu lực |
| Model không biết ưu tiên luật nào | ✅ **System Prompt Builder** - Tạo prompt động |
| 32+ văn bản khác nhau | ✅ **Laws Registry** - Database văn bản |
| Cần học STYLE không CONTENT | ✅ **ChatML Formatter** - Format chuẩn |

## 📁 Cấu trúc thư mục

```
dataset_pipeline/
├── config_v2.yaml           # 🆕 Config V2 (triết lý style-focused)
├── pipeline_v2.py           # 🆕 Pipeline V2 chính
├── config.yaml              # Config V1 (legacy)
├── run.py                   # CLI entry point
├── pipeline.py              # Pipeline V1 (legacy)
│
├── core/
│   ├── config.py            # Config loader
│   ├── logger.py            # Logging
│   └── utils.py             # Utilities
│
├── providers/               # LLM Providers
│   ├── base.py              # Base class
│   ├── gemini.py            # Google Gemini
│   ├── openai.py            # OpenAI GPT
│   ├── anthropic.py         # Anthropic Claude
│   ├── custom.py            # Custom API (vLLM, Ollama...)
│   └── factory.py           # Provider factory
│
├── steps/                   # Pipeline steps
│   ├── extractor.py         # Extract text từ documents
│   ├── generator.py         # Generate Q&A pairs
│   ├── evaluator.py         # Đánh giá chất lượng
│   ├── system_prompt_builder.py  # 🆕 Xây dựng system prompt động
│   └── temporal_validator.py     # 🆕 Lọc theo hiệu lực văn bản
│
├── output/                  # Output V1
└── output_v2/               # 🆕 Output V2
    ├── extracted/
    ├── generated/
    ├── validated/
    ├── formatted/
    └── final/
        ├── train.json
        ├── train.jsonl
        ├── validation.json
        └── test.json
```

## 🆕 Pipeline V2: Style-Focused Approach

### Triết lý core

```yaml
training:
  philosophy: "style_focused"
  
  objectives:
    - "Học cách format câu trả lời pháp luật"
    - "Học thuật ngữ chuyên ngành BHXH"
    - "Học cách trích dẫn điều khoản"
    - "Học cách tổng hợp từ context (RAG-ready)"
  
  non_objectives:
    - "Nhớ nội dung cụ thể của từng điều luật"
    - "Nhớ số liệu cụ thể (có thể thay đổi)"
```

### ChatML Output Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam.\n\nNguyên tắc:\n1. Luôn trích dẫn căn cứ pháp lý\n2. Giải thích rõ ràng, dễ hiểu"
    },
    {
      "role": "user",
      "content": "Mức đóng BHXH của người lao động là bao nhiêu?"
    },
    {
      "role": "assistant",
      "content": "Căn cứ Điều 85, Khoản 1 Luật BHXH số 41/2024/QH15:\n\nMức đóng BHXH bắt buộc của người lao động là 8% tiền lương tháng."
    }
  ],
  "metadata": {
    "source": "41_2024_QH15",
    "question_type": "factual",
    "cited_articles": ["Điều 85"],
    "temporal_status": "current"
  }
}
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE V2 FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EXTRACT          Trích xuất text từ văn bản pháp luật      │
│       ↓                                                        │
│  2. GENERATE         Sinh Q&A với STYLE chuẩn                  │
│       ↓              (format trích dẫn, thuật ngữ, cấu trúc)   │
│       ↓                                                        │
│  3. VALIDATE         - Temporal: Lọc văn bản hết hiệu lực      │
│       ↓              - Style: Kiểm tra format câu trả lời      │
│       ↓                                                        │
│  4. FORMAT           Chuyển sang ChatML                        │
│       ↓                                                        │
│  5. EXPORT           Split train/val/test + deduplicate        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🆕 Các module mới (v2.0)

### 1. System Prompt Builder (`steps/system_prompt_builder.py`)

**Vấn đề**: Dataset cũ không có system prompt → Model không biết vai trò, không có context thời gian.

**Giải pháp**: Tự động xây dựng system prompt dựa trên văn bản được trích dẫn trong answer.

```python
from steps.system_prompt_builder import SystemPromptBuilder

builder = SystemPromptBuilder()

# Auto-detect văn bản từ answer
answer = "Theo Điều 102 Luật BHXH số 41/2024/QH15 và Nghị định 158/2025/NĐ-CP..."
cited_docs = builder.detect_cited_documents(answer)
# → [{"doc_id": "41/2024/QH15", "doc_type": "LUẬT"}, 
#    {"doc_id": "158/2025/NĐ-CP", "doc_type": "NGHỊ ĐỊNH"}]

# Build system prompt phù hợp
system_prompt = builder.build_system_prompt(cited_docs, mode="specific")
```

**Các format hỗ trợ**:

| Format | Mô tả | Use case |
|--------|-------|----------|
| `chat` | ChatML (`<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`) | Chat models |
| `instruction` | `### Hệ thống:`, `### Câu hỏi:`, `### Trả lời:` | Instruction tuning |
| `alpaca` | Alpaca format | Alpaca-style models |

### 2. Temporal Validator (`steps/temporal_validator.py`)

**Vấn đề**: Dataset có cả Luật 58/2014 (hết hiệu lực) và Luật 41/2024 (mới) → Model học cả hai, trả lời theo luật cũ.

**Giải pháp**: Lọc QA pairs chỉ giữ những cái trích dẫn văn bản còn hiệu lực.

```python
from steps.temporal_validator import TemporalLegalValidator

validator = TemporalLegalValidator("../legal_knowledge/laws_registry.json")

# Validate một QA pair
result = validator.validate_qa(qa_pair, reference_date="2025-12-01")
# → {"is_valid": True, "status": "current", "cited_laws": [...]}

# Validate toàn bộ dataset
valid_qa, invalid_qa, stats = validator.validate_dataset(qa_pairs)
```

**Các status**:
- `current`: Chỉ trích dẫn văn bản còn hiệu lực ✅
- `mixed`: Có cả văn bản cũ và mới ⚠️
- `expired`: Chỉ trích dẫn văn bản hết hiệu lực ❌
- `unknown`: Không phát hiện văn bản nào

### 3. Laws Registry (`../legal_knowledge/laws_registry.json`)

Database 32 văn bản pháp luật BHXH với thông tin:
- Ngày hiệu lực, ngày hết hiệu lực
- Status: `active`, `expired`, `pending`
- Topics: Mapping chủ đề → văn bản

```json
{
  "documents": [
    {
      "id": "41/2024/QH15",
      "name": "Luật Bảo hiểm xã hội",
      "type": "Luật",
      "effective_date": "2025-07-01",
      "status": "active",
      "replaces": ["58/2014/QH13"],
      "topics": ["BHXH", "hưu trí", "thai sản"]
    }
  ],
  "topic_mapping": {
    "BHXH bắt buộc": ["41/2024/QH15", "158/2025/NĐ-CP"],
    "BHTN": ["41/2024/QH15", "157/2025/NĐ-CP"],
    "BHYT": ["51/2024/QH15", "146/2018/NĐ-CP"]
  }
}
```

## 🚀 Quick Start

### Cài đặt

```bash
cd dataset_pipeline
pip install -r requirements.txt
```

### Cấu hình API keys

```bash
# Gemini API keys (mỗi key 1 dòng)
echo "AIzaSy..." > ../gemini_keys.txt
```

### Chạy Pipeline V2

```bash
# Chạy toàn bộ pipeline V2
python pipeline_v2.py

# Hoặc từng bước
python pipeline_v2.py -s extract
python pipeline_v2.py -s generate
python pipeline_v2.py -s validate format export

# Với config custom
python pipeline_v2.py --config config_v2.yaml
```

### Chạy Pipeline V1 (Legacy)

```bash
python run.py
python run.py --steps extract generate evaluate
```

---

## ⚙️ Config V2 (`config_v2.yaml`)

```yaml
project:
  name: "bhxh_qa_dataset_v2"
  version: "2.0"
  philosophy: "style_focused"  # 🆕 Triết lý mới

output_format:
  type: "chatml"  # chatml | instruction | alpaca
  
  chatml:
    system_prompt: |
      Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam.
      Nguyên tắc:
      1. Luôn trích dẫn căn cứ pháp lý
      2. Giải thích rõ ràng, dễ hiểu

qa_generation:
  answer_style:
    require_citation: true
    citation_format: "Căn cứ {article}, {clause} của {document}"
    
quality_evaluation:
  # 🆕 Style-focused criteria
  criteria:
    - name: "has_citation_format"
      weight: 3
    - name: "has_structure"
      weight: 2
    - name: "natural_question"
      weight: 2
  
  min_score: 7.0

temporal_validation:
  enabled: true
  filter_expired: true
```

---

## 📊 Đánh giá chất lượng (Style-focused)

V2 đánh giá **STYLE** không phải đúng/sai nội dung:

| Tiêu chí | Weight | Mô tả |
|----------|--------|-------|
| `has_citation_format` | 3 | Có format "Căn cứ Điều X, Khoản Y..." |
| `has_structure` | 2 | Có cấu trúc rõ ràng |
| `has_specifics` | 2 | Có số liệu cụ thể (%, năm, tiền) |
| `appropriate_length` | 1 | Độ dài 150-800 ký tự |
| `natural_question` | 2 | Câu hỏi tự nhiên |

**Ngưỡng đạt**: 7/10

---

## ⏰ Temporal Validation

Lọc Q&A trích dẫn văn bản hết hiệu lực:

| Status | Xử lý | Ví dụ |
|--------|-------|-------|
| `current` | ✅ Giữ | Trích Luật 41/2024 |
| `expired` | ❌ Loại | Trích Luật 58/2014 |
| `mixed` | ⚠️ Review | Trích cả 2 luật |
| `unknown` | ✅ Giữ | Không phát hiện văn bản |

---

## 📊 Pipeline Flow (Chi tiết)

```
┌─────────────────┐
│  1. EXTRACT     │  Trích xuất text từ .txt/.docx/.pdf
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. GENERATE    │  Sinh Q&A pairs từ LLM (multi-thread)
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. EVALUATE    │  Đánh giá chất lượng + căn cứ pháp lý
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────┐
│  4. TEMPORAL VALIDATE (🆕)                       │
│  - Phát hiện văn bản trích dẫn                  │
│  - Kiểm tra hiệu lực theo laws_registry.json    │
│  - Lọc bỏ QA trích dẫn luật hết hiệu lực        │
└────────┬────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────┐
│  5. FORMAT WITH SYSTEM PROMPT (🆕)              │
│  - Auto-detect văn bản trong answer             │
│  - Build system prompt động theo văn bản        │
│  - Output: chat/instruction/alpaca format       │
└────────┬────────────────────────────────────────┘
         ▼
┌─────────────────┐
│  6. EXPORT      │  Split train/val/test, save JSON/JSONL
└─────────────────┘
```

## 📝 Output Format

### Trước (Format cũ - không có system prompt)

```json
{
  "question": "Mức đóng BHXH bắt buộc của người lao động là bao nhiêu?",
  "answer": "Theo Điều 85 Luật BHXH số 41/2024/QH15..."
}
```

### Sau (Format mới - có system prompt động)

```json
{
  "text": "### Hệ thống:\nBạn là chuyên gia tư vấn BHXH Việt Nam.\nNgày hiện tại: 01/12/2025\n\nVăn bản pháp luật áp dụng:\n- Luật số 41/2024/QH15 (Còn hiệu lực)\n\n### Câu hỏi:\nMức đóng BHXH bắt buộc của người lao động là bao nhiêu?\n\n### Trả lời:\nTheo Điều 85 Luật BHXH số 41/2024/QH15...",
  "metadata": {
    "cited_documents": ["41/2024/QH15"],
    "format_type": "instruction",
    "temporal_status": "current"
  }
}
```

## ⚠️ Quy tắc đánh giá chất lượng

### Bắt buộc có căn cứ pháp lý

| Tiêu chí | Điểm |
|----------|------|
| ❌ Thiếu căn cứ pháp lý | **-5 điểm** |
| ✅ Có Điều + Khoản cụ thể | +1 điểm |
| ✅ Có số liệu (%, VNĐ, năm) | +1 điểm |
| ❌ Độ dài < 100 ký tự | -2 điểm |
| ❌ Trích dẫn luật hết hiệu lực | **Loại** (🆕) |

### Ngưỡng đạt
- Điểm tối thiểu: **6/10**
- Câu trả lời thiếu căn cứ pháp lý: 10 - 5 = 5 < 6 → **Loại**
- Câu trả lời trích luật hết hiệu lực → **Loại** (temporal validation)

## 📚 Danh sách văn bản hỗ trợ (32 văn bản)

### Luật
| ID | Tên | Hiệu lực |
|----|-----|----------|
| 41/2024/QH15 | Luật BHXH | 01/07/2025 ✅ |
| 51/2024/QH15 | Luật sửa đổi BHYT | 01/07/2025 ✅ |
| 45/2019/QH14 | Bộ luật Lao động | 01/01/2021 ✅ |
| 58/2014/QH13 | Luật BHXH (cũ) | ❌ Hết hiệu lực |

### Nghị định chính
| ID | Nội dung |
|----|----------|
| 158/2025/NĐ-CP | BHXH bắt buộc (ốm đau, thai sản, hưu trí) |
| 159/2025/NĐ-CP | BHXH tự nguyện |
| 157/2025/NĐ-CP | Bảo hiểm thất nghiệp |
| 176/2025/NĐ-CP | Điều chỉnh lương hưu |
| 188/2025/NĐ-CP | Mức lương hưu tối thiểu |

*Xem đầy đủ tại `legal_knowledge/laws_registry.json`*

## 🔧 Troubleshooting

### Model trả lời theo luật cũ
```bash
# Chạy temporal validation để lọc
python run.py --steps validate-temporal

# Hoặc dùng module độc lập
python -m steps.temporal_validator --input data.json --filter-expired
```

### Muốn thêm văn bản mới
1. Thêm vào `legal_knowledge/laws_registry.json`
2. Cập nhật `topic_mapping` nếu cần
3. Chạy lại pipeline

### Rate limit API
- Thêm nhiều API keys vào file
- Tăng `request_delay` trong config

### Quality thấp
- Kiểm tra văn bản đầu vào có phải pháp luật không
- Dùng model tốt hơn (gemini-1.5-pro, gpt-4)
- Điều chỉnh prompt template

## 📈 Thống kê mẫu

```
📊 PIPELINE STATISTICS
================================================================================
📄 Documents extracted:        15 files
📝 Chunks processed:          245 chunks
❓ QA pairs generated:      1,225 pairs
✅ QA passed evaluation:      892 pairs (72.8%)
🕐 Temporal validation:
   - Current (valid):         756 pairs (84.8%)
   - Mixed (review needed):    89 pairs (10.0%)
   - Expired (removed):        47 pairs (5.2%)
📦 Final dataset:             756 pairs
   - Train:                   604 pairs
   - Validation:               76 pairs
   - Test:                     76 pairs
================================================================================
```

## 🔗 Links liên quan

- **Legal Knowledge**: `../legal_knowledge/` - Database văn bản pháp luật
- **Dataset Output**: `../dataset_output/` - Dataset đã xử lý
- **Tokenize & Upload**: `../02_tokenize_upload.py` - Upload lên HuggingFace

## License

MIT
