# 📋 HƯỚNG DẪN: Temporal Legal Knowledge cho BHXH Model

## 1. VẤN ĐỀ

Luật BHXH Việt Nam có đặc điểm:
- Nhiều văn bản chồng chéo (Luật, Nghị định, Thông tư, Quyết định...)
- Văn bản mới thay thế/sửa đổi văn bản cũ
- Có ngày hiệu lực khác ngày ban hành
- Cùng một chủ đề có thể áp dụng luật khác nhau tùy thời điểm

## 2. GIẢI PHÁP: Temporal Context

### 2.1 Cấu trúc dữ liệu

```
legal_knowledge/
├── laws_registry.json      # Danh sách văn bản + metadata
├── README.md              # File này
└── (tương lai) embeddings/ # Vector embeddings cho RAG
```

### 2.2 Format trong Dataset Training

**TRƯỚC (format cũ - có vấn đề):**
```
### Câu hỏi:
Mức đóng BHXH là bao nhiêu?

### Trả lời:
Theo Điều X Luật 58/2014...
```

**SAU (format mới - có temporal context):**
```
### Ngày tham chiếu: 01/12/2025
### Văn bản áp dụng: Luật 41/2024/QH15 (có hiệu lực từ 01/07/2025)

### Câu hỏi:
Mức đóng BHXH là bao nhiêu?

### Trả lời:
Theo Điều X Luật 41/2024/QH15...
```

### 2.3 Quy trình khi Inference

```python
def get_applicable_laws(query_date, laws_registry):
    """Lấy danh sách luật đang có hiệu lực tại thời điểm query"""
    applicable = []
    for law in laws_registry["laws"]:
        effective = datetime.strptime(law["effective_date"], "%Y-%m-%d")
        expiry = law["expiry_date"]
        
        if effective <= query_date:
            if expiry is None or datetime.strptime(expiry, "%Y-%m-%d") > query_date:
                applicable.append(law)
    
    return applicable

def build_prompt_with_context(question, query_date):
    """Xây dựng prompt với context về luật áp dụng"""
    applicable_laws = get_applicable_laws(query_date, laws_registry)
    
    # Lọc luật liên quan đến BHXH
    bhxh_laws = [l for l in applicable_laws if "BHXH" in l["name"] or "bảo hiểm xã hội" in l["name"].lower()]
    
    context = f"### Ngày tham chiếu: {query_date.strftime('%d/%m/%Y')}\n"
    context += "### Văn bản pháp luật đang có hiệu lực:\n"
    for law in bhxh_laws[:5]:  # Top 5 relevant
        context += f"- {law['id']}: {law['name']} (hiệu lực từ {law['effective_date']})\n"
    
    prompt = f"{context}\n### Câu hỏi:\n{question}\n\n### Trả lời:\n"
    return prompt
```

## 3. ÁP DỤNG CHO TRAINING LẦN SAU

### 3.1 Tiền xử lý Dataset

1. **Phân loại QA theo luật:**
   - Scan tất cả QA, detect văn bản được trích dẫn
   - Gắn tag luật cho mỗi QA

2. **Lọc theo hiệu lực:**
   ```python
   # Chỉ giữ QA trích dẫn luật còn hiệu lực
   def is_current_law(law_id, reference_date):
       law = find_law_by_id(law_id)
       if law["status"] == "expired":
           return False
       return True
   ```

3. **Thêm temporal context:**
   - Mỗi QA cần có ngày tham chiếu
   - Mỗi QA cần list văn bản áp dụng

### 3.2 Format Dataset Mới

```json
{
  "question": "Mức đóng BHXH bắt buộc của người lao động?",
  "answer": "Theo Điều 33 Luật 41/2024/QH15...",
  "reference_date": "2025-12-01",
  "applicable_laws": ["41/2024/QH15", "135/2024/NĐ-CP"],
  "metadata": {
    "topic": "mức đóng",
    "law_version": "2024"
  }
}
```

### 3.3 Training với Temporal Awareness

```
### Hệ thống: Bạn là trợ lý pháp luật về BHXH. Hãy trả lời dựa trên văn bản pháp luật được cung cấp.

### Ngày hiện tại: 01/12/2025

### Văn bản pháp luật áp dụng:
- Luật BHXH số 41/2024/QH15 (hiệu lực từ 01/07/2025)
- Nghị định 135/2024/NĐ-CP (hiệu lực từ 01/07/2025)

### Câu hỏi:
{question}

### Trả lời:
{answer}
```

## 4. TƯƠNG LAI: RAG SYSTEM

Để giải quyết triệt để, cần RAG:

```
User Question
     ↓
[Query Understanding] → Extract intent, entities, date
     ↓
[Law Retrieval] → Tìm văn bản liên quan từ laws_registry.json
     ↓
[Context Building] → Build prompt với luật applicable
     ↓
[Generation] → Model generate với full context
     ↓
[Validation] → Check answer có đúng luật không
     ↓
Answer
```

## 5. CHECKLIST CHO LẦN TRAINING SAU

- [ ] Cập nhật `laws_registry.json` với đầy đủ văn bản
- [ ] Scan dataset, tag luật cho mỗi QA
- [ ] Lọc bỏ QA trích dẫn luật hết hiệu lực
- [ ] Thêm temporal context vào mỗi sample
- [ ] Format training data theo cấu trúc mới
- [ ] Test model với câu hỏi có yêu cầu thời điểm khác nhau
