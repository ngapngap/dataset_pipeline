"""
Script hỗ trợ xử lý Temporal Legal Context cho BHXH Model
Sử dụng cho lần training tiếp theo
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Load laws registry
LAWS_REGISTRY_PATH = Path(__file__).parent / "laws_registry.json"

def load_laws_registry() -> Dict:
    """Load danh sách văn bản pháp luật"""
    with open(LAWS_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_applicable_laws(query_date: datetime, registry: Dict = None) -> List[Dict]:
    """
    Lấy danh sách văn bản đang có hiệu lực tại thời điểm query
    
    Args:
        query_date: Ngày cần kiểm tra
        registry: Laws registry dict (optional, sẽ load nếu không truyền)
    
    Returns:
        List các văn bản đang có hiệu lực
    """
    if registry is None:
        registry = load_laws_registry()
    
    applicable = []
    for law in registry["laws"]:
        effective = datetime.strptime(law["effective_date"], "%Y-%m-%d")
        expiry = law.get("expiry_date")
        
        # Kiểm tra đã có hiệu lực chưa
        if effective <= query_date:
            # Kiểm tra đã hết hiệu lực chưa
            if expiry is None:
                applicable.append(law)
            elif datetime.strptime(expiry, "%Y-%m-%d") > query_date:
                applicable.append(law)
    
    return applicable

def detect_law_citations(text: str) -> List[str]:
    """
    Detect các văn bản pháp luật được trích dẫn trong text
    
    Args:
        text: Văn bản cần scan
    
    Returns:
        List các law_id được trích dẫn
    """
    patterns = [
        r'(\d+/\d{4}/QH\d+)',           # Luật: 41/2024/QH15
        r'(\d+/\d{4}/NĐ-CP)',           # Nghị định: 135/2024/NĐ-CP
        r'(\d+/\d{4}/TT-[A-Z]+)',       # Thông tư: 59/2024/TT-BLĐTBXH
        r'(\d+/QĐ-[A-Z]+)',             # Quyết định: 1234/QĐ-BHXH
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))

def is_law_valid(law_id: str, reference_date: datetime, registry: Dict = None) -> bool:
    """
    Kiểm tra văn bản có còn hiệu lực tại thời điểm reference không
    
    Args:
        law_id: ID văn bản (VD: "41/2024/QH15")
        reference_date: Ngày tham chiếu
        registry: Laws registry
    
    Returns:
        True nếu còn hiệu lực, False nếu hết hoặc không tìm thấy
    """
    if registry is None:
        registry = load_laws_registry()
    
    for law in registry["laws"]:
        if law["id"] == law_id:
            effective = datetime.strptime(law["effective_date"], "%Y-%m-%d")
            expiry = law.get("expiry_date")
            
            if effective > reference_date:
                return False  # Chưa có hiệu lực
            
            if expiry and datetime.strptime(expiry, "%Y-%m-%d") <= reference_date:
                return False  # Đã hết hiệu lực
            
            return True
    
    # Không tìm thấy trong registry - coi như valid (có thể là văn bản mới)
    return True

def filter_qa_by_law_validity(qa_list: List[Dict], reference_date: datetime) -> List[Dict]:
    """
    Lọc danh sách QA, chỉ giữ những QA trích dẫn luật còn hiệu lực
    
    Args:
        qa_list: List các QA dict có field "question" và "answer"
        reference_date: Ngày tham chiếu
    
    Returns:
        List QA đã được lọc
    """
    registry = load_laws_registry()
    valid_qa = []
    invalid_qa = []
    
    for qa in qa_list:
        text = qa.get("question", "") + " " + qa.get("answer", "")
        citations = detect_law_citations(text)
        
        # Kiểm tra tất cả citations có valid không
        all_valid = True
        invalid_citations = []
        
        for citation in citations:
            if not is_law_valid(citation, reference_date, registry):
                all_valid = False
                invalid_citations.append(citation)
        
        if all_valid:
            valid_qa.append(qa)
        else:
            qa_copy = qa.copy()
            qa_copy["_invalid_citations"] = invalid_citations
            invalid_qa.append(qa_copy)
    
    print(f"✅ Valid QA: {len(valid_qa)}")
    print(f"❌ Invalid QA (trích dẫn luật hết hiệu lực): {len(invalid_qa)}")
    
    return valid_qa, invalid_qa

def build_temporal_prompt(question: str, reference_date: datetime = None) -> str:
    """
    Xây dựng prompt với temporal context
    
    Args:
        question: Câu hỏi
        reference_date: Ngày tham chiếu (default: hôm nay)
    
    Returns:
        Prompt với đầy đủ context
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    registry = load_laws_registry()
    applicable = get_applicable_laws(reference_date, registry)
    
    # Lọc văn bản quan trọng nhất
    main_laws = [l for l in applicable if l["type"] == "Luật"]
    main_decrees = [l for l in applicable if l["type"] == "Nghị định"][:3]
    
    context_lines = [
        f"### Ngày tham chiếu: {reference_date.strftime('%d/%m/%Y')}",
        "### Văn bản pháp luật đang có hiệu lực:"
    ]
    
    for law in main_laws:
        context_lines.append(f"- {law['id']}: {law['name']}")
    for decree in main_decrees:
        context_lines.append(f"- {decree['id']}: {decree['name']}")
    
    context = "\n".join(context_lines)
    
    prompt = f"""{context}

### Câu hỏi:
{question}

### Lưu ý: Trả lời theo văn bản pháp luật đang có hiệu lực tại ngày tham chiếu.

### Trả lời:
"""
    return prompt

def add_temporal_context_to_dataset(dataset: List[Dict], reference_date: datetime) -> List[Dict]:
    """
    Thêm temporal context vào toàn bộ dataset
    
    Args:
        dataset: List các QA
        reference_date: Ngày tham chiếu
    
    Returns:
        Dataset đã được thêm context
    """
    registry = load_laws_registry()
    applicable = get_applicable_laws(reference_date, registry)
    
    # Build context string
    main_laws = [l for l in applicable if l["type"] == "Luật"]
    
    context_header = f"### Ngày tham chiếu: {reference_date.strftime('%d/%m/%Y')}\n"
    context_header += "### Văn bản áp dụng: "
    context_header += ", ".join([f"{l['id']}" for l in main_laws])
    context_header += "\n\n"
    
    new_dataset = []
    for qa in dataset:
        new_qa = qa.copy()
        
        # Thêm context vào đầu question
        original_q = qa.get("question", "")
        new_qa["question"] = context_header + "### Câu hỏi:\n" + original_q
        
        # Thêm metadata
        new_qa["reference_date"] = reference_date.strftime("%Y-%m-%d")
        new_qa["applicable_laws"] = [l["id"] for l in main_laws]
        
        new_dataset.append(new_qa)
    
    return new_dataset


# ============================================================================
# MAIN: Ví dụ sử dụng
# ============================================================================
if __name__ == "__main__":
    # Test 1: Lấy luật hiện hành
    print("=" * 60)
    print("TEST 1: Lấy văn bản đang có hiệu lực")
    print("=" * 60)
    
    today = datetime(2025, 12, 1)
    applicable = get_applicable_laws(today)
    
    for law in applicable:
        print(f"✅ {law['id']}: {law['name']}")
    
    # Test 2: Detect citations
    print("\n" + "=" * 60)
    print("TEST 2: Detect trích dẫn luật trong text")
    print("=" * 60)
    
    sample_text = """
    Theo Điều 33 Luật Bảo hiểm xã hội số 41/2024/QH15 và 
    Nghị định 135/2024/NĐ-CP, mức đóng BHXH...
    Trước đây theo Luật 58/2014/QH13 thì quy định khác.
    """
    
    citations = detect_law_citations(sample_text)
    print(f"Tìm thấy: {citations}")
    
    for c in citations:
        valid = is_law_valid(c, today)
        status = "✅ Còn hiệu lực" if valid else "❌ Hết hiệu lực"
        print(f"  {c}: {status}")
    
    # Test 3: Build temporal prompt
    print("\n" + "=" * 60)
    print("TEST 3: Build prompt với temporal context")
    print("=" * 60)
    
    question = "Mức đóng BHXH bắt buộc của người lao động là bao nhiêu?"
    prompt = build_temporal_prompt(question, today)
    print(prompt)
