# -*- coding: utf-8 -*-
"""
Rescue Bad Q&A - Tự động thêm số hiệu văn bản vào câu trả lời thiếu
"""

import json
import re
import os

def extract_doc_number(source_doc: str) -> dict:
    """
    Trích xuất số hiệu văn bản từ tên file
    Ví dụ: "11_2025_TT-BNV_663712.txt" -> {"number": "11/2025/TT-BNV", "type": "Thông tư"}
    """
    # Pattern: XX_YYYY_LOAI-COQUAN_ID.txt
    patterns = [
        # Thông tư: 11_2025_TT-BNV_663712.txt
        (r'^(\d+)_(\d{4})_(TT)-([A-Z]+)_', 'Thông tư'),
        # Nghị định: 274_2025_ND-CP_653507.txt
        (r'^(\d+)_(\d{4})_(ND|NĐ)-CP_', 'Nghị định'),
        # Luật/QH: 74_2025_QH15_530912.txt
        (r'^(\d+)_(\d{4})_(QH\d+)_', 'Luật số'),
        # Quyết định: 2222_QD-BHXH_667488.txt
        (r'^(\d+)_(QD|QĐ)-([A-Z]+)_', 'Quyết định'),
    ]
    
    for pattern, doc_type in patterns:
        match = re.match(pattern, source_doc)
        if match:
            groups = match.groups()
            if doc_type == 'Thông tư':
                return {
                    "number": f"{groups[0]}/{groups[1]}/{groups[2]}-{groups[3]}",
                    "type": doc_type
                }
            elif doc_type == 'Nghị định':
                return {
                    "number": f"{groups[0]}/{groups[1]}/NĐ-CP",
                    "type": doc_type
                }
            elif doc_type == 'Luật số':
                return {
                    "number": f"{groups[0]}/{groups[1]}/{groups[2]}",
                    "type": doc_type
                }
            elif doc_type == 'Quyết định':
                return {
                    "number": f"{groups[0]}/{groups[1]}-{groups[2]}",
                    "type": doc_type
                }
    
    return None


def add_doc_number_to_answer(answer: str, doc_info: dict) -> str:
    """
    Thêm số hiệu văn bản vào câu trả lời
    """
    if not doc_info:
        return answer
    
    doc_number = doc_info["number"]
    doc_type = doc_info["type"]
    
    # Đã có số hiệu rồi thì bỏ qua
    if re.search(r'\d+/\d{4}/[A-Za-zĐ]+-?[A-Za-z]*', answer):
        return answer
    
    # Pattern để tìm vị trí cần chèn
    # "Căn cứ Điều X" -> "Căn cứ Điều X Thông tư 11/2025/TT-BNV"
    # "Căn cứ Điều X Khoản Y" -> "Căn cứ Điều X Khoản Y Thông tư 11/2025/TT-BNV"
    
    patterns_to_fix = [
        # Căn cứ Điều X Khoản Y: -> Căn cứ Điều X Khoản Y Thông tư XX:
        (r'(Căn cứ\s+Điều\s+\d+[a-z]?\s*,?\s*Khoản\s+\d+[a-z]?)(\s*:)', 
         rf'\1 {doc_type} {doc_number}\2'),
        
        # Căn cứ Điều X: -> Căn cứ Điều X Thông tư XX:
        (r'(Căn cứ\s+Điều\s+\d+[a-z]?)(\s*:)', 
         rf'\1 {doc_type} {doc_number}\2'),
        
        # Căn cứ khoản X Điều Y: 
        (r'(Căn cứ\s+[Kk]hoản\s+\d+[a-z]?\s*,?\s*Điều\s+\d+[a-z]?)(\s*:)', 
         rf'\1 {doc_type} {doc_number}\2'),
        
        # Căn cứ khoản X:
        (r'(Căn cứ\s+[Kk]hoản\s+\d+[a-z]?)(\s*:)', 
         rf'\1 {doc_type} {doc_number}\2'),
         
        # Theo Điều X Khoản Y,
        (r'(Theo\s+Điều\s+\d+[a-z]?\s*,?\s*[Kk]hoản\s+\d+[a-z]?)(\s*,)', 
         rf'\1 {doc_type} {doc_number}\2'),
        
        # Theo Điều X,
        (r'(Theo\s+Điều\s+\d+[a-z]?)(\s*,)', 
         rf'\1 {doc_type} {doc_number}\2'),
         
        # Căn cứ Ví dụ X Điều Y -> Căn cứ Ví dụ X Điều Y Thông tư
        (r'(Căn cứ\s+Ví dụ\s+\d+\s+Điều\s+\d+)(\s+)', 
         rf'\1 {doc_type} {doc_number}\2'),
    ]
    
    modified = answer
    for pattern, replacement in patterns_to_fix:
        modified = re.sub(pattern, replacement, modified, count=1)
        if modified != answer:
            break
    
    return modified


def rescue_bad_qa(bad_file: str, output_file: str):
    """
    Xử lý file bad Q&A và cứu những câu có thể fix được
    """
    print(f"📂 Loading: {bad_file}")
    
    with open(bad_file, 'r', encoding='utf-8') as f:
        bad_data = json.load(f)
    
    print(f"📊 Total bad Q&A: {len(bad_data)}")
    
    rescued = []
    still_bad = []
    
    for item in bad_data:
        source_doc = item.get('source_doc', '')
        answer = item.get('answer', '')
        question = item.get('question', '')
        
        # Trích xuất số hiệu từ source_doc
        doc_info = extract_doc_number(source_doc)
        
        if not doc_info:
            still_bad.append(item)
            continue
        
        # Thử fix answer
        fixed_answer = add_doc_number_to_answer(answer, doc_info)
        
        if fixed_answer != answer:
            # Đã fix được
            item['answer'] = fixed_answer
            item['rescued'] = True
            item['doc_added'] = doc_info['number']
            rescued.append(item)
        else:
            still_bad.append(item)
    
    print(f"\n✅ Rescued: {len(rescued)}")
    print(f"❌ Still bad: {len(still_bad)}")
    
    # Lưu rescued
    if rescued:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rescued, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved rescued to: {output_file}")
    
    # Show samples
    print("\n" + "="*60)
    print("📝 RESCUED SAMPLES (first 5):")
    print("="*60)
    for i, item in enumerate(rescued[:5]):
        print(f"\n[{i+1}] Doc: {item.get('doc_added', '')}")
        print(f"Q: {item['question'][:80]}...")
        print(f"A: {item['answer'][:150]}...")
    
    return rescued, still_bad


def merge_with_good(good_file: str, rescued_file: str, output_file: str):
    """
    Merge rescued Q&A với good Q&A
    """
    print(f"\n📂 Loading good: {good_file}")
    with open(good_file, 'r', encoding='utf-8') as f:
        good_data = json.load(f)
    
    print(f"📂 Loading rescued: {rescued_file}")
    with open(rescued_file, 'r', encoding='utf-8') as f:
        rescued_data = json.load(f)
    
    # Merge
    merged = good_data + rescued_data
    
    print(f"\n📊 Good: {len(good_data)}")
    print(f"📊 Rescued: {len(rescued_data)}")
    print(f"📊 Total merged: {len(merged)}")
    
    # Lưu
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved merged to: {output_file}")
    
    return merged


if __name__ == "__main__":
    # Paths
    bad_file = "./output/evaluated/qa_bad.json"
    rescued_file = "./output/evaluated/qa_rescued.json"
    good_file = "./output/evaluated/qa_good.json"
    merged_file = "./output/evaluated/qa_good_merged.json"
    
    # Step 1: Rescue bad Q&A
    rescued, still_bad = rescue_bad_qa(bad_file, rescued_file)
    
    # Step 2: Merge with good
    if rescued:
        merge_with_good(good_file, rescued_file, merged_file)
