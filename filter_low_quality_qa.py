#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để filter và loại bỏ các QA có câu hỏi không có ngữ cảnh rõ ràng
"""

import json
import re
from pathlib import Path
from typing import List, Dict

from core.utils import load_json, save_json

def is_low_quality_question(question: str) -> bool:
    """
    Kiểm tra câu hỏi có phải là câu hỏi lý thuyết, không có ngữ cảnh rõ ràng không
    
    Returns:
        True nếu là câu hỏi chất lượng thấp (cần loại bỏ)
    """
    question_lower = question.lower()
    
    # Patterns cho câu hỏi lý thuyết, không có ngữ cảnh
    low_quality_patterns = [
        # "Trong văn bản này, X được viết tắt cho cụm từ gì?"
        r'trong\s+văn\s+bản\s+này.*được\s+viết\s+tắt',
        r'được\s+viết\s+tắt\s+cho\s+cụm\s+từ',
        r'chữ\s+viết\s+tắt',
        
        # "Văn bản này quy định về nội dung gì?"
        r'văn\s+bản\s+này\s+quy\s+định\s+về',
        r'văn\s+bản\s+này\s+nói\s+về',
        r'văn\s+bản\s+này\s+có\s+nội\s+dung',
        r'nội\s+dung.*văn\s+bản\s+này',
        
        # "Văn bản này là gì?"
        r'văn\s+bản\s+này\s+là\s+gì',
        
        # "Điều X của văn bản này quy định gì?"
        r'điều.*của\s+văn\s+bản\s+này\s+quy\s+định',
        
        # Câu hỏi quá chung chung
        r'văn\s+bản\s+này.*là\s+gì',
        r'văn\s+bản\s+này.*quy\s+định\s+gì',
    ]
    
    return any(re.search(pattern, question_lower) for pattern in low_quality_patterns)

def filter_qa_list(qa_list: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    Filter QA list, loại bỏ những câu hỏi chất lượng thấp
    
    Returns:
        (good_qa, removed_qa)
    """
    good_qa = []
    removed_qa = []
    
    for qa in qa_list:
        question = qa.get("question", "")
        if is_low_quality_question(question):
            removed_qa.append(qa)
        else:
            good_qa.append(qa)
    
    return good_qa, removed_qa

def main():
    # Paths
    train_file = Path("output/split/train.json")
    validation_file = Path("output/split/validation.json")
    test_file = Path("output/split/test.json")
    
    files_to_process = []
    if train_file.exists():
        files_to_process.append(("train", train_file))
    if validation_file.exists():
        files_to_process.append(("validation", validation_file))
    if test_file.exists():
        files_to_process.append(("test", test_file))
    
    if not files_to_process:
        print("❌ Không tìm thấy file nào để xử lý")
        return
    
    total_removed = 0
    total_kept = 0
    
    for split_name, file_path in files_to_process:
        print(f"\n📂 Đang xử lý {split_name} ({file_path})...")
        
        # Load
        qa_list = load_json(file_path)
        print(f"   Tổng: {len(qa_list)} QA")
        
        # Filter
        good_qa, removed_qa = filter_qa_list(qa_list)
        
        print(f"   ✅ Giữ lại: {len(good_qa)} QA")
        print(f"   ❌ Loại bỏ: {len(removed_qa)} QA")
        
        if removed_qa:
            # Hiển thị một vài ví dụ
            print(f"\n   📋 Ví dụ câu hỏi bị loại bỏ:")
            for i, qa in enumerate(removed_qa[:3]):
                print(f"      [{i+1}] {qa.get('question', '')[:80]}...")
        
        # Lưu lại
        if len(removed_qa) > 0:
            # Backup
            backup_file = file_path.with_suffix('.json.bak2')
            if not backup_file.exists():
                print(f"   💾 Đang backup -> {backup_file}")
                save_json(qa_list, backup_file)
            
            # Lưu file đã filter
            print(f"   💾 Đang lưu file đã filter -> {file_path}")
            save_json(good_qa, file_path)
            
            # Lưu removed QA để review
            removed_file = file_path.parent / f"{split_name}_removed_low_quality.json"
            save_json(removed_qa, removed_file)
            print(f"   📝 Đã lưu removed QA -> {removed_file}")
        
        total_kept += len(good_qa)
        total_removed += len(removed_qa)
    
    print(f"\n📊 Tổng kết:")
    print(f"   ✅ Giữ lại: {total_kept} QA")
    print(f"   ❌ Loại bỏ: {total_removed} QA")
    print(f"   📉 Tỷ lệ loại bỏ: {total_removed/(total_kept+total_removed)*100:.1f}%")

if __name__ == "__main__":
    main()

