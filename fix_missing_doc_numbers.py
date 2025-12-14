#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để fix các QA có "văn bản không có số hiệu" trong train.json
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Import Rescuer để dùng logic fix
import sys
sys.path.insert(0, str(Path(__file__).parent))

from steps.rescuer import QARescuer
from core.utils import load_json, save_json

def main():
    # Paths
    train_file = Path("output/split/train.json")
    extracted_docs_file = Path("output/extracted/extracted_documents.json")
    
    if not train_file.exists():
        print(f"❌ Không tìm thấy {train_file}")
        return
    
    if not extracted_docs_file.exists():
        print(f"❌ Không tìm thấy {extracted_docs_file}")
        return
    
    # Load data
    print(f"📂 Đang load {train_file}...")
    qa_list = load_json(train_file)
    print(f"✅ Đã load {len(qa_list)} QA")
    
    # Tạo Rescuer để dùng logic fix
    rescuer_config = {
        "output_dir": "./output/evaluated",
        "min_score": 8,
        "extracted_docs_path": str(extracted_docs_file)
    }
    rescuer = QARescuer(rescuer_config)
    
    # Tìm và fix các QA có "văn bản không có số hiệu"
    fixed_count = 0
    for qa in qa_list:
        answer = qa.get("answer", "")
        if "văn bản không có số hiệu" in answer.lower():
            source_doc = qa.get("source_doc", "")
            if not source_doc:
                continue
            
            # Thử rescue
            doc_info = rescuer._extract_doc_number(source_doc)
            if doc_info:
                fixed_answer = rescuer._add_doc_number_to_answer(answer, doc_info)
                if fixed_answer != answer:
                    qa["answer"] = fixed_answer
                    qa["fixed_by_postprocess"] = True
                    fixed_count += 1
                    print(f"✅ Fixed: {source_doc} - chunk {qa.get('chunk_id', '?')}")
    
    print(f"\n📊 Đã fix {fixed_count}/{len(qa_list)} QA")
    
    # Lưu lại
    if fixed_count > 0:
        # Backup file cũ
        backup_file = train_file.with_suffix('.json.bak')
        if not backup_file.exists():
            print(f"💾 Đang backup -> {backup_file}")
            save_json(qa_list, backup_file)
        
        # Lưu file mới
        print(f"💾 Đang lưu file đã fix -> {train_file}")
        save_json(qa_list, train_file)
        print("✅ Hoàn thành!")
    else:
        print("ℹ️ Không có QA nào cần fix")

if __name__ == "__main__":
    main()

