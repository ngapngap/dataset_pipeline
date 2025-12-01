# -*- coding: utf-8 -*-
"""
Pipeline Enhancement: Temporal Legal Validator
Tích hợp vào dataset_pipeline để lọc QA dựa trên hiệu lực văn bản pháp luật

CÁCH SỬ DỤNG:
1. Copy file này vào dataset_pipeline/steps/
2. Import và gọi trong pipeline.py
3. Hoặc chạy standalone để lọc dataset đã có
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.utils import save_json, load_json

logger = get_logger(__name__)


class TemporalLegalValidator:
    """
    Validator kiểm tra và lọc QA dựa trên hiệu lực thời gian của văn bản pháp luật.
    
    Giải quyết vấn đề:
    - Dataset có cả luật cũ (58/2014) và luật mới (41/2024)
    - Model học cả 2 → không biết trả lời theo luật nào
    - Cần lọc chỉ giữ QA trích dẫn luật còn hiệu lực
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: {
                "registry_path": path to laws_registry.json,
                "reference_date": "YYYY-MM-DD" (ngày tham chiếu),
                "output_dir": output directory,
                "strict_mode": True = loại bỏ tất cả QA có luật cũ
            }
        """
        self.config = config or {}
        
        # Load laws registry
        registry_path = self.config.get(
            "registry_path", 
            Path(__file__).parent.parent.parent / "legal_knowledge" / "laws_registry.json"
        )
        self.registry = self._load_registry(registry_path)
        
        # Reference date
        ref_date_str = self.config.get("reference_date", "2025-12-01")
        self.reference_date = datetime.strptime(ref_date_str, "%Y-%m-%d")
        
        # Output
        self.output_dir = self.config.get("output_dir", "./output/validated")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Mode
        self.strict_mode = self.config.get("strict_mode", True)
        
        # Build lookup
        self._build_law_lookup()
        
        logger.info(f"TemporalLegalValidator initialized")
        logger.info(f"  Reference date: {self.reference_date.strftime('%d/%m/%Y')}")
        logger.info(f"  Laws in registry: {len(self.registry.get('laws', []))}")
        logger.info(f"  Active laws: {len(self.active_laws)}")
        logger.info(f"  Expired laws: {len(self.expired_laws)}")
    
    def _load_registry(self, path) -> Dict:
        """Load laws registry từ file"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Registry not found: {path}")
            return {"laws": []}
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _build_law_lookup(self):
        """Build lookup tables cho active/expired laws"""
        self.active_laws = {}  # law_id -> law_info
        self.expired_laws = {}
        self.all_laws = {}
        
        for law in self.registry.get("laws", []):
            law_id = law["id"]
            self.all_laws[law_id] = law
            
            # Check if active at reference date
            effective = datetime.strptime(law["effective_date"], "%Y-%m-%d")
            expiry = law.get("expiry_date")
            
            is_active = effective <= self.reference_date
            if expiry:
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                is_active = is_active and expiry_date > self.reference_date
            
            if is_active:
                self.active_laws[law_id] = law
            else:
                self.expired_laws[law_id] = law
        
        # Build regex patterns for detection
        self._build_detection_patterns()
    
    def _build_detection_patterns(self):
        """Build regex patterns để detect law citations trong text"""
        self.law_patterns = []
        
        # Pattern cho từng loại văn bản
        patterns = [
            (r'(\d+/\d{4}/QH\d+)', 'Luật'),           # 41/2024/QH15
            (r'(\d+/\d{4}/NĐ-CP)', 'Nghị định'),     # 135/2024/NĐ-CP
            (r'(\d+/\d{4}/TT-[A-Z]+)', 'Thông tư'),  # 59/2024/TT-BLĐTBXH
            (r'(\d+/QĐ-[A-Z]+)', 'Quyết định'),      # 1234/QĐ-BHXH
        ]
        
        for pattern, law_type in patterns:
            self.law_patterns.append((re.compile(pattern), law_type))
    
    def detect_citations(self, text: str) -> List[Dict]:
        """
        Detect tất cả law citations trong text
        
        Returns:
            List[{
                "id": "41/2024/QH15",
                "type": "Luật",
                "status": "active" | "expired" | "unknown",
                "info": law info from registry (if found)
            }]
        """
        citations = []
        seen = set()
        
        for pattern, law_type in self.law_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    
                    # Lookup in registry
                    status = "unknown"
                    info = None
                    
                    if match in self.active_laws:
                        status = "active"
                        info = self.active_laws[match]
                    elif match in self.expired_laws:
                        status = "expired"
                        info = self.expired_laws[match]
                    
                    citations.append({
                        "id": match,
                        "type": law_type,
                        "status": status,
                        "info": info
                    })
        
        return citations
    
    def validate_qa(self, qa: Dict) -> Tuple[bool, Dict]:
        """
        Validate một QA pair
        
        Args:
            qa: {"question": ..., "answer": ...} hoặc {"instruction": ..., "output": ...}
        
        Returns:
            (is_valid, details)
        """
        # Get text from QA
        question = qa.get("question") or qa.get("instruction", "")
        answer = qa.get("answer") or qa.get("output", "")
        text = f"{question}\n{answer}"
        
        # Detect citations
        citations = self.detect_citations(text)
        
        # Analyze
        active_citations = [c for c in citations if c["status"] == "active"]
        expired_citations = [c for c in citations if c["status"] == "expired"]
        unknown_citations = [c for c in citations if c["status"] == "unknown"]
        
        # Determine validity
        if self.strict_mode:
            # Strict: Loại bỏ nếu có BẤT KỲ luật hết hiệu lực
            is_valid = len(expired_citations) == 0
        else:
            # Lenient: Chỉ loại bỏ nếu CHỈ CÓ luật hết hiệu lực
            is_valid = len(active_citations) > 0 or len(expired_citations) == 0
        
        details = {
            "is_valid": is_valid,
            "total_citations": len(citations),
            "active": [c["id"] for c in active_citations],
            "expired": [c["id"] for c in expired_citations],
            "unknown": [c["id"] for c in unknown_citations],
            "reason": None
        }
        
        if not is_valid:
            details["reason"] = f"Trích dẫn luật hết hiệu lực: {', '.join(details['expired'])}"
        
        return is_valid, details
    
    def validate_dataset(self, qa_list: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Validate toàn bộ dataset
        
        Args:
            qa_list: List các QA pairs
        
        Returns:
            (valid_qa, invalid_qa, stats)
        """
        logger.info(f"Validating {len(qa_list)} QA pairs...")
        
        valid_qa = []
        invalid_qa = []
        
        # Stats
        stats = {
            "total": len(qa_list),
            "valid": 0,
            "invalid": 0,
            "expired_laws_found": {},
            "active_laws_found": {}
        }
        
        for qa in qa_list:
            is_valid, details = self.validate_qa(qa)
            
            # Add validation details to QA
            qa_with_details = qa.copy()
            qa_with_details["_validation"] = details
            
            if is_valid:
                valid_qa.append(qa_with_details)
                stats["valid"] += 1
                
                # Count active laws
                for law_id in details["active"]:
                    stats["active_laws_found"][law_id] = stats["active_laws_found"].get(law_id, 0) + 1
            else:
                invalid_qa.append(qa_with_details)
                stats["invalid"] += 1
                
                # Count expired laws
                for law_id in details["expired"]:
                    stats["expired_laws_found"][law_id] = stats["expired_laws_found"].get(law_id, 0) + 1
        
        # Summary
        stats["valid_rate"] = stats["valid"] / max(1, stats["total"]) * 100
        
        logger.info(f"Validation complete:")
        logger.info(f"  ✅ Valid: {stats['valid']} ({stats['valid_rate']:.1f}%)")
        logger.info(f"  ❌ Invalid: {stats['invalid']}")
        
        if stats["expired_laws_found"]:
            logger.info(f"  📜 Expired laws found:")
            for law_id, count in sorted(stats["expired_laws_found"].items(), key=lambda x: -x[1])[:5]:
                logger.info(f"     - {law_id}: {count} QA pairs")
        
        return valid_qa, invalid_qa, stats
    
    def add_temporal_context(self, qa_list: List[Dict]) -> List[Dict]:
        """
        Thêm temporal context vào mỗi QA
        
        Format mới:
        {
            "question": "### Ngày tham chiếu: 01/12/2025\n### Câu hỏi:\n{original_question}",
            "answer": "...",
            "reference_date": "2025-12-01",
            "applicable_laws": ["41/2024/QH15", ...]
        }
        """
        logger.info(f"Adding temporal context to {len(qa_list)} QA pairs...")
        
        # Get list of active laws for context
        main_laws = [l["id"] for l in self.active_laws.values() if l["type"] == "Luật"][:3]
        
        context_header = f"### Ngày tham chiếu: {self.reference_date.strftime('%d/%m/%Y')}\n"
        context_header += f"### Văn bản áp dụng: {', '.join(main_laws)}\n\n"
        
        enhanced_qa = []
        for qa in qa_list:
            new_qa = qa.copy()
            
            # Add context to question
            question = qa.get("question") or qa.get("instruction", "")
            new_qa["question"] = context_header + "### Câu hỏi:\n" + question
            
            # Add metadata
            new_qa["reference_date"] = self.reference_date.strftime("%Y-%m-%d")
            
            # Detect and add applicable laws
            citations = self.detect_citations(qa.get("answer") or qa.get("output", ""))
            new_qa["applicable_laws"] = [c["id"] for c in citations if c["status"] == "active"]
            
            enhanced_qa.append(new_qa)
        
        return enhanced_qa
    
    def process(self, qa_list: List[Dict], add_context: bool = True) -> Dict:
        """
        Full processing pipeline:
        1. Validate & filter
        2. Add temporal context (optional)
        3. Save results
        
        Returns:
            Summary with paths to output files
        """
        # Step 1: Validate
        valid_qa, invalid_qa, stats = self.validate_dataset(qa_list)
        
        # Step 2: Add context
        if add_context and valid_qa:
            valid_qa = self.add_temporal_context(valid_qa)
        
        # Step 3: Save
        output_files = {}
        
        if valid_qa:
            valid_path = os.path.join(self.output_dir, "qa_valid.json")
            save_json(valid_qa, valid_path)
            output_files["valid"] = valid_path
            
            # Also save without validation metadata for training
            clean_qa = []
            for qa in valid_qa:
                clean = {k: v for k, v in qa.items() if not k.startswith("_")}
                clean_qa.append(clean)
            
            clean_path = os.path.join(self.output_dir, "qa_clean_for_training.json")
            save_json(clean_qa, clean_path)
            output_files["clean"] = clean_path
        
        if invalid_qa:
            invalid_path = os.path.join(self.output_dir, "qa_invalid.json")
            save_json(invalid_qa, invalid_path)
            output_files["invalid"] = invalid_path
        
        # Save stats
        stats_path = os.path.join(self.output_dir, "validation_stats.json")
        save_json(stats, stats_path)
        output_files["stats"] = stats_path
        
        return {
            "stats": stats,
            "output_files": output_files,
            "valid_count": len(valid_qa),
            "invalid_count": len(invalid_qa)
        }


# =============================================================================
# STANDALONE USAGE
# =============================================================================
def main():
    """Chạy validator standalone trên dataset đã có"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate QA dataset against law validity")
    parser.add_argument("--input", "-i", required=True, help="Input QA JSON file")
    parser.add_argument("--output", "-o", default="./output/validated", help="Output directory")
    parser.add_argument("--registry", "-r", help="Path to laws_registry.json")
    parser.add_argument("--date", "-d", default="2025-12-01", help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--lenient", action="store_true", help="Use lenient mode (keep mixed citations)")
    parser.add_argument("--no-context", action="store_true", help="Don't add temporal context")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    
    print(f"📂 Loaded {len(qa_list)} QA pairs from {args.input}")
    
    # Config
    config = {
        "output_dir": args.output,
        "reference_date": args.date,
        "strict_mode": not args.lenient
    }
    if args.registry:
        config["registry_path"] = args.registry
    
    # Process
    validator = TemporalLegalValidator(config)
    result = validator.process(qa_list, add_context=not args.no_context)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"✅ Valid:   {result['valid_count']}")
    print(f"❌ Invalid: {result['invalid_count']}")
    print(f"📈 Rate:    {result['stats']['valid_rate']:.1f}%")
    print("\n📁 Output files:")
    for name, path in result["output_files"].items():
        print(f"   {name}: {path}")


if __name__ == "__main__":
    main()
