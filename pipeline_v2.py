# -*- coding: utf-8 -*-
"""
Dataset Pipeline V2 - Style-focused Training
============================================

Pipeline tạo dataset Q&A theo hướng tiếp cận mới:
- Dạy STYLE trả lời, không dạy CONTENT
- Format: ChatML (chuẩn công nghiệp)
- RAG-ready: Model học cách tổng hợp từ context
- Temporal validation: Lọc văn bản hết hiệu lực
"""

import os
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.config import PipelineConfig
from core.logger import get_logger, setup_logger
from core.utils import save_json, load_json, save_jsonl
from steps.extractor import TextExtractor
from steps.generator import QAGenerator
from steps.evaluator import QualityEvaluator


logger = get_logger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    CHATML = "chatml"
    INSTRUCTION = "instruction"
    ALPACA = "alpaca"


@dataclass
class QAPair:
    """Structured Q&A pair"""
    question: str
    answer: str
    question_type: str = ""
    cited_articles: List[str] = None
    source: str = ""
    temporal_status: str = "unknown"
    score: float = 0.0
    
    def to_chatml(self, system_prompt: str, include_context: bool = False, context: str = "") -> Dict:
        """Convert to ChatML format"""
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if include_context and context:
            messages.append({
                "role": "user", 
                "content": f"Văn bản tham khảo:\n{context}\n\nCâu hỏi: {self.question}"
            })
        else:
            messages.append({"role": "user", "content": self.question})
        
        messages.append({"role": "assistant", "content": self.answer})
        
        result = {"messages": messages}
        
        # Add metadata
        if self.source or self.question_type:
            result["metadata"] = {
                "source": self.source,
                "question_type": self.question_type,
                "cited_articles": self.cited_articles or [],
                "temporal_status": self.temporal_status
            }
        
        return result
    
    def to_instruction(self, system_prompt: str) -> Dict:
        """Convert to instruction format"""
        return {
            "system": system_prompt,
            "instruction": self.question,
            "input": "",
            "output": self.answer,
            "metadata": {
                "source": self.source,
                "question_type": self.question_type
            }
        }
    
    def to_alpaca(self) -> Dict:
        """Convert to Alpaca format"""
        return {
            "instruction": self.question,
            "input": "",
            "output": self.answer
        }


class DatasetPipelineV2:
    """
    Pipeline V2 - Style-focused Training
    
    Triết lý:
    - Fine-tune = Dạy CÁCH trả lời, không dạy NỘI DUNG
    - Model học: format, thuật ngữ, cấu trúc câu trả lời
    - RAG lo: cung cấp nội dung chính xác, cập nhật
    
    Các bước:
    1. Extract: Trích xuất text từ documents
    2. Generate: Sinh Q&A pairs với STYLE chuẩn
    3. Validate: Kiểm tra temporal + quality
    4. Format: Chuyển sang ChatML
    5. Export: Xuất dataset final
    """
    
    def __init__(self, config_path: str = "config_v2.yaml"):
        self.config_path = os.path.abspath(config_path)
        self.config = PipelineConfig(config_path)
        
        # Setup logging
        log_file = self.config.get("general.log_file", "./logs/pipeline_v2.log")
        if not os.path.isabs(log_file):
            log_file = os.path.join(os.path.dirname(self.config_path), log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        setup_logger(log_file)
        
        self.project_name = self.config.get("general.project_name", "bhxh-qa-v2")
        self.version = self.config.get("general.version", "2.0")
        
        logger.info(f"=== Pipeline V2: {self.project_name} (v{self.version}) ===")
        logger.info(f"Philosophy: Style-focused training")
        
        # Output format
        format_type = self.config.get("output_format.type", "chatml")
        self.output_format = OutputFormat(format_type)
        
        # System prompt
        self.system_prompt = self.config.get(
            "output_format.chatml.system_prompt",
            "Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam."
        )
        
        # State
        self.state = {
            "started_at": None,
            "steps_completed": [],
            "documents": [],
            "qa_pairs": [],
            "validated_qa": [],
            "final_dataset": []
        }
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize pipeline components"""
        base_dir = os.path.dirname(self.config_path)
        
        # Resolve paths
        input_dir = self._resolve_path(self.config.get("general.input_dir", "../Luat"))
        output_dir = self._resolve_path(self.config.get("general.output_dir", "./output_v2"))
        
        # Create directories
        for subdir in ["extracted", "generated", "validated", "formatted", "final"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        self.output_dir = output_dir
        
        # Text Extractor
        self.extractor = TextExtractor({
            "input_dir": input_dir,
            "output_dir": os.path.join(output_dir, "extracted"),
            "chunk_size": self.config.get("processing.chunk_size", 4000),
            "chunk_overlap": self.config.get("processing.chunk_overlap", 200),
            "document_mappings": self.config.get("documents.mappings", {}),
            "ignore_patterns": self.config.get("documents.ignore_patterns", [])
        })
        
        # QA Generator
        self.generator = QAGenerator(self._build_generator_config())
        
        # Quality Evaluator
        self.evaluator = QualityEvaluator(self._build_evaluator_config())
        
        # Laws Registry (for temporal validation)
        laws_registry_path = self._resolve_path(
            self.config.get("general.laws_registry", "../legal_knowledge/laws_registry.json")
        )
        self.laws_registry = self._load_laws_registry(laws_registry_path)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative path"""
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(self.config_path), path)
    
    def _load_laws_registry(self, path: str) -> Dict:
        """Load laws registry for temporal validation"""
        if os.path.exists(path):
            return load_json(path)
        logger.warning(f"Laws registry not found: {path}")
        return {"documents": []}
    
    def _build_generator_config(self) -> Dict[str, Any]:
        """Build QA Generator config"""
        provider_name = self.config.get("llm.provider", "gemini")
        provider_cfg = self.config.get(f"llm.providers.{provider_name}", {})
        
        api_keys = self._load_api_keys(provider_cfg.get("api_keys_file", ""))
        
        return {
            "provider": provider_name,
            "model": provider_cfg.get("model", "gemini-2.0-flash"),
            "api_keys": api_keys,
            "generation": {
                "max_tokens": provider_cfg.get("max_output_tokens", 4000),
                "temperature": provider_cfg.get("temperature", 0.7),
            },
            "num_qa_per_chunk": self.config.get("qa_generation.qa_per_chunk", 5),
            "chunk_size": self.config.get("processing.chunk_size", 4000),
            "prompt_template": self.config.get("qa_generation.prompt_template"),
            "output_dir": os.path.join(self.output_dir, "generated"),
            "request_delay": 60.0 / provider_cfg.get("rate_limit_per_minute", 60)
        }
    
    def _build_evaluator_config(self) -> Dict[str, Any]:
        """Build Quality Evaluator config"""
        provider_name = self.config.get("llm.provider", "gemini")
        provider_cfg = self.config.get(f"llm.providers.{provider_name}", {})
        
        return {
            "mode": "rule",  # V2 uses rule-based for style checking
            "min_score": self.config.get("quality.min_score", 7),
            "output_dir": os.path.join(self.output_dir, "validated")
        }
    
    def _load_api_keys(self, keys_file: str) -> List[str]:
        """Load API keys from file"""
        if not keys_file:
            return []
        
        path = self._resolve_path(keys_file)
        if not os.path.exists(path):
            return []
        
        keys = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keys.append(line)
        
        logger.info(f"Loaded {len(keys)} API keys from {path}")
        return keys
    
    def run(self, steps: List[str] = None) -> Dict[str, Any]:
        """
        Run pipeline
        
        Args:
            steps: Specific steps to run. None = all steps.
                   Steps: extract, generate, validate, format, export
        """
        self.state["started_at"] = datetime.now().isoformat()
        
        all_steps = ["extract", "generate", "validate", "format", "export"]
        steps_to_run = steps or all_steps
        
        logger.info(f"Pipeline V2 starting with steps: {steps_to_run}")
        
        try:
            for step in steps_to_run:
                logger.info(f"\n{'='*60}")
                logger.info(f"STEP: {step.upper()}")
                logger.info(f"{'='*60}")
                
                if step == "extract":
                    self._step_extract()
                elif step == "generate":
                    self._step_generate()
                elif step == "validate":
                    self._step_validate()
                elif step == "format":
                    self._step_format()
                elif step == "export":
                    self._step_export()
                
                self.state["steps_completed"].append(step)
            
            return self._get_summary()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
    
    def _step_extract(self):
        """Step 1: Extract text from documents"""
        self.state["documents"] = self.extractor.extract_all()
        logger.info(f"Extracted {len(self.state['documents'])} documents")
    
    def _step_generate(self):
        """Step 2: Generate Q&A pairs with proper STYLE"""
        if not self.state["documents"]:
            self.state["documents"] = self.extractor.load_extracted()
        
        if not self.state["documents"]:
            logger.error("No documents! Run 'extract' step first.")
            return
        
        self.state["qa_pairs"] = self.generator.generate(self.state["documents"])
        logger.info(f"Generated {len(self.state['qa_pairs'])} Q&A pairs")
    
    def _step_validate(self):
        """Step 3: Validate Q&A (temporal + quality)"""
        if not self.state["qa_pairs"]:
            qa_file = os.path.join(self.output_dir, "generated", "qa_generated.json")
            if os.path.exists(qa_file):
                self.state["qa_pairs"] = load_json(qa_file)
        
        if not self.state["qa_pairs"]:
            logger.error("No Q&A pairs! Run 'generate' step first.")
            return
        
        validated = []
        rejected = []
        
        for qa in self.state["qa_pairs"]:
            # Temporal validation
            temporal_status = self._check_temporal_validity(qa)
            qa["temporal_status"] = temporal_status
            
            # Style validation
            style_score = self._check_style_quality(qa)
            qa["style_score"] = style_score
            
            # Decision
            min_score = self.config.get("quality.min_score", 7)
            temporal_action = self.config.get(f"temporal_validation.actions.{temporal_status}", "keep")
            
            if temporal_action == "remove":
                qa["reject_reason"] = f"temporal:{temporal_status}"
                rejected.append(qa)
            elif style_score < min_score:
                qa["reject_reason"] = f"style_score:{style_score}"
                rejected.append(qa)
            else:
                validated.append(qa)
        
        self.state["validated_qa"] = validated
        
        # Save
        save_json(validated, os.path.join(self.output_dir, "validated", "qa_validated.json"))
        save_json(rejected, os.path.join(self.output_dir, "validated", "qa_rejected.json"))
        
        logger.info(f"Validation: {len(validated)} passed, {len(rejected)} rejected")
    
    def _check_temporal_validity(self, qa: Dict) -> str:
        """Check if Q&A cites valid (non-expired) laws"""
        import re
        
        answer = qa.get("answer", "") or qa.get("output", "")
        
        # Extract cited law IDs
        patterns = [
            r"(\d+/\d+/QH\d+)",
            r"(\d+/\d+/NĐ-CP)",
            r"(\d+/\d+/TT-[A-Z]+)",
        ]
        
        cited_ids = set()
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            cited_ids.update(matches)
        
        if not cited_ids:
            return "unknown"
        
        # Check against registry
        has_active = False
        has_expired = False
        
        for doc in self.laws_registry.get("documents", []):
            doc_id = doc.get("id", "")
            if doc_id in cited_ids:
                status = doc.get("status", "active")
                if status == "expired":
                    has_expired = True
                elif status == "active":
                    has_active = True
        
        if has_expired and has_active:
            return "mixed"
        elif has_expired:
            return "expired"
        elif has_active:
            return "current"
        else:
            return "unknown"
    
    def _check_style_quality(self, qa: Dict) -> float:
        """Check style quality (format, structure, terminology)"""
        import re
        
        answer = qa.get("answer", "") or qa.get("output", "")
        question = qa.get("question", "") or qa.get("instruction", "")
        
        score = 5.0  # Base score
        
        criteria = self.config.get("quality.criteria", [])
        
        for criterion in criteria:
            name = criterion.get("name", "")
            weight = criterion.get("weight", 1)
            check_type = criterion.get("check", "regex")
            
            if check_type == "regex":
                patterns = criterion.get("patterns", [])
                for pattern in patterns:
                    if re.search(pattern, answer, re.IGNORECASE):
                        score += weight
                        break
                        
            elif check_type == "length":
                min_len = criterion.get("min", 100)
                max_len = criterion.get("max", 1000)
                if min_len <= len(answer) <= max_len:
                    score += weight
                    
            elif check_type == "patterns":
                # Check question patterns
                good_patterns = criterion.get("good_patterns", [])
                bad_patterns = criterion.get("bad_patterns", [])
                
                has_good = any(re.search(p, question) for p in good_patterns)
                has_bad = any(re.search(p, question) for p in bad_patterns)
                
                if has_good and not has_bad:
                    score += weight
                elif has_bad:
                    score -= weight
        
        return min(10.0, max(0.0, score))
    
    def _step_format(self):
        """Step 4: Format to ChatML"""
        if not self.state["validated_qa"]:
            validated_file = os.path.join(self.output_dir, "validated", "qa_validated.json")
            if os.path.exists(validated_file):
                self.state["validated_qa"] = load_json(validated_file)
        
        if not self.state["validated_qa"]:
            logger.error("No validated Q&A! Run 'validate' step first.")
            return
        
        formatted = []
        
        for qa in self.state["validated_qa"]:
            qa_pair = QAPair(
                question=qa.get("question") or qa.get("instruction", ""),
                answer=qa.get("answer") or qa.get("output", ""),
                question_type=qa.get("question_type", ""),
                cited_articles=qa.get("cited_articles", []),
                source=qa.get("source_doc") or qa.get("source", ""),
                temporal_status=qa.get("temporal_status", "unknown"),
                score=qa.get("style_score", 0)
            )
            
            if self.output_format == OutputFormat.CHATML:
                formatted.append(qa_pair.to_chatml(self.system_prompt))
            elif self.output_format == OutputFormat.INSTRUCTION:
                formatted.append(qa_pair.to_instruction(self.system_prompt))
            else:
                formatted.append(qa_pair.to_alpaca())
        
        self.state["final_dataset"] = formatted
        
        # Save
        save_json(formatted, os.path.join(self.output_dir, "formatted", "dataset_chatml.json"))
        logger.info(f"Formatted {len(formatted)} samples to {self.output_format.value}")
    
    def _step_export(self):
        """Step 5: Export final dataset with train/val/test split"""
        if not self.state["final_dataset"]:
            formatted_file = os.path.join(self.output_dir, "formatted", "dataset_chatml.json")
            if os.path.exists(formatted_file):
                self.state["final_dataset"] = load_json(formatted_file)
        
        if not self.state["final_dataset"]:
            logger.error("No formatted dataset! Run 'format' step first.")
            return
        
        dataset = self.state["final_dataset"]
        
        # Deduplicate
        if self.config.get("output.deduplicate.enabled", True):
            dataset = self._deduplicate(dataset)
        
        # Shuffle
        random.shuffle(dataset)
        
        # Split
        train_ratio = self.config.get("output.train_ratio", 0.85)
        val_ratio = self.config.get("output.val_ratio", 0.10)
        
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            "train": dataset[:train_end],
            "validation": dataset[train_end:val_end],
            "test": dataset[val_end:]
        }
        
        # Save
        final_dir = os.path.join(self.output_dir, "final")
        
        for split_name, split_data in splits.items():
            if split_data:
                save_json(split_data, os.path.join(final_dir, f"{split_name}.json"))
                save_jsonl(split_data, os.path.join(final_dir, f"{split_name}.jsonl"))
        
        # Save all
        save_json(dataset, os.path.join(final_dir, "dataset_all.json"))
        save_jsonl(dataset, os.path.join(final_dir, "dataset_all.jsonl"))
        
        logger.info(f"Exported: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
    
    def _deduplicate(self, dataset: List[Dict]) -> List[Dict]:
        """Remove duplicate Q&A pairs"""
        seen = set()
        unique = []
        
        for item in dataset:
            # Get question text
            if "messages" in item:
                q = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
            else:
                q = item.get("instruction", item.get("question", ""))
            
            # Simple hash
            q_hash = hash(q.lower().strip())
            
            if q_hash not in seen:
                seen.add(q_hash)
                unique.append(item)
        
        removed = len(dataset) - len(unique)
        if removed > 0:
            logger.info(f"Deduplicated: removed {removed} duplicates")
        
        return unique
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary"""
        return {
            "project": self.project_name,
            "version": self.version,
            "format": self.output_format.value,
            "started_at": self.state["started_at"],
            "steps_completed": self.state["steps_completed"],
            "documents": len(self.state["documents"]),
            "qa_generated": len(self.state["qa_pairs"]),
            "qa_validated": len(self.state["validated_qa"]),
            "final_dataset": len(self.state["final_dataset"])
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dataset Pipeline V2 - Style-focused Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_v2.py                    # Run all steps
  python pipeline_v2.py -s extract         # Extract only
  python pipeline_v2.py -s generate validate format export
  python pipeline_v2.py -c custom_config.yaml
        """
    )
    
    parser.add_argument("-c", "--config", default="config_v2.yaml", help="Config file")
    parser.add_argument("-s", "--steps", nargs="+", 
                       choices=["extract", "generate", "validate", "format", "export"],
                       help="Steps to run")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        return
    
    pipeline = DatasetPipelineV2(args.config)
    result = pipeline.run(steps=args.steps)
    
    print("\n" + "="*60)
    print("📊 PIPELINE V2 SUMMARY")
    print("="*60)
    print(f"  Format: {result['format']}")
    print(f"  Documents: {result['documents']}")
    print(f"  Q&A Generated: {result['qa_generated']}")
    print(f"  Q&A Validated: {result['qa_validated']}")
    print(f"  Final Dataset: {result['final_dataset']}")
    print("="*60)


if __name__ == "__main__":
    main()
