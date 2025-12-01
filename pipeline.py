# -*- coding: utf-8 -*-
"""
Dataset Pipeline - Orchestrator chính
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.config import PipelineConfig
from core.logger import get_logger, setup_logger
from core.utils import save_json, load_json
from steps.extractor import TextExtractor
from steps.generator import QAGenerator
from steps.evaluator import QualityEvaluator


logger = get_logger(__name__)


class DatasetPipeline:
    """
    Pipeline tạo dataset Q&A từ tài liệu.
    
    Các bước:
    1. Extract: Trích xuất text từ documents
    2. Generate: Sinh Q&A pairs bằng LLM
    3. Evaluate: Đánh giá và lọc chất lượng
    4. Export: Xuất dataset cuối cùng
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Đường dẫn tới file config
        """
        # Lưu config path để resolve relative paths
        self.config_path = os.path.abspath(config_path)
        
        # Load config
        self.config = PipelineConfig(config_path)
        
        # Setup logger với đường dẫn từ config
        log_file = self.config.get("general.log_file", "./logs/pipeline.log")
        if not os.path.isabs(log_file):
            log_file = os.path.join(os.path.dirname(self.config_path), log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        setup_logger(log_file)
        
        self.project_name = self.config.get("general.project_name", "dataset_project")
        
        logger.info(f"=== Pipeline: {self.project_name} ===")
        logger.info(f"Config: {config_path}")
        
        # State tracking
        self.state = {
            "started_at": None,
            "steps_completed": [],
            "current_step": None,
            "documents": [],
            "qa_pairs": [],
            "good_qa": [],
            "bad_qa": []
        }
        
        # Initialize steps
        self._init_steps()
    
    def _init_steps(self):
        """Khởi tạo các steps"""
        # Resolve paths
        base_dir = os.path.dirname(self.config_path)
        
        input_dir = self.config.get("general.input_dir", "../Luat")
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(base_dir, input_dir)
        
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_dir, output_dir)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "extracted"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "evaluated"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "final"), exist_ok=True)
        
        # Text Extractor
        proc_cfg = self.config.get("processing", {})
        docs_cfg = self.config.get("documents", {})
        
        extract_config = {
            "input_dir": input_dir,
            "output_dir": os.path.join(output_dir, "extracted"),
            "chunk_size": proc_cfg.get("chunk_size", 4000),
            "chunk_overlap": proc_cfg.get("chunk_overlap", 200),
            "document_mappings": docs_cfg.get("mappings", {}),
            "ignore_patterns": docs_cfg.get("ignore_patterns", [])
        }
        self.extractor = TextExtractor(extract_config)
        
        # QA Generator
        gen_config = self._build_generator_config()
        self.generator = QAGenerator(gen_config)
        
        # Quality Evaluator
        eval_config = self._build_evaluator_config()
        self.evaluator = QualityEvaluator(eval_config)
    
    def _build_generator_config(self) -> Dict[str, Any]:
        """Build config cho QA Generator từ YAML config"""
        # Lấy provider chính từ llm.provider
        provider_name = self.config.get("llm.provider", "gemini")
        
        # Lấy config của provider đó
        provider_cfg = self.config.get(f"llm.providers.{provider_name}", {})
        
        # Lấy config QA generation
        qa_cfg = self.config.get("qa_generation", {})
        
        # Lấy config processing
        proc_cfg = self.config.get("processing", {})
        
        # Load API keys từ file hoặc inline
        api_keys = []
        api_keys_file = provider_cfg.get("api_keys_file", "")
        if api_keys_file:
            # Resolve relative path
            if not os.path.isabs(api_keys_file):
                api_keys_file = os.path.join(
                    os.path.dirname(self.config_path), 
                    api_keys_file
                )
            api_keys = self._load_api_keys(api_keys_file)
        
        # Nếu không có file keys, check inline api_key
        if not api_keys:
            inline_key = provider_cfg.get("api_key", "")
            if inline_key:
                api_keys = [inline_key]
        
        # Get base_url cho custom providers
        base_url = provider_cfg.get("base_url", None)
        
        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        return {
            "provider": provider_name,
            "model": provider_cfg.get("model", "gemini-2.0-flash"),
            "api_keys": api_keys,
            "base_url": base_url,
            "generation": {
                "max_tokens": provider_cfg.get("max_output_tokens", provider_cfg.get("max_tokens", 3000)),
                "temperature": provider_cfg.get("temperature", 0.7),
                "top_p": provider_cfg.get("top_p", 0.9),
                "max_retries": proc_cfg.get("max_retries", 5),
                "retry_delay": proc_cfg.get("retry_delay", 2)
            },
            "num_qa_per_chunk": qa_cfg.get("qa_per_chunk", 5),
            "chunk_size": proc_cfg.get("chunk_size", 4000),
            "chunk_overlap": proc_cfg.get("chunk_overlap", 200),
            "prompt_template": qa_cfg.get("prompt_template"),
            "output_dir": os.path.join(output_dir, "generated"),
            "save_interval": proc_cfg.get("checkpoint_interval", 50),
            "request_delay": 60.0 / provider_cfg.get("rate_limit_per_minute", 60)
        }
    
    def _build_evaluator_config(self) -> Dict[str, Any]:
        """Build config cho Quality Evaluator từ YAML config"""
        # Lấy provider chính từ llm.provider
        provider_name = self.config.get("llm.provider", "gemini")
        
        # Lấy config của provider đó
        provider_cfg = self.config.get(f"llm.providers.{provider_name}", {})
        
        # Lấy config quality evaluation
        quality_cfg = self.config.get("quality", {})
        
        # Load API keys từ file hoặc inline
        api_keys = []
        api_keys_file = provider_cfg.get("api_keys_file", "")
        if api_keys_file:
            if not os.path.isabs(api_keys_file):
                api_keys_file = os.path.join(
                    os.path.dirname(self.config_path), 
                    api_keys_file
                )
            api_keys = self._load_api_keys(api_keys_file)
        
        if not api_keys:
            inline_key = provider_cfg.get("api_key", "")
            if inline_key:
                api_keys = [inline_key]
        
        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        return {
            "mode": "hybrid" if quality_cfg.get("use_llm_evaluation", False) else "rule",
            "min_score": quality_cfg.get("min_score", 3) * 2,  # Convert 1-5 scale to 1-10
            "min_question_length": 10,
            "min_answer_length": quality_cfg.get("criteria", [{}])[2].get("min_chars", 100) if len(quality_cfg.get("criteria", [])) > 2 else 100,
            "max_answer_length": 5000,
            "provider": provider_name,
            "model": provider_cfg.get("model", "gemini-2.0-flash"),
            "api_keys": api_keys,
            "generation": {
                "max_tokens": 512,
                "temperature": 0.3
            },
            "output_dir": os.path.join(output_dir, "evaluated"),
            "request_delay": 60.0 / provider_cfg.get("rate_limit_per_minute", 60)
        }
    
    def _load_api_keys(self, keys_file: str) -> List[str]:
        """Load API keys từ file"""
        if not keys_file or not os.path.exists(keys_file):
            return []
        
        keys = []
        with open(keys_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keys.append(line)
        
        logger.info(f"Loaded {len(keys)} API keys từ {keys_file}")
        return keys
    
    def run(self, steps: List[str] = None) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline hoặc các bước cụ thể
        
        Args:
            steps: List các bước cần chạy. None = tất cả.
                   Các bước: "extract", "generate", "evaluate", "export"
        
        Returns:
            Kết quả pipeline
        """
        self.state["started_at"] = datetime.now().isoformat()
        
        all_steps = ["extract", "generate", "evaluate", "export"]
        steps_to_run = steps or all_steps
        
        logger.info(f"Pipeline bắt đầu với các bước: {steps_to_run}")
        
        try:
            for step in steps_to_run:
                if step not in all_steps:
                    logger.warning(f"Bước không hợp lệ: {step}")
                    continue
                
                self.state["current_step"] = step
                logger.info(f"\n{'='*50}")
                logger.info(f"STEP: {step.upper()}")
                logger.info(f"{'='*50}")
                
                if step == "extract":
                    self._run_extract()
                elif step == "generate":
                    self._run_generate()
                elif step == "evaluate":
                    self._run_evaluate()
                elif step == "export":
                    self._run_export()
                
                self.state["steps_completed"].append(step)
            
            self._save_state()
            logger.info("\n=== Pipeline hoàn thành! ===")
            
            return self._get_summary()
            
        except Exception as e:
            logger.error(f"Pipeline lỗi: {e}")
            self._save_state()
            raise
    
    def _run_extract(self):
        """Bước 1: Extract text từ documents"""
        self.state["documents"] = self.extractor.extract_all()
        logger.info(f"Đã extract {len(self.state['documents'])} documents")
    
    def _run_generate(self):
        """Bước 2: Generate Q&A pairs"""
        # Load documents nếu chưa có
        if not self.state["documents"]:
            self.state["documents"] = self.extractor.load_extracted()
            if not self.state["documents"]:
                logger.error("Không có documents! Hãy chạy bước 'extract' trước.")
                return
        
        self.state["qa_pairs"] = self.generator.generate(self.state["documents"])
        logger.info(f"Đã generate {len(self.state['qa_pairs'])} Q&A pairs")
    
    def _run_evaluate(self):
        """Bước 3: Evaluate và filter"""
        # Load Q&A nếu chưa có
        if not self.state["qa_pairs"]:
            output_dir = self.config.get("general.output_dir", "./output")
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
            
            qa_file = os.path.join(output_dir, "generated", "qa_generated.json")
            if os.path.exists(qa_file):
                self.state["qa_pairs"] = load_json(qa_file)
            else:
                logger.error("Không có Q&A pairs! Hãy chạy bước 'generate' trước.")
                return
        
        good, bad = self.evaluator.evaluate(self.state["qa_pairs"])
        self.state["good_qa"] = good
        self.state["bad_qa"] = bad
        
        logger.info(f"Evaluation: {len(good)} good, {len(bad)} bad")
    
    def _run_export(self):
        """Bước 4: Export dataset cuối cùng"""
        # Load good Q&A nếu chưa có
        if not self.state["good_qa"]:
            self.state["good_qa"], self.state["bad_qa"] = self.evaluator.load_evaluated()
        
        if not self.state["good_qa"]:
            logger.error("Không có good Q&A! Hãy chạy bước 'evaluate' trước.")
            return
        
        # Format theo yêu cầu
        output_cfg = self.config.get("output", {})
        structure = output_cfg.get("structure", {})
        
        # Detect format type
        if structure.get("instruction_field"):
            export_format = "instruction"
        else:
            export_format = "chat"
        
        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        # Create final directory
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        final_dataset = self._format_dataset(self.state["good_qa"], export_format)
        
        # Lưu theo các formats được cấu hình
        formats = output_cfg.get("formats", ["json", "jsonl"])
        
        if "json" in formats:
            output_json = os.path.join(final_dir, "dataset_final.json")
            save_json(final_dataset, output_json)
        
        if "jsonl" in formats:
            output_jsonl = os.path.join(final_dir, "dataset_final.jsonl")
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                import json
                for item in final_dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Split train/val/test
        self._split_dataset(final_dataset, final_dir)
        
        logger.info(f"Exported {len(final_dataset)} samples -> {final_dir}")
    
    def _format_dataset(self, qa_pairs: List[Dict], format_type: str) -> List[Dict]:
        """Format dataset theo format type"""
        # Get system prompt từ config
        qa_cfg = self.config.get("qa_generation", {})
        system_prompt = "Bạn là trợ lý AI chuyên về lĩnh vực pháp luật Việt Nam, đặc biệt về bảo hiểm xã hội."
        
        # Get output structure từ config
        output_cfg = self.config.get("output", {})
        structure = output_cfg.get("structure", {})
        include_metadata = "source" in structure.get("metadata_fields", [])
        
        formatted = []
        
        for qa in qa_pairs:
            # Support both field naming conventions
            question = qa.get("question") or qa.get("instruction", "")
            answer = qa.get("answer") or qa.get("output", "")
            
            if format_type == "chat":
                # Chat format cho fine-tuning
                item = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
            elif format_type == "instruction":
                # Instruction format
                item = {
                    structure.get("instruction_field", "instruction"): question,
                    structure.get("input_field", "input"): qa.get("input", ""),
                    structure.get("output_field", "output"): answer
                }
            elif format_type == "simple":
                # Simple Q&A format
                item = {
                    "question": question,
                    "answer": answer
                }
            else:
                item = qa
            
            # Thêm metadata nếu cần
            if include_metadata:
                item["source"] = qa.get("source_doc") or qa.get("source", "")
                if "perspective" in structure.get("metadata_fields", []):
                    item["perspective"] = qa.get("perspective", "")
            
            formatted.append(item)
        
        return formatted
    
    def _split_dataset(self, dataset: List[Dict], output_dir: str):
        """Split thành train/val/test"""
        import random
        
        train_ratio = self.config.get("output.train_ratio", 0.8)
        val_ratio = self.config.get("output.val_ratio", 0.1)
        
        # Shuffle
        random.shuffle(dataset)
        
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = dataset[:train_end]
        val_data = dataset[train_end:val_end]
        test_data = dataset[val_end:]
        
        # Save
        for name, data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
            if data:
                save_json(data, os.path.join(output_dir, f"{name}.json"))
                # JSONL
                with open(os.path.join(output_dir, f"{name}.jsonl"), 'w', encoding='utf-8') as f:
                    import json
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    def _save_state(self):
        """Lưu state của pipeline"""
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        state_file = os.path.join(output_dir, "pipeline_state.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # Không lưu data lớn
        state_to_save = {
            "started_at": self.state["started_at"],
            "steps_completed": self.state["steps_completed"],
            "current_step": self.state["current_step"],
            "documents_count": len(self.state["documents"]),
            "qa_pairs_count": len(self.state["qa_pairs"]),
            "good_qa_count": len(self.state["good_qa"]),
            "bad_qa_count": len(self.state["bad_qa"])
        }
        
        save_json(state_to_save, state_file)
    
    def _get_summary(self) -> Dict[str, Any]:
        """Lấy summary của pipeline"""
        return {
            "project": self.project_name,
            "started_at": self.state["started_at"],
            "steps_completed": self.state["steps_completed"],
            "documents": len(self.state["documents"]),
            "qa_pairs_generated": len(self.state["qa_pairs"]),
            "good_qa": len(self.state["good_qa"]),
            "bad_qa": len(self.state["bad_qa"]),
            "success_rate": (
                len(self.state["good_qa"]) / max(1, len(self.state["qa_pairs"])) * 100
                if self.state["qa_pairs"] else 0
            )
        }
    
    def resume(self):
        """Resume pipeline từ state đã lưu"""
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        state_file = os.path.join(output_dir, "pipeline_state.json")
        
        if not os.path.exists(state_file):
            logger.warning("Không tìm thấy state file để resume")
            return self.run()
        
        saved_state = load_json(state_file)
        completed = saved_state.get("steps_completed", [])
        
        all_steps = ["extract", "generate", "evaluate", "export"]
        remaining = [s for s in all_steps if s not in completed]
        
        if not remaining:
            logger.info("Pipeline đã hoàn thành trước đó!")
            return self._get_summary()
        
        logger.info(f"Resume pipeline từ bước: {remaining[0]}")
        return self.run(steps=remaining)
