# -*- coding: utf-8 -*-
"""
Dataset Pipeline V2 - Orchestrator chính

THAY ĐỔI SO VỚI V1:
1. Thêm bước "split" - Document-based split để tránh data leakage
2. Thêm bước "tokenize" - Tokenization với labels cho Causal LM
3. Cải thiện export với nhiều format hơn
4. Input validation trước khi chạy
5. Error handling và graceful degradation
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.config import PipelineConfig
from core.logger import get_logger, setup_logger
from core.utils import save_json, load_json
from core.validators import ConfigValidator, ValidationResult
from core.errors import (
    ConfigurationError, 
    PipelineError, 
    ErrorSummary
)
from steps.extractor import TextExtractor
from steps.generator import QAGenerator
from steps.evaluator import QualityEvaluator
from steps.rescuer import QARescuer
from steps.regenerator import ChunkRegenerator
from steps.filter import QAFilter
from steps.splitter import DatasetSplitter

# Optional: Dashboard metrics
try:
    from dashboard.metrics import MetricsCollector, get_metrics_collector
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    MetricsCollector = None


logger = get_logger(__name__)


class DatasetPipeline:
    """
    Pipeline V2 tạo dataset Q&A từ tài liệu.

    Các bước:
    1. Extract: Trích xuất text từ documents
    2. Generate: Sinh Q&A pairs bằng LLM
    3. Evaluate: Đánh giá chất lượng + Rescue + Regenerate (all-in-one)
       - Evaluate: Đánh giá và lọc chất lượng Q&A
       - Rescue: Cứu vớt Q&A có thể sửa được (thêm số hiệu văn bản)
       - Regenerate: Tái tạo Q&A cho chunks kém chất lượng
       - Loop cho đến khi hết chunks cần regenerate hoặc max iterations
    4. Split: Chia dataset theo document (tránh data leakage)
    5. Export: Xuất dataset cuối cùng (JSON, JSONL)

    Note: Tokenize được bỏ qua vì mỗi model có tokenizer riêng.
    """
    
    def __init__(
        self, 
        config_path: str = "config.yaml",
        skip_validation: bool = False
    ) -> None:
        """
        Args:
            config_path: Đường dẫn tới file config
            skip_validation: Bỏ qua validation (không khuyến khích)
            
        Raises:
            ConfigurationError: Nếu config không hợp lệ
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
        
        # Validate config
        if not skip_validation:
            self._validate_config()
        
        # Error tracking
        self.error_summary = ErrorSummary()
        
        # Metrics collector for dashboard
        self.metrics: Optional[Any] = None
        if DASHBOARD_AVAILABLE:
            self.metrics = get_metrics_collector()
        
        # State tracking
        self.state: Dict[str, Any] = {
            "started_at": None,
            "steps_completed": [],
            "current_step": None,
            "documents": [],
            "qa_pairs": [],
            "good_qa": [],
            "bad_qa": [],
            "rescued_qa": [],
            "splits": {},  # V2: train/val/test splits
        }

        # Initialize steps
        self._init_steps()
    
    def _validate_config(self) -> None:
        """Validate config trước khi chạy
        
        Raises:
            ConfigurationError: Nếu có lỗi validation
        """
        logger.info("Validating config...")
        
        validator = ConfigValidator(self.config)
        result = validator.validate_all()
        
        # Log warnings
        for warning in result.warnings:
            logger.warning(f"[{warning.field}] {warning.message}")
            if warning.suggestion:
                logger.warning(f"  Suggestion: {warning.suggestion}")
        
        # Check errors
        if not result.is_valid:
            error_messages = result.get_error_messages()
            for err in result.errors:
                logger.error(f"[{err.field}] {err.message}")
                if err.suggestion:
                    logger.error(f"  Suggestion: {err.suggestion}")
            
            raise ConfigurationError(
                message=f"Config validation failed with {len(result.errors)} error(s)",
                field="config"
            )
        
        logger.info("✅ Config validation passed")
    
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
        os.makedirs(os.path.join(output_dir, "split"), exist_ok=True)  # V2
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
        
        # Q&A Rescuer
        rescuer_config = self._build_rescuer_config()
        self.rescuer = QARescuer(rescuer_config)
        
        # Chunk Regenerator
        regenerator_config = self._build_regenerator_config()
        self.regenerator = ChunkRegenerator(regenerator_config)

        # QA Filter (loại bỏ câu hỏi không có ngữ cảnh rõ ràng)
        filter_config = self._build_filter_config()
        self.filter = QAFilter(filter_config)

        # V2: Dataset Splitter (document-based split)
        splitter_config = self._build_splitter_config()
        self.splitter = DatasetSplitter(splitter_config)

    
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
            # Threading config
            "threads_per_key": proc_cfg.get("threads_per_key", 1),
            "max_threads": proc_cfg.get("max_threads", 50),
            # QA config
            "num_qa_per_chunk": qa_cfg.get("qa_per_chunk", 5),
            "chunk_size": proc_cfg.get("chunk_size", 4000),
            "chunk_overlap": proc_cfg.get("chunk_overlap", 200),
            "prompt_template": qa_cfg.get("prompt_template"),
            "output_dir": os.path.join(output_dir, "generated"),
            "save_interval": proc_cfg.get("checkpoint_interval", 50),
            "request_delay": 60.0 / provider_cfg.get("rate_limit_per_minute", 60),
            "cache": proc_cfg.get("cache", {"enabled": True, "cache_dir": "./cache", "ttl_days": 30})
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
    
    def _build_rescuer_config(self) -> Dict[str, Any]:
        """Build config cho Q&A Rescuer từ YAML config"""
        quality_cfg = self.config.get("quality", {})
        
        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
        
        # Path đến extracted documents để trích số hiệu từ nội dung
        extracted_docs_path = os.path.join(output_dir, "extracted", "extracted_documents.json")
        
        return {
            "output_dir": os.path.join(output_dir, "evaluated"),
            "min_score": quality_cfg.get("min_score", 4) * 2,  # Convert 1-5 scale to 1-10
            "extracted_docs_path": extracted_docs_path,
        }
    
    def _build_regenerator_config(self) -> Dict[str, Any]:
        """Build config cho Chunk Regenerator từ YAML config"""
        quality_cfg = self.config.get("quality", {})
        regenerate_cfg = self.config.get("regenerate", {})

        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        return {
            "output_dir": output_dir,
            "min_good_rate": regenerate_cfg.get("min_good_rate", 0.5),  # 50%
            "min_good_count": regenerate_cfg.get("min_good_count", 1),
        }

    def _build_filter_config(self) -> Dict[str, Any]:
        """Build config cho QA Filter"""
        filter_cfg = self.config.get("filter", {})
        return {
            "enabled": filter_cfg.get("enabled", True)
        }
    
    def _build_splitter_config(self) -> Dict[str, Any]:
        """Build config cho Dataset Splitter từ YAML config"""
        output_cfg = self.config.get("output", {})

        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        return {
            "train_ratio": output_cfg.get("train_ratio", 0.70),
            "val_ratio": output_cfg.get("val_ratio", 0.15),
            "test_ratio": output_cfg.get("test_ratio", 0.15),
            "seed": self.config.get("general.seed", 42),
            "output_dir": os.path.join(output_dir, "split")
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

    def run_auto(self) -> Dict[str, Any]:
        """
        Chạy pipeline hoàn toàn tự động.

        Tự động kiểm tra từng bước:
        - Nếu đã có output → skip
        - Nếu chưa có → chạy

        Returns:
            Kết quả pipeline
        """
        self.state["started_at"] = datetime.now().isoformat()

        all_steps = ["extract", "generate", "evaluate", "split", "export"]

        logger.info("Pipeline V2 AUTO MODE - Tự động kiểm tra và chạy các bước cần thiết")

        try:
            for step in all_steps:
                self.state["current_step"] = step

                # Kiểm tra step đã hoàn thành chưa
                if self._is_step_completed(step):
                    logger.info(f"\n⏭️  SKIP: {step.upper()} (đã hoàn thành)")
                    self._load_step_results(step)
                    continue

                logger.info(f"\n{'='*50}")
                logger.info(f"STEP: {step.upper()}")
                logger.info(f"{'='*50}")

                if step == "extract":
                    self._run_extract()
                elif step == "generate":
                    self._run_generate()
                elif step == "evaluate":
                    self._run_evaluate()
                elif step == "split":
                    self._run_split()
                elif step == "export":
                    self._run_export()

                self.state["steps_completed"].append(step)

            self._save_state()
            logger.info("\n=== Pipeline V2 hoàn thành! ===")

            return self._get_summary()

        except Exception as e:
            logger.error(f"Pipeline lỗi: {e}")
            self._save_state()
            raise

    def _is_step_completed(self, step: str) -> bool:
        """Kiểm tra step đã hoàn thành chưa dựa vào output files"""
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        if step == "extract":
            # Kiểm tra có file extracted_documents.json
            extracted_file = os.path.join(output_dir, "extracted", "extracted_documents.json")
            if os.path.exists(extracted_file):
                data = load_json(extracted_file)
                return data and len(data) > 0
            return False

        elif step == "generate":
            # Kiểm tra có file qa_generated.json
            qa_file = os.path.join(output_dir, "generated", "qa_generated.json")
            if os.path.exists(qa_file):
                data = load_json(qa_file)
                return data and len(data) > 0
            return False

        elif step == "evaluate":
            # Kiểm tra có file qa_good.json
            good_file = os.path.join(output_dir, "evaluated", "qa_good.json")
            if os.path.exists(good_file):
                data = load_json(good_file)
                return data and len(data) > 0
            return False

        elif step == "split":
            # Kiểm tra có file train.json trong split/
            train_file = os.path.join(output_dir, "split", "train.json")
            if os.path.exists(train_file):
                data = load_json(train_file)
                return data and len(data) > 0
            return False

        elif step == "export":
            # Kiểm tra có file dataset_final.json
            final_file = os.path.join(output_dir, "final", "dataset_final.json")
            return os.path.exists(final_file)

        return False

    def _load_step_results(self, step: str):
        """Load kết quả của step đã hoàn thành vào state"""
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        if step == "extract":
            self.state["documents"] = self.extractor.load_extracted()
            logger.info(f"  Loaded {len(self.state['documents'])} documents")

        elif step == "generate":
            qa_file = os.path.join(output_dir, "generated", "qa_generated.json")
            self.state["qa_pairs"] = load_json(qa_file) or []
            logger.info(f"  Loaded {len(self.state['qa_pairs'])} Q&A pairs")

        elif step == "evaluate":
            self.state["good_qa"], self.state["bad_qa"] = self.evaluator.load_evaluated()
            rescued_file = os.path.join(output_dir, "evaluated", "qa_rescued.json")
            if os.path.exists(rescued_file):
                self.state["rescued_qa"] = load_json(rescued_file) or []
            logger.info(f"  Loaded {len(self.state['good_qa'])} good, {len(self.state['bad_qa'])} bad Q&A")

        elif step == "split":
            self.state["splits"] = self.splitter.load_splits()
            total = sum(len(v) for v in self.state["splits"].values())
            logger.info(f"  Loaded splits: {total} total samples")

    def run(self, steps: List[str] = None) -> Dict[str, Any]:
        """
        Chạy pipeline với các bước được chỉ định (force mode - không skip).

        Dùng cho:
        - Debug/test từng step riêng lẻ
        - Force chạy lại từ đầu

        Args:
            steps: List các bước cần chạy. None = tất cả.

        Returns:
            Kết quả pipeline
        """
        self.state["started_at"] = datetime.now().isoformat()

        # V2: Pipeline steps (evaluate đã gộp rescue + regenerate)
        all_steps = ["extract", "generate", "evaluate", "split", "export"]
        steps_to_run = steps or all_steps

        logger.info(f"Pipeline V2 bắt đầu với các bước: {steps_to_run}")

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
                elif step == "split":
                    self._run_split()
                elif step == "export":
                    self._run_export()

                self.state["steps_completed"].append(step)

            self._save_state()
            logger.info("\n=== Pipeline V2 hoàn thành! ===")

            return self._get_summary()

        except Exception as e:
            logger.error(f"Pipeline lỗi: {e}")
            self._save_state()
            raise
    
    def _run_extract(self):
        """Bước 1: Extract text từ documents"""
        # Live status
        self._save_live_status("extract", 0, 0)
        
        # Notify metrics
        if self.metrics:
            self.metrics.start_step("extract", 0)
        
        self.state["documents"] = self.extractor.extract_all()
        logger.info(f"Đã extract {len(self.state['documents'])} documents")
        
        # Update live status
        self._save_live_status("extract", len(self.state["documents"]), len(self.state["documents"]),
                               docs_extracted=len(self.state["documents"]))
        
        # End step
        if self.metrics:
            self.metrics.end_step()
    
    def _run_generate(self):
        """Bước 2: Generate Q&A pairs"""
        # Load documents nếu chưa có
        if not self.state["documents"]:
            self.state["documents"] = self.extractor.load_extracted()
            if not self.state["documents"]:
                logger.error("Không có documents! Hãy chạy bước 'extract' trước.")
                return
        
        # Notify metrics - tính tổng chunks
        if self.metrics:
            from core.utils import chunk_text
            total_chunks = 0
            chunk_size = self.config.get("processing.chunk_size", 4000)
            chunk_overlap = self.config.get("processing.chunk_overlap", 200)
            for doc in self.state["documents"]:
                content = doc.get("content", "")
                if content:
                    chunks = chunk_text(content, chunk_size, chunk_overlap)
                    total_chunks += len(chunks)
            self.metrics.start_step("generate", total_chunks)
            # Pass metrics to generator
            self.generator.set_metrics(self.metrics)
        
        self.state["qa_pairs"] = self.generator.generate(self.state["documents"])
        logger.info(f"Đã generate {len(self.state['qa_pairs'])} Q&A pairs")
        
        # Update live status
        self._save_live_status("generate_done", 
                               qa_generated=len(self.state["qa_pairs"]),
                               docs_extracted=len(self.state["documents"]))
        
        # End step
        if self.metrics:
            self.metrics.end_step()
    
    def _run_evaluate(self):
        """
        Bước 3: Evaluate + Rescue + Regenerate (All-in-one)

        Quy trình:
        1. Evaluate tất cả Q&A pairs
        2. Rescue bad Q&A bằng cách tự động thêm số hiệu văn bản
        3. Phân tích chunks có chất lượng kém
        4. Regenerate chunks kém (nếu có)
        5. Lặp lại cho đến khi không còn chunks cần regenerate hoặc đạt max iterations
        """
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        # Load Q&A cần evaluate
        if not self.state["qa_pairs"]:
            qa_file = os.path.join(output_dir, "generated", "qa_generated.json")
            if os.path.exists(qa_file):
                self.state["qa_pairs"] = load_json(qa_file)
            else:
                logger.error("Không có Q&A pairs! Hãy chạy bước 'generate' trước.")
                return
        
        # Live status
        self._save_live_status("evaluate", 0, len(self.state["qa_pairs"]),
                               qa_generated=len(self.state["qa_pairs"]))
        
        # Notify metrics
        if self.metrics:
            self.metrics.start_step("evaluate", len(self.state["qa_pairs"]))

        # Config cho regenerate loop
        max_iterations = self.config.get("quality.max_regenerate_iterations", 3)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"EVALUATE ITERATION {iteration}/{max_iterations}")
            logger.info(f"{'='*50}")

            # ========== STEP 1: EVALUATE ==========
            logger.info(f"\n📊 [1/3] Evaluating {len(self.state['qa_pairs'])} Q&A pairs...")
            good, bad = self.evaluator.evaluate(self.state["qa_pairs"])

            # Iteration 1: Ghi đè hoàn toàn
            # Iteration 2+: Merge với kết quả cũ (đã loại bỏ Q&A của chunks regenerate ở bước trước)
            if iteration == 1:
                self.state["good_qa"] = good
                self.state["bad_qa"] = bad
            else:
                # Merge kết quả mới vào kết quả cũ
                self.state["good_qa"].extend(good)
                self.state["bad_qa"].extend(bad)

            logger.info(f"Kết quả: {len(good)} good ({len(good)/len(self.state['qa_pairs'])*100:.1f}%), {len(bad)} bad")
            logger.info(f"Tổng tích lũy: {len(self.state['good_qa'])} good, {len(self.state['bad_qa'])} bad")

            if not bad:
                logger.info("✅ Không có bad Q&A! Hoàn thành evaluate.")
                break

            # ========== STEP 2: RESCUE ==========
            logger.info(f"\n🔧 [2/3] Rescuing {len(bad)} bad Q&A pairs...")
            rescued, still_bad = self.rescuer.rescue(bad)

            if rescued:
                self.state["good_qa"].extend(rescued)
                self.state["bad_qa"] = still_bad
                self.state["rescued_qa"] = self.state.get("rescued_qa", []) + rescued

                logger.info(f"Rescued: {len(rescued)} ({len(rescued)/len(bad)*100:.1f}%)")
                logger.info(f"Tổng good sau rescue: {len(self.state['good_qa'])}")
                logger.info(f"Còn lại bad: {len(still_bad)}")
            else:
                logger.info("Không rescue được Q&A nào")

            # Lưu kết quả sau mỗi iteration
            self._save_evaluate_results(output_dir)

            if not still_bad:
                logger.info("✅ Đã rescue hết bad Q&A! Hoàn thành evaluate.")
                break

            # ========== STEP 3: ANALYZE & REGENERATE ==========
            logger.info(f"\n🔄 [3/3] Phân tích chunks cần regenerate...")

            stats = self.regenerator.analyze(
                self.state["good_qa"],
                still_bad
            )

            chunks_to_regen = self.regenerator.get_chunks_to_regenerate()

            if not chunks_to_regen:
                logger.info("✅ Tất cả chunks đều có chất lượng OK, không cần regenerate!")
                break

            logger.info(f"Có {len(chunks_to_regen)} chunks cần regenerate:")
            for i, chunk in enumerate(chunks_to_regen[:5]):
                logger.info(f"  [{i+1}] {chunk['source_doc']} - chunk {chunk['chunk_id']}: {chunk['reason']}")
            if len(chunks_to_regen) > 5:
                logger.info(f"  ... và {len(chunks_to_regen) - 5} chunks khác")

            # Load documents nếu chưa có
            if not self.state["documents"]:
                self.state["documents"] = self.extractor.load_extracted()

            if not self.state["documents"]:
                logger.error("Không có documents để regenerate!")
                break

            # Regenerate
            chunk_ids_to_regen = self.regenerator.get_regenerate_chunk_ids()

            logger.info(f"Đang regenerate {len(chunk_ids_to_regen)} chunks...")
            new_qa = self.generator.regenerate_chunks(
                self.state["documents"],
                chunk_ids_to_regen
            )

            if not new_qa:
                logger.warning("Không regenerate được Q&A nào mới!")
                break

            logger.info(f"Đã regenerate {len(new_qa)} Q&A pairs mới")

            # Loại bỏ Q&A cũ của chunks đã regenerate và thay bằng Q&A mới
            regen_chunk_ids = set(chunk_ids_to_regen)

            # Loại Q&A cũ khỏi good và bad
            self.state["good_qa"] = [
                qa for qa in self.state["good_qa"]
                if (qa.get("source_doc"), qa.get("chunk_id")) not in regen_chunk_ids
            ]

            # Cập nhật qa_pairs với Q&A mới để evaluate lại
            self.state["qa_pairs"] = [
                qa for qa in self.state["qa_pairs"]
                if (qa.get("source_doc"), qa.get("chunk_id")) not in regen_chunk_ids
            ]
            self.state["qa_pairs"].extend(new_qa)

            # Chỉ evaluate lại Q&A mới
            self.state["qa_pairs"] = new_qa

            logger.info(f"Sẽ evaluate lại {len(new_qa)} Q&A mới ở iteration tiếp theo...")

        # Lưu kết quả cuối cùng
        self._save_evaluate_results(output_dir)

        # Log summary
        total_qa = len(self.state["good_qa"]) + len(self.state["bad_qa"])
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATE HOÀN THÀNH")
        logger.info(f"{'='*50}")
        logger.info(f"Tổng Q&A: {total_qa}")
        logger.info(f"Good: {len(self.state['good_qa'])} ({len(self.state['good_qa'])/total_qa*100:.1f}%)")
        logger.info(f"Rescued: {len(self.state.get('rescued_qa', []))}")
        logger.info(f"Bad: {len(self.state['bad_qa'])} ({len(self.state['bad_qa'])/total_qa*100:.1f}%)")
        
        # Update live status
        self._save_live_status("evaluate_done", 
                               processed=total_qa, total=total_qa,
                               qa_generated=total_qa,
                               qa_good=len(self.state['good_qa']),
                               qa_bad=len(self.state['bad_qa']))
        
        # Update metrics
        if self.metrics:
            self.metrics.record_qa_evaluated(
                good=len(self.state['good_qa']),
                bad=len(self.state['bad_qa'])
            )
            self.metrics.end_step()
        logger.info(f"Iterations: {iteration}")

    def _save_evaluate_results(self, output_dir: str):
        """Lưu kết quả evaluate"""
        eval_dir = os.path.join(output_dir, "evaluated")
        os.makedirs(eval_dir, exist_ok=True)

        # Lưu good Q&A
        good_file = os.path.join(eval_dir, "qa_good.json")
        save_json(self.state["good_qa"], good_file)

        # Lưu bad Q&A
        bad_file = os.path.join(eval_dir, "qa_bad.json")
        save_json(self.state["bad_qa"], bad_file)

        # Lưu rescued Q&A (nếu có)
        if self.state.get("rescued_qa"):
            rescued_file = os.path.join(eval_dir, "qa_rescued.json")
            save_json(self.state["rescued_qa"], rescued_file)
    
    def _run_split(self):
        """Bước 4: Split dataset theo document để tránh data leakage"""
        # Load good Q&A nếu chưa có
        if not self.state["good_qa"]:
            self.state["good_qa"], self.state["bad_qa"] = self.evaluator.load_evaluated()

        if not self.state["good_qa"]:
            logger.error("Không có good Q&A! Hãy chạy bước 'evaluate' trước.")
            return
        
        # Live status
        self._save_live_status("split", 0, len(self.state["good_qa"]),
                               qa_good=len(self.state["good_qa"]))
        
        # Notify metrics
        if self.metrics:
            self.metrics.start_step("split", len(self.state["good_qa"]))

        # Filter QA có câu hỏi không có ngữ cảnh rõ ràng
        logger.info(f"Bắt đầu filter {len(self.state['good_qa'])} Q&A pairs...")
        filtered_qa, removed_qa = self.filter.filter(self.state["good_qa"])
        
        if removed_qa:
            logger.info(f"Đã loại bỏ {len(removed_qa)} QA có câu hỏi không có ngữ cảnh rõ ràng")
            # Lưu removed QA để review
            output_dir = self.config.get("general.output_dir", "./output")
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
            removed_file = os.path.join(output_dir, "split", "qa_removed_low_quality.json")
            os.makedirs(os.path.dirname(removed_file), exist_ok=True)
            save_json(removed_qa, removed_file)
            logger.info(f"Đã lưu removed QA -> {removed_file}")
        
        logger.info(f"Bắt đầu split {len(filtered_qa)} Q&A pairs theo document...")

        # Split theo document
        self.state["splits"] = self.splitter.split(filtered_qa)

        # Validate không có data leakage
        is_valid = self.splitter.validate_no_leakage(self.state["splits"])

        if not is_valid:
            logger.error("Phát hiện data leakage! Kiểm tra lại split.")

        logger.info(f"Split hoàn thành: train={len(self.state['splits'].get('train', []))}, "
                    f"val={len(self.state['splits'].get('validation', []))}, "
                    f"test={len(self.state['splits'].get('test', []))}")
        
        # Live status
        total_split = sum(len(v) for v in self.state['splits'].values())
        self._save_live_status("split_done", processed=total_split, total=total_split,
                               qa_good=len(self.state["good_qa"]))
        
        # End metrics
        if self.metrics:
            self.metrics.end_step()

    
    def _run_export(self):
        """Bước 5: Export dataset cuối cùng"""
        # Live status
        self._save_live_status("export", 0, 0)
        
        # Notify metrics
        if self.metrics:
            self.metrics.start_step("export", 0)
        
        # Get output dir
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        # Create final directory
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        # V2: Ưu tiên sử dụng splits đã được chia theo document
        if self.state["splits"] and any(self.state["splits"].values()):
            logger.info("Exporting từ document-based splits...")
            self._export_from_splits(final_dir)
        else:
            # Fallback: Load splits từ file nếu có
            splits = self.splitter.load_splits()
            if any(splits.values()):
                logger.info("Exporting từ saved splits...")
                self.state["splits"] = splits
                self._export_from_splits(final_dir)
            else:
                # Fallback cuối: Load good Q&A và tự split (như v1)
                logger.warning("Không tìm thấy splits, sử dụng random split (có thể gây data leakage)")
                self._export_legacy(final_dir)
        
        # Live status - completed
        self._save_live_status("completed", 100, 100)
        
        # End metrics
        if self.metrics:
            self.metrics.end_step()

    def _export_from_splits(self, final_dir: str):
        """Export từ document-based splits (V2) với ChatML format"""
        output_cfg = self.config.get("output", {})

        # Mặc định dùng ChatML format
        export_format = self.config.get("export.format", "chat")

        formats = output_cfg.get("formats", ["json", "jsonl"])
        total_exported = 0

        for split_name, qa_pairs in self.state["splits"].items():
            if not qa_pairs:
                continue

            # Format dataset
            formatted_data = self._format_dataset(qa_pairs, export_format)
            total_exported += len(formatted_data)

            # Save
            if "json" in formats:
                save_json(formatted_data, os.path.join(final_dir, f"{split_name}.json"))

            if "jsonl" in formats:
                jsonl_file = os.path.join(final_dir, f"{split_name}.jsonl")
                with open(jsonl_file, 'w', encoding='utf-8') as f:
                    import json
                    for item in formatted_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

            logger.info(f"  {split_name}: {len(formatted_data)} samples")

        # Lưu dataset_final.json (merge tất cả)
        all_data = []
        for qa_pairs in self.state["splits"].values():
            if qa_pairs:
                all_data.extend(self._format_dataset(qa_pairs, export_format))

        if "json" in formats:
            save_json(all_data, os.path.join(final_dir, "dataset_final.json"))

        if "jsonl" in formats:
            with open(os.path.join(final_dir, "dataset_final.jsonl"), 'w', encoding='utf-8') as f:
                import json
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Exported {total_exported} samples -> {final_dir}")
        logger.info("Document-based split: NO data leakage!")

    def _export_legacy(self, final_dir: str):
        """Export kiểu cũ với random split (fallback)"""
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

        # Random split train/val/test (legacy - có thể gây data leakage)
        self._split_dataset(final_dataset, final_dir)

        logger.info(f"Exported {len(final_dataset)} samples -> {final_dir}")
        logger.warning("Used random split - may have data leakage!")
    
    def _format_dataset(self, qa_pairs: List[Dict], format_type: str) -> List[Dict]:
        """Format dataset theo format type"""
        # System prompt cho chatbot BHXH
        default_system_prompt = """Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam.

Nguyên tắc trả lời:
1. Luôn trích dẫn căn cứ pháp lý (số hiệu văn bản, điều, khoản)
2. Giải thích rõ ràng, dễ hiểu cho người dân
3. Nếu không chắc chắn, hướng dẫn liên hệ cơ quan BHXH địa phương"""

        # Lấy system prompt từ config nếu có
        system_prompt = self.config.get("export.system_prompt", default_system_prompt)

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
                # ChatML format cho fine-tuning
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
    
    def _save_live_status(self, step: str, processed: int = 0, total: int = 0, **kwargs):
        """
        Ghi file live status cho Dashboard real-time
        
        Args:
            step: Tên bước hiện tại (extract, generate, evaluate, split, export)
            processed: Số items đã xử lý
            total: Tổng số items
            **kwargs: Các thông tin bổ sung (qa_generated, cache_hits, etc.)
        """
        try:
            output_dir = self.config.get("general.output_dir", "./output")
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)
            
            status_file = os.path.join(output_dir, "live_status.json")
            
            # Tính elapsed time
            elapsed_seconds = 0
            started_at = self.state.get("started_at")
            if started_at:
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(started_at)
                    elapsed_seconds = (datetime.now() - start_time).total_seconds()
                except:
                    pass

            status = {
                "step": step,
                "chunks_processed": processed,
                "chunks_total": total,
                "qa_generated": kwargs.get("qa_generated", len(self.state.get("qa_pairs", []))),
                "qa_good": kwargs.get("qa_good", len(self.state.get("good_qa", []))),
                "qa_bad": kwargs.get("qa_bad", len(self.state.get("bad_qa", []))),
                "qa_rescued": kwargs.get("qa_rescued", len(self.state.get("rescued_qa", []))),
                "docs_extracted": kwargs.get("docs_extracted", len(self.state.get("documents", []))),
                "cache_hits": kwargs.get("cache_hits", 0),
                "cache_misses": kwargs.get("cache_misses", 0),
                "failed_chunks": kwargs.get("failed_chunks", 0),
                "started_at": started_at,
                "elapsed_seconds": elapsed_seconds,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False)
        except Exception:
            pass  # Không để lỗi ảnh hưởng pipeline

    def _save_state(self):
        """Lưu state của pipeline"""
        output_dir = self.config.get("general.output_dir", "./output")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(self.config_path), output_dir)

        state_file = os.path.join(output_dir, "pipeline_state.json")
        os.makedirs(output_dir, exist_ok=True)

        # Không lưu data lớn
        state_to_save = {
            "version": "2.0",  # V2
            "started_at": self.state["started_at"],
            "steps_completed": self.state["steps_completed"],
            "current_step": self.state["current_step"],
            "documents_count": len(self.state["documents"]),
            "qa_pairs_count": len(self.state["qa_pairs"]),
            "good_qa_count": len(self.state["good_qa"]),
            "bad_qa_count": len(self.state["bad_qa"]),
            "splits": {
                k: len(v) for k, v in self.state.get("splits", {}).items()
            }
        }

        save_json(state_to_save, state_file)

    def _get_summary(self) -> Dict[str, Any]:
        """Lấy summary của pipeline"""
        summary = {
            "version": "2.0",
            "project": self.project_name,
            "started_at": self.state["started_at"],
            "steps_completed": self.state["steps_completed"],
            "documents": len(self.state["documents"]),
            "qa_pairs_generated": len(self.state["qa_pairs"]),
            "good_qa": len(self.state["good_qa"]),
            "rescued_qa": len(self.state.get("rescued_qa", [])),
            "bad_qa": len(self.state["bad_qa"]),
            "success_rate": (
                len(self.state["good_qa"]) / max(1, len(self.state["qa_pairs"])) * 100
                if self.state["qa_pairs"] else 0
            )
        }

        # V2: Thêm thông tin splits
        if self.state.get("splits"):
            summary["splits"] = {
                k: len(v) for k, v in self.state["splits"].items()
            }

        return summary

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

        # V2: Pipeline steps (evaluate đã gộp rescue + regenerate)
        all_steps = ["extract", "generate", "evaluate", "split", "export"]
        remaining = [s for s in all_steps if s not in completed]

        if not remaining:
            logger.info("Pipeline V2 đã hoàn thành trước đó!")
            return self._get_summary()

        logger.info(f"Resume pipeline V2 từ bước: {remaining[0]}")
        return self.run(steps=remaining)
