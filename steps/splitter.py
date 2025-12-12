# -*- coding: utf-8 -*-
"""
Dataset Splitter - Chia dataset theo document để tránh data leakage

QUAN TRỌNG:
- Split theo DOCUMENT, không phải theo sample
- Q&A từ cùng một document luôn nằm trong cùng một split
- Tránh data leakage khi train/val/test có Q&A từ cùng nguồn
"""

import os
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from core.logger import get_logger
from core.utils import save_json, load_json, save_jsonl

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Chia dataset thành train/val/test theo document.

    Cách hoạt động:
    1. Group Q&A pairs theo document (source_doc)
    2. Shuffle documents (không phải samples)
    3. Chia documents theo tỷ lệ
    4. Collect Q&A pairs từ mỗi nhóm documents
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho splitting
        """
        self.train_ratio = config.get("train_ratio", 0.70)
        self.val_ratio = config.get("val_ratio", 0.15)
        self.test_ratio = config.get("test_ratio", 0.15)
        self.seed = config.get("seed", 42)
        self.output_dir = config.get("output_dir", "./output/split")

        os.makedirs(self.output_dir, exist_ok=True)

        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Split ratios không bằng 1.0: {total}")

    def split(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Split dataset theo document

        Args:
            qa_pairs: List Q&A pairs, mỗi item có field "source_doc"

        Returns:
            Dict với keys: "train", "validation", "test"
        """
        if not qa_pairs:
            logger.warning("Không có Q&A pairs để split!")
            return {"train": [], "validation": [], "test": []}

        logger.info(f"Bắt đầu split {len(qa_pairs)} Q&A pairs theo document...")

        # Step 1: Group by document
        doc_qa = defaultdict(list)
        for qa in qa_pairs:
            # Support both field names
            doc_id = qa.get("source_doc") or qa.get("doc_id") or qa.get("document") or "unknown"
            doc_qa[doc_id].append(qa)

        doc_ids = list(doc_qa.keys())
        n_docs = len(doc_ids)

        logger.info(f"Tổng số documents: {n_docs}")

        # Log document distribution
        for doc_id in sorted(doc_ids):
            count = len(doc_qa[doc_id])
            logger.debug(f"  {doc_id}: {count} Q&A pairs")

        # Step 2: Shuffle documents
        random.seed(self.seed)
        random.shuffle(doc_ids)

        # Step 3: Split documents
        train_end = int(n_docs * self.train_ratio)
        val_end = train_end + int(n_docs * self.val_ratio)

        train_docs = doc_ids[:train_end]
        val_docs = doc_ids[train_end:val_end]
        test_docs = doc_ids[val_end:]

        # Step 4: Collect Q&A pairs
        splits = {
            "train": [qa for doc in train_docs for qa in doc_qa[doc]],
            "validation": [qa for doc in val_docs for qa in doc_qa[doc]],
            "test": [qa for doc in test_docs for qa in doc_qa[doc]],
        }

        # Log statistics
        logger.info(f"Split hoàn thành:")
        logger.info(f"  Train: {len(splits['train'])} samples ({len(train_docs)} docs)")
        logger.info(f"  Validation: {len(splits['validation'])} samples ({len(val_docs)} docs)")
        logger.info(f"  Test: {len(splits['test'])} samples ({len(test_docs)} docs)")

        # Save split info
        split_info = {
            "seed": self.seed,
            "total_samples": len(qa_pairs),
            "total_documents": n_docs,
            "train": {
                "samples": len(splits["train"]),
                "documents": len(train_docs),
                "doc_ids": train_docs
            },
            "validation": {
                "samples": len(splits["validation"]),
                "documents": len(val_docs),
                "doc_ids": val_docs
            },
            "test": {
                "samples": len(splits["test"]),
                "documents": len(test_docs),
                "doc_ids": test_docs
            }
        }
        save_json(split_info, os.path.join(self.output_dir, "split_info.json"))

        # Save splits
        self._save_splits(splits)

        return splits

    def _save_splits(self, splits: Dict[str, List[Dict]]):
        """Lưu các splits thành files"""
        for split_name, data in splits.items():
            if data:
                # JSON
                json_file = os.path.join(self.output_dir, f"{split_name}.json")
                save_json(data, json_file)

                # JSONL
                jsonl_file = os.path.join(self.output_dir, f"{split_name}.jsonl")
                save_jsonl(data, jsonl_file)

                logger.info(f"Saved {split_name} -> {json_file}")

    def validate_no_leakage(self, splits: Dict[str, List[Dict]]) -> bool:
        """
        Validate rằng không có document overlap giữa các splits

        Returns:
            True nếu không có data leakage
        """
        train_docs = set(qa.get("source_doc", "") for qa in splits.get("train", []))
        val_docs = set(qa.get("source_doc", "") for qa in splits.get("validation", []))
        test_docs = set(qa.get("source_doc", "") for qa in splits.get("test", []))

        train_val_overlap = train_docs & val_docs
        train_test_overlap = train_docs & test_docs
        val_test_overlap = val_docs & test_docs

        has_leakage = False

        if train_val_overlap:
            logger.error(f"❌ Data leakage: Train-Val overlap: {train_val_overlap}")
            has_leakage = True

        if train_test_overlap:
            logger.error(f"❌ Data leakage: Train-Test overlap: {train_test_overlap}")
            has_leakage = True

        if val_test_overlap:
            logger.error(f"❌ Data leakage: Val-Test overlap: {val_test_overlap}")
            has_leakage = True

        if not has_leakage:
            logger.info("✅ No data leakage detected")

        return not has_leakage

    def load_splits(self) -> Dict[str, List[Dict]]:
        """Load splits đã lưu từ files"""
        splits = {}
        for split_name in ["train", "validation", "test"]:
            json_file = os.path.join(self.output_dir, f"{split_name}.json")
            if os.path.exists(json_file):
                splits[split_name] = load_json(json_file)
            else:
                splits[split_name] = []
        return splits
