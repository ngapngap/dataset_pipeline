# -*- coding: utf-8 -*-
"""
Dataset Tokenizer - Tokenization với labels cho Causal LM training

QUAN TRỌNG:
- Tạo labels = input_ids.copy() cho causal language modeling
- Mask phần prompt (chỉ train trên phần response)
- Hỗ trợ nhiều chat template formats
- Dynamic padding thay vì max_length padding
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.logger import get_logger
from core.utils import save_json, load_json, save_jsonl

logger = get_logger(__name__)


# Template mặc định cho Vistral/Vietnamese models
VISTRAL_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant}</s>"""

# Template cho các model khác
CHAT_TEMPLATES = {
    "vistral": VISTRAL_CHAT_TEMPLATE,
    "llama2": """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant}</s>""",
    "llama3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>""",
    "chatml": """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>""",
    "gemma": """<start_of_turn>user
{user}<end_of_turn>
<start_of_turn>model
{assistant}<end_of_turn>""",
    "qwen": """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>""",
}

# System prompt mặc định cho pháp luật Việt Nam
DEFAULT_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về lĩnh vực pháp luật Việt Nam, đặc biệt về bảo hiểm xã hội (BHXH), bảo hiểm y tế (BHYT), và bảo hiểm thất nghiệp (BHTN).
Hãy trả lời câu hỏi dựa trên các văn bản pháp luật hiện hành, trích dẫn điều khoản cụ thể khi cần thiết."""


@dataclass
class TokenizerConfig:
    """Config cho tokenizer"""
    model_name: str = "Viet-Mistral/Vistral-7B-Chat"
    template_name: str = "vistral"
    max_length: int = 2048
    padding: str = "max_length"  # "max_length" hoặc "dynamic"
    truncation: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    mask_prompt: bool = True  # Mask phần prompt, chỉ train trên response
    add_eos_token: bool = True
    output_dir: str = "./output/tokenized"


class DatasetTokenizer:
    """
    Tokenize dataset cho fine-tuning.

    Cách hoạt động:
    1. Format Q&A thành chat format
    2. Tokenize với tokenizer của model
    3. Tạo labels (mask prompt nếu cần)
    4. Lưu dataset đã tokenize

    QUAN TRỌNG về labels:
    - labels = input_ids.copy() cho causal LM
    - Mask (=-100) các token của prompt nếu mask_prompt=True
    - Model chỉ tính loss trên phần response
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho tokenization
        """
        self.model_name = config.get("model_name", "Viet-Mistral/Vistral-7B-Chat")
        self.template_name = config.get("template_name", "vistral")
        self.max_length = config.get("max_length", 2048)
        self.padding = config.get("padding", "max_length")
        self.truncation = config.get("truncation", True)
        self.system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.mask_prompt = config.get("mask_prompt", True)
        self.add_eos_token = config.get("add_eos_token", True)
        self.output_dir = config.get("output_dir", "./output/tokenized")

        os.makedirs(self.output_dir, exist_ok=True)

        # Template
        self.template = CHAT_TEMPLATES.get(self.template_name, VISTRAL_CHAT_TEMPLATE)

        # Tokenizer sẽ được load khi cần
        self._tokenizer = None

        logger.info(f"DatasetTokenizer initialized:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Template: {self.template_name}")
        logger.info(f"  Max length: {self.max_length}")
        logger.info(f"  Mask prompt: {self.mask_prompt}")

    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                logger.info(f"Loading tokenizer: {self.model_name}...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Đảm bảo có pad token
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

                logger.info(f"Tokenizer loaded. Vocab size: {len(self._tokenizer)}")

            except ImportError:
                logger.error("Cần cài đặt transformers: pip install transformers")
                raise
            except Exception as e:
                logger.error(f"Không thể load tokenizer: {e}")
                raise

        return self._tokenizer

    def format_chat(self, question: str, answer: str) -> str:
        """
        Format Q&A thành chat string theo template.

        Args:
            question: Câu hỏi
            answer: Câu trả lời

        Returns:
            Formatted chat string
        """
        return self.template.format(
            system=self.system_prompt,
            user=question,
            assistant=answer
        )

    def format_prompt_only(self, question: str) -> str:
        """
        Format chỉ phần prompt (để tính độ dài mask).

        Args:
            question: Câu hỏi

        Returns:
            Formatted prompt string (không có response)
        """
        # Tìm vị trí bắt đầu response trong template
        if self.template_name in ["vistral", "llama2"]:
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{question} [/INST] "
        elif self.template_name == "llama3":
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.template_name in ["chatml", "qwen"]:
            return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        elif self.template_name == "gemma":
            return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Fallback: tìm placeholder {assistant}
            template_parts = self.template.split("{assistant}")
            if len(template_parts) > 1:
                return template_parts[0].format(
                    system=self.system_prompt,
                    user=question
                )
            return self.format_chat(question, "")

    def tokenize_sample(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Tokenize một sample Q&A.

        Args:
            question: Câu hỏi
            answer: Câu trả lời

        Returns:
            Dict với input_ids, attention_mask, labels
        """
        # Format full text
        full_text = self.format_chat(question, answer)

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=self.padding if self.padding == "max_length" else False,
            truncation=self.truncation,
            return_tensors=None  # Return lists
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # === TẠO LABELS ===
        # Đây là phần quan trọng nhất cho Causal LM training
        labels = input_ids.copy()

        if self.mask_prompt:
            # Tính độ dài phần prompt để mask
            prompt_text = self.format_prompt_only(question)
            prompt_tokens = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors=None
            )["input_ids"]

            prompt_length = len(prompt_tokens)

            # Mask prompt tokens (set to -100)
            # -100 là ignore_index của CrossEntropyLoss
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100

        # Mask padding tokens
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def tokenize_dataset(self,
                         qa_pairs: List[Dict[str, Any]],
                         split_name: str = "train") -> List[Dict[str, Any]]:
        """
        Tokenize toàn bộ dataset.

        Args:
            qa_pairs: List Q&A pairs với keys "question", "answer"
            split_name: Tên split (train/validation/test)

        Returns:
            List tokenized samples
        """
        if not qa_pairs:
            logger.warning(f"Không có Q&A pairs để tokenize cho {split_name}!")
            return []

        logger.info(f"Tokenizing {len(qa_pairs)} samples cho {split_name}...")

        tokenized_samples = []
        skipped = 0

        for i, qa in enumerate(qa_pairs):
            # Lấy question và answer
            question = qa.get("question") or qa.get("instruction", "")
            answer = qa.get("answer") or qa.get("output", "")

            if not question or not answer:
                skipped += 1
                continue

            try:
                tokenized = self.tokenize_sample(question, answer)

                # Thêm metadata
                tokenized["source_doc"] = qa.get("source_doc", "")
                tokenized["chunk_id"] = qa.get("chunk_id", -1)

                tokenized_samples.append(tokenized)

            except Exception as e:
                logger.warning(f"Lỗi tokenize sample {i}: {e}")
                skipped += 1

        if skipped > 0:
            logger.warning(f"Đã skip {skipped} samples do lỗi hoặc thiếu dữ liệu")

        # Thống kê
        if tokenized_samples:
            avg_len = sum(len(s["input_ids"]) for s in tokenized_samples) / len(tokenized_samples)
            max_len = max(len(s["input_ids"]) for s in tokenized_samples)

            logger.info(f"Tokenization hoàn thành cho {split_name}:")
            logger.info(f"  Samples: {len(tokenized_samples)}")
            logger.info(f"  Avg length: {avg_len:.0f} tokens")
            logger.info(f"  Max length: {max_len} tokens")

        return tokenized_samples

    def tokenize_splits(self,
                        splits: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Tokenize tất cả splits (train/validation/test).

        Args:
            splits: Dict với keys "train", "validation", "test"

        Returns:
            Dict tokenized splits
        """
        tokenized_splits = {}

        for split_name, qa_pairs in splits.items():
            if qa_pairs:
                tokenized_splits[split_name] = self.tokenize_dataset(qa_pairs, split_name)
            else:
                tokenized_splits[split_name] = []

        # Lưu tokenized data
        self._save_tokenized(tokenized_splits)

        return tokenized_splits

    def _save_tokenized(self, tokenized_splits: Dict[str, List[Dict]]):
        """Lưu tokenized data"""
        for split_name, data in tokenized_splits.items():
            if data:
                # Lưu JSON (cho inspection)
                json_file = os.path.join(self.output_dir, f"{split_name}_tokenized.json")
                save_json(data, json_file)

                # Lưu JSONL (cho training)
                jsonl_file = os.path.join(self.output_dir, f"{split_name}_tokenized.jsonl")
                save_jsonl(data, jsonl_file)

                logger.info(f"Saved {split_name} -> {jsonl_file}")

        # Lưu config tokenization
        config_info = {
            "model_name": self.model_name,
            "template_name": self.template_name,
            "max_length": self.max_length,
            "padding": self.padding,
            "mask_prompt": self.mask_prompt,
            "system_prompt": self.system_prompt,
            "splits": {
                name: len(data) for name, data in tokenized_splits.items()
            }
        }
        save_json(config_info, os.path.join(self.output_dir, "tokenization_config.json"))

    def create_hf_dataset(self,
                          tokenized_splits: Dict[str, List[Dict]]) -> Any:
        """
        Tạo HuggingFace Dataset từ tokenized data.

        Args:
            tokenized_splits: Dict tokenized splits

        Returns:
            DatasetDict hoặc None nếu không có datasets library
        """
        try:
            from datasets import Dataset, DatasetDict

            hf_splits = {}

            for split_name, data in tokenized_splits.items():
                if data:
                    # Chuyển đổi format cho HF Dataset
                    hf_data = {
                        "input_ids": [s["input_ids"] for s in data],
                        "attention_mask": [s["attention_mask"] for s in data],
                        "labels": [s["labels"] for s in data],
                    }
                    hf_splits[split_name] = Dataset.from_dict(hf_data)

            dataset_dict = DatasetDict(hf_splits)

            # Lưu dataset
            dataset_path = os.path.join(self.output_dir, "hf_dataset")
            dataset_dict.save_to_disk(dataset_path)
            logger.info(f"Saved HuggingFace Dataset -> {dataset_path}")

            return dataset_dict

        except ImportError:
            logger.warning("Cần cài đặt datasets: pip install datasets")
            return None

    def get_stats(self, tokenized_samples: List[Dict]) -> Dict[str, Any]:
        """Lấy thống kê về tokenized data"""
        if not tokenized_samples:
            return {}

        lengths = [len(s["input_ids"]) for s in tokenized_samples]

        # Đếm số tokens được mask
        total_tokens = sum(lengths)
        masked_tokens = sum(
            sum(1 for l in s["labels"] if l == -100)
            for s in tokenized_samples
        )

        return {
            "total_samples": len(tokenized_samples),
            "total_tokens": total_tokens,
            "masked_tokens": masked_tokens,
            "trainable_tokens": total_tokens - masked_tokens,
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "truncated": sum(1 for l in lengths if l >= self.max_length)
        }


def create_tokenizer_from_config(config: Dict[str, Any]) -> DatasetTokenizer:
    """Factory function để tạo tokenizer từ config"""
    return DatasetTokenizer(config)
