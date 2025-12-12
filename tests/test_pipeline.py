# -*- coding: utf-8 -*-
"""
Test Suite cho Dataset Pipeline
Chạy: python -m pytest tests/ -v
Hoặc: python tests/test_pipeline.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Thêm parent dir vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestCoreConfig(unittest.TestCase):
    """Test core/config.py"""
    
    def setUp(self):
        """Tạo temp config file"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        
        config_content = """
general:
  project_name: "test-project"
  input_dir: "./test_input"
  output_dir: "./test_output"

llm:
  provider: "gemini"
  providers:
    gemini:
      model: "gemini-2.0-flash"
      api_keys_file: "./keys.txt"
      max_tokens: 1000

processing:
  chunk_size: 2000
  max_retries: 3
"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config(self):
        """Test load config từ YAML"""
        from core.config import PipelineConfig
        
        config = PipelineConfig(self.config_file)
        
        self.assertEqual(config.get("general.project_name"), "test-project")
        self.assertEqual(config.get("llm.provider"), "gemini")
        self.assertEqual(config.get("processing.chunk_size"), 2000)
    
    def test_get_nested_value(self):
        """Test lấy giá trị nested"""
        from core.config import PipelineConfig
        
        config = PipelineConfig(self.config_file)
        
        self.assertEqual(config.get("llm.providers.gemini.model"), "gemini-2.0-flash")
        self.assertEqual(config.get("llm.providers.gemini.max_tokens"), 1000)
    
    def test_get_default_value(self):
        """Test giá trị default khi key không tồn tại"""
        from core.config import PipelineConfig
        
        config = PipelineConfig(self.config_file)
        
        self.assertEqual(config.get("not.exist.key", "default"), "default")
        self.assertIsNone(config.get("not.exist.key"))


class TestCoreUtils(unittest.TestCase):
    """Test core/utils.py"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chunk_text_short(self):
        """Test chunk text ngắn hơn chunk_size"""
        from core.utils import chunk_text
        
        text = "Đây là văn bản ngắn."
        chunks = chunk_text(text, chunk_size=1000)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_chunk_text_long(self):
        """Test chunk text dài"""
        from core.utils import chunk_text
        
        text = "A" * 5000
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        
        self.assertGreater(len(chunks), 1)
        # Mỗi chunk không quá chunk_size + một chút buffer
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1200)
    
    def test_save_load_json(self):
        """Test save và load JSON"""
        from core.utils import save_json, load_json
        
        data = [{"question": "Test?", "answer": "Answer"}]
        filepath = os.path.join(self.temp_dir, "test.json")
        
        save_json(data, filepath)
        loaded = load_json(filepath)
        
        self.assertEqual(data, loaded)
    
    def test_save_load_jsonl(self):
        """Test save và load JSONL"""
        from core.utils import save_jsonl, load_jsonl
        
        data = [
            {"q": "Q1", "a": "A1"},
            {"q": "Q2", "a": "A2"}
        ]
        filepath = os.path.join(self.temp_dir, "test.jsonl")
        
        save_jsonl(data, filepath)
        loaded = load_jsonl(filepath)
        
        self.assertEqual(data, loaded)
    
    def test_load_api_keys(self):
        """Test load API keys từ file"""
        from core.utils import load_api_keys
        
        keys_file = os.path.join(self.temp_dir, "keys.txt")
        with open(keys_file, 'w') as f:
            f.write("key1\n")
            f.write("# comment\n")
            f.write("key2\n")
            f.write("\n")
            f.write("key3\n")
        
        keys = load_api_keys(keys_file)
        
        self.assertEqual(keys, ["key1", "key2", "key3"])
    
    def test_deduplicate_qa(self):
        """Test loại bỏ Q&A trùng lặp"""
        from core.utils import deduplicate_qa
        
        qa_list = [
            {"question": "Câu hỏi 1?", "answer": "A1"},
            {"question": "Câu hỏi 1?", "answer": "A2"},  # Trùng question
            {"question": "Câu hỏi 2?", "answer": "A3"},
        ]
        
        unique = deduplicate_qa(qa_list)
        
        self.assertEqual(len(unique), 2)


class TestProviders(unittest.TestCase):
    """Test providers/"""
    
    def test_provider_registry(self):
        """Test registry có đủ providers"""
        from providers.factory import PROVIDER_REGISTRY, PROVIDER_PRESETS
        
        self.assertIn("gemini", PROVIDER_REGISTRY)
        self.assertIn("openai", PROVIDER_REGISTRY)
        self.assertIn("anthropic", PROVIDER_REGISTRY)
        self.assertIn("custom", PROVIDER_REGISTRY)
        
        self.assertIn("ollama", PROVIDER_PRESETS)
        self.assertIn("groq", PROVIDER_PRESETS)
        self.assertIn("together", PROVIDER_PRESETS)
    
    def test_get_available_providers(self):
        """Test lấy danh sách providers"""
        from providers import get_available_providers
        
        providers = get_available_providers()
        
        self.assertIn("gemini", providers)
        self.assertIn("openai", providers)
        self.assertIn("ollama", providers)
    
    @patch('providers.gemini.genai', create=True)
    def test_gemini_provider_init(self, mock_genai):
        """Test khởi tạo Gemini provider"""
        from providers import GeminiProvider
        
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-2.0-flash",
            max_tokens=1000
        )
        
        self.assertEqual(provider.model, "gemini-2.0-flash")
        self.assertEqual(provider.max_tokens, 1000)
    
    def test_create_provider_invalid(self):
        """Test tạo provider không hợp lệ"""
        from providers import create_provider
        
        with self.assertRaises(ValueError):
            create_provider("invalid_provider", "key", "model")
    
    def test_create_custom_provider(self):
        """Test tạo custom provider với base_url"""
        from providers import create_provider
        
        try:
            import openai
            with patch.object(openai, 'OpenAI'):
                provider = create_provider(
                    provider_name="custom",
                    api_key="test-key",
                    model="test-model",
                    base_url="http://localhost:8000/v1"
                )
                
                self.assertEqual(provider.base_url, "http://localhost:8000/v1")
        except ImportError:
            # Skip nếu openai chưa cài
            self.skipTest("openai package not installed")


class TestTextExtractor(unittest.TestCase):
    """Test steps/extractor.py"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_txt_files(self):
        """Test extract file .txt"""
        from steps.extractor import TextExtractor
        
        # Tạo test files
        test_content = "Điều 1. Nội dung văn bản pháp luật test."
        with open(os.path.join(self.input_dir, "test.txt"), 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        extractor = TextExtractor({
            "input_dir": self.input_dir,
            "output_dir": self.output_dir
        })
        
        documents = extractor.extract_all()
        
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], test_content)
        self.assertEqual(documents[0]["file_name"], "test.txt")
    
    def test_extract_empty_dir(self):
        """Test extract từ thư mục rỗng"""
        from steps.extractor import TextExtractor
        
        extractor = TextExtractor({
            "input_dir": self.input_dir,
            "output_dir": self.output_dir
        })
        
        documents = extractor.extract_all()
        
        self.assertEqual(len(documents), 0)
    
    def test_extract_multiple_encodings(self):
        """Test extract file với encoding khác nhau"""
        from steps.extractor import TextExtractor
        
        # UTF-8
        with open(os.path.join(self.input_dir, "utf8.txt"), 'w', encoding='utf-8') as f:
            f.write("Tiếng Việt UTF-8")
        
        # UTF-8 BOM
        with open(os.path.join(self.input_dir, "utf8bom.txt"), 'w', encoding='utf-8-sig') as f:
            f.write("Tiếng Việt UTF-8 BOM")
        
        extractor = TextExtractor({
            "input_dir": self.input_dir,
            "output_dir": self.output_dir
        })
        
        documents = extractor.extract_all()
        
        self.assertEqual(len(documents), 2)


class TestQAGenerator(unittest.TestCase):
    """Test steps/generator.py"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_without_api_keys(self):
        """Test khởi tạo generator không có API keys"""
        from steps.generator import QAGenerator
        
        generator = QAGenerator({
            "provider": "gemini",
            "api_keys": [],
            "output_dir": self.temp_dir
        })
        
        self.assertEqual(len(generator.providers), 0)
    
    def test_prepare_chunks(self):
        """Test chuẩn bị chunks từ documents"""
        from steps.generator import QAGenerator
        
        generator = QAGenerator({
            "provider": "gemini",
            "api_keys": [],
            "output_dir": self.temp_dir,
            "chunk_size": 100,
            "chunk_overlap": 20
        })
        
        documents = [
            {"file_name": "test.txt", "content": "A" * 300}
        ]
        
        chunks = generator._prepare_chunks(documents)
        
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["doc_name"], "test.txt")
    
    def test_parse_qa_response_valid(self):
        """Test parse JSON response hợp lệ"""
        from steps.generator import QAGenerator
        
        generator = QAGenerator({
            "provider": "gemini",
            "api_keys": [],
            "output_dir": self.temp_dir
        })
        
        response = '''```json
[
  {"question": "Câu hỏi 1?", "answer": "Trả lời 1"},
  {"question": "Câu hỏi 2?", "answer": "Trả lời 2"}
]
```'''
        
        qa_pairs = generator._parse_qa_response(response)
        
        self.assertEqual(len(qa_pairs), 2)
        self.assertEqual(qa_pairs[0]["question"], "Câu hỏi 1?")
    
    def test_parse_qa_response_invalid(self):
        """Test parse JSON response không hợp lệ"""
        from steps.generator import QAGenerator
        
        generator = QAGenerator({
            "provider": "gemini",
            "api_keys": [],
            "output_dir": self.temp_dir
        })
        
        response = "Đây không phải JSON"
        
        qa_pairs = generator._parse_qa_response(response)
        
        self.assertIsNone(qa_pairs)


class TestQualityEvaluator(unittest.TestCase):
    """Test steps/evaluator.py"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rule_score_with_legal_citation(self):
        """Test điểm cao khi có căn cứ pháp lý"""
        from steps.evaluator import QualityEvaluator
        
        evaluator = QualityEvaluator({
            "mode": "rule",
            "output_dir": self.temp_dir
        })
        
        qa = {
            "question": "Mức đóng BHXH là bao nhiêu phần trăm?",
            "answer": "Căn cứ Điều 5, Khoản 2, Luật BHXH 2024, mức đóng BHXH bắt buộc là 8% tiền lương hàng tháng của người lao động."
        }
        
        score, reason = evaluator._rule_score(qa)
        
        self.assertGreaterEqual(score, 7)
        self.assertIn("căn cứ", reason.lower())
    
    def test_rule_score_without_legal_citation(self):
        """Test điểm thấp khi KHÔNG có căn cứ pháp lý"""
        from steps.evaluator import QualityEvaluator
        
        evaluator = QualityEvaluator({
            "mode": "rule",
            "output_dir": self.temp_dir
        })
        
        qa = {
            "question": "Mức đóng BHXH là bao nhiêu?",
            "answer": "Mức đóng BHXH là 8% tiền lương."  # Không có căn cứ
        }
        
        score, reason = evaluator._rule_score(qa)
        
        self.assertLess(score, 7)
        self.assertIn("THIẾU CĂN CỨ", reason)
    
    def test_rule_score_short_answer(self):
        """Test điểm thấp khi câu trả lời quá ngắn"""
        from steps.evaluator import QualityEvaluator
        
        evaluator = QualityEvaluator({
            "mode": "rule",
            "output_dir": self.temp_dir,
            "min_answer_length": 50
        })
        
        qa = {
            "question": "Hỏi gì đó?",
            "answer": "Ngắn"
        }
        
        score, reason = evaluator._rule_score(qa)
        
        self.assertLess(score, 7)
        self.assertIn("ngắn", reason.lower())
    
    def test_evaluate_rule_based(self):
        """Test evaluate với mode rule"""
        from steps.evaluator import QualityEvaluator
        
        evaluator = QualityEvaluator({
            "mode": "rule",
            "min_score": 7,
            "output_dir": self.temp_dir
        })
        
        qa_pairs = [
            {
                "question": "Điều kiện hưởng lương hưu là gì?",
                "answer": "Căn cứ Điều 54, Luật BHXH 2024, người lao động đóng BHXH đủ 20 năm được hưởng lương hưu khi đủ tuổi nghỉ hưu."
            },
            {
                "question": "Test?",
                "answer": "Ngắn quá"  # Sẽ bị loại
            }
        ]
        
        good, bad = evaluator.evaluate(qa_pairs)
        
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 1)


class TestPipelineIntegration(unittest.TestCase):
    """Test tích hợp pipeline"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir)

        # Tạo config file
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        config_content = f"""
general:
  project_name: "test-pipeline"
  input_dir: "{self.input_dir}"
  output_dir: "{self.output_dir}"

llm:
  provider: "gemini"
  providers:
    gemini:
      model: "gemini-2.0-flash"
      api_keys_file: ""

processing:
  chunk_size: 1000

output:
  base_dir: "{self.output_dir}"
"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)

        # Tạo test document
        test_doc = """
LUẬT BẢO HIỂM XÃ HỘI (TEST)

Điều 1. Phạm vi điều chỉnh
Luật này quy định về chế độ, chính sách bảo hiểm xã hội.

Điều 2. Đối tượng áp dụng
1. Người lao động làm việc theo hợp đồng lao động.
2. Người sử dụng lao động.

Điều 5. Mức đóng BHXH
Người lao động đóng 8% tiền lương hàng tháng.
Người sử dụng lao động đóng 17.5% quỹ tiền lương.
"""
        with open(os.path.join(self.input_dir, "luat_bhxh_test.txt"), 'w', encoding='utf-8') as f:
            f.write(test_doc)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extractor_step(self):
        """Test bước extract hoạt động"""
        from steps.extractor import TextExtractor

        extractor = TextExtractor({
            "input_dir": self.input_dir,
            "output_dir": self.output_dir
        })

        documents = extractor.extract_all()

        self.assertEqual(len(documents), 1)
        self.assertIn("Điều 1", documents[0]["content"])
        self.assertIn("Điều 5", documents[0]["content"])


class TestDatasetSplitter(unittest.TestCase):
    """Test steps/splitter.py - V2"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_split_by_document(self):
        """Test split theo document"""
        from steps.splitter import DatasetSplitter

        splitter = DatasetSplitter({
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "output_dir": self.temp_dir
        })

        qa_pairs = [
            {"question": "Q1 from Doc A", "answer": "A1", "source_doc": "doc_a.txt"},
            {"question": "Q2 from Doc A", "answer": "A2", "source_doc": "doc_a.txt"},
            {"question": "Q3 from Doc A", "answer": "A3", "source_doc": "doc_a.txt"},
            {"question": "Q1 from Doc B", "answer": "A1", "source_doc": "doc_b.txt"},
            {"question": "Q2 from Doc B", "answer": "A2", "source_doc": "doc_b.txt"},
            {"question": "Q1 from Doc C", "answer": "A1", "source_doc": "doc_c.txt"},
            {"question": "Q2 from Doc C", "answer": "A2", "source_doc": "doc_c.txt"},
            {"question": "Q1 from Doc D", "answer": "A1", "source_doc": "doc_d.txt"},
            {"question": "Q1 from Doc E", "answer": "A1", "source_doc": "doc_e.txt"},
        ]

        splits = splitter.split(qa_pairs)

        # Check có đủ splits
        self.assertIn("train", splits)
        self.assertIn("validation", splits)
        self.assertIn("test", splits)

        # Check tổng số samples đúng
        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        self.assertEqual(total, len(qa_pairs))

    def test_no_document_overlap(self):
        """Test không có document overlap giữa các splits"""
        from steps.splitter import DatasetSplitter

        splitter = DatasetSplitter({
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": 42,
            "output_dir": self.temp_dir
        })

        qa_pairs = [
            {"question": "Q1", "answer": "A1", "source_doc": "doc_a"},
            {"question": "Q2", "answer": "A2", "source_doc": "doc_a"},
            {"question": "Q3", "answer": "A3", "source_doc": "doc_b"},
            {"question": "Q4", "answer": "A4", "source_doc": "doc_c"},
            {"question": "Q5", "answer": "A5", "source_doc": "doc_d"},
        ]

        splits = splitter.split(qa_pairs)

        # Validate không có overlap
        is_valid = splitter.validate_no_leakage(splits)
        self.assertTrue(is_valid)

    def test_validate_detects_leakage(self):
        """Test phát hiện data leakage"""
        from steps.splitter import DatasetSplitter

        splitter = DatasetSplitter({
            "output_dir": self.temp_dir
        })

        # Tạo splits có overlap (giả lập lỗi)
        splits = {
            "train": [{"question": "Q1", "source_doc": "same_doc"}],
            "validation": [{"question": "Q2", "source_doc": "same_doc"}],  # Overlap!
            "test": [{"question": "Q3", "source_doc": "other_doc"}]
        }

        is_valid = splitter.validate_no_leakage(splits)
        self.assertFalse(is_valid)  # Phải phát hiện overlap


class TestDatasetTokenizer(unittest.TestCase):
    """Test steps/tokenizer.py - V2"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_chat(self):
        """Test format chat template"""
        from steps.tokenizer import DatasetTokenizer

        tokenizer = DatasetTokenizer({
            "model_name": "Viet-Mistral/Vistral-7B-Chat",
            "template_name": "vistral",
            "output_dir": self.temp_dir
        })

        formatted = tokenizer.format_chat(
            question="Mức đóng BHXH là bao nhiêu?",
            answer="Mức đóng là 8% tiền lương."
        )

        # Check format đúng
        self.assertIn("[INST]", formatted)
        self.assertIn("[/INST]", formatted)
        self.assertIn("Mức đóng BHXH là bao nhiêu?", formatted)
        self.assertIn("Mức đóng là 8% tiền lương.", formatted)

    def test_format_prompt_only(self):
        """Test format phần prompt (để tính mask length)"""
        from steps.tokenizer import DatasetTokenizer

        tokenizer = DatasetTokenizer({
            "template_name": "vistral",
            "output_dir": self.temp_dir
        })

        prompt = tokenizer.format_prompt_only("Test question?")

        self.assertIn("[INST]", prompt)
        self.assertIn("Test question?", prompt)
        self.assertIn("[/INST]", prompt)
        # Prompt không chứa answer
        self.assertTrue(prompt.endswith(" "))


def run_quick_test():
    """Chạy test nhanh không cần pytest"""
    print("=" * 60)
    print("DATASET PIPELINE V2 - QUICK TEST")
    print("=" * 60)

    # Test imports
    print("\n[1] Testing imports...")
    try:
        from core.config import PipelineConfig
        from core.logger import get_logger
        from core.utils import chunk_text, save_json, load_json
        from providers import get_available_providers, create_provider
        from steps.extractor import TextExtractor
        from steps.generator import QAGenerator
        from steps.evaluator import QualityEvaluator
        from steps.splitter import DatasetSplitter  # V2
        from steps.tokenizer import DatasetTokenizer  # V2
        print("   All imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False
    
    # Test utils
    print("\n[2] Testing utils...")
    try:
        chunks = chunk_text("A" * 5000, chunk_size=1000)
        assert len(chunks) > 1, "Chunk text failed"
        print(f"   [OK] chunk_text: {len(chunks)} chunks created")
        
        providers = get_available_providers()
        assert "gemini" in providers, "Gemini not in providers"
        print(f"   [OK] Providers available: {providers}")
    except Exception as e:
        print(f"   [FAIL] Utils error: {e}")
        return False
    
    # Test evaluator rules
    print("\n[3] Testing evaluator rules...")
    try:
        temp_dir = tempfile.mkdtemp()
        evaluator = QualityEvaluator({
            "mode": "rule",
            "output_dir": temp_dir
        })
        
        # Test với căn cứ pháp lý
        qa_good = {
            "question": "Mức đóng BHXH là bao nhiêu?",
            "answer": "Căn cứ Điều 5, Khoản 1, Luật BHXH 2024, mức đóng BHXH của người lao động là 8% tiền lương hàng tháng."
        }
        score1, reason1 = evaluator._rule_score(qa_good)
        
        # Test không có căn cứ
        qa_bad = {
            "question": "Mức đóng BHXH là bao nhiêu?",
            "answer": "Mức đóng là 8%."
        }
        score2, reason2 = evaluator._rule_score(qa_bad)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"   [OK] With legal citation: score={score1:.1f} ({reason1})")
        print(f"   [OK] Without citation: score={score2:.1f} ({reason2})")
        
        assert score1 > score2, "Scoring logic incorrect"
        print("   Scoring logic correct (with citation > without)")
    except Exception as e:
        print(f"   Evaluator error: {e}")
        return False

    # V2: Test splitter
    print("\nTesting V2 Splitter...")
    try:
        temp_dir = tempfile.mkdtemp()
        splitter = DatasetSplitter({
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "output_dir": temp_dir
        })

        qa_pairs = [
            {"question": "Q1", "answer": "A1", "source_doc": "doc_a"},
            {"question": "Q2", "answer": "A2", "source_doc": "doc_a"},
            {"question": "Q3", "answer": "A3", "source_doc": "doc_b"},
            {"question": "Q4", "answer": "A4", "source_doc": "doc_c"},
            {"question": "Q5", "answer": "A5", "source_doc": "doc_d"},
        ]

        splits = splitter.split(qa_pairs)
        is_valid = splitter.validate_no_leakage(splits)

        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"   Splits: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
        assert is_valid, "Data leakage detected!"
        print("   No data leakage - document-based split working!")
    except Exception as e:
        print(f"   Splitter error: {e}")
        return False

    # V2: Test tokenizer
    print("\nTesting V2 Tokenizer...")
    try:
        temp_dir = tempfile.mkdtemp()
        tokenizer = DatasetTokenizer({
            "template_name": "vistral",
            "output_dir": temp_dir
        })

        formatted = tokenizer.format_chat("Test question?", "Test answer.")
        assert "[INST]" in formatted, "Template format wrong"
        assert "Test question?" in formatted, "Question not in output"

        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"   Format chat: OK (contains [INST] and question)")
    except Exception as e:
        print(f"   Tokenizer error: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL QUICK TESTS PASSED!")
    print("=" * 60)
    print("\nPipeline V2 Ready!")
    print("  Run: python run.py --steps split tokenize export")

    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test không cần pytest
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # Full test với unittest
        unittest.main(verbosity=2)
