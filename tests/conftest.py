# -*- coding: utf-8 -*-
"""
Pytest Fixtures - Shared fixtures cho tất cả tests
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Generator

import pytest

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# TEMP DIRECTORY FIXTURES
# ==============================================================================

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Tạo temp directory cho test, cleanup sau khi test xong"""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def temp_input_dir(temp_dir: str) -> str:
    """Tạo input directory trong temp"""
    input_dir = os.path.join(temp_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    return input_dir


@pytest.fixture
def temp_output_dir(temp_dir: str) -> str:
    """Tạo output directory trong temp"""
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==============================================================================
# CONFIG FIXTURES
# ==============================================================================

@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample config dict cho testing"""
    return {
        "general": {
            "project_name": "test-project",
            "description": "Test project",
            "language": "vi",
            "input_dir": "./input",
            "output_dir": "./output",
            "log_level": "INFO",
            "log_file": "./logs/test.log"
        },
        "llm": {
            "provider": "gemini",
            "providers": {
                "gemini": {
                    "model": "gemini-2.0-flash",
                    "api_key": "test-api-key",
                    "max_tokens": 3000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "rate_limit_per_minute": 60
                }
            }
        },
        "processing": {
            "threads_per_key": 1,
            "max_threads": 5,
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "max_retries": 3,
            "retry_delay": 1,
            "requests_per_minute": 60,
            "checkpoint_enabled": True,
            "checkpoint_interval": 10
        },
        "qa_generation": {
            "qa_per_chunk": 3,
            "perspectives": [],
            "prompt_template": "Generate {num_qa} Q&A pairs from: {content}"
        },
        "quality": {
            "enabled": True,
            "min_score": 4,
            "criteria": [],
            "use_llm_evaluation": False
        },
        "output": {
            "formats": ["json", "jsonl"],
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        }
    }


@pytest.fixture
def config_file(temp_dir: str, sample_config_dict: Dict[str, Any]) -> str:
    """Tạo config file từ sample dict"""
    import yaml
    
    # Update paths to use temp dir
    sample_config_dict["general"]["input_dir"] = os.path.join(temp_dir, "input")
    sample_config_dict["general"]["output_dir"] = os.path.join(temp_dir, "output")
    sample_config_dict["general"]["log_file"] = os.path.join(temp_dir, "logs", "test.log")
    
    # Create directories
    os.makedirs(sample_config_dict["general"]["input_dir"], exist_ok=True)
    os.makedirs(sample_config_dict["general"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(sample_config_dict["general"]["log_file"]), exist_ok=True)
    
    config_path = os.path.join(temp_dir, "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config_dict, f, allow_unicode=True)
    
    return config_path


@pytest.fixture
def invalid_config_file(temp_dir: str) -> str:
    """Config file với các giá trị không hợp lệ"""
    import yaml
    
    config = {
        "general": {
            "project_name": "invalid-test",
            "input_dir": "/nonexistent/path",
            "output_dir": os.path.join(temp_dir, "output")
        },
        "llm": {
            "provider": "gemini",
            "providers": {}  # Thiếu provider config
        },
        "processing": {
            "chunk_size": -100,  # Invalid
            "chunk_overlap": 5000  # > chunk_size
        },
        "output": {
            "train_ratio": 0.5,
            "val_ratio": 0.5,
            "test_ratio": 0.5  # Sum > 1
        }
    }
    
    os.makedirs(config["general"]["output_dir"], exist_ok=True)
    
    config_path = os.path.join(temp_dir, "invalid_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    return config_path


# ==============================================================================
# QA PAIR FIXTURES
# ==============================================================================

@pytest.fixture
def sample_qa_pairs() -> List[Dict[str, Any]]:
    """Sample Q&A pairs cho testing"""
    return [
        {
            "question": "Mức đóng BHXH của người lao động là bao nhiêu phần trăm?",
            "answer": "Căn cứ Điều 5, Khoản 1, Luật BHXH số 58/2014/QH13, mức đóng BHXH của người lao động là 8% tiền lương hàng tháng.",
            "source_doc": "luat_bhxh.txt",
            "chunk_id": 0
        },
        {
            "question": "Điều kiện hưởng lương hưu là gì?",
            "answer": "Căn cứ Điều 54, Luật BHXH số 58/2014/QH13, người lao động được hưởng lương hưu khi đủ tuổi nghỉ hưu và đóng BHXH đủ 20 năm.",
            "source_doc": "luat_bhxh.txt",
            "chunk_id": 1
        },
        {
            "question": "Tôi có được hưởng BHYT không?",
            "answer": "Căn cứ Điều 12, Luật BHYT số 46/2014/QH13, người lao động tham gia BHXH bắt buộc đều được hưởng BHYT.",
            "source_doc": "luat_bhyt.txt",
            "chunk_id": 0
        }
    ]


@pytest.fixture
def sample_qa_good() -> List[Dict[str, Any]]:
    """Q&A pairs chất lượng tốt"""
    return [
        {
            "question": "Mức đóng BHXH của người lao động là bao nhiêu?",
            "answer": "Căn cứ Điều 5, Khoản 1, Luật BHXH số 58/2014/QH13, mức đóng BHXH của người lao động là 8% tiền lương hàng tháng. Người sử dụng lao động đóng 17.5% quỹ tiền lương.",
            "source_doc": "doc_a.txt",
            "eval_score": 9.0
        },
        {
            "question": "Điều kiện để được hưởng lương hưu theo quy định mới?",
            "answer": "Căn cứ Điều 54, Luật BHXH số 41/2024/QH15, người lao động được hưởng lương hưu khi đủ tuổi nghỉ hưu (nam 62, nữ 60) và đóng BHXH đủ 15 năm.",
            "source_doc": "doc_a.txt",
            "eval_score": 8.5
        }
    ]


@pytest.fixture
def sample_qa_bad() -> List[Dict[str, Any]]:
    """Q&A pairs chất lượng kém"""
    return [
        {
            "question": "Test?",
            "answer": "Ngắn quá",
            "source_doc": "doc_b.txt",
            "eval_score": 2.0,
            "eval_reason": "Câu trả lời quá ngắn"
        },
        {
            "question": "Quy định về BHXH?",
            "answer": "Theo quy định, người lao động phải đóng BHXH.",  # Thiếu căn cứ
            "source_doc": "doc_b.txt",
            "eval_score": 4.0,
            "eval_reason": "Thiếu căn cứ pháp lý"
        }
    ]


# ==============================================================================
# DOCUMENT FIXTURES
# ==============================================================================

@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents cho testing"""
    return [
        {
            "file_name": "luat_bhxh.txt",
            "file_path": "/path/to/luat_bhxh.txt",
            "content": """
LUẬT BẢO HIỂM XÃ HỘI

Điều 1. Phạm vi điều chỉnh
Luật này quy định về chế độ, chính sách bảo hiểm xã hội.

Điều 5. Mức đóng BHXH
1. Người lao động đóng 8% tiền lương hàng tháng.
2. Người sử dụng lao động đóng 17.5% quỹ tiền lương.

Điều 54. Điều kiện hưởng lương hưu
Người lao động được hưởng lương hưu khi:
a) Đủ tuổi nghỉ hưu theo quy định
b) Đóng BHXH đủ 20 năm
""",
            "num_chunks": 3
        },
        {
            "file_name": "luat_bhyt.txt",
            "file_path": "/path/to/luat_bhyt.txt",
            "content": """
LUẬT BẢO HIỂM Y TẾ

Điều 12. Đối tượng tham gia BHYT
Người lao động tham gia BHXH bắt buộc đều được hưởng BHYT.

Điều 22. Mức hưởng BHYT
Người tham gia BHYT được thanh toán 80% chi phí khám chữa bệnh.
""",
            "num_chunks": 2
        }
    ]


@pytest.fixture
def sample_document_file(temp_input_dir: str) -> str:
    """Tạo một file document thật"""
    content = """
LUẬT BẢO HIỂM XÃ HỘI SỐ 58/2014/QH13

Điều 1. Phạm vi điều chỉnh
Luật này quy định về chế độ, chính sách bảo hiểm xã hội.

Điều 5. Mức đóng BHXH bắt buộc
Khoản 1. Người lao động đóng bằng 8% mức tiền lương tháng vào quỹ hưu trí và tử tuất.
Khoản 2. Người sử dụng lao động đóng 17.5% trên quỹ tiền lương tháng.

Điều 54. Điều kiện hưởng lương hưu
Người lao động được hưởng lương hưu khi thuộc một trong các trường hợp sau:
a) Nam đủ 60 tuổi, nữ đủ 55 tuổi và có đủ 20 năm đóng BHXH trở lên.
b) Nam đủ 55 tuổi, nữ đủ 50 tuổi và có đủ 20 năm đóng BHXH trở lên, trong đó có đủ 15 năm làm công việc nặng nhọc, độc hại, nguy hiểm.
"""
    file_path = os.path.join(temp_input_dir, "luat_bhxh_58_2014.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path


# ==============================================================================
# MOCK FIXTURES
# ==============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response cho Q&A generation"""
    return '''```json
[
  {
    "question": "Mức đóng BHXH của người lao động là bao nhiêu?",
    "answer": "Căn cứ Điều 5, Khoản 1, Luật BHXH số 58/2014/QH13, mức đóng BHXH của người lao động là 8% tiền lương tháng."
  },
  {
    "question": "Điều kiện hưởng lương hưu là gì?",
    "answer": "Căn cứ Điều 54, Luật BHXH số 58/2014/QH13, người lao động được hưởng lương hưu khi đủ tuổi và đóng BHXH đủ 20 năm."
  }
]
```'''


@pytest.fixture
def mock_eval_response():
    """Mock LLM response cho evaluation"""
    return '''{"score": 8.5, "has_legal_citation": true, "reason": "Có căn cứ pháp lý đầy đủ", "keep": true}'''

