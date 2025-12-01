# -*- coding: utf-8 -*-
"""
System Prompt Builder - Xây dựng system prompt động cho training

Module này giải quyết vấn đề:
1. Dataset cũ không có system prompt → model không biết vai trò
2. Không thể hardcode một luật cụ thể vì BHXH có 30+ văn bản
3. Cần thêm context về thời gian và văn bản đang trích dẫn

Cách sử dụng:
- Khi generate QA: Tự động extract văn bản đang trích dẫn từ answer
- Khi training: Format lại dataset với system prompt phù hợp
"""

import re
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class LegalDocument:
    """Thông tin văn bản pháp luật"""
    doc_id: str           # "41/2024/QH15"
    doc_type: str         # "LUẬT", "NGHỊ ĐỊNH", "THÔNG TƯ"
    issuer: str           # "QUỐC HỘI", "CHÍNH PHỦ"
    effective_date: str   # "01/07/2025"
    status: str           # "Còn hiệu lực", "Hết hiệu lực"
    description: str      # Mô tả ngắn


class SystemPromptBuilder:
    """
    Builder để tạo system prompt động dựa trên nội dung QA.
    
    Thay vì hardcode "Luật 41/2024", module này sẽ:
    1. Phát hiện văn bản được trích dẫn trong answer
    2. Tạo system prompt phù hợp với văn bản đó
    3. Thêm context về thời gian và hiệu lực
    """
    
    # Patterns để phát hiện các loại văn bản
    DOCUMENT_PATTERNS = {
        "LUẬT": [
            r"Luật\s+(?:số\s+)?(\d+/\d+/QH\d+)",
            r"Luật\s+BHXH\s+(?:số\s+)?(\d+/\d+/QH\d+)",
            r"Luật\s+Bảo hiểm xã hội\s+(?:số\s+)?(\d+/\d+/QH\d+)",
            r"Luật\s+Bảo hiểm y tế\s+(?:số\s+)?(\d+/\d+/QH\d+)",
            r"Bộ luật\s+Lao động\s+(?:số\s+)?(\d+/\d+/QH\d+)",
        ],
        "NGHỊ ĐỊNH": [
            r"Nghị định\s+(?:số\s+)?(\d+/\d+/NĐ-CP)",
            r"NĐ\s+(\d+/\d+/NĐ-CP)",
        ],
        "THÔNG TƯ": [
            r"Thông tư\s+(?:số\s+)?(\d+/\d+/TT-[A-Z]+)",
            r"TT\s+(\d+/\d+/TT-[A-Z]+)",
        ],
        "QUYẾT ĐỊNH": [
            r"Quyết định\s+(?:số\s+)?(\d+/QĐ-BHXH)",
            r"QĐ\s+(\d+/QĐ-BHXH)",
        ],
    }
    
    # Database văn bản (load từ CSV hoặc JSON)
    _document_db: Dict[str, LegalDocument] = {}
    
    def __init__(self, document_registry_path: Optional[str] = None):
        """
        Args:
            document_registry_path: Path đến file JSON chứa thông tin văn bản
        """
        if document_registry_path:
            self.load_document_registry(document_registry_path)
    
    def load_document_registry(self, path: str):
        """Load database văn bản từ file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for doc in data.get("documents", []):
                    self._document_db[doc["id"]] = LegalDocument(
                        doc_id=doc["id"],
                        doc_type=doc.get("type", ""),
                        issuer=doc.get("issuer", ""),
                        effective_date=doc.get("effective_date", ""),
                        status=doc.get("status", "Còn hiệu lực"),
                        description=doc.get("description", "")
                    )
        except Exception as e:
            print(f"Warning: Could not load document registry: {e}")
    
    def detect_cited_documents(self, text: str) -> List[Dict[str, str]]:
        """
        Phát hiện các văn bản được trích dẫn trong text.
        
        Args:
            text: Nội dung câu trả lời
            
        Returns:
            List các văn bản được trích dẫn
        """
        cited = []
        seen = set()
        
        for doc_type, patterns in self.DOCUMENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    doc_id = match.group(1)
                    if doc_id not in seen:
                        seen.add(doc_id)
                        cited.append({
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "full_match": match.group(0)
                        })
        
        return cited
    
    def build_system_prompt(
        self, 
        cited_documents: List[Dict[str, str]] = None,
        reference_date: str = None,
        mode: str = "general"
    ) -> str:
        """
        Xây dựng system prompt dựa trên văn bản trích dẫn.
        
        Args:
            cited_documents: Danh sách văn bản được trích dẫn (từ detect_cited_documents)
            reference_date: Ngày tham chiếu (format: DD/MM/YYYY)
            mode: Chế độ - "general", "specific", "minimal"
            
        Returns:
            System prompt phù hợp
        """
        if reference_date is None:
            reference_date = datetime.now().strftime("%d/%m/%Y")
        
        if mode == "minimal":
            return "Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam."
        
        # Base system prompt
        base = """Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam.
Nhiệm vụ: Trả lời câu hỏi về BHXH, BHYT, BHTN dựa trên các văn bản pháp luật hiện hành.
Ngày hiện tại: {date}""".format(date=reference_date)
        
        if mode == "general" or not cited_documents:
            # Prompt chung - không biết trước văn bản nào
            return base + """

Nguyên tắc trả lời:
1. Luôn trích dẫn điều khoản cụ thể: "Căn cứ Điều X, Khoản Y của [Tên văn bản]"
2. Ưu tiên văn bản mới nhất đang có hiệu lực
3. Nếu có quy định chuyển tiếp, nêu rõ thời điểm áp dụng
4. Giải thích dễ hiểu, kèm số liệu cụ thể (%, số năm, mức tiền)"""
        
        elif mode == "specific" and cited_documents:
            # Prompt cụ thể - biết văn bản nào
            doc_list = []
            for doc in cited_documents[:3]:  # Tối đa 3 văn bản
                doc_info = self._document_db.get(doc["doc_id"])
                if doc_info:
                    doc_list.append(f"- {doc_info.doc_type} số {doc_info.doc_id} ({doc_info.status})")
                else:
                    doc_list.append(f"- {doc['doc_type']} số {doc['doc_id']}")
            
            docs_str = "\n".join(doc_list)
            
            return base + f"""

Văn bản pháp luật áp dụng:
{docs_str}

Yêu cầu:
1. Trích dẫn chính xác điều khoản từ văn bản trên
2. Giải thích rõ ràng, dễ hiểu
3. Nêu số liệu cụ thể khi có"""
        
        return base
    
    def format_qa_with_system_prompt(
        self,
        question: str,
        answer: str,
        format_type: str = "chat",
        reference_date: str = None
    ) -> str:
        """
        Format QA pair với system prompt.
        
        Args:
            question: Câu hỏi
            answer: Câu trả lời
            format_type: "chat" (ChatML), "instruction" (### format), "alpaca"
            reference_date: Ngày tham chiếu
            
        Returns:
            Text đã format với system prompt
        """
        # Detect văn bản trích dẫn từ answer
        cited_docs = self.detect_cited_documents(answer)
        
        # Build system prompt phù hợp
        if cited_docs:
            system_prompt = self.build_system_prompt(
                cited_documents=cited_docs,
                reference_date=reference_date,
                mode="specific"
            )
        else:
            system_prompt = self.build_system_prompt(
                reference_date=reference_date,
                mode="general"
            )
        
        # Format theo loại
        if format_type == "chat":
            # ChatML format (phổ biến cho chat models)
            return f"""<|system|>
{system_prompt}
<|user|>
{question}
<|assistant|>
{answer}"""
        
        elif format_type == "instruction":
            # Instruction format (như dataset hiện tại nhưng có system)
            return f"""### Hệ thống:
{system_prompt}

### Câu hỏi:
{question}

### Trả lời:
{answer}"""
        
        elif format_type == "alpaca":
            # Alpaca format
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{system_prompt}

{question}

### Response:
{answer}"""
        
        else:
            # Default - simple format
            return f"""{system_prompt}

Câu hỏi: {question}

Trả lời: {answer}"""
    
    def process_dataset(
        self,
        qa_pairs: List[Dict[str, str]],
        format_type: str = "instruction",
        reference_date: str = None
    ) -> List[Dict[str, Any]]:
        """
        Xử lý toàn bộ dataset để thêm system prompt.
        
        Args:
            qa_pairs: List các QA pairs [{"question": ..., "answer": ...}]
            format_type: Loại format
            reference_date: Ngày tham chiếu
            
        Returns:
            List các QA đã format với system prompt và metadata
        """
        processed = []
        
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            # Detect cited documents
            cited_docs = self.detect_cited_documents(answer)
            
            # Format với system prompt
            formatted_text = self.format_qa_with_system_prompt(
                question=question,
                answer=answer,
                format_type=format_type,
                reference_date=reference_date
            )
            
            processed.append({
                "question": question,
                "answer": answer,
                "formatted_text": formatted_text,
                "cited_documents": [d["doc_id"] for d in cited_docs],
                "format_type": format_type,
                "metadata": qa.get("metadata", {})
            })
        
        return processed


# ============================================================================
# SYSTEM PROMPT TEMPLATES - Các mẫu system prompt cho từng loại câu hỏi
# ============================================================================

SYSTEM_PROMPT_TEMPLATES = {
    "default": """Bạn là chuyên gia tư vấn Bảo hiểm xã hội Việt Nam.
Ngày hiện tại: {date}

Nhiệm vụ: Trả lời câu hỏi về BHXH, BHYT, BHTN dựa trên các văn bản pháp luật hiện hành.

Yêu cầu khi trả lời:
1. Trích dẫn căn cứ pháp lý: "Căn cứ Điều X, Khoản Y của [Tên văn bản số hiệu]"
2. Ưu tiên văn bản mới nhất đang có hiệu lực
3. Giải thích rõ ràng, kèm số liệu cụ thể (%, số năm, số tiền)
4. Nếu có quy định chuyển tiếp, nêu rõ thời điểm áp dụng""",

    "bhxh_bat_buoc": """Bạn là chuyên gia tư vấn Bảo hiểm xã hội bắt buộc.
Ngày hiện tại: {date}

Văn bản áp dụng chính:
- Luật BHXH số 41/2024/QH15 (hiệu lực từ 01/07/2025)
- Nghị định 158/2025/NĐ-CP hướng dẫn BHXH bắt buộc

Chủ đề: Mức đóng, chế độ ốm đau, thai sản, hưu trí, tử tuất

Yêu cầu: Trích dẫn điều khoản cụ thể, nêu số liệu %/năm/tiền.""",

    "bhxh_tu_nguyen": """Bạn là chuyên gia tư vấn Bảo hiểm xã hội tự nguyện.
Ngày hiện tại: {date}

Văn bản áp dụng:
- Luật BHXH số 41/2024/QH15
- Nghị định 159/2025/NĐ-CP hướng dẫn BHXH tự nguyện

Chủ đề: Đối tượng tham gia, mức đóng, phương thức đóng, chế độ hưởng.""",

    "bhtn": """Bạn là chuyên gia tư vấn Bảo hiểm thất nghiệp.
Ngày hiện tại: {date}

Văn bản áp dụng:
- Luật BHXH số 41/2024/QH15 (Chương VII)
- Nghị định 157/2025/NĐ-CP hướng dẫn BHTN

Chủ đề: Điều kiện hưởng, mức hưởng, thời gian hưởng trợ cấp thất nghiệp.""",

    "bhyt": """Bạn là chuyên gia tư vấn Bảo hiểm y tế.
Ngày hiện tại: {date}

Văn bản áp dụng:
- Luật BHYT sửa đổi số 51/2024/QH15
- Nghị định 146/2018/NĐ-CP
- Thông tư 01/2025/TT-BYT về chuyển tuyến

Chủ đề: Mức đóng, mức hưởng, quyền lợi KCB, chuyển tuyến.""",

    "luong_huu": """Bạn là chuyên gia tư vấn về lương hưu và trợ cấp.
Ngày hiện tại: {date}

Văn bản áp dụng:
- Luật BHXH số 41/2024/QH15 (Mục 4, 5 Chương III)
- Nghị định 176/2025/NĐ-CP điều chỉnh lương hưu
- Nghị định 188/2025/NĐ-CP mức lương hưu tối thiểu

Chủ đề: Điều kiện hưởng, cách tính, mức hưởng lương hưu.""",
}


def get_system_prompt(category: str = "default", date: str = None) -> str:
    """
    Lấy system prompt theo chủ đề.
    
    Args:
        category: Loại câu hỏi (default, bhxh_bat_buoc, bhtn, bhyt, luong_huu)
        date: Ngày hiện tại (format DD/MM/YYYY)
        
    Returns:
        System prompt
    """
    if date is None:
        date = datetime.now().strftime("%d/%m/%Y")
    
    template = SYSTEM_PROMPT_TEMPLATES.get(category, SYSTEM_PROMPT_TEMPLATES["default"])
    return template.format(date=date)


# ============================================================================
# CLI - Chạy độc lập để test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System Prompt Builder for BHXH QA")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--input", type=str, help="Input JSON file to process")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--format", type=str, default="instruction", 
                       choices=["chat", "instruction", "alpaca"],
                       help="Output format type")
    
    args = parser.parse_args()
    
    if args.test:
        # Test với một QA mẫu
        builder = SystemPromptBuilder()
        
        test_answer = """Căn cứ Điều 102, Khoản 2 của Luật Bảo hiểm xã hội số 41/2024/QH15:

Mức hưởng BHXH một lần được tính như sau:
- Mỗi năm đóng TRƯỚC năm 2014: 1,5 lần mức bình quân thu nhập tháng đóng BHXH
- Mỗi năm đóng TỪ năm 2014: 2 lần mức bình quân thu nhập tháng đóng BHXH

Theo Nghị định 158/2025/NĐ-CP, tháng lẻ được quy đổi theo tỷ lệ tương ứng."""
        
        test_question = "Cách tính BHXH một lần nếu có thời gian đóng trước và sau 2014?"
        
        print("="*80)
        print("🧪 TEST SYSTEM PROMPT BUILDER")
        print("="*80)
        
        # Detect documents
        cited = builder.detect_cited_documents(test_answer)
        print(f"\n📄 Văn bản trích dẫn: {cited}")
        
        # Build system prompt
        system_prompt = builder.build_system_prompt(cited, mode="specific")
        print(f"\n📝 System Prompt:\n{system_prompt}")
        
        # Format QA
        formatted = builder.format_qa_with_system_prompt(
            test_question, test_answer, format_type="instruction"
        )
        print(f"\n📋 Formatted QA:\n{formatted}")
        
    elif args.input and args.output:
        # Process dataset
        builder = SystemPromptBuilder()
        
        with open(args.input, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        processed = builder.process_dataset(qa_pairs, format_type=args.format)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Processed {len(processed)} QA pairs -> {args.output}")
    
    else:
        parser.print_help()
