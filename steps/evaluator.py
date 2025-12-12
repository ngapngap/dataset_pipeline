# -*- coding: utf-8 -*-
"""
Quality Evaluator - Đánh giá và lọc chất lượng Q&A
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from core.logger import get_logger
from core.utils import save_json, load_json, save_jsonl
from providers import create_providers_from_config


logger = get_logger(__name__)


# Prompt đánh giá mặc định - CHO DATASET PHÁP LUẬT
DEFAULT_EVAL_PROMPT = """Đánh giá chất lượng cặp câu hỏi-trả lời PHÁP LUẬT sau theo thang điểm 1-10.

CÂU HỎI: {question}

CÂU TRẢ LỜI: {answer}

TIÊU CHÍ ĐÁNH GIÁ (quan trọng theo thứ tự):

1. ⭐ CĂN CỨ PHÁP LÝ (40% trọng số):
   - Có trích dẫn điều, khoản, tên văn bản pháp luật không?
   - Ví dụ tốt: "Căn cứ Điều 5, Khoản 2, Luật BHXH 2024..."
   - Nếu KHÔNG có căn cứ → điểm tối đa = 4

2. ĐỘ CHÍNH XÁC (30% trọng số):
   - Thông tin có chính xác theo văn bản gốc không?
   - Có số liệu cụ thể (%, năm, tiền) không?

3. CÂU HỎI THỰC TẾ (20% trọng số):
   - Câu hỏi có như người dân thật sự hỏi không?
   - Hay chỉ là câu hỏi lý thuyết chung chung?

4. RÕ RÀNG, DỄ HIỂU (10% trọng số):
   - Ngôn ngữ có dễ hiểu cho người dân không?

OUTPUT FORMAT (JSON):
{{"score": <1-10>, "has_legal_citation": <true/false>, "reason": "<lý do>", "keep": <true/false>}}

⚠️ keep=true CHỈ KHI: score >= 7 VÀ has_legal_citation=true"""


class QualityEvaluator:
    """
    Đánh giá và lọc chất lượng Q&A pairs.
    Có thể dùng LLM hoặc rule-based.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Config cho evaluation
        """
        # Evaluation mode
        self.mode = config.get("mode", "hybrid")  # "llm", "rule", "hybrid"
        self.min_score = config.get("min_score", 7.0)
        
        # Rule-based config
        self.min_question_length = config.get("min_question_length", 10)
        self.min_answer_length = config.get("min_answer_length", 20)
        self.max_answer_length = config.get("max_answer_length", 5000)
        
        # LLM config (nếu mode != "rule")
        self.provider_name = config.get("provider", "gemini")
        self.model = config.get("model", "gemini-2.0-flash")
        self.api_keys = config.get("api_keys", [])
        self.generation_config = config.get("generation", {})
        self.eval_prompt = config.get("eval_prompt", DEFAULT_EVAL_PROMPT)
        
        # Output
        self.output_dir = config.get("output_dir", "data/evaluated")
        self.request_delay = config.get("request_delay", 0.5)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tạo providers nếu cần LLM evaluation
        self.providers = []
        if self.mode in ["llm", "hybrid"] and self.api_keys:
            self.providers = create_providers_from_config(
                provider_name=self.provider_name,
                api_keys=self.api_keys,
                model=self.model,
                generation_config=self.generation_config
            )
            logger.info(f"Evaluator: {len(self.providers)} providers ({self.provider_name})")
        
        # Tracking
        self._results_lock = Lock()
        self._evaluated = []
        self._good = []
        self._bad = []
    
    def evaluate(self, qa_pairs: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Đánh giá danh sách Q&A pairs

        Args:
            qa_pairs: List Q&A pairs cần đánh giá

        Returns:
            (good_qa, bad_qa): Tuple 2 lists
        """
        logger.info(f"Bắt đầu đánh giá {len(qa_pairs)} Q&A pairs (mode={self.mode})")

        if not qa_pairs:
            return [], []

        # STEP 1: Deduplicate - loại bỏ Q&A trùng lặp
        qa_pairs, duplicates = self._deduplicate(qa_pairs)
        if duplicates:
            logger.info(f"Đã loại bỏ {len(duplicates)} Q&A trùng lặp")

        # Reset
        self._evaluated = []
        self._good = []
        self._bad = []

        if self.mode == "rule":
            # Rule-based only
            self._evaluate_rule_based(qa_pairs)
        elif self.mode == "llm":
            # LLM only
            self._evaluate_llm(qa_pairs)
        else:
            # Hybrid: Rule first, then LLM for borderline cases
            self._evaluate_hybrid(qa_pairs)

        # Lưu kết quả
        self._save_results()

        # Lưu duplicates nếu có
        if duplicates:
            self._save_duplicates(duplicates)

        logger.info(f"Kết quả: {len(self._good)} good, {len(self._bad)} bad")

        return self._good, self._bad

    def _deduplicate(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Loại bỏ Q&A trùng lặp

        Tiêu chí trùng lặp:
        - Cùng question (normalize lowercase, strip)
        - Hoặc cùng answer (normalize)

        Giữ lại Q&A có answer dài hơn trong trường hợp trùng question.

        Returns:
            (unique_qa, duplicates)
        """
        seen_questions = {}  # question -> qa với answer dài nhất
        seen_answers = set()
        unique = []
        duplicates = []

        for qa in qa_pairs:
            q = qa.get("question", "").strip().lower()
            a = qa.get("answer", "").strip().lower()

            # Check duplicate question
            if q in seen_questions:
                existing_qa = seen_questions[q]
                # Giữ cái có answer dài hơn
                if len(qa.get("answer", "")) > len(existing_qa.get("answer", "")):
                    # Thay thế cái cũ bằng cái mới
                    duplicates.append(existing_qa)
                    seen_questions[q] = qa
                    # Cập nhật unique list
                    for i, u in enumerate(unique):
                        if u.get("question", "").strip().lower() == q:
                            unique[i] = qa
                            break
                else:
                    duplicates.append(qa)
                continue

            # Check duplicate answer (exact match)
            if a in seen_answers:
                duplicates.append(qa)
                continue

            # Unique Q&A
            seen_questions[q] = qa
            seen_answers.add(a)
            unique.append(qa)

        return unique, duplicates

    def _save_duplicates(self, duplicates: List[Dict]):
        """Lưu các Q&A trùng lặp để review"""
        dup_file = os.path.join(self.output_dir, "qa_duplicates.json")
        save_json(duplicates, dup_file)
        logger.info(f"Đã lưu {len(duplicates)} duplicates -> {dup_file}")
    
    def _evaluate_rule_based(self, qa_pairs: List[Dict]):
        """Đánh giá bằng rules"""
        for qa in tqdm(qa_pairs, desc="Rule-based evaluation"):
            score, reason = self._rule_score(qa)
            qa["eval_score"] = score
            qa["eval_reason"] = reason
            qa["eval_method"] = "rule"
            
            if score >= self.min_score:
                self._good.append(qa)
            else:
                self._bad.append(qa)
    
    def _rule_score(self, qa: Dict) -> Tuple[float, str]:
        """
        Tính điểm bằng rules - ĐẶC BIỆT CHO DATASET PHÁP LUẬT
        
        Returns:
            (score, reason)
        """
        import re
        
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        answer_lower = answer.lower()
        
        score = 10.0
        reasons = []
        
        # ========== KIỂM TRA SỐ HIỆU VĂN BẢN (BẮT BUỘC - QUAN TRỌNG NHẤT) ==========
        # Pattern số hiệu văn bản: XX/YYYY/XX-XX hoặc XX-XXXX
        # CHỈ match số hiệu chính thức, KHÔNG match "năm 2025" chung chung
        document_number_patterns = [
            r'\d+/\d{4}/[A-Za-zĐ]+[-/]?[A-Za-z]*',  # 41/2024/QH15, 25/2025/TT-BYT
            r'\d+/\d{4}/NĐ-CP',                       # Nghị định
            r'\d+/\d{4}/TT-[A-Z]+',                   # Thông tư
            r'\d+/\d{4}/QH\d+',                       # Luật
            r'số\s+\d+/\d{4}',                        # "số 41/2024"
            # XÓA: r'năm\s+\d{4}' - quá rộng, match cả "năm 2025" trong nội dung
        ]
        
        has_document_number = any(re.search(pattern, answer, re.IGNORECASE) for pattern in document_number_patterns)
        
        # ========== KIỂM TRA CÓ TRÍCH DẪN ĐIỀU KHOẢN ==========
        has_article = bool(re.search(r'(điều|dieu)\s+\d+', answer_lower))
        has_clause = bool(re.search(r'(khoản|khoan)\s+\d+', answer_lower))
        
        # ========== KIỂM TRA TỪ KHÓA CHUNG (yếu hơn) ==========
        general_legal_keywords = [
            # Có dấu
            "căn cứ", "theo quy định", "quy định tại",
            "luật ", "nghị định", "thông tư", "quyết định",
            # Không dấu (tiếng Việt ASCII)
            "can cu", "theo quy dinh", "quy dinh tai",
            "luat ", "nghi dinh", "thong tu", "quyet dinh"
        ]
        has_general_keywords = any(kw in answer_lower for kw in general_legal_keywords)
        
        # ========== TÍNH ĐIỂM ==========
        
        # Trường hợp tệ nhất: không có gì
        if not has_document_number and not has_article and not has_general_keywords:
            score -= 6
            reasons.append("❌ KHÔNG CÓ CĂN CỨ PHÁP LÝ")
        
        # Có điều khoản + từ khóa chung nhưng KHÔNG có số hiệu văn bản cụ thể
        # Ví dụ: "Căn cứ Điều 21, Khoản 2 Thông tư này" -> thiếu số hiệu!
        elif has_article and has_general_keywords and not has_document_number:
            score -= 5  # Trừ nặng: có điều khoản nhưng không có số hiệu = KHÔNG ĐÁNG TIN
            reasons.append("⚠️ Có điều khoản nhưng THIẾU SỐ HIỆU văn bản (chỉ có 'Thông tư này')")
        
        # Có từ khóa chung nhưng KHÔNG có số hiệu và KHÔNG có điều khoản
        elif has_general_keywords and not has_document_number and not has_article:
            score -= 4
            reasons.append("⚠️ Thiếu số hiệu văn bản và điều khoản cụ thể")
        
        # Có số hiệu nhưng không có điều khoản
        elif has_document_number and not has_article:
            score -= 1
            reasons.append("Có số hiệu VB nhưng thiếu trích dẫn điều khoản")
        
        # Trường hợp tốt: có đầy đủ số hiệu + điều khoản
        elif has_document_number and has_article:
            # Bonus nếu có cả khoản
            if has_clause:
                score += 0.5
        
        # ========== KIỂM TRA SỐ LIỆU CỤ THỂ ==========
        has_numbers = bool(re.search(r'\d+(\.\d+)?\s*(%|năm|tháng|ngày|đồng|triệu|tỷ|VNĐ)', answer))
        if not has_numbers:
            # Không trừ nhiều nếu đã có căn cứ pháp lý tốt
            if has_document_number:
                score -= 0.5
            else:
                score -= 1
            reasons.append("Thiếu số liệu cụ thể")
        
        # ========== KIỂM TRA ĐỘ DÀI ==========
        if len(question) < self.min_question_length:
            score -= 2
            reasons.append("Câu hỏi quá ngắn")
        
        if len(answer) < self.min_answer_length:
            score -= 3
            reasons.append("Câu trả lời quá ngắn")
        elif len(answer) > self.max_answer_length:
            score -= 1
            reasons.append("Câu trả lời quá dài")
        
        # ========== KIỂM TRA FORMAT ==========
        if not question.endswith("?"):
            score -= 0.5
            reasons.append("Câu hỏi thiếu dấu ?")
        
        # ========== KIỂM TRA CÂU HỎI CHUNG CHUNG ==========
        # Câu hỏi kiểu "văn bản này có hiệu lực..." không có giá trị thực tế
        vague_question_patterns = [
            r'(thông tư|nghị định|luật|quyết định|văn bản)\s+(này|trên)\s+(có hiệu lực|quy định|áp dụng)',
            r'^(thông tư|nghị định|luật)\s+(này|số)\s+',
            r'(văn bản|thông tư|nghị định)\s+này\s+',
            r'^danh mục\s+',
            r'^trình tự\s+',
            # Không dấu
            r'(thong tu|nghi dinh|luat|quyet dinh|van ban)\s+(nay|tren)\s+',
            r'^(thong tu|nghi dinh|luat)\s+(nay|so)\s+',
            r'(van ban|thong tu|nghi dinh)\s+nay\s+',
            r'^danh muc\s+',
            r'^trinh tu\s+',
        ]
        question_lower = question.lower()
        is_vague_question = any(re.search(p, question_lower) for p in vague_question_patterns)
        
        # Câu hỏi TỐT phải có từ khóa cá nhân như "tôi", "của tôi"
        # "khi nào", "bao nhiêu" một mình không đủ - phải đi kèm context cá nhân
        personal_keywords = ['tôi ', ' tôi', 'của tôi', 'cho tôi', 'tôi có', 'tôi được', 'tôi phải', 
                           'toi ', ' toi', 'cua toi', 'cho toi', 'toi co', 'toi duoc', 'toi phai',
                           'doanh nghiệp tôi', 'công ty tôi', 'doanh nghiep toi', 'cong ty toi']
        has_personal = any(kw in question_lower for kw in personal_keywords)
        
        if is_vague_question and not has_personal:
            score -= 5  # Trừ nặng để FAIL
            reasons.append("⚠️ Câu hỏi quá chung chung (không như người dân thật hỏi)")
        
        # Check empty/placeholder
        if "..." in answer and len(answer) < 50:
            score -= 3
            reasons.append("Câu trả lời không đầy đủ")
        
        # Check gibberish (basic)
        if len(set(question)) < 5 or len(set(answer)) < 10:
            score -= 5
            reasons.append("Nội dung có vẻ bất thường")
        
        # ========== CHECK "THÔNG TƯ NÀY", "LUẬT NÀY" ==========
        vague_references = ["thông tư này", "luật này", "nghị định này", "quyết định này", "văn bản này"]
        if any(vr in answer_lower for vr in vague_references) and not has_document_number:
            score -= 2
            reasons.append("⚠️ Dùng 'văn bản này' mà không ghi số hiệu")
        
        score = max(0, min(10, score))
        
        # Tạo reason text
        if not reasons:
            if has_document_number and has_article:
                reason = "✅ Đầy đủ: số hiệu VB + điều khoản"
            elif has_document_number:
                reason = "✅ Có số hiệu văn bản"
            else:
                reason = "✅ OK"
        else:
            reason = "; ".join(reasons)
        
        return score, reason
    
    def _evaluate_llm(self, qa_pairs: List[Dict]):
        """Đánh giá bằng LLM"""
        if not self.providers:
            logger.error("Không có provider để đánh giá bằng LLM!")
            return
        
        # Phân phối cho providers
        num_providers = len(self.providers)
        queues = [[] for _ in range(num_providers)]
        
        for i, qa in enumerate(qa_pairs):
            queues[i % num_providers].append(qa)
        
        # Chạy đa luồng
        with ThreadPoolExecutor(max_workers=num_providers) as executor:
            futures = []
            for i, provider in enumerate(self.providers):
                future = executor.submit(
                    self._worker_evaluate_llm,
                    provider=provider,
                    qa_list=queues[i],
                    worker_id=i + 1
                )
                futures.append(future)
            
            # Progress
            pbar = tqdm(total=len(qa_pairs), desc="LLM evaluation")
            
            completed = 0
            while completed < len(qa_pairs):
                time.sleep(0.5)
                new_completed = len(self._good) + len(self._bad)
                if new_completed > completed:
                    pbar.update(new_completed - completed)
                    completed = new_completed
            
            pbar.close()
            
            for future in futures:
                future.result()
    
    def _worker_evaluate_llm(self, provider, qa_list: List[Dict], worker_id: int):
        """Worker đánh giá bằng LLM"""
        for qa in qa_list:
            try:
                score, reason = self._llm_score(provider, qa)
                qa["eval_score"] = score
                qa["eval_reason"] = reason
                qa["eval_method"] = "llm"
                
                with self._results_lock:
                    if score >= self.min_score:
                        self._good.append(qa)
                    else:
                        self._bad.append(qa)
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} lỗi: {e}")
                # Fallback to rule-based
                score, reason = self._rule_score(qa)
                qa["eval_score"] = score
                qa["eval_reason"] = reason
                qa["eval_method"] = "rule_fallback"
                
                with self._results_lock:
                    if score >= self.min_score:
                        self._good.append(qa)
                    else:
                        self._bad.append(qa)
    
    def _llm_score(self, provider, qa: Dict) -> Tuple[float, str]:
        """
        Đánh giá 1 Q&A bằng LLM
        
        Returns:
            (score, reason)
        """
        prompt = self.eval_prompt.format(
            question=qa.get("question", ""),
            answer=qa.get("answer", "")
        )
        
        response = provider.generate(prompt)
        
        if not response:
            raise ValueError("Empty LLM response")
        
        # Parse JSON
        return self._parse_eval_response(response)
    
    def _parse_eval_response(self, response: str) -> Tuple[float, str]:
        """Parse evaluation response từ LLM"""
        try:
            response = response.strip()
            
            # Xử lý markdown
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            # Tìm JSON object
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                score = float(data.get("score", 5))
                reason = data.get("reason", "N/A")
                
                return score, reason
            
            return 5.0, "Parse error"
            
        except Exception as e:
            logger.warning(f"Eval parse error: {e}")
            return 5.0, f"Parse error: {e}"
    
    def _evaluate_hybrid(self, qa_pairs: List[Dict]):
        """
        Đánh giá hybrid: Rule first, LLM cho borderline cases
        """
        borderline = []
        
        # Rule-based first pass
        for qa in tqdm(qa_pairs, desc="Rule-based (pass 1)"):
            score, reason = self._rule_score(qa)
            qa["eval_score"] = score
            qa["eval_reason"] = reason
            qa["eval_method"] = "rule"
            
            if score >= 8:  # Clearly good
                self._good.append(qa)
            elif score < 5:  # Clearly bad
                self._bad.append(qa)
            else:  # Borderline (5-7.9) -> LLM evaluate
                borderline.append(qa)
        
        logger.info(f"Pass 1: {len(self._good)} good, {len(self._bad)} bad, {len(borderline)} borderline")
        
        # LLM for borderline cases
        if borderline and self.providers:
            logger.info(f"LLM evaluating {len(borderline)} borderline cases...")
            
            # Temporary swap
            temp_good = self._good
            temp_bad = self._bad
            self._good = []
            self._bad = []
            
            self._evaluate_llm(borderline)
            
            # Merge back
            temp_good.extend(self._good)
            temp_bad.extend(self._bad)
            self._good = temp_good
            self._bad = temp_bad
        elif borderline:
            # No LLM, use rule score
            for qa in borderline:
                if qa["eval_score"] >= self.min_score:
                    self._good.append(qa)
                else:
                    self._bad.append(qa)
    
    def _save_results(self):
        """Lưu kết quả đánh giá"""
        # Good Q&A
        good_json = os.path.join(self.output_dir, "qa_good.json")
        save_json(self._good, good_json)
        
        good_jsonl = os.path.join(self.output_dir, "qa_good.jsonl")
        save_jsonl(self._good, good_jsonl)
        
        # Bad Q&A
        bad_json = os.path.join(self.output_dir, "qa_bad.json")
        save_json(self._bad, bad_json)
        
        # Stats
        stats = {
            "total": len(self._good) + len(self._bad),
            "good": len(self._good),
            "bad": len(self._bad),
            "good_rate": len(self._good) / max(1, len(self._good) + len(self._bad)) * 100,
            "min_score_threshold": self.min_score,
            "evaluation_mode": self.mode
        }
        
        stats_file = os.path.join(self.output_dir, "evaluation_stats.json")
        save_json(stats, stats_file)
        
        logger.info(f"Đã lưu: {len(self._good)} good -> {good_json}")
        logger.info(f"Good rate: {stats['good_rate']:.1f}%")
    
    def load_evaluated(self) -> Tuple[List[Dict], List[Dict]]:
        """Load kết quả đã evaluate"""
        good = load_json(os.path.join(self.output_dir, "qa_good.json"))
        bad = load_json(os.path.join(self.output_dir, "qa_bad.json"))
        return good or [], bad or []
