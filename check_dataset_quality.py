# -*- coding: utf-8 -*-
"""
Dataset Quality Checker
Kiem tra chat luong dataset Q&A

Usage:
    python check_dataset_quality.py
    python check_dataset_quality.py --file output/evaluated/qa_good.json
"""

import json
import os
import re
import sys
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_json(file_path: str) -> List[Dict]:
    """Load JSON file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_duplicates(qa_list: List[Dict]) -> Dict:
    """Kiem tra cau hoi/tra loi trung lap"""
    questions = [qa.get('question', '').strip().lower() for qa in qa_list]
    answers = [qa.get('answer', '').strip().lower() for qa in qa_list]

    q_counter = Counter(questions)
    a_counter = Counter(answers)

    dup_questions = {q: c for q, c in q_counter.items() if c > 1}
    dup_answers = {a[:100]: c for a, c in a_counter.items() if c > 1}

    return {
        'duplicate_questions': len(dup_questions),
        'duplicate_answers': len(dup_answers),
        'top_dup_questions': list(dup_questions.items())[:5],
        'top_dup_answers': [(a[:80] + '...', c) for a, c in list(dup_answers.items())[:5]]
    }


def check_length(qa_list: List[Dict]) -> Dict:
    """Kiem tra do dai Q&A"""
    q_lengths = [len(qa.get('question', '')) for qa in qa_list]
    a_lengths = [len(qa.get('answer', '')) for qa in qa_list]

    short_questions = sum(1 for l in q_lengths if l < 15)
    short_answers = sum(1 for l in a_lengths if l < 50)
    long_answers = sum(1 for l in a_lengths if l > 3000)

    return {
        'avg_question_length': sum(q_lengths) / len(q_lengths) if q_lengths else 0,
        'avg_answer_length': sum(a_lengths) / len(a_lengths) if a_lengths else 0,
        'min_question_length': min(q_lengths) if q_lengths else 0,
        'max_question_length': max(q_lengths) if q_lengths else 0,
        'min_answer_length': min(a_lengths) if a_lengths else 0,
        'max_answer_length': max(a_lengths) if a_lengths else 0,
        'short_questions': short_questions,
        'short_answers': short_answers,
        'long_answers': long_answers
    }


def check_legal_citations(qa_list: List[Dict]) -> Dict:
    """Kiem tra can cu phap ly"""
    patterns = {
        'doc_number': r'\d+/\d{4}/[A-Za-z\-]+',  # 41/2024/QH15
        'article': r'[Dd]iều\s+\d+',  # Dieu X
        'clause': r'[Kk]hoản\s+\d+',  # Khoan Y
        'point': r'điểm\s+[a-zđ]',  # diem a
    }

    stats = {k: 0 for k in patterns}
    no_citation = 0

    for qa in qa_list:
        answer = qa.get('answer', '')
        has_any = False

        for name, pattern in patterns.items():
            if re.search(pattern, answer, re.IGNORECASE):
                stats[name] += 1
                has_any = True

        if not has_any:
            no_citation += 1

    total = len(qa_list)
    return {
        'with_doc_number': f"{stats['doc_number']} ({stats['doc_number']/total*100:.1f}%)",
        'with_article': f"{stats['article']} ({stats['article']/total*100:.1f}%)",
        'with_clause': f"{stats['clause']} ({stats['clause']/total*100:.1f}%)",
        'with_point': f"{stats['point']} ({stats['point']/total*100:.1f}%)",
        'no_citation': f"{no_citation} ({no_citation/total*100:.1f}%)"
    }


def check_format(qa_list: List[Dict]) -> Dict:
    """Kiem tra format"""
    no_question_mark = 0
    starts_lowercase = 0
    has_placeholder = 0  # ... hoac ____

    for qa in qa_list:
        q = qa.get('question', '')
        a = qa.get('answer', '')

        if not q.strip().endswith('?'):
            no_question_mark += 1

        if q and q[0].islower():
            starts_lowercase += 1

        if '...' in a or '____' in a or '…' in a:
            has_placeholder += 1

    total = len(qa_list)
    return {
        'no_question_mark': f"{no_question_mark} ({no_question_mark/total*100:.1f}%)",
        'starts_lowercase': f"{starts_lowercase} ({starts_lowercase/total*100:.1f}%)",
        'has_placeholder': f"{has_placeholder} ({has_placeholder/total*100:.1f}%)"
    }


def check_distribution(qa_list: List[Dict]) -> Dict:
    """Kiem tra phan bo theo document"""
    doc_counter = Counter(qa.get('source_doc', 'unknown') for qa in qa_list)

    docs = list(doc_counter.items())
    docs.sort(key=lambda x: x[1], reverse=True)

    return {
        'total_documents': len(doc_counter),
        'avg_qa_per_doc': len(qa_list) / len(doc_counter) if doc_counter else 0,
        'top_5_docs': docs[:5],
        'bottom_5_docs': docs[-5:] if len(docs) >= 5 else docs,
        'max_qa_in_doc': docs[0] if docs else ('N/A', 0),
        'min_qa_in_doc': docs[-1] if docs else ('N/A', 0)
    }


def check_content_quality(qa_list: List[Dict]) -> Dict:
    """Kiem tra chat luong noi dung"""
    vague_answers = 0
    generic_questions = 0

    vague_patterns = [
        r'không thể trả lời',
        r'cần thêm thông tin',
        r'tùy thuộc vào',
        r'liên hệ.*để biết thêm',
    ]

    generic_patterns = [
        r'^thông tư này',
        r'^nghị định này',
        r'^luật này',
        r'^văn bản này',
    ]

    for qa in qa_list:
        q = qa.get('question', '').lower()
        a = qa.get('answer', '').lower()

        if any(re.search(p, a) for p in vague_patterns):
            vague_answers += 1

        if any(re.search(p, q) for p in generic_patterns):
            generic_questions += 1

    total = len(qa_list)
    return {
        'vague_answers': f"{vague_answers} ({vague_answers/total*100:.1f}%)",
        'generic_questions': f"{generic_questions} ({generic_questions/total*100:.1f}%)"
    }


def print_report(qa_list: List[Dict], file_path: str):
    """In bao cao tong hop"""
    print("\n" + "="*60)
    print("DATASET QUALITY REPORT")
    print("="*60)
    print(f"File: {file_path}")
    print(f"Total Q&A pairs: {len(qa_list)}")
    print()

    # 1. Duplicates
    print("-"*40)
    print("1. DUPLICATES CHECK")
    print("-"*40)
    dup = check_duplicates(qa_list)
    print(f"  Duplicate questions: {dup['duplicate_questions']}")
    print(f"  Duplicate answers: {dup['duplicate_answers']}")
    if dup['top_dup_questions']:
        print("  Top duplicate questions:")
        for q, c in dup['top_dup_questions'][:3]:
            print(f"    - [{c}x] {q[:60]}...")
    print()

    # 2. Length
    print("-"*40)
    print("2. LENGTH CHECK")
    print("-"*40)
    length = check_length(qa_list)
    print(f"  Avg question length: {length['avg_question_length']:.0f} chars")
    print(f"  Avg answer length: {length['avg_answer_length']:.0f} chars")
    print(f"  Question range: {length['min_question_length']} - {length['max_question_length']} chars")
    print(f"  Answer range: {length['min_answer_length']} - {length['max_answer_length']} chars")
    print(f"  Short questions (<15 chars): {length['short_questions']}")
    print(f"  Short answers (<50 chars): {length['short_answers']}")
    print(f"  Long answers (>3000 chars): {length['long_answers']}")
    print()

    # 3. Legal citations
    print("-"*40)
    print("3. LEGAL CITATIONS CHECK")
    print("-"*40)
    legal = check_legal_citations(qa_list)
    print(f"  With doc number (XX/YYYY/XX): {legal['with_doc_number']}")
    print(f"  With article (Dieu X): {legal['with_article']}")
    print(f"  With clause (Khoan Y): {legal['with_clause']}")
    print(f"  With point (diem a): {legal['with_point']}")
    print(f"  NO citation at all: {legal['no_citation']}")
    print()

    # 4. Format
    print("-"*40)
    print("4. FORMAT CHECK")
    print("-"*40)
    fmt = check_format(qa_list)
    print(f"  Missing question mark: {fmt['no_question_mark']}")
    print(f"  Starts with lowercase: {fmt['starts_lowercase']}")
    print(f"  Has placeholder (...): {fmt['has_placeholder']}")
    print()

    # 5. Distribution
    print("-"*40)
    print("5. DISTRIBUTION BY DOCUMENT")
    print("-"*40)
    dist = check_distribution(qa_list)
    print(f"  Total documents: {dist['total_documents']}")
    print(f"  Avg Q&A per doc: {dist['avg_qa_per_doc']:.1f}")
    print(f"  Max: {dist['max_qa_in_doc'][0]} ({dist['max_qa_in_doc'][1]} Q&A)")
    print(f"  Min: {dist['min_qa_in_doc'][0]} ({dist['min_qa_in_doc'][1]} Q&A)")
    print()

    # 6. Content quality
    print("-"*40)
    print("6. CONTENT QUALITY CHECK")
    print("-"*40)
    content = check_content_quality(qa_list)
    print(f"  Vague answers: {content['vague_answers']}")
    print(f"  Generic questions: {content['generic_questions']}")
    print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    issues = []
    if dup['duplicate_questions'] > 0:
        issues.append(f"- {dup['duplicate_questions']} duplicate questions")
    if length['short_answers'] > 0:
        issues.append(f"- {length['short_answers']} short answers")
    if 'no_citation' in legal and int(legal['no_citation'].split()[0]) > len(qa_list) * 0.1:
        issues.append(f"- {legal['no_citation']} without legal citation")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("No major issues found. Dataset looks good!")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Check dataset quality")
    parser.add_argument(
        "--file", "-f",
        default="output/evaluated/qa_good.json",
        help="Path to Q&A JSON file"
    )
    args = parser.parse_args()

    qa_list = load_json(args.file)
    if qa_list:
        print_report(qa_list, args.file)


if __name__ == "__main__":
    main()
