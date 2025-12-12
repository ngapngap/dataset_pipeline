# -*- coding: utf-8 -*-
import json

# Load still bad Q&A
with open('./output/evaluated/qa_still_bad.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"=== STILL BAD Q&A SAMPLES (first 20) ===\n")
for i, item in enumerate(data[:20]):
    q = item.get('question', '')[:80]
    a = item.get('answer', '')[:180]
    source = item.get('source_doc', '')
    print(f"[{i+1}] Source: {source}")
    print(f"Q: {q}")
    print(f"A: {a}...")
    print("-"*70)
