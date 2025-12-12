# Dataset Pipeline V2 - BHXH Viet Nam

> **Pipeline tu dong tao dataset Q&A chat luong cao cho fine-tuning LLM tu tai lieu phap luat Viet Nam**

## Tinh nang noi bat

- **Tu dong hoan toan**: Chi can chay `python run.py`
- **Multi-provider**: Gemini, OpenAI, Anthropic, Ollama, DeepSeek, vLLM...
- **Smart Caching**: Tiet kiem chi phi API, ho tro resume
- **Document-based Split**: Tranh data leakage
- **ChatML Format**: Chuan cong nghiep cho fine-tuning

## Quick Start

```bash
# 1. Cai dat dependencies
pip install -r requirements.txt

# 2. Cau hinh API keys
# Sua file config.yaml, them API keys vao file tuong ung

# 3. Chay pipeline (tu dong)
python run.py

# 4. Kiem tra chat luong dataset
python check_dataset_quality.py
```

## Installation

```bash
# Clone repo
git clone <repo-url>
cd dataset_pipeline_v2

# Tao virtual environment (khuyen nghi)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Cai dat dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Muc dich |
|---------|----------|
| PyYAML | Doc config files |
| tqdm | Progress bars |
| google-generativeai | Gemini API |
| openai | OpenAI/Custom API |
| anthropic | Claude API |
| python-docx | Doc .docx files |
| PyPDF2 | Doc .pdf files |

## Pipeline Steps

```
+-------------------------------------------------------------+
|                    PIPELINE V2 (5 STEPS)                    |
+-------------------------------------------------------------+
|                                                             |
|  1. EXTRACT      Trich xuat text tu PDF/documents          |
|       |                                                     |
|       v                                                     |
|  2. GENERATE     Sinh Q&A pairs bang LLM                   |
|       |          (multi-thread, cache, retry)              |
|       v                                                     |
|  3. EVALUATE     ALL-IN-ONE:                               |
|       |          +-- Deduplicate (loai bo trung lap)       |
|       |          +-- Evaluate (danh gia chat luong)        |
|       |          +-- Rescue (cuu vot bad Q&A)              |
|       |          +-- Regenerate (tai tao chunks kem)       |
|       v                                                     |
|  4. SPLIT        Chia theo DOCUMENT (tranh data leakage)   |
|       |                                                     |
|       v                                                     |
|  5. EXPORT       Xuat JSON/JSONL (train, val, test)        |
|                  ChatML format voi system prompt           |
+-------------------------------------------------------------+
```

## CLI Options

| Command | Mo ta |
|---------|-------|
| `python run.py` | Chay tu dong (skip cac buoc da xong) |
| `python run.py --force` | Chay lai tu dau |
| `python run.py --step evaluate` | Chi chay 1 step (debug/test) |
| `python run.py -c custom.yaml` | Dung config khac |
| `python run.py --retry-failed` | Retry chunks that bai |

## Auto Mode

Pipeline **tu dong kiem tra va bo qua** cac buoc da hoan thanh:

```bash
$ python run.py

Pipeline V2 AUTO MODE - Tu dong kiem tra va chay cac buoc can thiet

[SKIP] SKIP: EXTRACT (da hoan thanh)
[SKIP] SKIP: GENERATE (da hoan thanh)

==================================================
STEP: EVALUATE
==================================================
[1/3] Evaluating 7638 Q&A pairs...
Da loai bo 1608 Q&A trung lap
[2/3] Rescuing 1068 bad Q&A pairs...
[3/3] Phan tich chunks can regenerate...

==================================================
STEP: SPLIT
==================================================
...
```

## Supported LLM Providers

### Standard Providers

| Provider | Model Example |
|----------|---------------|
| `gemini` | gemini-2.0-flash |
| `openai` | gpt-4o-mini |
| `anthropic` | claude-3-haiku |

### Custom Providers (OpenAI-compatible)

| Provider | Base URL |
|----------|----------|
| `ollama` | http://localhost:11434/v1 |
| `lmstudio` | http://localhost:1234/v1 |
| `vllm` | http://localhost:8000/v1 |
| `deepseek` | https://api.deepseek.com/v1 |
| `together` | https://api.together.xyz/v1 |
| `groq` | https://api.groq.com/openai/v1 |
| `openrouter` | https://openrouter.ai/api/v1 |
| `nvidia` | https://integrate.api.nvidia.com/v1 |
| `megallm` | https://ai.megallm.io/v1 |

**Tu dinh nghia provider:**

```yaml
llm:
  provider: "my_custom_api"
  providers:
    my_custom_api:
      base_url: "https://my-api.example.com/v1"
      model: "my-model"
      api_key: "your-api-key"
      max_tokens: 3000
```

## Output Format: ChatML

Dataset duoc export theo **ChatML format** - chuan cong nghiep cho fine-tuning chat models:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Ban la chuyen gia tu van Bao hiem xa hoi Viet Nam..."
    },
    {
      "role": "user",
      "content": "Toi phai dong BHXH bao nhieu nam de duoc huong luong huu?"
    },
    {
      "role": "assistant",
      "content": "Can cu Dieu 21, Khoan 2 Luat BHXH so 41/2024/QH15..."
    }
  ]
}
```

### Tai sao ChatML?

- Chuan cong nghiep (OpenAI, Anthropic, Google)
- Hau het model chat deu ho tro
- De tich hop voi RAG pipeline
- System prompt dinh nghia persona ro rang

### Custom System Prompt

```yaml
export:
  format: "chat"  # chat | instruction | simple
  system_prompt: |
    Ban la chuyen gia tu van BHXH...
    Custom prompt cua ban o day
```

## Step 3: Evaluate (All-in-One)

Step `evaluate` gop 4 sub-steps:

### 3.1 Deduplicate
- Loai bo Q&A co **question trung lap** (giu cai co answer dai hon)
- Loai bo Q&A co **answer trung lap** (exact match)
- Output: `output/evaluated/qa_duplicates.json`

### 3.2 Evaluate
Danh gia chat luong theo rules:

| Tieu chi | Diem |
|----------|------|
| Co so hieu van ban (XX/YYYY/XX) | +3 |
| Co Dieu/Khoan cu the | +2 |
| Do dai phu hop (100-2000 chars) | +1 |
| Thieu can cu phap ly | **-5** |
| Co placeholder (...) | -2 |

**Nguong dat**: 6/10

### 3.3 Rescue
Tu dong **cuu vot** bad Q&A bang cach:
- Them so hieu van ban tu `source_doc`
- Re-evaluate sau khi sua

### 3.4 Regenerate
- Phan tich chunks co ty le good < 50%
- Tu dong regenerate cac chunks kem
- Loop cho den khi het chunks can regenerate hoac max iterations

## Document-based Split

**Tai sao quan trong?**

```
Van de Random Split (V1):
Document A -> Q1, Q2, Q3
Train: [Q1, Q3]  <- Q&A tu cung document
Test:  [Q2]      <- Model da "thay" document nay! (DATA LEAKAGE)

Giai phap Document Split (V2):
Documents: [A, B, C, D, E]
            | shuffle documents
Train: [A, C, E] -> all Q&A tu 3 docs nay
Val:   [B]       -> all Q&A tu doc nay
Test:  [D]       -> all Q&A tu doc nay
```

## Check Dataset Quality

```bash
$ python check_dataset_quality.py

============================================================
DATASET QUALITY REPORT
============================================================
File: output/evaluated/qa_good.json
Total Q&A pairs: 5087

1. DUPLICATES CHECK
  Duplicate questions: 0      [OK]
  Duplicate answers: 1        [OK]

2. LENGTH CHECK
  Avg question length: 122 chars
  Avg answer length: 392 chars
  Short questions (<15 chars): 0
  Short answers (<50 chars): 0

3. LEGAL CITATIONS CHECK
  With doc number (XX/YYYY/XX): 99.0%  [OK]
  With clause (Khoan Y): 83.5%         [OK]
  NO citation at all: 0.1%             [OK]

4. FORMAT CHECK
  Missing question mark: 1.8%
  Has placeholder (...): 1.7%

5. DISTRIBUTION BY DOCUMENT
  Total documents: 37
  Avg Q&A per doc: 137.5

SUMMARY
============================================================
No major issues found. Dataset looks good!
```

## Output Structure

```
output/
+-- extracted/
|   +-- extracted_documents.json     # Extracted text
|
+-- generated/
|   +-- qa_generated.json            # All generated Q&A
|   +-- failed_chunks.json           # Failed chunks for retry
|
+-- evaluated/
|   +-- qa_good.json                 # Good Q&A (passed)
|   +-- qa_bad.json                  # Bad Q&A (failed)
|   +-- qa_rescued.json              # Rescued Q&A
|   +-- qa_duplicates.json           # Removed duplicates
|   +-- chunks_to_regenerate.json    # Chunks needing regen
|
+-- split/
|   +-- train.json                   # Training set
|   +-- validation.json              # Validation set
|   +-- test.json                    # Test set
|
+-- final/
    +-- dataset_final.json           # Combined dataset
    +-- train.jsonl                  # JSONL format
    +-- validation.jsonl
    +-- test.jsonl
```

## Configuration (config.yaml)

```yaml
general:
  project_name: "bhxh_qa_dataset"
  input_dir: "../Luat"
  output_dir: "./output"

llm:
  provider: "gemini"
  providers:
    gemini:
      model: "gemini-2.0-flash"
      api_keys_file: "../gemini_keys.txt"

processing:
  threads_per_key: 5
  chunk_size: 4000
  cache:
    enabled: true
    cache_dir: "./cache"

qa_generation:
  qa_per_chunk: 5

quality:
  min_score: 4
  max_regenerate_iterations: 3

export:
  format: "chat"
  system_prompt: |
    Ban la chuyen gia tu van BHXH...
```

## Project Structure

```
dataset_pipeline_v2/
+-- run.py                    # CLI entry point
+-- pipeline.py               # Main pipeline class
+-- config.yaml               # Configuration
+-- requirements.txt          # Dependencies
+-- check_dataset_quality.py  # Quality checker script
|
+-- core/
|   +-- config.py             # Config loader
|   +-- logger.py             # Logging
|   +-- utils.py              # Utilities, CacheManager
|
+-- providers/                # LLM Providers
|   +-- base.py               # Base class
|   +-- gemini.py             # Google Gemini
|   +-- openai.py             # OpenAI GPT
|   +-- anthropic.py          # Anthropic Claude
|   +-- custom.py             # Custom API (vLLM, Ollama, etc.)
|   +-- factory.py            # Provider factory
|
+-- steps/                    # Pipeline steps
|   +-- extractor.py          # Extract text from documents
|   +-- generator.py          # Generate Q&A pairs
|   +-- evaluator.py          # Evaluate + Deduplicate
|   +-- rescuer.py            # Rescue bad Q&A
|   +-- regenerator.py        # Analyze chunks to regenerate
|   +-- splitter.py           # Document-based split
|
+-- output/                   # Output directory
+-- cache/                    # API response cache
+-- logs/
    +-- pipeline.log          # Pipeline logs
```

## FAQ

### Pipeline bi dung giua chung?
```bash
# Chay lai - tu dong skip cac buoc da xong
python run.py
```

### Muon chay lai tu dau?
```bash
python run.py --force
```

### Kiem tra tien do?
```bash
# Xem log (Windows)
Get-Content logs/pipeline.log -Tail 50

# Xem log (Linux/Mac)
tail -50 logs/pipeline.log

# Xem state
cat output/pipeline_state.json
```

### Tai sao khong co step Tokenize?
Moi model co tokenizer rieng. Tokenize nen de qua trinh fine-tuning xu ly, khong lam trong pipeline tao dataset.

### Rate limit API?
- Them nhieu API keys vao file (vd: `gemini_keys.txt`)
- Tang `threads_per_key` trong config
- Pipeline co cache de tranh goi lai

### Lam sao them provider moi?
Them vao config.yaml voi `base_url`:
```yaml
llm:
  provider: "my_provider"
  providers:
    my_provider:
      base_url: "https://api.example.com/v1"
      model: "model-name"
      api_key: "your-key"
```

## Changelog

### v2.0.0 (2024-12)
- **Document-based Split**: Tranh data leakage
- **ChatML Format**: Chuan cong nghiep cho fine-tuning
- **All-in-One Evaluate**: Gop Deduplicate + Evaluate + Rescue + Regenerate
- **Auto Mode**: Tu dong kiem tra va skip cac buoc da xong
- **Smart Caching**: Luu API response, ho tro resume
- **Multi-thread per Key**: Cau hinh `threads_per_key`
- **Custom Providers**: Ho tro bat ky API OpenAI-compatible

### v1.1.1
- Sua loi va them legal_knowledge

### v1.1.0
- Them Cache, Temporal Validation va System Prompt Builder

### v1.0.0
- Initial release

## License

MIT
