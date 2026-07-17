# MultiSense V2

MultiSense V2 is a multi-task NLP system that analyzes English text across four dimensions:

- **Sentiment:** positive, negative, neutral
- **Emotion:** joy, anger, sadness, fear, surprise, disgust
- **Intensity:** strong, medium, weak
- **Topic:** six short-form text categories

The system combines **Multi-Head BERT**, **ChromaDB RAG**, **LangChain**, and **Gemini** to provide predictions, sarcasm detection, retrieved sentiment evidence, and natural-language explanations.

---

## Architecture

```text
Input Text
    ↓
Text Preprocessing
    ↓
Shared BERT Encoder
    ├── Sentiment Head
    ├── Emotion Head
    ├── Intensity Head
    └── Topic Head
    ↓
ChromaDB RAG
(VADER + NRC + SentiWordNet)
    ↓
Gemini Reasoning
(sarcasm detection and correction)
    ↓
Validated Final Result
    ↓
Gradio Interface
```

---

## Main Features

- One shared BERT encoder with four classification heads
- Missing-label training with `-1` ignored per task
- Leakage-safe train, validation, and test splits
- Duplicate-text merging across multiple datasets
- ChromaDB knowledge base with approximately 26,660 lexicon documents
- Gemini-based sarcasm and irony detection
- Initial BERT vs final prediction comparison
- VADER, Naive Bayes, and Single-Head BERT baselines
- Gradio web interface and command-line support
- Automatic fallback to BERT results when Gemini is unavailable

---

## Evaluation Results

Final Multi-Head BERT results on the held-out test set:

| Task | Macro F1 |
|---|---:|
| Sentiment | 0.680 |
| Emotion | 0.690 |
| Intensity | 0.573 |
| Topic | 0.687 |

Sentiment model comparison:

| Model | Macro F1 |
|---|---:|
| Single-Head BERT | 0.691 |
| Multi-Head BERT | 0.680 |
| VADER | 0.535 |
| Naive Bayes | 0.521 |

The Single-Head BERT performs slightly better on sentiment alone, while Multi-Head BERT predicts all four tasks through one shared encoder.

The complete report is available in:

```text
evaluation_report.html
```

---

## Project Structure

```text
multisense-v2-final/
├── app/                    # Gradio web interface
├── checkpoints/            # Final checkpoint location
├── config/                 # Model and training configuration
├── data/                   # Lexicons and generated datasets
├── docs/                   # System design documentation
├── evaluation_figures/     # Evaluation charts
├── scripts/                # Data, training, evaluation, and RAG scripts
├── src/
│   ├── data/               # Dataset and preprocessing
│   ├── evaluation/         # Metrics and baselines
│   ├── models/             # Multi-Head BERT and trainer
│   ├── pipeline/           # BERT, RAG, Gemini orchestration
│   ├── rag/                # ChromaDB knowledge base
│   └── utils/              # Config, logging, seed, and device helpers
├── tests/                  # Unit and integration tests
├── cli.py
├── evaluation_report.html
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.11–3.13 is recommended.

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd multisense-v2-final
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install an appropriate PyTorch build for your system, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Model Checkpoint

Download `best_model.pt` from the repository's **Releases** page and place it at:

```text
checkpoints/best_model.pt
```

---

## Gemini API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-api-key
```

The real `.env` file is ignored by Git and must not be uploaded.

Gemini can be disabled with:

```text
--disable-llm
```

---

## Build the RAG Knowledge Base

```bash
python scripts/build_rag.py --config config/config.yaml
```

This creates:

```text
data/chroma_db/
```

The knowledge base contains VADER, NRC Emotion Lexicon, and SentiWordNet entries.

---

## Run the Demo

### Full system

```bash
python app/demo.py --config config/config.yaml --checkpoint checkpoints/best_model.pt
```

Open:

```text
http://127.0.0.1:7860
```

### BERT only

```bash
python app/demo.py --config config/config.yaml --checkpoint checkpoints/best_model.pt --disable-rag --disable-llm
```

### VADER fallback

```bash
python app/demo.py --config config/config.yaml --vader --disable-rag --disable-llm
```

---

## Command-Line Usage

Analyze one text:

```bash
python cli.py analyze --text "The movie was absolutely brilliant!"
```

Batch analysis:

```bash
python cli.py batch --input reviews.csv --output results.json
```

---

## Reproduce the Dataset

```bash
python scripts/prepare_multisource_data.py --config config/config.yaml
python scripts/validate_dataset.py --data-dir data/processed_v2
```

The project uses:

- TweetEval Sentiment
- GoEmotions
- SemEval 2025 Emotion Intensity
- Tweet Topic Single

The V2 preparation pipeline removes duplicate text leakage across train, validation, and test splits.

---

## Training

Smoke test:

```bash
python scripts/create_smoke_dataset.py
python scripts/train.py --config config/config_smoke.yaml --output-dir checkpoints/smoke
```

Full training:

```bash
python scripts/train.py --config config/config.yaml --output-dir checkpoints
```

The best model is saved as:

```text
checkpoints/best_model.pt
```

---

## Evaluation

```bash
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pt
```

This evaluates:

- VADER
- TF-IDF + Naive Bayes
- Single-Head BERT
- Multi-Head BERT

---

## Testing

```bash
pytest tests -v
```

Optional project checks:

```bash
python -m compileall app src scripts
python scripts/preflight.py --config config/config.yaml --checkpoint checkpoints/best_model.pt
```

---

## Notes

- Gemini may occasionally return temporary `503 UNAVAILABLE` errors. The application falls back to the original BERT predictions instead of crashing.
- Displayed confidence values are softmax scores and are not probability-calibrated.
- The intensity task is currently the weakest task and remains a target for improvement.
- Model weights, `.env`, generated datasets, ChromaDB files, and `.venv` are intentionally excluded from normal Git commits.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- LangChain
- ChromaDB
- Google Gemini
- Gradio
- Sentence Transformers
- scikit-learn
- VADER
- NRC Emotion Lexicon
- SentiWordNet
