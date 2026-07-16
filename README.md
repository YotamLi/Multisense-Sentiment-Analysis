# MultiSense V2

**Multi-task sentiment analysis with Multi-Head BERT, ChromaDB RAG, LangChain, and Gemini reasoning.**

MultiSense predicts four attributes from English short-form text:

- sentiment polarity: positive, negative, neutral
- emotion: joy, anger, sadness, fear, surprise, disgust
- emotion intensity: strong, medium, weak
- topic: six fixed topic categories

The final V2 pipeline uses one shared BERT encoder with four task-specific heads, retrieves sentiment evidence from VADER, NRC, and SentiWordNet through ChromaDB, and optionally uses Gemini for sarcasm detection and explainable correction.

## Final evaluation

| Task | Macro F1 |
|---|---:|
| Sentiment | 0.680 |
| Emotion | 0.690 |
| Intensity | 0.573 |
| Topic | 0.687 |

For sentiment, the dedicated Single-Head BERT baseline achieved 0.691 Macro F1, while Multi-Head BERT achieved 0.680. The multi-task model's main advantage is joint four-task prediction through one shared encoder rather than guaranteed superiority on every individual task.

Open `evaluation_report.html` for the full report.

## Project structure

```text
app/                    Gradio interface
config/config.yaml      final V2 configuration
config/config_smoke.yaml
src/data/               preprocessing and PyTorch datasets
src/models/             Multi-Head BERT, trainer, checkpoint validation
src/rag/                ChromaDB knowledge base and retriever
src/pipeline/           LangChain orchestration and Gemini reasoning
src/evaluation/         metrics and baseline models
scripts/                data, training, evaluation, RAG, and validation workflows
tests/                  unit and integration tests
evaluation_figures/     report figures
evaluation_report.html  final evaluation report
```

## New-computer setup

Python 3.11–3.13 is recommended. A virtual environment must be recreated; do not copy `.venv` from another computer.

### Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an RTX 50-series GPU, install a current CUDA-enabled PyTorch build before or after installing the remaining requirements.

Create `.env` from `.env.example`:

```env
GOOGLE_API_KEY=your-key-here
```

Do not commit `.env`.

## Required runtime assets

The GitHub/code package intentionally does not include secrets or large generated assets.

Place the final trained model at:

```text
checkpoints/best_model.pt
```

For a fully offline local copy, privately transfer these generated directories from the trained computer:

```text
data/chroma_db/
data/processed_v2/        # required for retraining/evaluation, not basic inference
```

Alternatively, rebuild them:

```powershell
python scripts\prepare_multisource_data.py --config config\config.yaml
python scripts\build_rag.py --config config\config.yaml
```

## Validate the project

```powershell
python scripts\preflight.py --config config\config.yaml --checkpoint checkpoints\best_model.pt
python scripts\validate_dataset.py --data-dir data\processed_v2
pytest tests -v
```

## Run the final demo

```powershell
python app\demo.py --config config\config.yaml --checkpoint checkpoints\best_model.pt
```

Open `http://127.0.0.1:7860`.

## Reproduce training

Smoke test:

```powershell
python scripts\create_smoke_dataset.py
python scripts\train.py --config config\config_smoke.yaml --output-dir checkpoints\smoke
```

Full training:

```powershell
python scripts\train.py --config config\config.yaml --output-dir checkpoints
```

Evaluation:

```powershell
python scripts\evaluate.py --config config\config.yaml --checkpoint checkpoints\best_model.pt
```

## Important limitations

- RAG and Gemini improve explainability and pragmatic handling, especially sarcasm, but the main test report does not claim a measured full-test-set F1 improvement from the LLM layer.
- Intensity classification is the weakest task and remains a target for future improvement.
- Gemini requires an API key and network access.
- Model weights should be distributed through a GitHub Release, Git LFS, or a Hugging Face model repository rather than normal Git.
