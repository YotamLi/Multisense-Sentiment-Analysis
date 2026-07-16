# MultiSense V2 Release Notes

This package contains the final V2 implementation only.

Key improvements:

- leakage-safe train, validation, and test splits
- duplicate-text merging with provenance tracking
- correct GoEmotions multi-label handling
- six-class topic head
- CUDA-safe mixed precision
- correct gradient accumulation and scheduler updates
- multi-task early stopping
- self-describing V2 checkpoints
- fair Naive Bayes evaluation
- validated Gemini output
- structured RAG evidence
- truthful runtime status in the Gradio interface
- BERT initial versus final LLM-assisted prediction comparison

Legacy five-topic compatibility files and migration utilities are intentionally excluded.
