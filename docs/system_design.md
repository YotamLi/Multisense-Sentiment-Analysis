# MultiSense V2 System Design

## Runtime path

```text
Input text
  -> preprocessing
  -> Multi-Head BERT or VADER initial prediction
  -> optional ChromaDB lexicon retrieval
  -> optional Gemini pragmatic reasoning
  -> output validation and aggregation
  -> Gradio / CLI
```

## Multi-task model

One `bert-base-uncased` encoder is shared by four heads:

- sentiment: 3 classes;
- emotion: 6 classes;
- intensity: 3 classes;
- topic: 6 classes in V2.

A label value of `-1` means that the sample has no supervision for that task.
The total loss is a weighted sum over tasks that have valid labels in the batch.

## V2 data policy

- Every row keeps `source`, `source_split`, `provenance`, `normalised_text` and `text_hash`.
- Duplicate normalized texts are merged.
- Compatible labels from different task sources are combined.
- Conflicting labels for the same task are marked missing.
- Official test membership has highest priority, then validation, then train.
- The topic dataset has no official validation split, so a deterministic hash-based subset of its official training data is reserved for validation.
- Cross-split text-hash intersections must be zero.

## Training policy

- CUDA FP16 is enabled only on CUDA.
- Optimizer updates include the final partial accumulation group.
- Scheduler steps use ceiling division.
- Early stopping monitors a weighted average of four task macro-F1 values.
- Checkpoints save model state, class configuration, label order, data fingerprints, metrics and training parameters.

## Evaluation policy

The held-out test set is used once for final reporting.
Naive Bayes is tuned on validation data and then evaluated on test.
Single-head BERT is selected using validation data.
Multi-Head BERT reports all four tasks.
