# MultiSense V2 技术要点总结

这份文档用于快速理解 MultiSense V2 的核心技术、系统结构和设计逻辑。它适合作为 GitHub 仓库中的技术说明，也可以用于项目展示、简历准备和面试复习。

---

## 1. 项目目标

MultiSense V2 是一个多任务英文文本分析系统。输入一段文本后，系统会同时预测：

- **Sentiment**：positive / negative / neutral
- **Emotion**：joy / anger / sadness / fear / surprise / disgust
- **Intensity**：strong / medium / weak
- **Topic**：6 个主题类别

系统不仅给出分类结果，还会：

- 检索情感词典证据
- 检测 sarcasm / irony
- 在必要时修正 BERT 结果
- 给出自然语言解释
- 展示初始预测和最终预测的差异

---

## 2. 整体架构

```text
Input Text
    ↓
Text Preprocessing
    ↓
Multi-Head BERT
    ├── Sentiment Head
    ├── Emotion Head
    ├── Intensity Head
    └── Topic Head
    ↓
ChromaDB RAG Retrieval
    ↓
Gemini Pragmatic Reasoning
    ↓
Output Validation
    ↓
Final Aggregation
    ↓
Gradio UI / CLI
```

可以把系统理解为三层：

```text
第一层：BERT 基础分类
第二层：RAG 外部知识增强
第三层：Gemini 语境推理与解释
```

---

## 3. Multi-Task Learning

项目使用一个共享的 BERT encoder，同时连接四个分类头：

```text
Shared BERT Encoder
    ├── Sentiment Head
    ├── Emotion Head
    ├── Intensity Head
    └── Topic Head
```

### 为什么使用多任务学习

相比训练四个独立 BERT 模型，多任务结构具有以下优势：

- 四个任务共享语言表示
- 只需要运行一次 BERT encoder
- 模型数量更少
- 推理速度更高
- 可以同时返回四种结果

### 需要注意

多任务学习不保证每一个任务都超过单任务模型。

在最终结果中：

```text
Single-Head BERT Sentiment Macro F1: 0.691
Multi-Head BERT Sentiment Macro F1: 0.680
```

因此正确结论是：

> Multi-Head BERT 在 Sentiment 上保持了接近单任务 BERT 的表现，同时通过一个共享 encoder 完成四个任务。

---

## 4. BERT

项目使用：

```text
bert-base-uncased
```

基本过程：

```text
Text
→ Tokenizer
→ input_ids + attention_mask
→ BERT Encoder
→ Shared Representation
→ Task Heads
→ Logits
→ Softmax
→ Labels and Confidence
```

### 主要概念

- **Tokenizer**：把文本转换为 token IDs
- **Encoder**：提取上下文语义表示
- **Classification Head**：完成具体分类任务
- **Logits**：模型原始输出分数
- **Softmax**：把 logits 转成概率分布

---

## 5. Pipeline

Pipeline 表示系统完整的执行流程，而不是单独某个模型。

正式 Pipeline：

```text
Preprocess
→ Multi-Head BERT
→ RAG Retrieval
→ Gemini Reasoning
→ Output Validation
→ Aggregation
```

项目中主要有两条 Pipeline：

### SentimentPipeline

正式模式：

```text
BERT → RAG → Gemini
```

### HybridPipeline

没有 BERT checkpoint 时的备用模式：

```text
VADER → RAG → Gemini
```

---

## 6. LangChain

LangChain 在本项目中主要负责流程编排，不负责训练 BERT。

它将多个步骤组织成 Chain：

```text
Preprocess Chain
BERT Inference Chain
RAG Retrieval Chain
LLM Reasoning Chain
Aggregation Chain
```

可以简单理解为：

> LangChain 把前一个步骤的输出传给下一个步骤，并将多个模块连接成一个完整工作流。

主要代码位置：

```text
src/pipeline/chains.py
src/pipeline/orchestrator.py
```

---

## 7. RAG

RAG 全称：

```text
Retrieval-Augmented Generation
```

中文是：

```text
检索增强生成
```

核心思想：

> 在调用 LLM 之前，先从外部知识库检索相关信息，再把这些证据交给 LLM。

本项目中的流程：

```text
Input Text
→ Retrieve Sentiment Evidence
→ Send Text + BERT Prediction + Evidence to Gemini
→ Generate Final Reasoning
```

RAG 的主要作用：

- 提供外部情感知识
- 增强解释能力
- 帮助判断 BERT 结果是否合理
- 为 Gemini 提供额外证据

---

## 8. Embedding 和 ChromaDB

### Embedding

Embedding 将文本转换为向量：

```text
"happy"
→ [0.13, -0.42, 0.88, ...]
```

语义相近的文本，其向量距离通常也更近。

项目使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

### ChromaDB

ChromaDB 是向量数据库，负责保存和检索：

- 文本
- Embedding
- Metadata
- 情感分数
- 词典来源

检索过程：

```text
User Text
→ Embedding
→ ChromaDB Similarity Search
→ Top-K Evidence
```

默认：

```text
top_k = 5
```

---

## 9. RAG 知识来源

知识库由以下资源构建：

### VADER

提供规则型情感分数。

例如：

```text
excellent → positive
terrible  → negative
```

### NRC Emotion Lexicon

提供单词与情绪的对应关系。

例如：

```text
happy   → joy
furious → anger
afraid  → fear
```

### SentiWordNet

提供 WordNet 词义层面的：

- positive score
- negative score
- objective score

最终知识库存储约：

```text
26,660 documents
```

---

## 10. Gemini

Gemini 不是项目的主分类模型，而是语境推理层。

Gemini 接收：

- 原始文本
- BERT 四任务预测
- BERT 概率
- RAG 检索结果
- 合法标签列表

主要负责：

- sarcasm detection
- irony detection
- pragmatic reasoning
- prediction correction
- natural-language explanation

示例：

```text
Oh, brilliant. Another flat tire.
```

BERT 可能预测：

```text
positive / joy
```

Gemini 根据语境修正为：

```text
negative / anger
sarcasm = true
```

---

## 11. Output Validation

LLM 输出不能直接使用，必须经过校验。

系统会检查：

```text
Sentiment 是否属于 3 个合法标签
Emotion 是否属于 6 个合法标签
Intensity 是否属于 3 个合法标签
Topic 是否属于 6 个训练类别
Confidence 是否在 0 到 1 之间
JSON 是否完整
```

如果 Gemini 输出非法内容，例如：

```text
frustration
very_strong
transportation
```

系统会拒绝这些标签，并保留或恢复 BERT 的合法结果。

---

## 12. Fallback 容错机制

系统支持多个 fallback。

### Gemini 不可用

例如出现：

```text
503 UNAVAILABLE
```

系统会：

```text
保留 BERT 初始预测
显示 LLM unavailable
网页继续运行
```

### RAG 不可用

可以运行：

```text
--disable-rag
```

### Gemini 被禁用

可以运行：

```text
--disable-llm
```

### 没有 BERT checkpoint

可以运行：

```text
--vader
```

容错设计保证：

> 某一个外部服务失败时，整个系统不会崩溃。

---

## 13. 多来源数据与缺失标签

项目的数据来自四个不同来源：

```text
TweetEval          → Sentiment
GoEmotions         → Emotion
SemEval            → Intensity
Tweet Topic Single → Topic
```

因此一条样本可能只有一个任务标签：

```text
sentiment = 0
emotion = -1
intensity = -1
topic = -1
```

其中：

```text
-1 = missing label
```

训练时使用：

```text
ignore_index = -1
```

这表示该样本只更新有标签的任务头。

---

## 14. 数据安全与防泄漏

V2 数据处理流程增加了：

- 文本标准化
- URL 和用户名占位符
- SHA-256 text hash
- 重复文本合并
- provenance 来源记录
- 标签冲突处理
- official test split priority
- train / val / test overlap validation

最终审计结果：

```text
train vs val:  0 overlap
train vs test: 0 overlap
val vs test:   0 overlap
```

这确保测试文本不会进入训练过程。

---

## 15. 训练技术

训练部分涉及：

### CrossEntropyLoss

四个任务均使用分类损失。

### Weighted Multi-Task Loss

```text
Total Loss =
Sentiment Loss × weight
+ Emotion Loss × weight
+ Intensity Loss × weight
+ Topic Loss × weight
```

### Different Learning Rates

```text
BERT Encoder → smaller learning rate
Task Heads   → larger learning rate
```

### AMP

Automatic Mixed Precision：

```text
FP16 + FP32
```

作用：

- 减少显存
- 加速 GPU 训练

### Gradient Accumulation

多个 mini-batch 累积后再更新参数，用来模拟更大的 batch size。

### Scheduler

学习率先 warmup，再逐渐下降。

### Gradient Clipping

防止梯度爆炸。

### Early Stopping

系统根据四个任务的综合 Macro F1 保存最佳 checkpoint。

---

## 16. 评估指标

项目使用：

- Accuracy
- Macro F1
- Weighted F1
- Cohen's Kappa
- Confusion Matrix
- Per-class F1

### Macro F1

每个类别权重相同，适合类别不平衡场景。

### Weighted F1

样本数量多的类别权重更高。

### Confusion Matrix

用于观察：

> 哪些类别最容易被模型混淆。

---

## 17. Baselines

项目比较了四种模型：

```text
VADER
Naive Bayes
Single-Head BERT
Multi-Head BERT
```

Baseline 的作用是：

> 判断复杂模型相较于简单方法是否真正有价值。

最终 Sentiment Macro F1：

| Model | Macro F1 |
|---|---:|
| Single-Head BERT | 0.691 |
| Multi-Head BERT | 0.680 |
| VADER | 0.535 |
| Naive Bayes | 0.521 |

---

## 18. Gradio

Gradio 负责网页界面，包括：

- 输入框
- Analyze 按钮
- 四任务结果卡片
- Initial vs Final 对比
- Gemini Explanation
- RAG Evidence
- Emotion Radar Chart

Gradio 只负责交互，不负责模型训练。

---

## 19. CLI

项目还提供命令行接口：

```text
cli.py
```

支持：

- 单条文本分析
- CSV 批量分析
- 启动 Demo
- 构建 RAG

CLI 使用：

```text
Click
```

作为命令行框架。

---

## 20. Checkpoint

最终模型文件：

```text
checkpoints/best_model.pt
```

checkpoint 包含：

- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- validation metrics
- monitor score
- model config
- training config
- label maps
- PyTorch version

需要注意：

> checkpoint 和 config 中的类别数量必须一致。

---

## 21. 技术栈关系

```text
PyTorch
└── 训练 Multi-Head BERT

Hugging Face Transformers
└── BERT tokenizer 和 encoder

Hugging Face Datasets
└── 下载和处理数据集

LangChain
└── 编排 Pipeline

Sentence Transformers
└── 生成 Embedding

ChromaDB
└── 保存和检索向量

RAG
└── 提供外部情感知识

Gemini
└── 讽刺判断、纠错和解释

Gradio
└── 网页界面

Click
└── CLI

scikit-learn
└── Naive Bayes、TF-IDF 和指标

NLTK
└── VADER 和 SentiWordNet
```

---

## 22. 面试或展示时最重要的六句话

1. **This is a multi-task NLP system with one shared BERT encoder and four task-specific classification heads.**

2. **The four tasks are sentiment, emotion, emotion intensity, and topic classification.**

3. **LangChain orchestrates preprocessing, BERT inference, RAG retrieval, Gemini reasoning, and result aggregation.**

4. **ChromaDB stores sentiment knowledge from VADER, NRC, and SentiWordNet as vector embeddings.**

5. **Gemini is used as a pragmatic reasoning layer for sarcasm detection, correction, and explanation, rather than as the primary classifier.**

6. **The system falls back to the original BERT predictions when Gemini or RAG is unavailable.**

---

## 23. 核心关键词

```text
Multi-Task Learning
BERT
Classification Heads
Missing Labels
Pipeline
LangChain
RAG
Embeddings
ChromaDB
Gemini
Sarcasm Detection
Output Validation
Fallback
Gradio
CLI
PyTorch Training
Baseline Evaluation
Data Leakage Prevention
```

---

## 24. 一句话总结

> MultiSense V2 is a robust multi-task NLP pipeline that combines a shared BERT encoder, lexicon-based RAG retrieval, and Gemini pragmatic reasoning to jointly analyze sentiment, emotion, intensity, and topic while supporting explainability and graceful fallback.
