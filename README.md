# Arabic Documents Summarization, NER & Topic Modeling

A comprehensive Arabic NLP pipeline that combines multiple state-of-the-art models for text summarization, Named Entity Recognition, sentiment analysis, and topic modeling with detailed benchmarking and evaluation metrics (including LLM-based approaches).

## Overview

This project implements a production-ready Arabic Natural Language Processing pipeline that handles the complete NLP workflow:

1. **Text Preprocessing**: Arabic normalization, tokenization, and lemmatization
2. **Named Entity Recognition**: Multi-model comparison (CAMeL, Hatmimoha, Stanza, LLM-Only)
3. **Summarization**: Both extractive (Sumy) and abstractive (AraBART, mT5-XLSum, LLM-Only)
4. **Sentiment Analysis**: Document-level sentiment classification (CAMeL, LLM-Only)
5. **Topic Modeling**: Automatic topic extraction with coherence scoring (LDA, LLM-Only)
6. **Performance Benchmarking**: Runtime and accuracy metrics for all models

All components include:

- ✅ Automatic evaluation metrics
- ✅ Multi-model comparison for fair benchmarking
- ✅ Real-world Arabic datasets with annotations
- ✅ Runtime and accuracy tracking for performance analysis
- ✅ LLM-based approaches using gemma3:4b on Ollama
- ✅ Graceful error handling and model fallbacks

---

## Features

### Text Preprocessing

- **Unicode Normalization**: Standardizes Arabic characters
- **Diacritical Removal**: Removes Tasdeed, Fatha, Damma, etc.
- **Character Normalization**:
  - Alef variants → ا
  - Maksura → ي
  - Teh Marbuta → ة
- **Lemmatization**: CAMeL morphological database analysis
- **Stopword Removal**: 100+ Arabic stopwords filtered

### Named Entity Recognition

Five models with standardized output:

| Model                     | Backend      | Entities                       |
| ------------------------- | ------------ | ------------------------------ |
| **CAMeL Tools**         | AraBERT      | PERS, LOC, ORG, MISC           |
| **Hatmimoha**           | BERT         | PERSON, LOCATION, ORGANIZATION |
| **Stanford Stanza**     | Multilingual | PER, LOC, ORG                  |
| **LangExtract**         | Google's Multilingual Model | PERS, LOC, ORG, MISC |
| **LLM-Only Benchmark**  | gemma3:4b on Ollama | PERS, LOC, ORG, MISC |

**Output Format**: Unified dictionary with text and label

### Text Summarization

**Extractive Methods** (Sumy):

- LexRank: Graph-based ranking
- LSA: Latent Semantic Analysis
- TextRank: PageRank adaptation

**Abstractive Methods** (Neural):

- **AraBART**: Arabic-specific BART model
- **mT5-XLSum**: Multilingual mT5 fine-tuned on XLSum
- **LangExtract**: Google's Multilingual Model
- **LLM-Only Benchmark**: gemma3:4b on Ollama

### Sentiment Analysis

- **Models**: CAMeL Tools Sentiment Analyzer, LLM-Only (gemma3:4b on Ollama)
- **Labels**: Positive, Negative, Neutral
- **Evaluation**: Accuracy on reference labels

### Topic Modeling

- **Algorithms**: Latent Dirichlet Allocation (LDA), LLM-Only (gemma3:4b on Ollama)
- **Topics**: 3 topics (configurable)
- **Metrics**: Coherence score (C_V measure)
- **Output**: Top 5 words per topic

### Performance Tracking

- **Runtime Measurement**: Execution time for each model and task
- **Accuracy Metrics**: ROUGE-1 for summarization, F1 for NER, accuracy for sentiment
- **Comprehensive Reporting**: Combined accuracy and runtime benchmarks

---

## Architecture

### Class Hierarchy

```
UltimatePipeline (Main Orchestrator)
├── ArabicPreprocessor (Text Normalization)
├── ArabicNER (Named Entity Recognition)
│   ├── CAMeL NER
│   ├── Hatmimoha NER
│   └── Stanza NER
├── ArabicSummarizer (Text Summarization)
│   ├── Extractive Models (Sumy)
│   ├── Abstractive Models (Neural)
│   └── Preprocessing Pipeline
├── TopicModeler (Topic Extraction)
│   └── LDA with Gensim
├── SentimentAnalyzer (CAMeL Tools)
├── LangExtractWrapper (Google's Multilingual Model)
├── LLMOnlyBenchmark (gemma3:4b on Ollama)
└── EvaluationMetrics (All Metrics)
    ├── ROUGE-1 Scorer
    ├── NER Metrics
    └── Text Normalization
├── Timing Tracker (Runtime Measurement)
└── Combined Results Formatter (Accuracy & Runtime)
```

### Data Flow

```
Raw Arabic Text
    ↓
[Preprocessing & Normalization]
    ↓
[Parallel Processing]
├─→ NER Extraction (5 models: CAMeL, Hatmimoha, Stanza, LangExtract, LLM-Only)
├─→ Summarization (6 methods: Sumy, AraBART, mT5-XLSum, LangExtract, LLM-Only)
├─→ Sentiment Analysis (2 models: CAMeL, LLM-Only)
├─→ Topic Modeling (2 algorithms: LDA, LLM-Only)
├─→ Runtime Measurement (for each model/task)
└─→ Accuracy Calculation
    ↓
[Evaluation & Metrics]
├─→ ROUGE Scores (Summarization)
├─→ NER F1, Precision, Recall
├─→ Sentiment Accuracy
├─→ Topic Coherence Score
├─→ Runtime Measurements
└─→ Combined Benchmark Report
    ↓
Benchmark Results (Accuracy & Runtime)
```

### Custom Dataset

```python
from Ar-SUM_NER import UltimatePipeline

# Prepare your dataset
my_data = [
    {
        'text': 'أعلنت الشركة...',  # Your Arabic text
        'reference_summary': 'ملخص مرجعي...',
        'entities': [{'text': 'الشركة', 'label': 'ORG'}, ...],
        'sentiment': 'positive'
    },
    ...
]

# Run pipeline
pipeline = UltimatePipeline()
pipeline.run(my_data)
```

### Individual Components

```python
from Ar-SUM_NER import ArabicPreprocessor, ArabicNER, ArabicSummarizer

# Initialize components
preprocessor = ArabicPreprocessor()
ner = ArabicNER()
summarizer = ArabicSummarizer(preprocessor)

# Use individually
text = "أعلنت شركة أرامكو..."
tokens = preprocessor.preprocess(text)
entities = ner.extract_all(text)
summaries = summarizer.summarize(text)
```

---

## Sample Output

### Pipeline Execution Output

```
======================================================================
ARABIC NLP PIPELINE: BENCHMARK EDITION
======================================================================
  Loading CAMeL Morphology...
  Loading NER Models...
  Loading Summarization Models...
  Loading Ollama with gemma3:4b (LLM-based Multilingual Model)...
  Topic Modeling: Gensim

======================================================================
📄 DETAILED ANALYSIS (LARGE DOCS)
======================================================================

Document 1 (317 words)
📝 Summarization:
   [AraBART]: 0.85s - وقعت شركة أرامكو اتفاقيات استراتيجية مع توتال وشل...
   [LLM-Only]: 2.45s - gemma3:4b summary output...

🏷️ NER:
   Entities found: أرامكو السعودية, أمين الناصر, الظهران, توتال إنرجيز, شل...
   [LLM-Only]: 1.98s - Entities extracted by LLM

😊 Sentiment:
   True: mixed | Pred: positive | Runtime: 0.12s
   [LLM-Only]: True: mixed | Pred: positive | Runtime: 0.87s

Document 2 (253 words)
📝 Summarization:
   [AraBART]: 0.78s - اختتمت القمة العربية بدعوات للتضامن في مواجهة التحديات...
   [LLM-Only]: 2.31s - gemma3:4b summary output...

🏷️ NER:
   Entities found: عمان, عبدالله الثاني, محمد بن سلمان, السيسي...
   [LLM-Only]: 1.85s - Entities extracted by LLM

😊 Sentiment:
   True: neutral | Pred: neutral | Runtime: 0.11s
   [LLM-Only]: True: neutral | Pred: neutral | Runtime: 0.82s

Document 3 (292 words)
📝 Summarization:
   [AraBART]: 0.82s - أطلقت جامعة كاوست مبادرة للذكاء الاصطناعي مع جوجل...
   [LLM-Only]: 2.38s - gemma3:4b summary output...

🏷️ NER:
   Entities found: كاوست, جوجل, مايكروسوفت, مستشفى الملك فيصل...
   [LLM-Only]: 1.92s - Entities extracted by LLM

😊 Sentiment:
   True: positive | Pred: positive | Runtime: 0.13s
   [LLM-Only]: True: positive | Pred: positive | Runtime: 0.85s

======================================================================
FINAL BENCHMARK SCORES (Accuracy & Runtime)
======================================================================

📝 SUMMARIZATION (ROUGE-1 & Runtime)
  mT5-XLSum        : Acc=0.5234, Time=0.95s
  AraBART          : Acc=0.4892, Time=0.82s
  LLM-Only         : Acc=0.5421, Time=2.38s
  Sumy-TextRank    : Acc=0.4156, Time=0.45s
  Sumy-LexRank     : Acc=0.4023, Time=0.42s
  Sumy-LSA         : Acc=0.3845, Time=0.38s

🏷️ NER (F1 Score & Runtime)
  CAMeL            : Acc=0.8234, Time=0.65s
  Hatmimoha        : Acc=0.7856, Time=0.72s
  LLM-Only         : Acc=0.8012, Time=1.89s
  Stanza           : Acc=0.7123, Time=0.89s

😊 SENTIMENT (Accuracy & Runtime): Acc=0.89, Time=0.12s

📊 TOPIC MODELING (Coherence & Runtime): Coherence=0.6234, Time=1.45s
```

---

## Performance Metrics

### Evaluation Metrics Used

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

```
ROUGE-1 = 2 × (Precision × Recall) / (Precision + Recall)

Where:
  Precision = Overlap / Hypothesis Length
  Recall = Overlap / Reference Length
  Overlap = Matching n-grams
```

**Interpretation**:

- 0.0-0.3: Poor
- 0.3-0.5: Fair
- 0.5-0.7: Good
- 0.7+: Excellent

#### NER Metrics

```
Precision = Correct Entities / Total Predicted
Recall = Correct Entities / Total Reference
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Partial Matching**: Entities with overlapping text are counted as partial matches.

#### Coherence Score (Topic Modeling)

```
Coherence (C_V) ∈ [0, 1]

- 0.0-0.3: Topics not well-separated
- 0.3-0.6: Moderate coherence
- 0.6+: Highly coherent topics
```

#### Sentiment Accuracy

```
Accuracy = Correct Predictions / Total Predictions
```

## Models & Components

### Model Sizes & Download Times

| Component     | Model          | Size   | Download Time |
| ------------- | -------------- | ------ | ------------- |
| NER           | AraBERT        | 541 MB | ~5 min        |
| Sentiment     | ARABERT        | 541 MB | ~5 min        |
| Morphology    | CALIMA-MSA-r13 | 40 MB  | <1 min        |
| Summarization | mT5-XLSum      | 2.8 GB | ~15 min       |
| Summarization | AraBART        | 1.8 GB | ~10 min       |
| LLM-Only      | gemma3:4b      | 3.3 GB | ~15 min       |

### Model Cards

#### CAMeL Tools NER (AraBERT)

- **Type**: Token Classification (BERT)
- **Training**: Arabic Wikipedia + News
- **Entities**: PERSON, LOCATION, ORGANIZATION, MISCELLANEOUS
- **Input**: Raw or preprocessed Arabic text
- **Output**: Token-level BIO tags

#### Hatmimoha NER

- **Type**: Token Classification (BERT)
- **Base Model**: BERT-base-arabic
- **Training**: Arabic Wikipedia
- **Entities**: PERSON, LOCATION, ORGANIZATION
- **Aggregation Strategy**: Simple (takes first token)

#### Stanford Stanza

- **Type**: Multilingual NLP Pipeline
- **Processors**: Tokenization, NER
- **Language**: Arabic (ar)
- **Architecture**: BiLSTM + Attention

#### AraBART

- **Type**: Sequence-to-Sequence (Transformer)
- **Base**: mBART (multilingual BART)
- **Fine-tuning**: Arabic Summarization datasets
- **Input**: Raw Arabic text
- **Output**: Summary text

#### mT5-XLSum

- **Type**: Sequence-to-Sequence (T5)
- **Multilingual**: 101 languages
- **Fine-tuning**: XLSum (cross-lingual)
- **Max Input**: 512 tokens
- **Max Output**: 150 tokens

#### LLM-Only Benchmark (gemma3:4b on Ollama)

- **Type**: Large Language Model (Decoder-only Transformer)
- **Backend**: Ollama inference engine
- **Capabilities**: Summarization, NER, Sentiment Analysis, Topic Modeling
- **Input**: Raw Arabic text with task-specific prompts
- **Output**: Structured responses in requested format
- **Advantages**: Multitask capability, contextual understanding
- **Considerations**: Higher computational requirements, potential latency
