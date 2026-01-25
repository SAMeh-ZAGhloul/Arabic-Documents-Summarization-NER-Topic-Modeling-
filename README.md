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
🚀 ARABIC NLP PIPELINE: BENCHMARK EDITION
======================================================================
  📚 Loading CAMeL Morphology...
  🏷️ Loading NER Models...
Some weights of the model checkpoint at /Users/user/.camel_tools/data/ner/arabert were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
- This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  📝 Loading Summarization Models...
You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
  📊 Topic Modeling: Gensim
  🌍 Loading Ollama with gemma3:4b (LLM-based Multilingual Model)...
  🤖 Loading LLM-Only Benchmark (gemma3:4b on Ollama)...

======================================================================
📄 DETAILED ANALYSIS (LARGE DOCS)
======================================================================

📂 Document 1 (148 words)
📝 Summarization:
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
   [LangExtract]: 25.41s - Here's a concise summary of the Arabic text, in Arabic:

**أعلنت أرامكو السعودية عن أرباح قياسية بلغ...
   [LLM-Only]: 18.67s - Here’s a concise summary of the text in Arabic:

**أعلنت أرامكو عن أرباح قياسية بلغت 115 مليار ريال،...
   [AraBART]: 36.47s - اعلنت شركه ارامكو السعوديه، عملاق النفط العالمي واكبر شركه طاقه في العالم من حيث القيمه السوقيه، الي...
🏷️ NER:
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
   Entities found: ارامكو السعودية, ريال, امين حسن الناصر, الظهران, الناصر, توتال انرجيز...
😊 Sentiment:
   True: mixed | Pred: positive | Runtime: 5.79s
   [LLM-Only]: True: mixed | Pred: positive | Runtime: 4.22s

📂 Document 2 (87 words)
📝 Summarization:
   [LangExtract]: 24.40s - Here's a concise summary of the Arabic text in Arabic:

**تلقت القمة العربية الطارئة في عمان زخمًا ك...
   [LLM-Only]: 34.89s - Here's a concise summary of the text in Arabic:

**تلقت القمة العربية الطارئة في عمان إكمال أعمالها ...
   [AraBART]: 27.12s - اختتمت القمه العربيه الطارئه اعمالها في العاصمه الاردنيه عمان، وسط حضور رفيع المستوي من قاده الدول ا...
🏷️ NER:
   Entities found: عمان, عبدالله الثاني, محمد بن سلمان, عبدالفتاح السيسي, سوريا, واليمن...
😊 Sentiment:
   True: neutral | Pred: neutral | Runtime: 4.45s
   [LLM-Only]: True: neutral | Pred: positive | Runtime: 3.45s

📂 Document 3 (86 words)
📝 Summarization:
   [LangExtract]: 40.58s - Here’s a concise summary of the text in Arabic:

**تُشهد السعودية طفرة في التقنية والابتكار، بفضل مب...
   [LLM-Only]: 18.69s - Here’s a concise summary of the text in Arabic:

**تشهد السعودية طفرة تقنية كبيرة، مدفوعة بمبادرة ال...
   [AraBART]: 41.53s - تشهد المملكه العربيه السعوديه طفره تقنيه هائله، حيث اعلنت جامعه الملكعبدالله للعلوم والتقنيه (كاوست)...
🏷️ NER:
   Entities found: المملكة العربية السعودية, جامعة الملكعبدالله للعلوم والتقنية, كاوست, جوجل, ومايكروسوفت, الملك فيصل التخصصي...
😊 Sentiment:
   True: positive | Pred: positive | Runtime: 4.07s
   [LLM-Only]: True: positive | Pred: positive | Runtime: 4.24s
📊 Topic Modeling:
🌍 LANGEXTRACT TOPIC ANALYSIS:
   Doc 1 Topics: Here’s a list of 3-5 key topics/phrases representing the main subjects of the Arabic text:, **Aramco Financial Results:** (النتائج المالية لأرامكو) - The core of the text is focused on Aramco’s exceptional profits and performance., **Strategic Partnerships:** (اتفاقيات شراكة استراتيجية) - The text highlights Aramco's agreements with TotalEnergies and Shell. | Runtime: 37.24s
   Doc 2 Topics: Here’s a list of 3-5 key topics/phrases representing the main subjects of the Arabic text:, **The Arab Summit (القمة العربية الطارئة):** The overarching event itself., **The Regional Crisis (الوضع المتفجر في المنطقة):**  This is the dominant concern driving the summit’s agenda. | Runtime: 22.14s
   Doc 3 Topics: Here’s a list of 3-5 key topics/phrases that represent the main subjects of the Arabic text:, **Artificial Intelligence (AI) Initiative:** This is the central focus, highlighted by the launch of a national AI program by KAUST., **Arabic Localization of AI:**  The text emphasizes the specific goal of adapting AI technologies for the Arab region. | Runtime: 20.14s
🤖 LLM-ONLY TOPIC ANALYSIS:
   Doc 1 Topics: Here’s a list of 3-5 key topics/phrases that represent the main subjects of the Arabic text:, **Aramco Financial Results:** (الأرباح المالية الاستثنائية) – The core of the text focuses on Aramco’s exceptional profits and growth., **Strategic Partnerships:** (اتفاقيات شراكة استراتيجية) – The text highlights Aramco’s new collaborations with TotalEnergies and Shell. | Runtime: 21.99s

   Doc 2 Topics: Here’s a list of 3-5 key topics/phrases represented in the Arabic text:, **Arab Summit/Emergency Arab Summit:** This refers to the main event itself, the concluding session of the urgent Arab summit held in Amman., **Regional Crisis/Situation in the Region:** The dominant theme is the "explosive situation" in the Middle East, highlighting the core concern driving the summit. | Runtime: 16.47s

   Doc 3 Topics: Here’s a list of 3-5 key topics/phrases representing the main subjects of the Arabic text:, **Artificial Intelligence (AI) Initiative:** This is the central theme, highlighted by the launch of the national AI program by KAUST., **Localization of AI Technology:**  The text emphasizes the goal of adapting AI technologies specifically for the Arab region and developing Arabic-language models. | Runtime: 11.99s

======================================================================
🏆 FINAL BENCHMARK SCORES (Accuracy & Runtime)
======================================================================

📝 SUMMARIZATION (ROUGE-1 & Runtime)
  mT5-XLSum       : Acc=0.2083, Time=35.04s
  Sumy-LexRank    : Acc=0.1828, Time=35.04s
  Sumy-LSA        : Acc=0.1828, Time=35.04s
  AraBART         : Acc=0.1798, Time=35.04s
  LangExtract     : Acc=0.1743, Time=30.13s
  LLM-Only        : Acc=0.1459, Time=24.08s

🏷️ NER (F1 Score & Runtime)
  CAMeL           : Acc=0.8413, Time=6.93s
  LLM-Only        : Acc=0.7774, Time=24.79s
  LangExtract     : Acc=0.7350, Time=24.94s
  Stanza          : Acc=0.7143, Time=6.93s
  Hatmimoha       : Acc=0.6088, Time=6.93s

😊 SENTIMENT (Accuracy & Runtime): Acc=0.67, Time=4.77s
   LLM-Only Sentiment Accuracy: 0.33

📊 TOPIC MODELING (Coherence & Runtime): Coherence=0.5343, Time=69.71s
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
