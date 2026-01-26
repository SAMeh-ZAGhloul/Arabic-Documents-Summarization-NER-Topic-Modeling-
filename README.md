# 🚀 Arabic NLP Pipeline: Benchmark Edition

This project provides a comprehensive Benchmarking Suite for Arabic Natural Language Processing (NLP). It aims to compare the performance of specialized traditional tools (such as CAMeL Tools and AraBERT) against modern Large Language Models (LLMs) like Gemma 3 and Qwen 3 via "Ollama" across tasks including Entity Extraction, Summarization, Sentiment Analysis, and Topic Modeling.

## 📋 Key Features

- **Summarization**: Comparison between Extractive methods (LexRank, LSA) and Abstractive models (mT5, AraBART) vs. LLMs.
- **Named Entity Recognition (NER)**: Accuracy evaluation of CAMeL Tools, Stanza, and Hatmimoha against Ollama-hosted models.
- **Sentiment Analysis**: Measuring prediction accuracy for emotional states in Arabic text.
- **Topic Modeling**: Utilizing LDA (Gensim) compared to semantic thematic analysis from Large Language Models.
- **Performance Metrics**: Calculation of Accuracy, F1 Score, ROUGE scores, and execution Runtime.

## 🛠️ Requirements

The following libraries are required to run the suite:

```bash
pip install camel-tools transformers torch scikit-learn gensim sumy nltk ollama
```

Additionally, Ollama must be installed with the following models pulled:

```bash
ollama pull gemma3:4b
ollama pull qwen2.5:3b  # Used as a surrogate for Qwen3 in testing
```

## 📊 Benchmark Results

Based on the latest execution of Ar-SUM_NER.py across three test documents (e.g., Aramco profits, Arab Summit, AI initiatives), the results are as follows:

### 1. Summarization

| Model | Accuracy (ROUGE-1) | Avg. Time (sec) | Sample Output Snippet |
|-------|-------------------|-----------------|----------------------|
| mT5-XLSum | 0.2083 (Best) | 26.11 | "أعلنت أرامكو السعودية عن أرباح..." |
| Sumy (LexRank/LSA) | 0.1828 | 26.11 | (Extracted Sentences) |
| AraBART | 0.1798 | 26.11 | "تشهد المملكة طفرة تقنية..." |
| LangExtract (Gemma 3) | 0.1619 | 26.60 | "ملخص للنتائج المالية لشركة..." |
| LLM-Only | 0.1193 | 21.22 | "إليك ملخص النص العربي..." |

### 2. Named Entity Recognition (NER)

| Model | F1 Score | Avg. Time (sec) | Sample Entities |
|-------|----------|-----------------|-----------------|
| CAMeL (AraBERT) | 0.8413 (Highest) | 7.23 | الظهران (LOC), أمين الناصر (PERS) |
| LangExtract / LLM | 0.7417 | 20.83 | أرامكو (ORG), الأردن (LOC) |
| Stanza | 0.7143 | 7.23 | عمان (LOC), محمد بن سلمان (PERS) |
| Hatmimoha | 0.6088 | 7.23 | سوريا (LOC) |

### 3. Sentiment Analysis

- Specialized Models Accuracy: 0.67 (e.g., correctly identifying إيجابي - Positive)
- LLM-Only Accuracy: 0.33
- Qwen3-Only Accuracy: 0.33

## 📝 Full Arabic Output Samples (المخرجات الكاملة)

Below are the detailed Arabic outputs generated during the benchmark execution:

### Summarization (التلخيص)

- [LangExtract - Gemma 3]: "أعلنت أرامكو السعودية عن أرباح قياسية بلغت 115 مليار ريال، مدفوعة بارتفاع أسعار النفط وزيادة الإنتاج، مما يعزز مكانتها كأكبر شركة طاقة في العالم."
- [AraBART]: "اعلنت شركه ارامكو السعوديه، عملاق النفط العالمي واكبر شركه طاقه في العالم من حيث القيمه السوقيه، الي تحقيق أرباح قياسية."
- [LLM-Only]: "خُصّصَت القمة العربية الطارئة في الأردن للبحث في الوضع المتفجر في المنطقة وسبل دعم الاستقرار الإقليمي."

### Named Entities (الكيانات المستخرجة)

Entities Found: أرامكو السعودية (ORG), أمين حسن الناصر (PERS), الظهران (LOC), عمان (LOC), عبدالله الثاني (PERS), جامعة الملك عبدالله للعلوم والتقنية (ORG), جوجل (ORG).

### Topic Analysis (تحليل المواضيع)

- Doc 1 Topics: النتائج المالية لأرامكو (Aramco Financial Results), اتفاقيات شراكة استراتيجية (Strategic Partnerships).
- Doc 2 Topics: القمة العربية الطارئة (Arab Emergency Summit), الوضع المتفجر في المنطقة (Regional Crisis).
- Doc 3 Topics: المبادرة الوطنية للذكاء الاصطناعي (National AI Initiative), الإنجاز الطبي (Medical Advancement).

### Sentiment Labels (تحليل المشاعر)

Labels: إيجابي (Positive), محايد (Neutral), مختلط (Mixed).

## 🚀 How to Run

Run the main script to initiate the comparison:

```bash
python arabic_nlp_benchmark.py
```
