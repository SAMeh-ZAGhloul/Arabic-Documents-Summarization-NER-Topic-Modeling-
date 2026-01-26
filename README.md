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
ollama pull qwen3:4b
ollama pull lfm2.5-thinking:latest
ollama pull tomng/lfm2.5-instruct:1.2b
```

## 📊 Benchmark Results

Based on the latest execution of Ar-SUM_NER.py across three test documents (e.g., Aramco profits, Arab Summit, AI initiatives), the results are as follows:

### 1. Summarization

| Model | Accuracy (ROUGE-1) | Avg. Time (sec) | Sample Output Snippet |
|-------|-------------------|-----------------|----------------------|
| mT5-XLSum | 0.2083 (Best) | 47.55 | "أعلنت أرامكو السعودية عن أرباح..." |
| Tomng-LFM2.5-Instruct | 0.1955 | 12.63 | "أعلنت شركة أرامكو السعودية عن تحقيق نتائج مالية استثنائية..." |
| Sumy-LexRank | 0.1828 | 47.55 | (Extracted Sentences) |
| Sumy-LSA | 0.1828 | 47.55 | (Extracted Sentences) |
| AraBART | 0.1798 | 47.55 | "تشهد المملكة طفرة تقنية..." |
| LLM-Only | 0.1627 | 32.04 | "إليك ملخص النص العربي..." |
| LangExtract | 0.1524 | 45.11 | "ملخص للنتائج المالية لشركة..." |

### 2. Named Entity Recognition (NER)

| Model | F1 Score | Avg. Time (sec) | Sample Entities |
|-------|----------|-----------------|-----------------|
| CAMeL (AraBERT) | 0.8413 (Highest) | 7.08 | ارامكو السعودية (ORG), الظهران (LOC), أمين الناصر (PERS) |
| LLM-Only | 0.7774 | 22.31 | أرامكو (ORG), الأردن (LOC) |
| LangExtract | 0.7298 | 23.89 | أرامكو (ORG), الأردن (LOC) |
| Stanza | 0.7143 | 7.08 | عمان (LOC), محمد بن سلمان (PERS) |
| Hatmimoha | 0.6088 | 7.08 | سوريا (LOC) |
| Tomng-LFM2.5-Instruct | 0.4551 | 11.94 | entities found |

### 3. Sentiment Analysis

- Specialized Models Accuracy: 0.67 (e.g., correctly identifying إيجابي - Positive)
- LLM-Only Accuracy: 0.33
- Qwen3-Only Accuracy: 0.33
- LFM2.5-Thinking Accuracy: 0.33
- Tomng LFM2.5-Instruct Accuracy: 0.33

### 4. Topic Modeling

- Gensim LDA Coherence: 0.5343, Avg. Time: 61.07s
- LangExtract Topics (Sample):
  - Doc 1: Saudi Aramco Financial Results, Strategic Partnerships
  - Doc 2: Arab Emergency Summit, Regional Crisis
  - Doc 3: AI Initiative, Arabic Localization of AI
- LLM-Only Topics (Sample):
  - Doc 1: Aramco Financial Results, Strategic Partnerships
  - Doc 2: Arab Summit, Regional Crisis
  - Doc 3: AI Initiative, Localization of AI Technology
- Tomng-LFM2.5-Instruct Topics (Sample):
  - Doc 1: نتائج مالية استثنائية, اتفاقيات شراكة استراتيجية
  - Doc 2: القمة العربية الطارئة في عمان, التضامن العربي في مواجهة التحديات
  - Doc 3: التقنية والذكاء الاصطناعي في المملكة العربية السعودية, مبادرة الذكاء الاصطناعي الوطنية

## 📝 Full Arabic Output Samples (المخرجات الكاملة)

Below are the detailed Arabic outputs generated during the benchmark execution:

### Summarization (التلخيص)

- [LangExtract - Gemma 3]: "Here's a concise summary of the text in Arabic:\n\n**أعلنت أرامكو عن أرباح قياسية بلغت أكثر من 150 ملي..."
- [AraBART]: "اعلنت شركه ارامuco السعوديه، عملاق النفط العالمي واكبر شركه طاقه في العالم من حيث القيمه السوциه، الي..."
- [LLM-Only]: "Here's a concise summary of the text in Arabic:\n\n**تلقت السعودية إعلانًا استثنائيًا من شركة أرامكو، ... "
- [Tomng-LFM2.5-Instruct]: "أعلنت شركة أرامكو السعودية عن تحقيق نتائج مالية استثنائية خلال الربع الثالث من العام، حيث بلغت الأرب..."

### Named Entities (الكيانات المستخرجة)

Entities Found: ارامكو السعودية (ORG), ريال (MONEY), امين حسن الناصر (PERS), الظهران (LOC), الناصر (PERS), توتال انرجيز (ORG), عمان (LOC), عبدالله الثاني (PERS), محمد بن سلمان (PERS), عبدالفتاح السيسي (PERS), سوريا (LOC), واليمن (LOC), المملكة العربية السعودية (LOC), جامعة الملكعبدالله للعلوم والتقنية (ORG), كاوست (ORG), جوجل (ORG), ومايكروسوفت (ORG), الملك فيصل التخصصي (ORG).

### Topic Analysis (تحليل المواضيع)

- Doc 1 Topics: Here's a list of 3-5 key topics/phrases representing the main subjects of the Arabic text:, **Saudi Aramco Financial Results:** (النتائج المالية الاستثنائية) – This is the central theme, focusing on the company's record profits., **Strategic Partnerships:** (اتفاقيات شراكة استراتيجية) – The text highlights Aramco's agreements with TotalEnergies and Shell.
- Doc 2 Topics: Here's a list of 3-5 key topics/phrases that represent the main subjects of the Arabic text:, **The Arab Emergency Summit:** This is the overarching event and the primary focus of the text., **The Situation in the Region (تحديات المنطقة):** This refers to the explosive situation in the Middle East, dominating the summit's agenda.
- Doc 3 Topics: Here's a list of 3-5 key topics/phrases representing the main subjects of the Arabic text:, **Artificial Intelligence (AI) Initiative:** This is the central theme, highlighted by the launch of a national AI program by KAUST., **Arabic Localization of AI:** The text specifically mentions the goal of adapting AI technologies for the Arab region.

### Sentiment Labels (تحليل المشاعر)

Labels: إيجابي (Positive), محايد (Neutral), مختلط (Mixed).

## 🚀 How to Run

Run the main script to initiate the comparison:

```bash
python Ar-SUM_NER.py
```

