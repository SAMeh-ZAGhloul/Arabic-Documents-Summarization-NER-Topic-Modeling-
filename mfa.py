"""
# Install CAMeL Tools
pip install camel-tools scikit-learn networkx numpy

# Download required models
camel_data -i ner-arabert                    # NER (541 MB)
camel_data -i sentiment-analysis-arabert     # Sentiment (541 MB)
camel_data -i morphology-db-msa-r13          # Morphology (40 MB)
camel_data -i disambig-mle-calima-msa-r13    # Disambiguation (88 MB)

Components:
- Preprocessing & Lemmatization (CAMeL Tools)
- Named Entity Recognition (AraBERT)
- Sentiment Analysis (CAMeL Tools)
- Topic Modeling (LDA)
- Extractive Summarization (TextRank)
- Abstractive Summarization (mT5 & AraT5 & AraBART)
- Accuracy benchmarks for all components
- ROUGE scores for summarization
- F1/Precision/Recall for NER
- Accuracy for Sentiment
- Coherence for Topic Modeling

Summarization:
  1. Sumy-LexRank, TextRank, LSA
  2. TF-IDF Baseline
  3. mT5-XLSum (Abstractive)
  4. AraBART (Abstractive)

NER Comparison:
  1. CAMeL Tools (AraBERT)
  2. Stanford Stanza
  3. Hatmimoha (BERT)

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import warnings
import numpy as np
import torch
import nltk

# =============================================
# 🛠️ SETUP & IMPORTS
# =============================================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("⬇️ Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForTokenClassification,
    pipeline as hf_pipeline
)

# CAMeL Tools
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.sentiment import SentimentAnalyzer

# Optional Libraries
try:
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False

warnings.filterwarnings('ignore')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

# =============================================
# 1. METRICS
# =============================================
class EvaluationMetrics:
    @staticmethod
    def normalize_arabic(text):
        text = normalize_unicode(text)
        text = dediac_ar(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)
        return re.sub(r'\bال', '', text).lower().strip()
    
    @staticmethod
    def rouge_scores(reference, hypothesis):
        def tokenize(text):
            text = EvaluationMetrics.normalize_arabic(text)
            return [t for t in simple_word_tokenize(text) if len(t) > 1 and not t.isdigit()]
        
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens: return 0.0
        
        overlap = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
        p = overlap / len(hyp_tokens)
        r = overlap / len(ref_tokens)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def ner_metrics(true_entities, pred_entities):
        def norm(e): return EvaluationMetrics.normalize_arabic(e['text'])
        true_set = set((norm(e), e['label']) for e in true_entities)
        partial_tp = 0
        for p in pred_entities:
            p_norm = norm(p)
            for t in true_entities:
                t_norm = norm(t)
                if p['label'] == t['label'] and (p_norm in t_norm or t_norm in p_norm):
                    partial_tp += 1
                    break
        prec = partial_tp / len(pred_entities) if pred_entities else 0
        rec = partial_tp / len(true_entities) if true_entities else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

class ArabicPreprocessor:
    STOPWORDS = set("في من إلى على أن هذا هذه الذي التي لكن كان بها هم بأن هناك عن حيث و لا ال ب ل ك م ن هنا لذا لأن حتى ومع دون أو وما كل بعد قبل عند بين كما أيضا ثم لم لن إذا كيف ما هل أي هو هي نحن أنا أنت يكون تكون كانت عليه إليه منه فيه به ذلك تلك هؤلاء أولئك عام بعض جميع أكثر معظم غير خلال ضمن نحو حول قد قال يقول كانوا وكان قالت يوم وقد ولا ولم ومن وهو وهي ولكن فإن إلا أما".split())
    def __init__(self):
        print("  📚 Loading CAMeL Morphology...")
        self.db = MorphologyDB.builtin_db('calima-msa-r13')
        self.analyzer = Analyzer(self.db)
    def normalize(self, text):
        return normalize_teh_marbuta_ar(normalize_alef_maksura_ar(normalize_alef_ar(dediac_ar(normalize_unicode(text)))))
    def preprocess(self, text):
        text = self.normalize(text)
        tokens = simple_word_tokenize(text)
        lemmas = []
        for t in tokens:
            if len(t) < 2 or t in self.STOPWORDS: continue
            analyses = self.analyzer.analyze(t)
            lemma = analyses[0].get('lex', t) if analyses else t
            lemmas.append(re.sub(r'_\d+$', '', lemma))
        return [dediac_ar(l) for l in lemmas]

# =============================================
# 2. MODELS (NER, Summ, Topics)
# =============================================
class ArabicNER:
    def __init__(self):
        self.models = {}
        print("  🏷️ Loading NER Models...")
        # CAMeL
        try:
            path = "/Users/user/.camel_tools/data/ner/arabert"
            self.c_tok = AutoTokenizer.from_pretrained(path)
            self.c_mod = AutoModelForTokenClassification.from_pretrained(path)
            self.c_lbl = {0: 'B-LOC', 1: 'B-MISC', 2: 'B-ORG', 3: 'B-PERS', 4: 'I-LOC', 5: 'I-MISC', 6: 'I-ORG', 7: 'I-PERS', 8: 'O'}
            self.models['CAMeL'] = True
        except: pass
        # Hatmimoha
        try:
            self.h_pipe = hf_pipeline("ner", model="hatmimoha/arabic-ner", aggregation_strategy="simple", device=DEVICE_ID)
            self.models['Hatmimoha'] = True
        except: pass
        # Stanza
        try:
            if STANZA_AVAILABLE:
                stanza.download('ar', processors='tokenize,ner', verbose=False)
                self.s_nlp = stanza.Pipeline('ar', processors='tokenize,ner', verbose=False)
                self.models['Stanza'] = True
        except: pass

    def extract_all(self, text):
        res = {}
        if 'CAMeL' in self.models: res['CAMeL'] = self._camel(text)
        if 'Hatmimoha' in self.models: res['Hatmimoha'] = self._hat(text)
        if 'Stanza' in self.models: res['Stanza'] = self._stanza(text)
        return res

    def _camel(self, text):
        ents = []
        inp = self.c_tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad(): preds = torch.argmax(self.c_mod(**inp).logits, dim=2)[0]
        toks = self.c_tok.convert_ids_to_tokens(inp['input_ids'][0])
        lbls = [self.c_lbl[p.item()] for p in preds]
        curr_t, curr_l = [], None
        for t, l in zip(toks, lbls):
            if t in ['[CLS]', '[SEP]']: continue
            if t.startswith('##'): 
                if curr_t: curr_t[-1] += t[2:]
            elif l.startswith('B-'):
                if curr_t: ents.append({'text': ' '.join(curr_t), 'label': curr_l})
                curr_t, curr_l = [t], l[2:]
            elif l.startswith('I-') and curr_l: curr_t.append(t)
            else:
                if curr_t: ents.append({'text': ' '.join(curr_t), 'label': curr_l})
                curr_t, curr_l = [], None
        return ents

    def _hat(self, text):
        try:
            r = self.h_pipe(text)
            return [{'text': x['word'], 'label': {'PERSON':'PERS','LOCATION':'LOC','ORGANIZATION':'ORG'}.get(x['entity_group'],'MISC')} for x in r]
        except: return []

    def _stanza(self, text):
        try:
            doc = self.s_nlp(text)
            return [{'text': e.text, 'label': {'PER':'PERS'}.get(e.type, e.type)} for e in doc.ents]
        except: return []

class ArabicSummarizer:
    def __init__(self, prep):
        self.prep = prep
        self.sumy = {}
        if SUMY_AVAILABLE:
            print("  📝 Loading Summarization Models...")
            self.sumy = {'Sumy-LexRank': LexRankSummarizer, 'Sumy-LSA': LsaSummarizer}
        self.neural = {}
        self._load('AraBART', 'moussaKam/AraBART', 'seq2seq')
        self._load('mT5-XLSum', 'csebuetnlp/mT5_multilingual_XLSum', 'pipeline')

    def _load(self, name, path, type):
        try:
            if type == 'pipeline': self.neural[name] = {'mod': hf_pipeline("summarization", model=path, device=DEVICE_ID), 'type': 'pipe'}
            else:
                tok = AutoTokenizer.from_pretrained(path)
                mod = AutoModelForSeq2SeqLM.from_pretrained(path)
                mod.to(DEVICE)
                self.neural[name] = {'tok': tok, 'mod': mod, 'type': 'seq'}
        except: pass

    def summarize(self, text):
        res = {}
        # Sumy
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        for n, cls in self.sumy.items():
            try: res[n] = ' '.join([str(s) for s in cls()(parser.document, 7)])
            except: pass
        # Neural
        clean = ' '.join(self.prep.normalize(text).split())[:4000]
        for n, c in self.neural.items():
            try:
                if c['type'] == 'pipe': res[n] = c['mod'](clean, max_length=150, min_length=50, truncation=True)[0]['summary_text']
                else:
                    inp = c['tok'](clean, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
                    out = c['mod'].generate(**inp, max_length=150, min_length=50, num_beams=4)
                    res[n] = c['tok'].decode(out[0], skip_special_tokens=True)
            except: pass
        return res

class TopicModeler:
    def __init__(self, prep):
        self.prep = prep
        print(f"  📊 Topic Modeling: {'Gensim' if GENSIM_AVAILABLE else 'N/A'}")

    def run(self, docs):
        if not GENSIM_AVAILABLE: return None, 0
        texts = [self.prep.preprocess(d) for d in docs]
        dic = corpora.Dictionary(texts)
        dic.filter_extremes(no_below=1, no_above=0.9)
        corpus = [dic.doc2bow(t) for t in texts]
        lda = LdaModel(corpus, num_topics=3, id2word=dic, passes=20, random_state=42)
        return lda.print_topics(num_words=5), CoherenceModel(model=lda, texts=texts, dictionary=dic, coherence='c_v').get_coherence()

# =============================================
# 3. PIPELINE EXECUTION
# =============================================
class UltimatePipeline:
    def __init__(self):
        print("="*70)
        print("🚀 ARABIC NLP PIPELINE: BENCHMARK EDITION")
        print("="*70)
        self.prep = ArabicPreprocessor()
        self.ner = ArabicNER()
        self.summ = ArabicSummarizer(self.prep)
        self.topics = TopicModeler(self.prep)
        self.metrics = EvaluationMetrics()
        self.sentiment = SentimentAnalyzer.pretrained()

    def run(self, data):
        scores = {'summ': {}, 'ner': {}, 'sent': []}
        
        print("\n" + "="*70)
        print("📄 DETAILED ANALYSIS (LARGE DOCS)")
        print("="*70)

        for i, d in enumerate(data):
            text = d['text']
            print(f"\n📂 Document {i+1} ({len(text.split())} words)")
            
            # 1. Summarization
            print("📝 Summarization:")
            sums = self.summ.summarize(text)
            for m, s in sums.items():
                r1 = self.metrics.rouge_scores(d['reference_summary'], s)
                scores['summ'].setdefault(m, []).append(r1)
                if m == 'AraBART': print(f"   [{m}]: {s[:100]}...")
            
            # 2. NER
            print("🏷️ NER (CAMeL):")
            ents = self.ner.extract_all(text)
            for m, e in ents.items():
                f1 = self.metrics.ner_metrics(d['entities'], e)
                scores['ner'].setdefault(m, []).append(f1)
            
            c_ents = [f"{x['text']}" for x in ents.get('CAMeL', [])[:6]]
            print(f"   Entities found: {', '.join(c_ents)}...")

            # 3. Sentiment
            print("😊 Sentiment:")
            pred_sent = self.sentiment.predict([text])[0]
            # Normalize prediction for comparison
            p_label = 'positive' if 'positive' in pred_sent or 'pos' in pred_sent else ('negative' if 'negative' in pred_sent or 'neg' in pred_sent else 'neutral')
            t_label = d.get('sentiment', 'neutral')
            print(f"   True: {t_label} | Pred: {p_label}")
            scores['sent'].append(1 if p_label == t_label else 0)

        # 4. Global Results
        print("\n" + "="*70)
        print("🏆 FINAL BENCHMARK SCORES")
        print("="*70)
        
        print("\n📝 SUMMARIZATION (ROUGE-1)")
        avgs = {k: np.mean(v) for k, v in scores['summ'].items()}
        for k, v in sorted(avgs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:<15} : {v:.4f}")

        print("\n🏷️ NER (F1 Score)")
        avgs_ner = {k: np.mean(v) for k, v in scores['ner'].items()}
        for k, v in sorted(avgs_ner.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:<15} : {v:.4f}")

        print(f"\n😊 SENTIMENT ACCURACY: {np.mean(scores['sent']):.2f}")
        
        topics, coh = self.topics.run([d['text'] for d in data])
        print(f"\n📊 TOPIC COHERENCE: {coh:.4f}")

# =============================================
# DATA (LARGE & FORMATTED)
# =============================================
def get_large_data():
    return [
        {
            'text': """أعلنت شركة أرامكو السعودية، عملاق النفط العالمي وأكبر شركة طاقة في العالم من حيث القيمة السوقية، اليوم عن تحقيق نتائج مالية استثنائية وغير مسبوقة خلال الربع الثالث من العام الحالي، حيث بلغت الأرباح الصافية أكثر من مائة وخمسين مليار ريال سعودي، بزيادة قدرها خمسة وعشرون بالمائة مقارنة بالفترة نفسها من العام الماضي. جاء هذا الإعلان المهم خلال المؤتمر الصحفي الذي عقده الرئيس التنفيذي للشركة المهندس أمين حسن الناصر في المقر الرئيسي للشركة بمدينة الظهران. وأوضح الناصر أن هذه النتائج تعكس قوة الأداء التشغيلي وقدرة الشركة على التكيف مع تقلبات الأسواق العالمية. وأضاف أن الشركة وقعت اتفاقيات شراكة استراتيجية ضخمة مع شركة توتال إنرجيز الفرنسية وشركة شل البريطانية لتطوير حقول الغاز الطبيعي. وفي سياق متصل بالاقتصاد الوطني، سجل مؤشر السوق السعودي (تداول) انخفاضاً حاداً بنسبة 2%، متأثراً بتراجع قطاع البنوك. وفي خطوة مفاجئة، أعلن مصرف الراجحي، أحد أكبر البنوك الإسلامية، عن انخفاض طفيف في أرباحه الفصلية بسبب زيادة المخصصات.""",
            'reference_summary': "أرباح قياسية لأرامكو وشراكات مع توتال وشل، وسط تراجع السوق السعودي وأرباح الراجحي.",
            'entities': [{'text': 'أرامكو السعودية', 'label': 'ORG'}, {'text': 'أمين الناصر', 'label': 'PERS'}, {'text': 'الظهران', 'label': 'LOC'}, {'text': 'توتال إنرجيز', 'label': 'ORG'}, {'text': 'شل', 'label': 'ORG'}, {'text': 'السوق السعودي', 'label': 'ORG'}, {'text': 'بنك الراجحي', 'label': 'ORG'}],
            'sentiment': 'mixed'
        },
        {
            'text': """اختتمت القمة العربية الطارئة أعمالها في العاصمة الأردنية عمان، وسط حضور رفيع المستوى من قادة الدول العربية. وقد هيمن على جدول الأعمال الوضع المتفجر في المنطقة. وأكد العاهل الأردني الملك عبدالله الثاني في كلمته الافتتاحية على ضرورة التضامن العربي لمواجهة التحديات الراهنة. وعلى هامش القمة، عقد الملك عبدالله اجتماعات ثنائية مغلقة مع ولي العهد السعودي الأمير محمد بن سلمان، والرئيس المصري عبدالفتاح السيسي، حيث تم بحث سبل تنسيق المواقف. وناقشت القمة باستفاضة الأوضاع المأساوية في سوريا واليمن، داعية المجتمع الدولي إلى تحمل مسؤولياته لإنهاء الصراعات ووقف نزيف الدم.""",
            'reference_summary': "ختام القمة العربية في عمان بدعوات للتضامن، ولقاءات بين الملك عبدالله ومحمد بن سلمان والسيسي.",
            'entities': [{'text': 'عمان', 'label': 'LOC'}, {'text': 'عبدالله الثاني', 'label': 'PERS'}, {'text': 'محمد بن سلمان', 'label': 'PERS'}, {'text': 'عبدالفتاح السيسي', 'label': 'PERS'}, {'text': 'سوريا', 'label': 'LOC'}, {'text': 'اليمن', 'label': 'LOC'}],
            'sentiment': 'neutral'
        },
        {
            'text': """تشهد المملكة العربية السعودية طفرة تقنية هائلة، حيث أعلنت جامعة الملك عبدالله للعلوم والتقنية (كاوست) عن إطلاق مبادرة وطنية للذكاء الاصطناعي بالتعاون مع شركات عالمية مثل جوجل ومايكروسوفت. وتهدف المبادرة إلى تعريب تقنيات الذكاء الاصطناعي وتطوير نماذج لغوية تخدم المنطقة العربية. وفي القطاع الصحي، حقق مستشفى الملك فيصل التخصصي ومركز الأبحاث إنجازاً طبياً غير مسبوق، حيث نجح فريق طبي بقيادة الدكتور سعود الشمري في تطبيق علاج جيني متطور لمرضى السرطان، مما أدى إلى نسب شفاء عالية جداً. وقد أشاد وزير الصحة فهد الجلاجل بهذا التقدم العلمي الكبير.""",
            'reference_summary': "أطلقت كاوست مبادرة للذكاء الاصطناعي مع جوجل ومايكروسوفت. حقق مستشفى الملك فيصل التخصصي إنجازاً طبياً بقيادة سعود الشمري.",
            'entities': [{'text': 'كاوست', 'label': 'ORG'}, {'text': 'جوجل', 'label': 'ORG'}, {'text': 'مايكروسوفت', 'label': 'ORG'}, {'text': 'مستشفى الملك فيصل التخصصي', 'label': 'ORG'}, {'text': 'سعود الشمري', 'label': 'PERS'}, {'text': 'فهد الجلاجل', 'label': 'PERS'}],
            'sentiment': 'positive'
        }
    ]

if __name__ == "__main__":
    UltimatePipeline().run(get_large_data())


