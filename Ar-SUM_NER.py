# # Arabic Documents Summarization, NER & Topic Modeling
#
# ### Install CAMeL Tools
# pip install camel-tools scikit-learn networkx numpy ollama
#
# ### Download required models
# camel_data -i ner-arabert
# camel_data -i sentiment-analysis-arabert
# camel_data -i morphology-db-msa-r13
# camel_data -i disambig-mle-calima-msa-r13
#
# # Install Ollama and pull gemma3:4b model
# ollama pull gemma3:4b
#
# ## Components:
# - Preprocessing & Lemmatization (CAMeL Tools)
# - Named Entity Recognition (AraBERT)
# - Sentiment Analysis (CAMeL Tools)
# - Topic Modeling (LDA)
# - Extractive Summarization (TextRank)
# - Abstractive Summarization (mT5 & AraT5 & AraBART)
# - LLM-Only Benchmark (gemma3:4b on Ollama)
# - Accuracy and Runtime benchmarks for all components
# - ROUGE scores for summarization
# - F1/Precision/Recall for NER
# - Accuracy for Sentiment
# - Coherence for Topic Modeling
#
# ### Summarization:
#   1. Sumy-LexRank, TextRank, LSA
#   2. TF-IDF Baseline
#   3. mT5-XLSum (Abstractive)
#   4. AraBART (Abstractive)
#   5. LangExtract (Google's Multilingual Model)
#   6. LLM-Only Benchmark (gemma3:4b on Ollama)
#
# ### NER Comparison:
#   1. CAMeL Tools (AraBERT)
#   2. Stanford Stanza
#   3. Hatmimoha (BERT)
#   4. LangExtract (Google's Multilingual Model)
#   5. LLM-Only Benchmark (gemma3:4b on Ollama)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import warnings
import numpy as np
import torch
import nltk
import time

# =============================================
# GLOBAL VARIABLES & CONSTANTS
# =============================================
# Define global constants to prevent unbound variable errors
STANZA_AVAILABLE = False
SUMY_AVAILABLE = False
GENSIM_AVAILABLE = False
LANGEXTRACT_AVAILABLE = False

# Import required modules and assign them to global variables
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, pipeline as hf_pipeline
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForTokenClassification = None
    hf_pipeline = None

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    stanza = None

try:
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False
    PlaintextParser = None
    SumyTokenizer = None
    LexRankSummarizer = None
    LsaSummarizer = None

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    corpora = None
    LdaModel = None
    CoherenceModel = None

try:
    import ollama
    # Test if ollama is accessible by attempting a simple operation
    try:
        ollama.list()
        LANGEXTRACT_AVAILABLE = True
    except:
        LANGEXTRACT_AVAILABLE = False
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    ollama = None

# CAMeL Tools imports
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.sentiment import SentimentAnalyzer
except ImportError:
    MorphologyDB = None
    Analyzer = None
    SentimentAnalyzer = None

# Define constants for device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

# =============================================
# SETUP & IMPORTS
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

# Transformers
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM, 
        AutoModelForTokenClassification,
        pipeline as hf_pipeline
    )
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForTokenClassification = None
    hf_pipeline = None

# CAMeL Tools
try:
    from camel_tools.utils.normalize import normalize_unicode, normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar
    from camel_tools.utils.dediac import dediac_ar
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.sentiment import SentimentAnalyzer
except ImportError:
    normalize_unicode = None
    normalize_alef_ar = None
    normalize_alef_maksura_ar = None
    normalize_teh_marbuta_ar = None
    dediac_ar = None
    simple_word_tokenize = None
    MorphologyDB = None
    Analyzer = None
    SentimentAnalyzer = None

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
    PlaintextParser = None
    SumyTokenizer = None
    LexRankSummarizer = None
    TextRankSummarizer = None
    LsaSummarizer = None

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    corpora = None
    LdaModel = None
    CoherenceModel = None

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    stanza = None

# LangExtract
try:
    import ollama
    # Test if ollama is accessible by attempting a simple operation
    ollama.list()
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    ollama = None
except Exception:
    LANGEXTRACT_AVAILABLE = False
    ollama = None

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
            global STANZA_AVAILABLE
            if STANZA_AVAILABLE:
                stanza.download('ar', processors='tokenize,ner', verbose=False)
                self.s_nlp = stanza.Pipeline('ar', processors='tokenize,ner', verbose=False)
                self.models['Stanza'] = True
        except:
            STANZA_AVAILABLE = False
            pass

    def extract_all(self, text):
        res = {}
        if 'CAMeL' in self.models: res['CAMeL'] = self._camel(text)
        if 'Hatmimoha' in self.models: res['Hatmimoha'] = self._hat(text)
        if 'Stanza' in self.models: res['Stanza'] = self._stanza(text)
        return res

    def _camel(self, text):
        ents = []
        try:
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
        except: pass
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
# 2.5 LANGEXTRACT INTEGRATION
# =============================================
class LangExtractWrapper:
    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE
        if self.available:
            print("  🌍 Loading Ollama with gemma3:4b (LLM-based Multilingual Model)...")
            try:
                # Test connection to Ollama
                ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 10})
            except Exception as e:
                print(f"  ❌ Error connecting to Ollama: {e}")
                self.available = False

    def summarize(self, text):
        if not self.available: return None
        try:
            # Prepare a prompt for summarization
            prompt = f"""Please provide a concise summary of the following Arabic text. The summary should be in Arabic and capture the main points:

{text[:2000]}"""  # Limit text length to prevent context overflow
            
            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 150}  # Limit response length
            )
            
            summary = response['message']['content'].strip()
            return summary if summary else None
        except Exception as e:
            print(f"Ollama summary error: {e}")
            return None

    def extract_entities(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for NER
            prompt = f"""Extract named entities from the following Arabic text. Return the results in JSON format with 'text' and 'label' fields. Labels should be one of: 'PERS' (person), 'ORG' (organization), 'LOC' (location), 'MISC' (miscellaneous).

Example format:
[
    {{"text": "أرامكو السعودية", "label": "ORG"}},
    {{"text": "أمين الناصر", "label": "PERS"}}
]

Text:
{text[:2000]}"""  # Limit text length to prevent context overflow
            
            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 300}  # Allow more space for entity extraction
            )
            
            result = response['message']['content'].strip()
            
            # Parse the response to extract entities
            entities = []
            import json as json_module
            import re as re_module
            
            # Look for JSON-like structure in the response
            json_match = re_module.search(r'\[(.*?)\]', result, re_module.DOTALL)
            if json_match:
                try:
                    # Attempt to parse the JSON portion
                    json_str = '[' + json_match.group(1) + ']'
                    # Clean up the JSON string to make it valid
                    json_str = re_module.sub(r'\\*', '', json_str)  # Remove extra escapes
                    entities = json_module.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract entities with regex
                    # Look for patterns that match the expected format
                    for line in result.split('\n'):
                        # Match patterns like: {"text": "...", "label": "..."}
                        matches = re_module.findall(r'"text":\s*"([^"]+)"[^}}}]*"label":\s*"([^"]+)"', line)
                        for text_val, label_val in matches:
                            entities.append({"text": text_val, "label": label_val})
            else:
                # If no JSON format found, try to extract using regex patterns
                # Look for patterns in the response
                lines = result.split('\n')
                for line in lines:
                    # Look for patterns that might contain entity information
                    if 'text' in line.lower() and 'label' in line.lower():
                        # Extract using regex
                        text_match = re_module.search(r'"text":\s*"([^"]+)"', line)
                        label_match = re_module.search(r'"label":\s*"([^"]+)"', line)
                        if text_match and label_match:
                            entities.append({
                                "text": text_match.group(1),
                                "label": label_match.group(1)
                            })
            
            return entities
        except Exception as e:
            print(f"Ollama NER error: {e}")
            return []

    def extract_topics(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""Identify the main topics discussed in the following Arabic text. Return a list of 3-5 key topics/phrases that represent the main subjects.

Text:
{text[:2000]}"""
            
            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 100}
            )
            
            result = response['message']['content'].strip()
            
            # Extract topics from the response
            topics = []
            import re as re_module
            for line in result.split('\n'):
                # Remove numbering or bullet points
                cleaned_line = re_module.sub(r'^[\d\-\*\)\.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:  # Meaningful topic
                    topics.append(cleaned_line)
            
            # Limit to top 3 topics
            return topics[:3]
        except Exception as e:
            print(f"Ollama topic extraction error: {e}")
            return []


# =============================================
# 2.6 LLM-ONLY BENCHMARK (GEMMA3:4B ON OLLAMA)
# =============================================
class LLMOnlyBenchmark:
    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE  # Reuse the ollama availability check
        if self.available:
            print("  🤖 Loading LLM-Only Benchmark (gemma3:4b on Ollama)...")
            try:
                # Test connection to Ollama
                ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 10})
            except Exception as e:
                print(f"  ❌ Error connecting to Ollama: {e}")
                self.available = False

    def run_all_tasks(self, text):
        """Run all NLP tasks using only the LLM (gemma3:4b)"""
        # Initialize default results
        results = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
        
        if not self.available:
            return results

        try:
            # 1. Summarization
            start_time = time.time()
            results['summary'] = self.summarize(text)
            results['summary_runtime'] = time.time() - start_time

            # 2. NER
            start_time = time.time()
            results['entities'] = self.extract_entities(text)
            results['ner_runtime'] = time.time() - start_time

            # 3. Topic Modeling
            start_time = time.time()
            results['topics'] = self.extract_topics(text)
            results['topic_runtime'] = time.time() - start_time

            # 4. Sentiment Analysis (integrated in topic extraction)
            start_time = time.time()
            results['sentiment'] = self.extract_sentiment(text)
            results['sentiment_runtime'] = time.time() - start_time
        except Exception as e:
            print(f"Error in LLMOnlyBenchmark.run_all_tasks: {e}")
            # Return default results in case of error

        return results

    def summarize(self, text):
        if not self.available: return None
        try:
            # Prepare a prompt for summarization
            prompt = f"""Please provide a concise summary of the following Arabic text. The summary should be in Arabic and capture the main points:

{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 150}  # Limit response length
            )

            summary = response['message']['content'].strip()
            return summary if summary else None
        except Exception as e:
            print(f"LLM-only summary error: {e}")
            return None

    def extract_entities(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for NER
            prompt = f"""Extract named entities from the following Arabic text. Return the results in JSON format with 'text' and 'label' fields. Labels should be one of: 'PERS' (person), 'ORG' (organization), 'LOC' (location), 'MISC' (miscellaneous).

Example format:
[
    {{"text": "أرامكو السعودية", "label": "ORG"}},
    {{"text": "أمين الناصر", "label": "PERS"}}
]

Text:
{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 300}  # Allow more space for entity extraction
            )

            result = response['message']['content'].strip()

            # Parse the response to extract entities
            entities = []
            import json as json_module
            import re as re_module

            # Look for JSON-like structure in the response
            json_match = re_module.search(r'\[(.*?)\]', result, re_module.DOTALL)
            if json_match:
                try:
                    # Attempt to parse the JSON portion
                    json_str = '[' + json_match.group(1) + ']'
                    # Clean up the JSON string to make it valid
                    json_str = re_module.sub(r'\\*', '', json_str)  # Remove extra escapes
                    entities = json_module.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract entities with regex
                    # Look for patterns that match the expected format
                    for line in result.split('\n'):
                        # Match patterns like: {"text": "...", "label": "..."}
                        matches = re_module.findall(r'"text":\s*"([^"]+)"[^}}}]*"label":\s*"([^"]+)"', line)
                        for text_val, label_val in matches:
                            entities.append({"text": text_val, "label": label_val})
            else:
                # If no JSON format found, try to extract using regex patterns
                # Look for patterns in the response
                lines = result.split('\n')
                for line in lines:
                    # Look for patterns that might contain entity information
                    if 'text' in line.lower() and 'label' in line.lower():
                        # Extract using regex
                        text_match = re_module.search(r'"text":\s*"([^"]+)"', line)
                        label_match = re_module.search(r'"label":\s*"([^"]+)"', line)
                        if text_match and label_match:
                            entities.append({
                                "text": text_match.group(1),
                                "label": label_match.group(1)
                            })

            return entities
        except Exception as e:
            print(f"LLM-only NER error: {e}")
            return []

    def extract_topics(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""Identify the main topics discussed in the following Arabic text. Return a list of 3-5 key topics/phrases that represent the main subjects.

Text:
{text[:2000]}"""

            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 100}
            )

            result = response['message']['content'].strip()

            # Extract topics from the response
            topics = []
            import re as re_module
            for line in result.split('\n'):
                # Remove numbering or bullet points
                cleaned_line = re_module.sub(r'^[\d\-\*\)\.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:  # Meaningful topic
                    topics.append(cleaned_line)

            # Limit to top 3 topics
            return topics[:3]
        except Exception as e:
            print(f"LLM-only topic extraction error: {e}")
            return []

    def extract_sentiment(self, text):
        if not self.available: return 'unknown'
        try:
            # Prepare a prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of the following Arabic text. Return only one word: 'positive', 'negative', or 'neutral'.

Text:
{text[:1000]}"""  # Limit text length

            response = ollama.chat(
                model='gemma3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 20}  # Very short response
            )

            sentiment = response['message']['content'].strip().lower()
            # Normalize the sentiment response
            if 'positive' in sentiment or 'pos' in sentiment:
                return 'positive'
            elif 'negative' in sentiment or 'neg' in sentiment:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"LLM-only sentiment analysis error: {e}")
            return 'unknown'

# =============================================
# 2.7 QWEN3-ONLY BENCHMARK (QWEN3:4B ON OLLAMA)
# =============================================
class Qwen3OnlyBenchmark:
    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE  # Reuse the ollama availability check
        if self.available:
            print("  🤖 Loading Qwen3-Only Benchmark (qwen3:4b on Ollama)...")
            try:
                # Test connection to Ollama
                ollama.chat(model='qwen3:4b', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 10})
            except Exception as e:
                print(f"  ❌ Error connecting to Ollama: {e}")
                self.available = False

    def run_all_tasks(self, text):
        """Run all NLP tasks using only the LLM (qwen3:4b)"""
        # Initialize default results
        results = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
        
        if not self.available:
            return results

        try:
            # 1. Summarization
            start_time = time.time()
            results['summary'] = self.summarize(text)
            results['summary_runtime'] = time.time() - start_time

            # 2. NER
            start_time = time.time()
            results['entities'] = self.extract_entities(text)
            results['ner_runtime'] = time.time() - start_time

            # 3. Topic Modeling
            start_time = time.time()
            results['topics'] = self.extract_topics(text)
            results['topic_runtime'] = time.time() - start_time

            # 4. Sentiment Analysis (integrated in topic extraction)
            start_time = time.time()
            results['sentiment'] = self.extract_sentiment(text)
            results['sentiment_runtime'] = time.time() - start_time
        except Exception as e:
            print(f"Error in Qwen3OnlyBenchmark.run_all_tasks: {e}")
            # Return default results in case of error

        return results

    def summarize(self, text):
        if not self.available: return None
        try:
            # Prepare a prompt for summarization
            prompt = f"""Please provide a concise summary of the following Arabic text. The summary should be in Arabic and capture the main points:

{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='qwen3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 150}  # Limit response length
            )

            summary = response['message']['content'].strip()
            return summary if summary else None
        except Exception as e:
            print(f"Qwen3-only summary error: {e}")
            return None

    def extract_entities(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for NER
            prompt = f"""Extract named entities from the following Arabic text. Return the results in JSON format with 'text' and 'label' fields. Labels should be one of: 'PERS' (person), 'ORG' (organization), 'LOC' (location), 'MISC' (miscellaneous).

Example format:
[
    {{"text": "أرامكو السعودية", "label": "ORG"}},
    {{"text": "أمين الناصر", "label": "PERS"}}
]

Text:
{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='qwen3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 300}  # Allow more space for entity extraction
            )

            result = response['message']['content'].strip()

            # Parse the response to extract entities
            entities = []
            import json as json_module
            import re as re_module

            # Look for JSON-like structure in the response
            json_match = re_module.search(r'\[(.*?)\]', result, re_module.DOTALL)
            if json_match:
                try:
                    # Attempt to parse the JSON portion
                    json_str = '[' + json_match.group(1) + ']'
                    # Clean up the JSON string to make it valid
                    json_str = re_module.sub(r'\\*', '', json_str)  # Remove extra escapes
                    entities = json_module.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract entities with regex
                    # Look for patterns that match the expected format
                    for line in result.split('\n'):
                        # Match patterns like: {"text": "...", "label": "..."}
                        matches = re_module.findall(r'"text":\s*"([^"]+)"[^}}}]*"label":\s*"([^"]+)"', line)
                        for text_val, label_val in matches:
                            entities.append({"text": text_val, "label": label_val})
            else:
                # If no JSON format found, try to extract using regex patterns
                # Look for patterns in the response
                lines = result.split('\n')
                for line in lines:
                    # Look for patterns that might contain entity information
                    if 'text' in line.lower() and 'label' in line.lower():
                        # Extract using regex
                        text_match = re_module.search(r'"text":\s*"([^"]+)"', line)
                        label_match = re_module.search(r'"label":\s*"([^"]+)"', line)
                        if text_match and label_match:
                            entities.append({
                                "text": text_match.group(1),
                                "label": label_match.group(1)
                            })

            return entities
        except Exception as e:
            print(f"Qwen3-only NER error: {e}")
            return []

    def extract_topics(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""Identify the main topics discussed in the following Arabic text. Return a list of 3-5 key topics/phrases that represent the main subjects.

Text:
{text[:2000]}"""

            response = ollama.chat(
                model='qwen3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 100}
            )

            result = response['message']['content'].strip()

            # Extract topics from the response
            topics = []
            import re as re_module
            for line in result.split('\n'):
                # Remove numbering or bullet points
                cleaned_line = re_module.sub(r'^[\d\-\*\)\.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:  # Meaningful topic
                    topics.append(cleaned_line)

            # Limit to top 3 topics
            return topics[:3]
        except Exception as e:
            print(f"Qwen3-only topic extraction error: {e}")
            return []

    def extract_sentiment(self, text):
        if not self.available: return 'unknown'
        try:
            # Prepare a prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of the following Arabic text. Return only one word: 'positive', 'negative', or 'neutral'.

Text:
{text[:1000]}"""  # Limit text length

            response = ollama.chat(
                model='qwen3:4b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 20}  # Very short response
            )

            sentiment = response['message']['content'].strip().lower()
            # Normalize the sentiment response
            if 'positive' in sentiment or 'pos' in sentiment:
                return 'positive'
            elif 'negative' in sentiment or 'neg' in sentiment:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"Qwen3-only sentiment analysis error: {e}")
            return 'unknown'


# =============================================
# 2.8 LFM2.5-THINKING BENCHMARK (lfm2.5-thinking:latest on Ollama)
# =============================================
class Lfm25ThinkingBenchmark:
    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE  # Reuse the ollama availability check
        if self.available:
            print("  🧠 Loading LFM2.5-Thinking Benchmark (lfm2.5-thinking:latest on Ollama)...")
            try:
                # Test connection to Ollama
                ollama.chat(model='lfm2.5-thinking:latest', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 10})
            except Exception as e:
                print(f"  ❌ Error connecting to Ollama: {e}")
                self.available = False

    def run_all_tasks(self, text):
        """Run all NLP tasks using only the LLM (lfm2.5-thinking:latest)"""
        # Initialize default results
        results = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
        
        if not self.available:
            return results

        try:
            # 1. Summarization
            start_time = time.time()
            results['summary'] = self.summarize(text)
            results['summary_runtime'] = time.time() - start_time

            # 2. NER
            start_time = time.time()
            results['entities'] = self.extract_entities(text)
            results['ner_runtime'] = time.time() - start_time

            # 3. Topic Modeling
            start_time = time.time()
            results['topics'] = self.extract_topics(text)
            results['topic_runtime'] = time.time() - start_time

            # 4. Sentiment Analysis (integrated in topic extraction)
            start_time = time.time()
            results['sentiment'] = self.extract_sentiment(text)
            results['sentiment_runtime'] = time.time() - start_time
        except Exception as e:
            print(f"Error in Lfm25ThinkingBenchmark.run_all_tasks: {e}")
            # Return default results in case of error

        return results

    def summarize(self, text):
        if not self.available: return None
        try:
            # Prepare a prompt for summarization
            prompt = f"""Please provide a concise summary of the following Arabic text. The summary should be in Arabic and capture the main points:

{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='lfm2.5-thinking:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 150}  # Limit response length
            )

            summary = response['message']['content'].strip()
            return summary if summary else None
        except Exception as e:
            print(f"LFM2.5-Thinking summary error: {e}")
            return None

    def extract_entities(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for NER
            prompt = f"""Extract named entities from the following Arabic text. Return the results in JSON format with 'text' and 'label' fields. Labels should be one of: 'PERS' (person), 'ORG' (organization), 'LOC' (location), 'MISC' (miscellaneous).

Example format:
[
    {{"text": "أرامكو السعودية", "label": "ORG"}},
    {{"text": "أمين الناصر", "label": "PERS"}}
]

Text:
{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='lfm2.5-thinking:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 300}  # Allow more space for entity extraction
            )

            result = response['message']['content'].strip()

            # Parse the response to extract entities
            entities = []
            import json as json_module
            import re as re_module

            # Look for JSON-like structure in the response
            json_match = re_module.search(r'\[(.*?)\]', result, re_module.DOTALL)
            if json_match:
                try:
                    # Attempt to parse the JSON portion
                    json_str = '[' + json_match.group(1) + ']'
                    # Clean up the JSON string to make it valid
                    json_str = re_module.sub(r'\\*', '', json_str)  # Remove extra escapes
                    entities = json_module.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract entities with regex
                    # Look for patterns that match the expected format
                    for line in result.split('\n'):
                        # Match patterns like: {"text": "...", "label": "..."}
                        matches = re_module.findall(r'"text":\s*"([^"]+)"[^}}}]*"label":\s*"([^"]+)"', line)
                        for text_val, label_val in matches:
                            entities.append({"text": text_val, "label": label_val})
            else:
                # If no JSON format found, try to extract using regex patterns
                # Look for patterns in the response
                lines = result.split('\n')
                for line in lines:
                    # Look for patterns that might contain entity information
                    if 'text' in line.lower() and 'label' in line.lower():
                        # Extract using regex
                        text_match = re_module.search(r'"text":\s*"([^"]+)"', line)
                        label_match = re_module.search(r'"label":\s*"([^"]+)"', line)
                        if text_match and label_match:
                            entities.append({
                                "text": text_match.group(1),
                                "label": label_match.group(1)
                            })

            return entities
        except Exception as e:
            print(f"LFM2.5-Thinking NER error: {e}")
            return []

    def extract_topics(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""Identify the main topics discussed in the following Arabic text. Return a list of 3-5 key topics/phrases that represent the main subjects.

Text:
{text[:2000]}"""

            response = ollama.chat(
                model='lfm2.5-thinking:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 100}
            )

            result = response['message']['content'].strip()

            # Extract topics from the response
            topics = []
            import re as re_module
            for line in result.split('\n'):
                # Remove numbering or bullet points
                cleaned_line = re_module.sub(r'^[\d\-\*\)\.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:  # Meaningful topic
                    topics.append(cleaned_line)

            # Limit to top 3 topics
            return topics[:3]
        except Exception as e:
            print(f"LFM2.5-Thinking topic extraction error: {e}")
            return []

    def extract_sentiment(self, text):
        if not self.available: return 'unknown'
        try:
            # Prepare a prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of the following Arabic text. Return only one word: 'positive', 'negative', or 'neutral'.

Text:
{text[:1000]}"""  # Limit text length

            response = ollama.chat(
                model='lfm2.5-thinking:latest',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 20}  # Very short response
            )

            sentiment = response['message']['content'].strip().lower()
            # Normalize the sentiment response
            if 'positive' in sentiment or 'pos' in sentiment:
                return 'positive'
            elif 'negative' in sentiment or 'neg' in sentiment:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"LFM2.5-Thinking sentiment analysis error: {e}")
            return 'unknown'


# =============================================
# 2.9 TOMNG-LFM2.5-INSTRUCT BENCHMARK (tomng/lfm2.5-instruct:1.2b on Ollama)
# =============================================
class TomngLfm25InstructBenchmark:
    def __init__(self):
        self.available = LANGEXTRACT_AVAILABLE  # Reuse the ollama availability check
        if self.available:
            print("  🧠 Loading Tomng LFM2.5-Instruct Benchmark (tomng/lfm2.5-instruct:1.2b on Ollama)...")
            try:
                # Test connection to Ollama
                ollama.chat(model='tomng/lfm2.5-instruct:1.2b', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 10})
            except Exception as e:
                print(f"  ❌ Error connecting to Ollama: {e}")
                self.available = False

    def run_all_tasks(self, text):
        """Run all NLP tasks using only the LLM (tomng/lfm2.5-instruct:1.2b)"""
        # Initialize default results
        results = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
        
        if not self.available:
            return results

        try:
            # 1. Summarization
            start_time = time.time()
            results['summary'] = self.summarize(text)
            results['summary_runtime'] = time.time() - start_time

            # 2. NER
            start_time = time.time()
            results['entities'] = self.extract_entities(text)
            results['ner_runtime'] = time.time() - start_time

            # 3. Topic Modeling
            start_time = time.time()
            results['topics'] = self.extract_topics(text)
            results['topic_runtime'] = time.time() - start_time

            # 4. Sentiment Analysis (integrated in topic extraction)
            start_time = time.time()
            results['sentiment'] = self.extract_sentiment(text)
            results['sentiment_runtime'] = time.time() - start_time
        except Exception as e:
            print(f"Error in TomngLfm25InstructBenchmark.run_all_tasks: {e}")
            # Return default results in case of error

        return results

    def summarize(self, text):
        if not self.available: return None
        try:
            # Prepare a prompt for summarization
            prompt = f"""Please provide a concise summary of the following Arabic text. The summary should be in Arabic and capture the main points:

{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='tomng/lfm2.5-instruct:1.2b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 150}  # Limit response length
            )

            summary = response['message']['content'].strip()
            return summary if summary else None
        except Exception as e:
            print(f"Tomng LFM2.5-Instruct summary error: {e}")
            return None

    def extract_entities(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for NER
            prompt = f"""Extract named entities from the following Arabic text. Return the results in JSON format with 'text' and 'label' fields. Labels should be one of: 'PERS' (person), 'ORG' (organization), 'LOC' (location), 'MISC' (miscellaneous).

Example format:
[
    {{"text": "أرامكو السعودية", "label": "ORG"}},
    {{"text": "أمين الناصر", "label": "PERS"}}
]

Text:
{text[:2000]}"""  # Limit text length to prevent context overflow

            response = ollama.chat(
                model='tomng/lfm2.5-instruct:1.2b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 300}  # Allow more space for entity extraction
            )

            result = response['message']['content'].strip()

            # Parse the response to extract entities
            entities = []
            import json as json_module
            import re as re_module

            # Look for JSON-like structure in the response
            json_match = re_module.search(r'\[(.*?)\]', result, re_module.DOTALL)
            if json_match:
                try:
                    # Attempt to parse the JSON portion
                    json_str = '[' + json_match.group(1) + ']'
                    # Clean up the JSON string to make it valid
                    json_str = re_module.sub(r'\\*', '', json_str)  # Remove extra escapes
                    entities = json_module.loads(json_str)
                except:
                    # If JSON parsing fails, try to extract entities with regex
                    # Look for patterns that match the expected format
                    for line in result.split('\n'):
                        # Match patterns like: {"text": "...", "label": "..."}
                        matches = re_module.findall(r'"text":\s*"([^"]+)"[^}}}]*"label":\s*"([^"]+)"', line)
                        for text_val, label_val in matches:
                            entities.append({"text": text_val, "label": label_val})
            else:
                # If no JSON format found, try to extract using regex patterns
                # Look for patterns in the response
                lines = result.split('\n')
                for line in lines:
                    # Look for patterns that might contain entity information
                    if 'text' in line.lower() and 'label' in line.lower():
                        # Extract using regex
                        text_match = re_module.search(r'"text":\s*"([^"]+)"', line)
                        label_match = re_module.search(r'"label":\s*"([^"]+)"', line)
                        if text_match and label_match:
                            entities.append({
                                "text": text_match.group(1),
                                "label": label_match.group(1)
                            })

            return entities
        except Exception as e:
            print(f"Tomng LFM2.5-Instruct NER error: {e}")
            return []

    def extract_topics(self, text):
        if not self.available: return []
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""Identify the main topics discussed in the following Arabic text. Return a list of 3-5 key topics/phrases that represent the main subjects.

Text:
{text[:2000]}"""

            response = ollama.chat(
                model='tomng/lfm2.5-instruct:1.2b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 100}
            )

            result = response['message']['content'].strip()

            # Extract topics from the response
            topics = []
            import re as re_module
            for line in result.split('\n'):
                # Remove numbering or bullet points
                cleaned_line = re_module.sub(r'^[\d\-\*\)\.]+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:  # Meaningful topic
                    topics.append(cleaned_line)

            # Limit to top 3 topics
            return topics[:3]
        except Exception as e:
            print(f"Tomng LFM2.5-Instruct topic extraction error: {e}")
            return []

    def extract_sentiment(self, text):
        if not self.available: return 'unknown'
        try:
            # Prepare a prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of the following Arabic text. Return only one word: 'positive', 'negative', or 'neutral'.

Text:
{text[:1000]}"""  # Limit text length

            response = ollama.chat(
                model='tomng/lfm2.5-instruct:1.2b',
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': 20}  # Very short response
            )

            sentiment = response['message']['content'].strip().lower()
            # Normalize the sentiment response
            if 'positive' in sentiment or 'pos' in sentiment:
                return 'positive'
            elif 'negative' in sentiment or 'neg' in sentiment:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"Tomng LFM2.5-Instruct sentiment analysis error: {e}")
            return 'unknown'


# =============================================
# 3. PIPELINE EXECUTION
# =============================================
class UltimatePipeline:
    def __init__(self):
        print("="*70)
        print("🚀 ARABIC NLP PIPELINE: BENCHMARK EDITION")
        print("="*70)
        self.prep = ArabicPreprocessor() if 'ArabicPreprocessor' in globals() and ArabicPreprocessor is not None else None
        self.ner = ArabicNER() if 'ArabicNER' in globals() and ArabicNER is not None else None
        self.summ = ArabicSummarizer(self.prep) if 'ArabicSummarizer' in globals() and ArabicSummarizer is not None and self.prep is not None else None
        self.topics = TopicModeler(self.prep) if 'TopicModeler' in globals() and TopicModeler is not None and self.prep is not None else None
        self.metrics = EvaluationMetrics() if 'EvaluationMetrics' in globals() and EvaluationMetrics is not None else None
        self.sentiment = SentimentAnalyzer.pretrained() if 'SentimentAnalyzer' in globals() and SentimentAnalyzer is not None else None
        self.lang_extract = LangExtractWrapper() if 'LangExtractWrapper' in globals() and LangExtractWrapper is not None else None
        self.llm_only = LLMOnlyBenchmark() if 'LLMOnlyBenchmark' in globals() and LLMOnlyBenchmark is not None else None
        self.qwen3_only = Qwen3OnlyBenchmark() if 'Qwen3OnlyBenchmark' in globals() and Qwen3OnlyBenchmark is not None else None
        self.lfm25_thinking = Lfm25ThinkingBenchmark() if 'Lfm25ThinkingBenchmark' in globals() and Lfm25ThinkingBenchmark is not None else None
        self.tomng_lfm25_instruct = TomngLfm25InstructBenchmark() if 'TomngLfm25InstructBenchmark' in globals() and TomngLfm25InstructBenchmark is not None else None

    def run(self, data):
        scores = {'summ': {}, 'ner': {}, 'sent': []}
        runtimes = {'summ': {}, 'ner': {}, 'sent': [], 'topics': 0}

        print("\n" + "="*70)
        print("📄 DETAILED ANALYSIS (LARGE DOCS)")
        print("="*70)

        # Track LLM-only sentiment scores separately
        llm_only_sent_scores = []
        qwen3_only_sent_scores = []
        lfm25_thinking_sent_scores = []
        tomng_lfm25_instruct_sent_scores = []

        for i, d in enumerate(data):
            text = d['text']
            print(f"\n📂 Document {i+1} ({len(text.split())} words)")

            # Run LLM-only benchmark for this document (to get all results at once)
            llm_result = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
            if self.llm_only and hasattr(self.llm_only, 'available') and self.llm_only.available:
                result = self.llm_only.run_all_tasks(text)
                if result:
                    llm_result = result
            
            # Run Qwen3-only benchmark for this document (to get all results at once)
            qwen3_result = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
            if self.qwen3_only and hasattr(self.qwen3_only, 'available') and self.qwen3_only.available:
                result = self.qwen3_only.run_all_tasks(text)
                if result:
                    qwen3_result = result
            
            # Run LFM2.5-Thinking benchmark for this document (to get all results at once)
            lfm25_thinking_result = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
            if self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and self.lfm25_thinking.available:
                result = self.lfm25_thinking.run_all_tasks(text)
                if result:
                    lfm25_thinking_result = result
            
            # Run Tomng LFM2.5-Instruct benchmark for this document (to get all results at once)
            tomng_lfm25_instruct_result = {'summary': None, 'entities': [], 'topics': [], 'sentiment': 'unknown', 'summary_runtime': 0, 'ner_runtime': 0, 'topic_runtime': 0, 'sentiment_runtime': 0}
            if self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and self.tomng_lfm25_instruct.available:
                result = self.tomng_lfm25_instruct.run_all_tasks(text)
                if result:
                    tomng_lfm25_instruct_result = result

            # 1. Summarization
            print("📝 Summarization:")
            start_time = time.time()
            sums = self.summ.summarize(text) if self.summ else {}
            summ_runtime = time.time() - start_time

            # Add LangExtract summarization
            if self.lang_extract and hasattr(self.lang_extract, 'available') and self.lang_extract.available:
                start_time = time.time()
                le_summary = self.lang_extract.summarize(text) if self.lang_extract and hasattr(self.lang_extract, 'summarize') else None
                le_summ_runtime = time.time() - start_time
                if le_summary:
                    sums['LangExtract'] = le_summary
                    r1 = self.metrics.rouge_scores(d['reference_summary'], le_summary) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                    scores['summ'].setdefault('LangExtract', []).append(r1)
                    runtimes['summ'].setdefault('LangExtract', []).append(le_summ_runtime)
                    print(f"   [LangExtract]: {le_summ_runtime:.2f}s - {le_summary[:100]}...")

            # Add LLM-only summarization
            if (self.llm_only and hasattr(self.llm_only, 'available') and 
                self.llm_only.available and 
                llm_result.get('summary')):
                llm_summary = llm_result.get('summary')
                llm_summ_runtime = llm_result.get('summary_runtime', 0)
                sums['LLM-Only'] = llm_summary
                r1 = self.metrics.rouge_scores(d['reference_summary'], llm_summary) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                scores['summ'].setdefault('LLM-Only', []).append(r1)
                runtimes['summ'].setdefault('LLM-Only', []).append(llm_summ_runtime)
                print(f"   [LLM-Only]: {llm_summ_runtime:.2f}s - {llm_summary[:100]}...")

            # Add Qwen3-only summarization
            if (self.qwen3_only and hasattr(self.qwen3_only, 'available') and 
                self.qwen3_only.available and 
                qwen3_result.get('summary')):
                qwen3_summary = qwen3_result.get('summary')
                qwen3_summ_runtime = qwen3_result.get('summary_runtime', 0)
                sums['Qwen3-Only'] = qwen3_summary
                r1 = self.metrics.rouge_scores(d['reference_summary'], qwen3_summary) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                scores['summ'].setdefault('Qwen3-Only', []).append(r1)
                runtimes['summ'].setdefault('Qwen3-Only', []).append(qwen3_summ_runtime)
                print(f"   [Qwen3-Only]: {qwen3_summ_runtime:.2f}s - {qwen3_summary[:100]}...")

            # Add LFM2.5-Thinking summarization
            if (self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and 
                self.lfm25_thinking.available and 
                lfm25_thinking_result.get('summary')):
                lfm25_thinking_summary = lfm25_thinking_result.get('summary')
                lfm25_thinking_summ_runtime = lfm25_thinking_result.get('summary_runtime', 0)
                sums['LFM2.5-Thinking'] = lfm25_thinking_summary
                r1 = self.metrics.rouge_scores(d['reference_summary'], lfm25_thinking_summary) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                scores['summ'].setdefault('LFM2.5-Thinking', []).append(r1)
                runtimes['summ'].setdefault('LFM2.5-Thinking', []).append(lfm25_thinking_summ_runtime)
                print(f"   [LFM2.5-Thinking]: {lfm25_thinking_summ_runtime:.2f}s - {lfm25_thinking_summary[:100]}...")

            # Add Tomng LFM2.5-Instruct summarization
            if (self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and 
                self.tomng_lfm25_instruct.available and 
                tomng_lfm25_instruct_result.get('summary')):
                tomng_lfm25_instruct_summary = tomng_lfm25_instruct_result.get('summary')
                tomng_lfm25_instruct_summ_runtime = tomng_lfm25_instruct_result.get('summary_runtime', 0)
                sums['Tomng-LFM2.5-Instruct'] = tomng_lfm25_instruct_summary
                r1 = self.metrics.rouge_scores(d['reference_summary'], tomng_lfm25_instruct_summary) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                scores['summ'].setdefault('Tomng-LFM2.5-Instruct', []).append(r1)
                runtimes['summ'].setdefault('Tomng-LFM2.5-Instruct', []).append(tomng_lfm25_instruct_summ_runtime)
                print(f"   [Tomng-LFM2.5-Instruct]: {tomng_lfm25_instruct_summ_runtime:.2f}s - {tomng_lfm25_instruct_summary[:100]}...")

            for m, s in sums.items():
                if m not in ['LangExtract', 'LLM-Only', 'Qwen3-Only', 'LFM2.5-Thinking', 'Tomng-LFM2.5-Instruct']:  # Already processed
                    r1 = self.metrics.rouge_scores(d['reference_summary'], s) if self.metrics and hasattr(self.metrics, 'rouge_scores') else 0
                    scores['summ'].setdefault(m, []).append(r1)
                    runtimes['summ'].setdefault(m, []).append(summ_runtime)
                    if m == 'AraBART': print(f"   [{m}]: {summ_runtime:.2f}s - {s[:100]}...")

            # 2. NER
            print("🏷️ NER:")
            start_time = time.time()
            ents = self.ner.extract_all(text) if self.ner and hasattr(self.ner, 'extract_all') else {}
            ner_runtime = time.time() - start_time

            # Add LangExtract NER
            if self.lang_extract and hasattr(self.lang_extract, 'available') and self.lang_extract.available:
                start_time = time.time()
                le_entities = self.lang_extract.extract_entities(text) if self.lang_extract and hasattr(self.lang_extract, 'extract_entities') else []
                le_ner_runtime = time.time() - start_time
                if le_entities:
                    ents['LangExtract'] = le_entities
                    f1 = self.metrics.ner_metrics(d['entities'], le_entities) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                    scores['ner'].setdefault('LangExtract', []).append(f1)
                    runtimes['ner'].setdefault('LangExtract', []).append(le_ner_runtime)

            # Add LLM-only NER
            if (self.llm_only and hasattr(self.llm_only, 'available') and 
                self.llm_only.available and 
                llm_result.get('entities')):
                llm_entities = llm_result.get('entities', [])
                ents['LLM-Only'] = llm_entities
                f1 = self.metrics.ner_metrics(d['entities'], llm_entities) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                scores['ner'].setdefault('LLM-Only', []).append(f1)
                runtimes['ner'].setdefault('LLM-Only', []).append(llm_result.get('ner_runtime', 0))

            # Add Qwen3-only NER
            if (self.qwen3_only and hasattr(self.qwen3_only, 'available') and 
                self.qwen3_only.available and 
                qwen3_result.get('entities')):
                qwen3_entities = qwen3_result.get('entities', [])
                ents['Qwen3-Only'] = qwen3_entities
                f1 = self.metrics.ner_metrics(d['entities'], qwen3_entities) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                scores['ner'].setdefault('Qwen3-Only', []).append(f1)
                runtimes['ner'].setdefault('Qwen3-Only', []).append(qwen3_result.get('ner_runtime', 0))

            # Add LFM2.5-Thinking NER
            if (self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and 
                self.lfm25_thinking.available and 
                lfm25_thinking_result.get('entities')):
                lfm25_thinking_entities = lfm25_thinking_result.get('entities', [])
                ents['LFM2.5-Thinking'] = lfm25_thinking_entities
                f1 = self.metrics.ner_metrics(d['entities'], lfm25_thinking_entities) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                scores['ner'].setdefault('LFM2.5-Thinking', []).append(f1)
                runtimes['ner'].setdefault('LFM2.5-Thinking', []).append(lfm25_thinking_result.get('ner_runtime', 0))

            # Add Tomng LFM2.5-Instruct NER
            if (self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and 
                self.tomng_lfm25_instruct.available and 
                tomng_lfm25_instruct_result.get('entities')):
                tomng_lfm25_instruct_entities = tomng_lfm25_instruct_result.get('entities', [])
                ents['Tomng-LFM2.5-Instruct'] = tomng_lfm25_instruct_entities
                f1 = self.metrics.ner_metrics(d['entities'], tomng_lfm25_instruct_entities) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                scores['ner'].setdefault('Tomng-LFM2.5-Instruct', []).append(f1)
                runtimes['ner'].setdefault('Tomng-LFM2.5-Instruct', []).append(tomng_lfm25_instruct_result.get('ner_runtime', 0))

            for m, e in ents.items():
                if m not in ['LangExtract', 'LLM-Only', 'Qwen3-Only', 'LFM2.5-Thinking', 'Tomng-LFM2.5-Instruct']:  # Already processed
                    f1 = self.metrics.ner_metrics(d['entities'], e) if self.metrics and hasattr(self.metrics, 'ner_metrics') else 0
                    scores['ner'].setdefault(m, []).append(f1)
                    runtimes['ner'].setdefault(m, []).append(ner_runtime)

            c_ents = [f"{x['text']}" for x in ents.get('CAMeL', [])[:6]] if ents and 'CAMeL' in ents else []
            print(f"   Entities found: {', '.join(c_ents)}...")

            # 3. Sentiment
            print("😊 Sentiment:")
            start_time = time.time()
            pred_sent = self.sentiment.predict([text])[0] if self.sentiment and hasattr(self.sentiment, 'predict') else 'neutral'
            sent_runtime = time.time() - start_time
            # Normalize prediction for comparison
            p_label = 'positive' if 'positive' in pred_sent or 'pos' in pred_sent else ('negative' if 'negative' in pred_sent or 'neg' in pred_sent else 'neutral')
            t_label = d.get('sentiment', 'neutral')
            print(f"   True: {t_label} | Pred: {p_label} | Runtime: {sent_runtime:.2f}s")
            scores['sent'].append(1 if p_label == t_label else 0)
            runtimes['sent'].append(sent_runtime)

            # Add LLM-only sentiment
            if self.llm_only and hasattr(self.llm_only, 'available') and self.llm_only.available:
                llm_sentiment = llm_result.get('sentiment', 'unknown')
                llm_sent_runtime = llm_result.get('sentiment_runtime', 0)
                print(f"   [LLM-Only]: True: {t_label} | Pred: {llm_sentiment} | Runtime: {llm_sent_runtime:.2f}s")
                # Track LLM-only sentiment accuracy
                llm_only_sent_scores.append(1 if llm_sentiment == t_label else 0)
            
            # Add Qwen3-only sentiment
            if self.qwen3_only and hasattr(self.qwen3_only, 'available') and self.qwen3_only.available:
                qwen3_sentiment = qwen3_result.get('sentiment', 'unknown')
                qwen3_sent_runtime = qwen3_result.get('sentiment_runtime', 0)
                print(f"   [Qwen3-Only]: True: {t_label} | Pred: {qwen3_sentiment} | Runtime: {qwen3_sent_runtime:.2f}s")
                # Track Qwen3-only sentiment accuracy
                qwen3_only_sent_scores.append(1 if qwen3_sentiment == t_label else 0)

            # Add LFM2.5-Thinking sentiment
            if self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and self.lfm25_thinking.available:
                lfm25_thinking_sentiment = lfm25_thinking_result.get('sentiment', 'unknown')
                lfm25_thinking_sent_runtime = lfm25_thinking_result.get('sentiment_runtime', 0)
                print(f"   [LFM2.5-Thinking]: True: {t_label} | Pred: {lfm25_thinking_sentiment} | Runtime: {lfm25_thinking_sent_runtime:.2f}s")
                # Track LFM2.5-Thinking sentiment accuracy
                lfm25_thinking_sent_scores.append(1 if lfm25_thinking_sentiment == t_label else 0)

            # Add Tomng LFM2.5-Instruct sentiment
            if self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and self.tomng_lfm25_instruct.available:
                tomng_lfm25_instruct_sentiment = tomng_lfm25_instruct_result.get('sentiment', 'unknown')
                tomng_lfm25_instruct_sent_runtime = tomng_lfm25_instruct_result.get('sentiment_runtime', 0)
                print(f"   [Tomng-LFM2.5-Instruct]: True: {t_label} | Pred: {tomng_lfm25_instruct_sentiment} | Runtime: {tomng_lfm25_instruct_sent_runtime:.2f}s")
                # Track Tomng LFM2.5-Instruct sentiment accuracy
                tomng_lfm25_instruct_sent_scores.append(1 if tomng_lfm25_instruct_sentiment == t_label else 0)

        # 4. Topic Modeling
        print("📊 Topic Modeling:")
        start_time = time.time()
        topics, coh = self.topics.run([d['text'] for d in data]) if self.topics and hasattr(self.topics, 'run') else (None, 0)
        topic_runtime = time.time() - start_time
        runtimes['topics'] = topic_runtime

        # 5. LangExtract Topic Modeling Evaluation
        if self.lang_extract and hasattr(self.lang_extract, 'available') and self.lang_extract.available:
            print("🌍 LANGEXTRACT TOPIC ANALYSIS:")
            for i, d in enumerate(data):
                start_time = time.time()
                le_topics = self.lang_extract.extract_topics(d['text']) if self.lang_extract and hasattr(self.lang_extract, 'extract_topics') else []
                le_topic_runtime = time.time() - start_time
                if le_topics:
                    print(f"   Doc {i+1} Topics: {', '.join(le_topics[:3])} | Runtime: {le_topic_runtime:.2f}s")

        # 6. LLM-Only Topic Modeling Evaluation
        if self.llm_only and hasattr(self.llm_only, 'available') and self.llm_only.available:
            print("🤖 LLM-ONLY TOPIC ANALYSIS:")
            for i, d in enumerate(data):
                llm_result_doc = self.llm_only.run_all_tasks(d['text'])  # Run again for each document
                llm_topics = llm_result_doc.get('topics', []) if llm_result_doc else []
                llm_topic_runtime = llm_result_doc.get('topic_runtime', 0) if llm_result_doc else 0
                if llm_topics:
                    print(f"   Doc {i+1} Topics: {', '.join(llm_topics[:3])} | Runtime: {llm_topic_runtime:.2f}s")

        # 6.5 Qwen3-Only Topic Modeling Evaluation
        if self.qwen3_only and hasattr(self.qwen3_only, 'available') and self.qwen3_only.available:
            print("🤖 QWEN3-ONLY TOPIC ANALYSIS:")
            for i, d in enumerate(data):
                qwen3_result_doc = self.qwen3_only.run_all_tasks(d['text'])  # Run again for each document
                qwen3_topics = qwen3_result_doc.get('topics', []) if qwen3_result_doc else []
                qwen3_topic_runtime = qwen3_result_doc.get('topic_runtime', 0) if qwen3_result_doc else 0
                if qwen3_topics:
                    print(f"   Doc {i+1} Topics: {', '.join(qwen3_topics[:3])} | Runtime: {qwen3_topic_runtime:.2f}s")

        # 6.6 LFM2.5-Thinking Topic Modeling Evaluation
        if self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and self.lfm25_thinking.available:
            print("🤖 LFM2.5-THINKING TOPIC ANALYSIS:")
            for i, d in enumerate(data):
                lfm25_thinking_result_doc = self.lfm25_thinking.run_all_tasks(d['text'])  # Run again for each document
                lfm25_thinking_topics = lfm25_thinking_result_doc.get('topics', []) if lfm25_thinking_result_doc else []
                lfm25_thinking_topic_runtime = lfm25_thinking_result_doc.get('topic_runtime', 0) if lfm25_thinking_result_doc else 0
                if lfm25_thinking_topics:
                    print(f"   Doc {i+1} Topics: {', '.join(lfm25_thinking_topics[:3])} | Runtime: {lfm25_thinking_topic_runtime:.2f}s")

        # 6.7 Tomng LFM2.5-Instruct Topic Modeling Evaluation
        if self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and self.tomng_lfm25_instruct.available:
            print("🤖 TOMNG-LFM2.5-INSTRUCT TOPIC ANALYSIS:")
            for i, d in enumerate(data):
                tomng_lfm25_instruct_result_doc = self.tomng_lfm25_instruct.run_all_tasks(d['text'])  # Run again for each document
                tomng_lfm25_instruct_topics = tomng_lfm25_instruct_result_doc.get('topics', []) if tomng_lfm25_instruct_result_doc else []
                tomng_lfm25_instruct_topic_runtime = tomng_lfm25_instruct_result_doc.get('topic_runtime', 0) if tomng_lfm25_instruct_result_doc else 0
                if tomng_lfm25_instruct_topics:
                    print(f"   Doc {i+1} Topics: {', '.join(tomng_lfm25_instruct_topics[:3])} | Runtime: {tomng_lfm25_instruct_topic_runtime:.2f}s")

        # 7. Global Results
        print("\n" + "="*70)
        print("🏆 FINAL BENCHMARK SCORES (Accuracy & Runtime)")
        print("="*70)

        print("\n📝 SUMMARIZATION (ROUGE-1 & Runtime)")
        avg_scores = {k: np.mean(v) for k, v in scores['summ'].items() if v}
        avg_runtimes = {k: np.mean(v) for k, v in runtimes['summ'].items() if v}
        combined_results = [(k, avg_scores[k], avg_runtimes[k]) for k in avg_scores.keys() if k in avg_runtimes]
        sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)  # Sort by accuracy
        for k, acc, rt in sorted_results:
            print(f"  {k:<15} : Acc={acc:.4f}, Time={rt:.2f}s")

        print("\n🏷️ NER (F1 Score & Runtime)")
        avg_scores_ner = {k: np.mean(v) for k, v in scores['ner'].items() if v}
        avg_runtimes_ner = {k: np.mean(v) for k, v in runtimes['ner'].items() if v}
        combined_results_ner = [(k, avg_scores_ner[k], avg_runtimes_ner[k]) for k in avg_scores_ner.keys() if k in avg_runtimes_ner]
        sorted_results_ner = sorted(combined_results_ner, key=lambda x: x[1], reverse=True)  # Sort by accuracy
        for k, acc, rt in sorted_results_ner:
            print(f"  {k:<15} : Acc={acc:.4f}, Time={rt:.2f}s")

        avg_sent_acc = np.mean(scores['sent']) if scores['sent'] else 0
        avg_sent_rt = np.mean(runtimes['sent']) if runtimes['sent'] else 0
        print(f"\n😊 SENTIMENT (Accuracy & Runtime): Acc={avg_sent_acc:.2f}, Time={avg_sent_rt:.2f}s")

        # Add LLM-only sentiment accuracy if available
        if self.llm_only and hasattr(self.llm_only, 'available') and self.llm_only.available and llm_only_sent_scores:
            llm_only_sent_acc = np.mean(llm_only_sent_scores)
            print(f"   LLM-Only Sentiment Accuracy: {llm_only_sent_acc:.2f}")
        
        # Add Qwen3-only sentiment accuracy if available
        if self.qwen3_only and hasattr(self.qwen3_only, 'available') and self.qwen3_only.available and qwen3_only_sent_scores:
            qwen3_only_sent_acc = np.mean(qwen3_only_sent_scores)
            print(f"   Qwen3-Only Sentiment Accuracy: {qwen3_only_sent_acc:.2f}")

        # Add LFM2.5-Thinking sentiment accuracy if available
        if self.lfm25_thinking and hasattr(self.lfm25_thinking, 'available') and self.lfm25_thinking.available and lfm25_thinking_sent_scores:
            lfm25_thinking_sent_acc = np.mean(lfm25_thinking_sent_scores)
            print(f"   LFM2.5-Thinking Sentiment Accuracy: {lfm25_thinking_sent_acc:.2f}")

        # Add Tomng LFM2.5-Instruct sentiment accuracy if available
        if self.tomng_lfm25_instruct and hasattr(self.tomng_lfm25_instruct, 'available') and self.tomng_lfm25_instruct.available and tomng_lfm25_instruct_sent_scores:
            tomng_lfm25_instruct_sent_acc = np.mean(tomng_lfm25_instruct_sent_scores)
            print(f"   Tomng LFM2.5-Instruct Sentiment Accuracy: {tomng_lfm25_instruct_sent_acc:.2f}")

        print(f"\n📊 TOPIC MODELING (Coherence & Runtime): Coherence={coh:.4f}, Time={topic_runtime:.2f}s")


# =============================================
# DATA
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
            'text': """اختتمت القمة العربية الطارئة أعمالها في العاصمة الأردنية عمان، وسط حضور رفيع المستوى من قادة الدول العربية. وقد هيمن على جدول الأعمال الوضع المتفجر في المنطقة. وأكد العاهل الأردني الملك عبدالله الثاني في كلمته الافتتاحية على ضرورة التضامن العربي لمواجهة التحديات الراهنة. وعلى هامش القمة، عقد الملكعبدالله اجتماعات ثنائية مغلقة مع ولي العهد السعودي الأمير محمد بن سلمان، والرئيس المصري عبدالفتاح السيسي، حيث تم بحث سبل تنسيق المواقف. وناقشت القمة باستفاضة الأوضاع المأساوية في سوريا واليمن، داعية المجتمع الدولي إلى تحمل مسؤولياته لإنهاء الصراعات ووقف نزيف الدم.""",
            'reference_summary': "ختام القمة العربية في عمان بدعوات للتضامن، ولقاءات بين الملكعبدالله ومحمد بن سلمان والسيسي.",
            'entities': [{'text': 'عمان', 'label': 'LOC'}, {'text': 'عبدالله الثاني', 'label': 'PERS'}, {'text': 'محمد بن سلمان', 'label': 'PERS'}, {'text': 'عبدالفتاح السيسي', 'label': 'PERS'}, {'text': 'سوريا', 'label': 'LOC'}, {'text': 'اليمن', 'label': 'LOC'}],
            'sentiment': 'neutral'
        },
        {
            'text': """تشهد المملكة العربية السعودية طفرة تقنية هائلة، حيث أعلنت جامعة الملكعبدالله للعلوم والتقنية (كاوست) عن إطلاق مبادرة وطنية للذكاء الاصطناعي بالتعاون مع شركات عالمية مثل جوجل ومايكروسوفت. وتهدف المبادرة إلى تعريب تقنيات الذكاء الاصطناعي وتطوير نماذج لغوية تخدم المنطقة العربية. وفي القطاع الصحي، حقق مستشفى الملك فيصل التخصصي ومركز الأبحاث إنجازاً طبياً غير مسبوق، حيث نجح فريق طبي بقيادة الدكتور سعود الشمري في تطبيق علاج جيني متطور لمرضى السرطان، مما أدى إلى نسب شفاء عالية جداً. وقد أشاد وزير الصحة فهد الجلاجل بهذا التقدم العلمي الكبير.""",
            'reference_summary': "أطلقت كاوست مبادرة للذكاء الاصطناعي مع جوجل ومايكروسوفت. حقق مستشفى الملك فيصل التخصصي إنجازاً طبياً بقيادة سعود الشمري.",
            'entities': [{'text': 'كاوست', 'label': 'ORG'}, {'text': 'جوجل', 'label': 'ORG'}, {'text': 'مايكروسوفت', 'label': 'ORG'}, {'text': 'مستشفى الملك فيصل التخصصي', 'label': 'ORG'}, {'text': 'سعود الشمري', 'label': 'PERS'}, {'text': 'فهد الجلاجل', 'label': 'PERS'}],
            'sentiment': 'positive'
        }
    ]


if __name__ == "__main__":
    UltimatePipeline().run(get_large_data())





