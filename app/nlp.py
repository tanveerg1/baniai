from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

def process_punjabi(text):
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("pa")
    normalized_text = normalizer.normalize(text)
    tokens = indic_tokenize.trivial_tokenize(normalized_text, lang="pa")
    return tokens

def process_english(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum()]

def detect_intent(tokens, language):
    search_keywords = {"pa": ["ਲੱਭੋ", "ਖੋਜ", "ਸਬਦ"], "en": ["find", "search", "shabad"]}
    recommend_keywords = {"pa": ["ਸਿਫਾਰਸ", "ਹੋਰ"], "en": ["recommend", "similar"]}
    if language == "pa":
        if any(t in search_keywords["pa"] for t in tokens):
            return "search"
        elif any(t in recommend_keywords["pa"] for t in tokens):
            return "recommend"
    else:
        if any(t in search_keywords["en"] for t in tokens):
            return "search"
        elif any(t in recommend_keywords["en"] for t in tokens):
            return "recommend"
    return "general"