# analysis.py
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure NLTK corpora installed (call nltk.download(...) once in your env)
STOP = set(stopwords.words('english'))

def sentiment_score(text):
    """Return polarity [-1..1] and subjectivity [0..1]."""
    if not text or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}
    tb = TextBlob(text)
    return {"polarity": tb.sentiment.polarity, "subjectivity": tb.sentiment.subjectivity}

def top_keywords(text, top_n=5):
    """Return top N frequent alpha tokens excluding stopwords."""
    if not text or not text.strip():
        return []
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP and len(t) > 2]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_n)]

if __name__ == "__main__":
    demo = "Client liked the demo, worried about pricing; asked about integrations and timeline."
    print("Sentiment:", sentiment_score(demo))
    print("Keywords:", top_keywords(demo))
