# text_cleaner.py
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available (no noisy output)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True, min_token_len=2):
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def _clean_one(self, doc):
        if not isinstance(doc, str):
            return ""
        s = doc.lower()
        s = re.sub(r'http\S+|www\.\S+', ' ', s)
        s = re.sub(r'\S+@\S+', ' ', s)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        try:
            tokens = nltk.word_tokenize(s)
        except LookupError:
            nltk.download('punkt')
            tokens = nltk.word_tokenize(s)
        toks = []
        for t in tokens:
            if self.remove_stopwords and t in self.stopwords:
                continue
            if len(t) < self.min_token_len:
                continue
            t = self.lemmatizer.lemmatize(t)
            toks.append(t)
        return " ".join(toks)

    def transform(self, X):
        return [self._clean_one(x) for x in X]
