# train_model.py
import argparse
import os
import joblib
import pandas as pd
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# -----------------------
# Custom Transformer
# -----------------------
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

# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        probs = None
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if probs is not None else None

    print("=== Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC:   {roc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

# -----------------------
# Main training pipeline
# -----------------------
def main(args):
    print("Loading data:", args.data)
    df = pd.read_csv(args.data)
    if 'note_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'note_text' and 'label' columns")

    X = df['note_text'].fillna('').astype(str).values
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('clean', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced'))
    ])

    param_dist = {
        'tfidf__max_features': [3000, 5000, 8000, 10000],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.01, 0.1, 1.0, 5.0, 10.0],
    }

    # Dynamic n_splits for safety
    n_splits = min(5, len(y_train))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring='f1',
        n_jobs=1,   # single process on Windows (safe)
        cv=cv,
        verbose=1,
        random_state=42
    )

    print("Starting RandomizedSearchCV... (this can take a bit)")
    t0 = time()
    search.fit(X_train, y_train)
    print(f"RandomizedSearchCV finished in {time() - t0:.1f}s")
    print("Best params:", search.best_params_)
    best_pipe = search.best_estimator_

    print("Calibrating probabilities (Platt scaling)...")
    calibrated = CalibratedClassifierCV(estimator=best_pipe, method='sigmoid', cv=cv)
    calibrated.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    evaluate_model(calibrated, X_test, y_test)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(calibrated, args.out)
    print("Saved calibrated pipeline to:", args.out)
    print("Done.")

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/clean_sales_data.csv", help="CSV with note_text,label")
    parser.add_argument("--out", type=str, default="models/lead_pipeline.joblib", help="Output pipeline file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--n-iter", type=int, default=8, help="RandomizedSearchCV iterations")
    args = parser.parse_args()
    main(args)
