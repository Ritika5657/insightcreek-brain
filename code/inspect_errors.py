# inspect_errors.py
"""
Inspect model mistakes.

Usage:
  python code\inspect_errors.py --data data/clean_sales_data.csv --model models/lead_pipeline.joblib --test-size 0.2 --max-show 10

What it does:
- Loads CSV with columns: note_text,label
- Splits into train/test (same defaults as train_model.py)
- Loads saved pipeline (joblib)
- Predicts on test set and prints metrics
- Prints top false positives and false negatives (up to max_show)
- Saves models/misclassified.csv with columns: note_text,label,pred,prob,error_type
"""

import argparse
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from train_model import TextCleaner

def main(args):
    print("Loading data:", args.data)
    df = pd.read_csv(args.data)
    if 'note_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'note_text' and 'label' columns")

    X = df['note_text'].fillna('').astype(str).values
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    print(f"Using test set of {len(X_test)} samples (test_size={args.test_size})")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print("Loading model:", args.model)
    model = joblib.load(args.model)

    print("Predicting...")
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        probs = None

    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if probs is not None else None

    print("\n=== Evaluation on test set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC:   {roc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Build DataFrame for analysis
    df_out = pd.DataFrame({
        "note_text": X_test,
        "label": y_test,
        "pred": preds
    })
    if probs is not None:
        df_out["prob"] = probs
    else:
        df_out["prob"] = None

    # error types
    def error_type(row):
        if row["label"] == row["pred"]:
            return "correct"
        if row["label"] == 1 and row["pred"] == 0:
            return "false_negative"
        return "false_positive"

    df_out["error_type"] = df_out.apply(error_type, axis=1)

    # Print sample false negatives and false positives
    fn = df_out[df_out["error_type"] == "false_negative"].sort_values(by="prob", ascending=True).head(args.max_show)
    fp = df_out[df_out["error_type"] == "false_positive"].sort_values(by="prob", ascending=False).head(args.max_show)

    print(f"\nTop {len(fn)} False Negatives (actual=1, predicted=0):")
    for i, r in fn.iterrows():
        print(f"- prob={r['prob']:.3f} | note: {r['note_text'][:200]}")

    print(f"\nTop {len(fp)} False Positives (actual=0, predicted=1):")
    for i, r in fp.iterrows():
        print(f"- prob={r['prob']:.3f} | note: {r['note_text'][:200]}")

    # Save CSV of misclassified
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved detailed results to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/clean_sales_data.csv")
    parser.add_argument("--model", type=str, default="models/lead_pipeline.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="models/misclassified.csv")
    parser.add_argument("--max-show", type=int, default=10)
    args = parser.parse_args()
    main(args)
