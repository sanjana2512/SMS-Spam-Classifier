"""Train one predetermined pipeline; evaluate it and a baseline on one split."""

import argparse
import hashlib
import json
import platform
import warnings
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spam_classifier.data import class_balance, clean_data, load_data
from spam_classifier.paths import DATA_PATH, MODEL_PATH, REPORT_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.20
LABELS = ["ham", "spam"]


def build_pipeline() -> Pipeline:
    # Fixed before evaluation. No search or test-driven parameter choices.
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                                          random_state=RANDOM_STATE)),
    ])


def split_data(data: pd.DataFrame):
    return train_test_split(
        data["message"], data["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=data["label"],
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "matrix_labels": LABELS,
        "matrix_orientation": "Rows = actual; columns = predicted.",
    }


def write_summary(report: dict):
    lines = ["# Evaluation from the actual run", "",
             "Fixed 80/20 stratified split, seed 42. No test-set tuning.", "",
             "Spam is the positive class. Metrics below are for spam.", "",
             "| Model | Precision | Recall | F1 | Accuracy |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in report["models"].items():
        metrics = result["classification_report"]
        spam = metrics["spam"]
        lines.append(f"| {name} | {spam['precision']:.4f} | {spam['recall']:.4f} | "
                     f"{spam['f1-score']:.4f} | {metrics['accuracy']:.4f} |")
    for name, result in report["models"].items():
        matrix = result["confusion_matrix"]
        lines.extend(["", f"## {name}: confusion matrix", "",
                      "Rows = actual, columns = predicted. Order: ham, spam.", "",
                      "| | Predicted ham | Predicted spam |",
                      "| --- | ---: | ---: |",
                      f"| Actual ham | {matrix[0][0]} | {matrix[0][1]} |",
                      f"| Actual spam | {matrix[1][0]} | {matrix[1][1]} |"])
    lines.extend(["", "Undefined precision is reported as 0 when a model predicts no spam.",
                  "See metrics.json for both classes, macro/weighted averages, and split details.", ""])
    (REPORT_DIR / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true", help="Write the data audit without training.")
    args = parser.parse_args()
    raw = load_data(DATA_PATH)
    data, audit, conflicts = clean_data(raw)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    conflicts.to_csv(REPORT_DIR / "conflicts.csv", index=False)
    print("Stage 1: data audit")
    print(json.dumps(audit, indent=2))
    if args.inspect_only:
        return
    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 5:
        raise ValueError("Need both classes and at least five unique messages per class.")

    X_train, X_test, y_train, y_test = split_data(data)
    if set(X_train) & set(X_test):
        raise RuntimeError("A duplicate message crossed the split.")

    print("Stage 2: fit TF-IDF and logistic regression on training messages only")
    pipeline = build_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    # Learn the most common label from y_train, not from the full dataset.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train.to_frame(), y_train)
    report = {
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in ["scikit-learn", "pandas", "numpy", "joblib"]}},
        "split": {"seed": RANDOM_STATE, "test_fraction": TEST_SIZE,
                  "stratified": True, "train_rows": len(X_train), "test_rows": len(X_test),
                  "train_balance": class_balance(y_train), "test_balance": class_balance(y_test),
                  "train_clean_row_indices": X_train.index.tolist(),
                  "test_clean_row_indices": X_test.index.tolist()},
        "parameters": {"tfidf": "Scikit-learn defaults", "C": 1.0,
                       "solver": "liblinear", "max_iter": 1000, "class_weight": None,
                       "decision_threshold": 0.5},
        "test_policy": "One fixed model and baseline; no hyperparameter or threshold tuning on test data.",
        "models": {
            "majority_baseline": evaluate(y_test, baseline.predict(X_test.to_frame())),
            "tfidf_logistic_regression": evaluate(y_test, pipeline.predict(X_test)),
        },
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Persist the fitted vectorizer AND classifier together. Do not refit on test rows.
    joblib.dump(pipeline, MODEL_PATH)
    print("Stage 3: held-out evaluation")
    print((REPORT_DIR / "evaluation.md").read_text(encoding="utf-8"))
    print(f"Saved complete pipeline: {MODEL_PATH}")


if __name__ == "__main__":
    main()
