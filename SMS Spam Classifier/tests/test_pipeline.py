import joblib
import numpy as np
import pandas as pd

from spam_classifier.train import build_pipeline, evaluate


def test_prediction_does_not_learn_test_only_vocabulary():
    pipeline = build_pipeline().fit(
        ["hello friend", "meet tonight", "free prize", "win reward"],
        ["ham", "ham", "spam", "spam"],
    )
    vocabulary_before = dict(pipeline.named_steps["tfidf"].vocabulary_)
    pipeline.predict(["quarantinedtesttoken"])
    assert "quarantinedtesttoken" not in pipeline.named_steps["tfidf"].vocabulary_
    assert pipeline.named_steps["tfidf"].vocabulary_ == vocabulary_before


def test_complete_pipeline_round_trip(model_path, fitted_pipeline):
    restored = joblib.load(model_path)
    messages = ["hello friend", "free prize", "new unseen vocabulary"]
    np.testing.assert_array_equal(restored.predict(messages), fitted_pipeline.predict(messages))
    np.testing.assert_allclose(restored.predict_proba(messages), fitted_pipeline.predict_proba(messages))


def test_metrics_have_explicit_class_order_and_correct_errors():
    result = evaluate(["ham", "ham", "spam", "spam"], ["ham", "spam", "ham", "spam"])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["classification_report"]["spam"]["precision"] == 0.5
    assert result["classification_report"]["spam"]["recall"] == 0.5
    assert result["classification_report"]["spam"]["f1-score"] == 0.5


def test_training_command_fits_only_training_rows(tmp_path, monkeypatch):
    import sys
    from spam_classifier import train
    from spam_classifier.data import clean_data, load_data

    dataset = tmp_path / "sms"
    rows = [("ham", f"hello friend meeting {i}") for i in range(20)]
    rows += [("spam", f"free prize claim {i}") for i in range(10)]
    rows += [rows[0]]  # Exercise deduplication in the actual command as well.
    dataset.write_text("".join(f"{label}\t{text}\n" for label, text in rows), encoding="utf-8")
    cleaned, _, _ = clean_data(load_data(dataset))
    expected_X, _, expected_y, _ = train.split_data(cleaned)

    pipeline = build_pipeline()
    original_fit = type(pipeline).fit
    observed = {}

    def capture_fit(self, X, y):
        observed["X"] = X.copy()
        observed["y"] = y.copy()
        return original_fit(self, X, y)

    monkeypatch.setattr(type(pipeline), "fit", capture_fit)
    monkeypatch.setattr(train, "build_pipeline", lambda: pipeline)
    monkeypatch.setattr(train, "DATA_PATH", dataset)
    monkeypatch.setattr(train, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "artifacts" / "pipeline.joblib")
    monkeypatch.setattr(sys, "argv", ["train"])
    train.main()
    pd.testing.assert_series_equal(observed["X"], expected_X)
    pd.testing.assert_series_equal(observed["y"], expected_y)
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "artifacts" / "pipeline.joblib").is_file()
