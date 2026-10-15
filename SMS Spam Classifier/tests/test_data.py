import pandas as pd
import pytest

from spam_classifier.data import clean_data, load_data
from spam_classifier.train import split_data


def test_cleaning_excludes_all_conflicts_and_deduplicates_before_split():
    rows = [("ham", " hello "), ("ham", "hello"),
            ("ham", "ambiguous"), ("spam", "ambiguous"),
            (None, "missing label"), ("spam", None), ("ham", "  ")]
    rows += [("ham", f"normal message {i}") for i in range(20)]
    rows += [("spam", f"prize message {i}") for i in range(10)]
    raw = pd.DataFrame(rows, columns=["label", "message"])
    data, audit, conflicts = clean_data(raw)
    assert audit["missing_values"] == {"label": 1, "message": 1}
    assert audit["missing_or_blank_rows_removed"] == 3
    assert audit["conflicting_message_groups"] == 1
    assert audit["conflicting_rows_excluded"] == 2
    assert audit["same_label_duplicate_rows_removed"] == 1
    assert set(conflicts["label"]) == {"ham", "spam"}
    assert "ambiguous" not in set(data["message"])
    assert len(data) == 31
    assert data["message"].is_unique
    first = split_data(data)
    second = split_data(data)
    for left, right in zip(first, second):
        pd.testing.assert_series_equal(left, right)
    X_train, X_test, y_train, y_test = first
    assert not set(X_train) & set(X_test)
    assert set(y_train) == set(y_test) == {"ham", "spam"}
    assert abs((y_train == "spam").mean() - (y_test == "spam").mean()) < 0.15


def test_unknown_labels_fail_instead_of_becoming_ham():
    raw = pd.DataFrame({"label": ["unknown"], "message": ["hello"]})
    with pytest.raises(ValueError, match="Unexpected labels"):
        clean_data(raw)


def test_loader_preserves_quotation_marks_and_internal_tabs(tmp_path):
    path = tmp_path / "sms"
    path.write_text('ham\tSay "hello"\tfriend\nspam\tWin a prize\n', encoding="utf-8")
    data = load_data(path)
    assert data.iloc[0]["message"] == 'Say "hello"\tfriend'


def test_malformed_rows_are_flagged(tmp_path):
    path = tmp_path / "sms"
    path.write_text("missing tab\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 1"):
        load_data(path)

