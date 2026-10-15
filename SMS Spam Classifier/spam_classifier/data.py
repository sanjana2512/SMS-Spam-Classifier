"""Inspect and clean labels/messages before any random split."""

from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python -m spam_classifier.download")
    # Split only at the first tab, preserving tabs and quotation marks in SMS text.
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        label, separator, message = line.partition("\t")
        if not separator:
            raise ValueError(f"Malformed dataset row {line_number}: expected label<TAB>message.")
        rows.append({"label": label, "message": message})
    return pd.DataFrame(rows, columns=["label", "message"])


def class_balance(labels: pd.Series) -> dict:
    counts = labels.value_counts().sort_index()
    total = len(labels)
    return {
        str(label): {"count": int(count), "fraction": float(count / total)}
        for label, count in counts.items()
    }


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Drop missing/blank rows, quarantine conflicts, deduplicate by message.

    Message identity ignores surrounding whitespace only. Case, punctuation,
    and internal spacing remain unchanged. This is not near-duplicate detection.
    """
    missing = raw[["label", "message"]].isna().sum().astype(int).to_dict()
    stripped = raw[["label", "message"]].fillna("").astype(str)
    stripped = stripped.apply(lambda column: column.str.strip())
    blank = stripped.eq("").sum().astype(int).to_dict()
    valid = stripped.loc[stripped.ne("").all(axis=1)].copy()
    invalid_labels = sorted(set(valid["label"]) - {"ham", "spam"})
    if invalid_labels:
        raise ValueError(f"Unexpected labels: {invalid_labels}")

    label_counts = valid.groupby("message")["label"].nunique()
    conflicting_messages = label_counts[label_counts > 1].index
    conflicts = valid.loc[valid["message"].isin(conflicting_messages)].copy()
    unambiguous = valid.loc[~valid["message"].isin(conflicting_messages)]
    cleaned = unambiguous.drop_duplicates(subset="message").reset_index(drop=True)

    audit = {
        "raw_rows": len(raw),
        "missing_values": missing,
        "blank_values_after_strip_including_missing": blank,
        "missing_or_blank_rows_removed": len(raw) - len(valid),
        "raw_class_balance": class_balance(raw["label"].dropna()),
        "valid_class_balance_before_deduplication": class_balance(valid["label"]),
        "duplicate_message_groups": int((valid["message"].value_counts() > 1).sum()),
        "duplicate_message_rows_beyond_first": int(valid.duplicated("message").sum()),
        "conflicting_message_groups": len(conflicting_messages),
        "conflicting_rows_excluded": len(conflicts),
        "same_label_duplicate_rows_removed": len(unambiguous) - len(cleaned),
        "clean_rows": len(cleaned),
        "clean_class_balance": class_balance(cleaned["label"]),
        "identity_rule": "Strip surrounding whitespace only; preserve case and internal text.",
        "conflict_policy": "Exclude all rows for any message with both labels; flag in conflicts.csv.",
    }
    return cleaned, audit, conflicts

