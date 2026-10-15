"""Paths relative to the project, independent of the shell's working folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "SMSSpamCollection"
MODEL_PATH = ROOT / "artifacts" / "spam_pipeline.joblib"
REPORT_DIR = ROOT / "reports"

