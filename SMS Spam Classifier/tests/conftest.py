import joblib
import pytest

from spam_classifier.train import build_pipeline


@pytest.fixture(scope="session")
def fitted_pipeline():
    # Tiny synthetic examples test behavior without depending on UCI or the network.
    messages = [
        "meet me at home", "hello friend how are you", "dinner tonight at home",
        "please call mum", "free prize claim now", "win cash prize now",
        "urgent free cash offer", "claim your free reward",
    ]
    labels = ["ham"] * 4 + ["spam"] * 4
    return build_pipeline().fit(messages, labels)


@pytest.fixture
def model_path(tmp_path, fitted_pipeline):
    path = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, path)
    return path

