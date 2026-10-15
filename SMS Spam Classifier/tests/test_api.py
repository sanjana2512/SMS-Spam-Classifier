import numpy as np
import pytest
from fastapi.testclient import TestClient

from spam_classifier.api import create_app


@pytest.fixture
def client(model_path):
    # The context manager runs startup so this tests actual joblib loading too.
    with TestClient(create_app(model_path)) as client:
        yield client


def test_prediction_matches_loaded_pipeline(client, fitted_pipeline):
    response = client.post("/predict", json={"message": "free prize claim now"})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == fitted_pipeline.predict(["free prize claim now"])[0]
    index = list(fitted_pipeline.classes_).index("spam")
    assert np.isclose(result["spam_probability"], fitted_pipeline.predict_proba(["free prize claim now"])[0][index])
    assert 0 <= result["spam_probability"] <= 1


@pytest.mark.parametrize("payload", [
    {}, {"message": ""}, {"message": " \t\n"}, {"message": None},
    {"message": 123}, {"message": ["hello"]}, {"message": "x" * 5001},
    {"message": "hello", "extra": "unexpected"},
])
def test_invalid_inputs_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_boundary_length_and_unicode_are_valid(client):
    for message in ["x", "x" * 5000, "Hello 👋 नमस्ते"]:
        assert client.post("/predict", json={"message": message}).status_code == 200


def test_interface_and_api_docs_load(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "fetch('/predict'" in page.text
    assert client.get("/docs").status_code == 200


def test_malformed_json_is_rejected(client):
    response = client.post("/predict", content="{", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_missing_model_has_actionable_startup_error(tmp_path):
    with pytest.raises(RuntimeError, match="python -m spam_classifier.train"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass

