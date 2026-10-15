"""FastAPI routes; load the trusted local pipeline once during startup."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from spam_classifier.paths import MODEL_PATH, ROOT


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(strict=True, min_length=1, max_length=5000,
                         description="SMS text, 1 to 5,000 characters; must contain non-whitespace.")

    @field_validator("message")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value.strip()


class PredictResponse(BaseModel):
    label: Literal["ham", "spam"]
    spam_probability: float = Field(ge=0, le=1)


def create_app(model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path if model_path is not None else Path(os.getenv("SMS_MODEL_PATH", str(MODEL_PATH)))
        if not path.is_file():
            raise RuntimeError("Model missing. Run: python -m spam_classifier.train")
        # joblib artifacts can execute code: only load files you created/trust.
        app.state.pipeline = joblib.load(path)
        yield
        del app.state.pipeline

    app = FastAPI(title="SMS Spam Classifier", version="1.0.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def interface():
        return FileResponse(ROOT / "static" / "index.html")

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        pipeline = app.state.pipeline
        # Do not assume that spam is always probability column 1.
        spam_index = list(pipeline.classes_).index("spam")
        probability = float(pipeline.predict_proba([payload.message])[0][spam_index])
        label = str(pipeline.predict([payload.message])[0])
        return PredictResponse(label=label, spam_probability=probability)

    return app


app = create_app()

