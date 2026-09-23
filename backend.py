"""Model inference adapter for the Vercel Health Condition Prediction app."""
from __future__ import annotations

import pickle
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.random import _pickle as numpy_random_pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path("/tmp/model.pkl")
SCALER_PATH = Path("/tmp/scaler.pkl")

MODEL_URL = "https://media.githubusercontent.com/media/Mohd-Faizaan/health-condition-prediction/main/model.pkl"
SCALER_URL = "https://media.githubusercontent.com/media/Mohd-Faizaan/health-condition-prediction/main/scaler.pkl"

FEATURE_ORDER = (
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake", "diet_type",
    "stress_level", "sleep_quality", "physical_activity_level",
    "smoking_alcohol", "gender",
)
NUMERIC_FEATURES = (
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
)
CATEGORY_ENCODINGS = {
    "diet_type": {"unknown": 0, "balanced": 1, "non-veg": 2, "veg": 3},
    "stress_level": {"unknown": 0, "high": 1, "low": 2, "medium": 3},
    "sleep_quality": {"unknown": 0, "average": 1, "good": 2, "poor": 3},
    "physical_activity_level": {"unknown": 0, "active": 1, "moderate": 2, "sedentary": 3},
    "smoking_alcohol": {"unknown": 0, "no": 1, "occasional": 2, "yes": 3},
    "gender": {"unknown": 0, "female": 1, "male": 2, "other": 3},
}

def _download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "health-condition-prediction-vercel/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=240) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

def _ensure_artifacts() -> None:
    _download_if_missing(MODEL_PATH, MODEL_URL)
    _download_if_missing(SCALER_PATH, SCALER_URL)

@lru_cache(maxsize=1)
def _load_model() -> Any:
    _ensure_artifacts()
    original_constructor = numpy_random_pickle.__bit_generator_ctor
    def compatible_bit_generator_constructor(bit_generator: Any = "MT19937") -> Any:
        if isinstance(bit_generator, type):
            return bit_generator()
        return original_constructor(bit_generator)
    numpy_random_pickle.__bit_generator_ctor = compatible_bit_generator_constructor
    try:
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
    finally:
        numpy_random_pickle.__bit_generator_ctor = original_constructor
    actual_features = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    if actual_features != list(FEATURE_ORDER):
        raise ValueError("The model uses an incompatible feature order.")
    return model

@lru_cache(maxsize=1)
def _load_scaler() -> Any | None:
    _ensure_artifacts()
    with SCALER_PATH.open("rb") as scaler_file:
        return pickle.load(scaler_file)

def _apply_scaler(frame: pd.DataFrame) -> pd.DataFrame:
    scaler = _load_scaler()
    if scaler is None:
        return frame
    scaled = frame.copy()
    scaler_features = list(getattr(scaler, "feature_names_in_", ()))
    feature_count = int(getattr(scaler, "n_features_in_", 0) or 0)
    if scaler_features:
        missing = [feature for feature in scaler_features if feature not in scaled.columns]
        if missing:
            raise ValueError(f"The scaler expects missing features: {', '.join(missing)}.")
        columns = scaler_features
    elif feature_count == len(FEATURE_ORDER):
        columns = list(FEATURE_ORDER)
    elif feature_count == len(NUMERIC_FEATURES):
        columns = list(NUMERIC_FEATURES)
    else:
        raise ValueError("The scaler uses an incompatible feature count.")
    scaled.loc[:, columns] = scaler.transform(scaled[columns])
    return scaled

def _normalize_prediction(prediction: Any) -> str:
    value = str(prediction).strip().casefold()
    aliases = {
        "healthy": "fit", "fit": "fit",
        "at risk": "at-risk", "at-risk": "at-risk", "risk": "at-risk",
        "unhealthy": "unhealthy", "not healthy": "unhealthy",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported model prediction: {prediction!r}.")
    return aliases[value]

def predict_health(input_data: dict[str, Any]) -> str:
    missing = [feature for feature in FEATURE_ORDER if feature not in input_data]
    if missing:
        raise ValueError(f"Missing input features: {', '.join(missing)}.")
    prepared = {feature: input_data[feature] for feature in FEATURE_ORDER}
    for feature, encoding in CATEGORY_ENCODINGS.items():
        value = str(prepared[feature]).strip().casefold()
        if value not in encoding:
            allowed = ", ".join(key.title() for key in encoding if key != "unknown")
            raise ValueError(f"Invalid {feature.replace('_', ' ').title()}. Use one of: {allowed}.")
        prepared[feature] = encoding[value]
    frame = pd.DataFrame([prepared], columns=FEATURE_ORDER, dtype=float)
    prediction = _load_model().predict(_apply_scaler(frame))[0]
    return _normalize_prediction(prediction)
