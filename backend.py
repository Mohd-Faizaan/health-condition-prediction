"""Inference adapter for the trained health-condition model."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.random import _pickle as numpy_random_pickle


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (BASE_DIR / "model.pkl", BASE_DIR / "model (1).pkl")
SCALER_CANDIDATES = (BASE_DIR / "scaler.pkl", BASE_DIR / "scaler (1).pkl")
TARGET_ENCODER_CANDIDATES = (BASE_DIR / "target_encoder.pkl", BASE_DIR / "label_encoder.pkl")

FEATURE_ORDER = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

NUMERIC_FEATURES = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORY_ENCODINGS = {
    "diet_type": {"unknown": 0, "balanced": 1, "non-veg": 2, "veg": 3},
    "stress_level": {"unknown": 0, "high": 1, "low": 2, "medium": 3},
    "sleep_quality": {"unknown": 0, "average": 1, "good": 2, "poor": 3},
    "physical_activity_level": {
        "unknown": 0,
        "active": 1,
        "moderate": 2,
        "sedentary": 3,
    },
    "smoking_alcohol": {"unknown": 0, "no": 1, "occasional": 2, "yes": 3},
    "gender": {"unknown": 0, "female": 1, "male": 2, "other": 3},
}


def _model_path() -> Path:
    """Return the first uploaded model filename available in the project."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Place model.pkl beside backend.py.")


def _scaler_path() -> Path | None:
    """Return the first uploaded scaler filename available in the project."""
    for path in SCALER_CANDIDATES:
        if path.exists():
            return path
    return None


def _target_encoder_path() -> Path | None:
    """Return an optional target encoder for numeric model outputs."""
    for path in TARGET_ENCODER_CANDIDATES:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the trusted local artifact once per Streamlit process."""
    # Compatibility for artifacts created by NumPy versions that serialized
    # the BitGenerator class instead of its registered string name.
    original_constructor = numpy_random_pickle.__bit_generator_ctor

    def compatible_bit_generator_constructor(bit_generator: Any = "MT19937") -> Any:
        if isinstance(bit_generator, type):
            return bit_generator()
        return original_constructor(bit_generator)

    numpy_random_pickle.__bit_generator_ctor = compatible_bit_generator_constructor
    with _model_path().open("rb") as model_file:
        try:
            model = pickle.load(model_file)
        finally:
            numpy_random_pickle.__bit_generator_ctor = original_constructor

    actual_features = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    if actual_features != list(FEATURE_ORDER):
        raise ValueError("The model uses an incompatible feature order.")
    return model


@lru_cache(maxsize=1)
def _load_scaler() -> Any | None:
    """Load the optional local scaler once per Streamlit process."""
    path = _scaler_path()
    if path is None:
        return None
    with path.open("rb") as scaler_file:
        return pickle.load(scaler_file)


@lru_cache(maxsize=1)
def _load_target_encoder() -> Any | None:
    """Load an optional target encoder once per Streamlit process."""
    path = _target_encoder_path()
    if path is None:
        return None
    with path.open("rb") as encoder_file:
        return pickle.load(encoder_file)


def _apply_scaler(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply scaler.pkl whether it was fitted on all features or numerics only."""
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

    transformed = scaler.transform(scaled[columns])
    scaled.loc[:, columns] = transformed
    return scaled


def _normalize_prediction(prediction: Any) -> str:
    """Map common numeric and text labels to frontend result keys."""
    encoder = _load_target_encoder()
    if encoder is not None:
        try:
            prediction = encoder.inverse_transform([prediction])[0]
        except Exception as exc:
            raise ValueError("target_encoder.pkl could not decode the model prediction.") from exc

    value = str(prediction).strip().casefold()
    aliases = {
        "healthy": "fit",
        "fit": "fit",
        "at risk": "at-risk",
        "at-risk": "at-risk",
        "risk": "at-risk",
        "unhealthy": "unhealthy",
        "not healthy": "unhealthy",
    }
    if value not in aliases:
        if value in {"0", "1", "2"}:
            raise ValueError(
                "The model returned a numeric class label. Add the target label mapping "
                "or a target_encoder.pkl file so 0/1/2 can be mapped correctly."
            )
        raise ValueError(f"Unsupported model prediction: {prediction!r}.")
    return aliases[value]


def predict_health(input_data: dict[str, Any]) -> str:
    """Validate, encode, and predict one health-condition record."""
    missing = [feature for feature in FEATURE_ORDER if feature not in input_data]
    if missing:
        raise ValueError(f"Missing input features: {', '.join(missing)}.")

    prepared = {feature: input_data[feature] for feature in FEATURE_ORDER}
    for feature, encoding in CATEGORY_ENCODINGS.items():
        value = str(prepared[feature]).strip().casefold()
        if value not in encoding:
            allowed = ", ".join(key.title() for key in encoding if key != "unknown")
            label = feature.replace("_", " ").title()
            raise ValueError(f"Invalid {label}. Use one of: {allowed}.")
        prepared[feature] = encoding[value]

    frame = pd.DataFrame([prepared], columns=FEATURE_ORDER, dtype=float)
    prediction_frame = _apply_scaler(frame)
    prediction = _load_model().predict(prediction_frame)[0]
    return _normalize_prediction(prediction)
