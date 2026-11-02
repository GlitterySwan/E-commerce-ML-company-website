"""Shared feature preparation and inference helpers for the ANN demand model."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "YEAR",
    "MONTH",
    "WAREHOUSE SALES",
    "RETAIL TRANSFERS",
]
CATEGORICAL_FEATURES = ["ITEM CODE", "ITEM TYPE", "SUPPLIER"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_ann_features(frame):
    """Return model features with consistent types and missing-value handling."""
    features = frame.reindex(columns=FEATURE_COLUMNS).copy()

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("UNKNOWN").astype(str).str.strip()
        features.loc[features[column] == "", column] = "UNKNOWN"

    return features


def create_ann_pipeline():
    """Create the scaled/encoded multilayer-perceptron regression pipeline."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    dtype=np.float32,
                ),
            )
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(48, 24),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=512,
                    learning_rate_init=0.001,
                    max_iter=50,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def predict_retail_sales(pipeline, features):
    """Predict non-negative retail sales from a pipeline trained on log1p sales."""
    log_sales = pipeline.predict(prepare_ann_features(features))
    return np.maximum(0.0, np.expm1(log_sales))
