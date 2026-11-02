"""Train and save the ANN retail-demand model using Retail_sales.csv."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ann_model import create_ann_pipeline, prepare_ann_features

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "Retail_sales.csv"
DEFAULT_OUTPUT = ROOT / "ann_model.pkl"


def train_model(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    frame.columns = frame.columns.str.strip()
    required_columns = {
        "YEAR",
        "MONTH",
        "SUPPLIER",
        "ITEM CODE",
        "ITEM TYPE",
        "RETAIL SALES",
        "RETAIL TRANSFERS",
        "WAREHOUSE SALES",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    target = pd.to_numeric(
        frame["RETAIL SALES"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    valid_rows = target.notna() & np.isfinite(target) & (target >= 0)
    frame = frame.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 valid retail-sales rows are required to train.")

    features = prepare_ann_features(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = create_ann_pipeline()
    pipeline.fit(X_train, np.log1p(y_train))

    predicted_sales = np.maximum(0.0, np.expm1(pipeline.predict(X_test)))
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted_sales))),
        "mae": float(mean_absolute_error(y_test, predicted_sales)),
        "r2": float(r2_score(y_test, predicted_sales)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": features.columns.tolist(),
        "target": "RETAIL SALES",
        "target_transform": "log1p",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(json.dumps({"model_path": str(output_path), "metrics": metrics}, indent=2))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train_model(args.dataset, args.output)


if __name__ == "__main__":
    main()
