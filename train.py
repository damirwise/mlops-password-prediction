"""Validate training data, train the model, and register it in MLflow."""

from pathlib import Path
import sys

import mlflow
import pandas as pd
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from config import PROD_ALIAS, REGISTERED_MODEL_NAME

EXPERIMENT_NAME = "password-frequency-training"
REFERENCE_DATA_PATH = Path(__file__).parent / "data" / "reference.csv"


def get_data():
    if len(sys.argv) < 2:
        raise ValueError("Provide a path to the training data.")
    path = sys.argv[1]
    return pd.read_csv(path)


def get_model(params: dict | None):
    params = params or {"model": "rf", "ngrams": {"min": 1, "max": 3}}
    if params["model"] == "linear":
        model = LinearRegression()
    elif params["model"] == "rf":
        model = RandomForestRegressor(random_state=42)
    elif params["model"] == "ridge":
        model = Ridge()
    else:
        raise ValueError(f"Unsupported model type: {params['model']}")

    # Create the pipeline
    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    ngram_range=(params["ngrams"]["min"], params["ngrams"]["max"]),
                    analyzer="char",
                ),
            ),
            ("tfidf", TfidfTransformer()),
            ("clf", model),
        ]
    )
    return pipeline


def check_data(df: pd.DataFrame) -> bool:
    """Check that data has not drifted."""
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    suite = TestSuite(tests=[DataDriftTestPreset(columns=["Times"])])
    suite.run(reference_data=reference, current_data=df)
    return bool(suite)


def main():
    data = get_data()
    if not check_data(data):
        print("Data is bad")
        sys.exit(1)

    # create sklearn pipeline and train it
    params = {"model": "rf", "ngrams": {"min": 1, "max": 3}}
    model = get_model(params)
    model.fit(data["Password"], data["Times"])

    # load or create mlflow experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(
            {
                "model": params["model"],
                "ngram_min": params["ngrams"]["min"],
                "ngram_max": params["ngrams"]["max"],
            }
        )

        # log new model to mlflow
        model = mlflow.sklearn.log_model(model, artifact_path="model")
        # register model version
        reg_model = mlflow.register_model(
            model_uri=model.model_uri, name=REGISTERED_MODEL_NAME
        )
        # set prod alias for new version
        client = MlflowClient()
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PROD_ALIAS, reg_model.version
        )


if __name__ == "__main__":
    main()
