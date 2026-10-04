"""Application configuration."""

import os

PROD_ALIAS = os.getenv("PROD_ALIAS", "prod")
REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "password-frequency-model",
)

GITHUB_REPOSITORY = os.getenv(
    "GITHUB_REPOSITORY",
    "damirwise/mlops-password-prediction",
)
