import os
import time
from typing import Annotated, List

import mlflow
import requests
import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import GITHUB_REPOSITORY, PROD_ALIAS, REGISTERED_MODEL_NAME

app = FastAPI()


# define models for prediction endpoint
class PredictRequest(BaseModel):
    passwords: List[str] = Field(alias="Password")


class PredictResponse(BaseModel):
    prediction: List[float] = Field(alias="Times")


# define model cache and dependency for FastAPI
_model = None


def get_model():
    global _model
    if _model is None:
        _reload_model()
    return _model


@app.post("/predict")
def predict(
    data: Annotated[PredictRequest, Body()], model=Depends(get_model)
) -> PredictResponse:
    """Prediction endpoint."""
    prediction = model.predict(data.passwords)
    return PredictResponse(Times=prediction)


admin_auth = HTTPBearer(auto_error=False)


def verify_admin_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(admin_auth),
    ],
):
    """Verify access to model administration endpoints."""
    expected_key = os.environ.get("MODEL_ADMIN_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Model administration API is not configured",
        )

    if credentials is None or credentials.credentials != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def _reload_model():
    """Load the production model from MLflow."""
    global _model

    start_time = time.time()
    print("Loading production model from MLflow...")

    _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PROD_ALIAS}")

    duration = time.time() - start_time
    print(f"Model loaded successfully in {duration:.2f} seconds!")


@app.post("/reload-model", dependencies=[Depends(verify_admin_api_key)])
def reload_model():
    """Model hot reload endpoint."""
    _reload_model()


# define model for trigger endpoint
class Trigger(BaseModel):
    data_url: str


@app.post("/trigger", dependencies=[Depends(verify_admin_api_key)])
def trigger_pipeline(data: Annotated[Trigger, Body()]):
    """Trigger the model training workflow asynchronously."""
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise HTTPException(
            status_code=503,
            detail="GitHub workflow trigger is not configured",
        )

    url = (
        f"https://api.github.com/repos/{GITHUB_REPOSITORY}"
        "/actions/workflows/train.yml/dispatches"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "ref": "main",
            "inputs": {
                "data_url": data.data_url,
            },
        },
        timeout=10,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to trigger GitHub training workflow",
        ) from exc


@app.get("/health")
def health():
    return "OK"


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == "__main__":
    main()
