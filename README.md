# Password Frequency Prediction — MLOps Pipeline

An end-to-end MLOps project for predicting password usage frequency, with automated data validation, model retraining, experiment tracking, model versioning, CI/CD deployment, and model hot reloading.

## Overview

The project exposes a FastAPI inference service backed by a scikit-learn model registered in MLflow.

Retraining can be triggered through the API by providing a URL to a new training dataset. A GitHub Actions workflow downloads and validates the data, retrains the model, logs the training configuration to MLflow, registers a new model version, assigns the `prod` alias, and instructs the running service to reload the production model.

MLflow tracking and the Model Registry are hosted on DagsHub. Application images are built by GitHub Actions, stored on Docker Hub, and deployed to Render.

**Repository:** [damirwise/mlops-password-prediction](https://github.com/damirwise/mlops-password-prediction)

**Live API:** [mlops-password-prediction.onrender.com](https://mlops-password-prediction.onrender.com)

## Architecture

```text
Model retraining:

FastAPI: POST /trigger (dataset URL)
                 |
                 v
GitHub Actions: download -> Evidently validation -> train
                 |
                 v
DagsHub / MLflow: training parameters -> model registry -> prod alias
                 |
                 v
FastAPI: POST /reload-model -> load production model

Application CI/CD:

GitHub push -> CI -> Docker build -> Docker Hub -> Render

FastAPI endpoints:
  POST /predict
  POST /trigger
  POST /reload-model
  GET  /health
```

## Tech Stack

- Python 3.10
- FastAPI
- scikit-learn
- MLflow
- DagsHub
- Evidently
- Docker
- Docker Hub
- GitHub Actions
- Render
- pytest
- Ruff

## Model

Passwords are converted into character-level n-gram features using `CountVectorizer`, followed by TF-IDF transformation.

The default estimator is a `RandomForestRegressor` with `random_state=42` for reproducible training.

Feature extraction, transformation, and regression are combined into a scikit-learn `Pipeline`.

## Data Validation

Before retraining, incoming data is compared with a reference dataset using Evidently.

The pipeline checks for distribution drift in the target variable (`Times`). Training is stopped if validation fails.

The reference dataset is stored in:

```text
data/reference.csv
```

## Automated Retraining

Retraining is initiated through:

```text
POST /trigger
```

The endpoint dispatches the `Train Model` GitHub Actions workflow with a URL pointing to the training dataset.

The workflow:

1. Downloads the dataset.
2. Validates it against the reference data with Evidently.
3. Trains the scikit-learn pipeline.
4. Logs training parameters and the model to MLflow.
5. Registers a new model version.
6. Assigns the `prod` alias to the new version.
7. Calls `/reload-model` on the deployed API.

This allows the running inference service to load the newly promoted model without requiring an application redeployment.

## API

### Health Check

```text
GET /health
```

### Prediction

```text
POST /predict
```

Example request:

```json
{
  "Password": [
    "password123",
    "hello"
  ]
}
```

Illustrative response:

```json
{
  "Times": [
    1.0,
    2.0
  ]
}
```

Prediction values above are examples only, not measured predictions.

### Trigger Retraining

```text
POST /trigger
```

Example request:

```json
{
  "data_url": "https://example.com/training-data.csv"
}
```

Replace the placeholder URL with your dataset URL. The dataset URL must be accessible to the GitHub Actions runner.

### Reload Production Model

```text
POST /reload-model
```

Reloads the model currently assigned to the MLflow `prod` alias.

### Authentication

The model administration endpoints, `POST /trigger` and `POST /reload-model`, are protected with HTTP Bearer authentication. Include the administration API key in the request header:

```http
Authorization: Bearer <ADMIN_API_KEY>
```

Replace `<ADMIN_API_KEY>` with your configured administration API key.

`GET /health` and `POST /predict` remain publicly accessible and do not require authentication.

The administration API key is supplied through environment variables and GitHub Actions Secrets and is not stored in the repository. The training workflow uses this key when calling `/reload-model`.

## CI/CD

### Continuous Integration

On every push and pull request, GitHub Actions:

- installs project dependencies;
- runs the pytest test suite;
- checks Python formatting with Ruff.

### Application Deployment

After a successful CI run on `main`, GitHub Actions:

1. Builds the application Docker image.
2. Pushes the image to Docker Hub.
3. Calls the Render deployment hook.
4. Render pulls and deploys the updated image.

The deployment workflow can also be started manually to rebuild the Docker image.

### Model Training

Model retraining is implemented as a separate GitHub Actions workflow and is independent of application deployment.

The FastAPI `/trigger` endpoint dispatches this workflow through the GitHub API. After successful validation and training, the new model is registered in MLflow and the running service reloads the model assigned to the `prod` alias.

## Testing

Run the tests with:

```bash
pytest
```

The suite contains seven unit tests covering:

1. The health endpoint.
2. Prediction using a mocked model.
3. Successful GitHub Actions workflow dispatch.
4. Behavior when the GitHub token is not configured.
5. Behavior when the GitHub API returns an error.
6. Authentication protection for the retraining endpoint.
7. Authentication protection for the model reload endpoint.

External MLflow and GitHub credentials are not required to run the unit tests.

## Configuration

Runtime configuration and credentials are supplied through environment variables, GitHub Actions Secrets, and GitHub Actions Variables.

The application and CI/CD pipelines support configuration for:

- the MLflow tracking server and authentication;
- the registered model name and production alias;
- the GitHub repository and API token;
- the administration API key for protected endpoints;
- the production model reload endpoint;
- the Render deployment hook;
- Docker Hub authentication.

Secrets are not stored in the repository.

See `config.py` and the workflows in `.github/workflows/` for the exact configuration used by the application and CI/CD pipelines.
