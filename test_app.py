from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from app_model import app, get_model


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == "OK"


def test_predict(client: TestClient):
    class FakeModel:
        def predict(self, passwords):
            return [1.0, 2.0]

    app.dependency_overrides[get_model] = lambda: FakeModel()

    try:
        response = client.post(
            "/predict",
            json={"Password": ["password123", "hello"]},
        )

        assert response.status_code == 200
        assert response.json() == {"Times": [1.0, 2.0]}
    finally:
        app.dependency_overrides.clear()


def test_trigger(client: TestClient, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test")
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "test-admin-key")

    with patch("app_model.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None

        response = client.post(
            "/trigger",
            headers={"Authorization": "Bearer test-admin-key"},
            json={"data_url": "https://example.com/data.csv"},
        )

        assert response.status_code == 200

        mock_post.assert_called_once_with(
            "https://api.github.com/repos/damirwise/mlops-password-prediction/actions/workflows/train.yml/dispatches",
            headers={
                "Authorization": "Bearer test",
                "Accept": "application/vnd.github+json",
            },
            json={
                "ref": "main",
                "inputs": {
                    "data_url": "https://example.com/data.csv",
                },
            },
            timeout=10,
        )


def test_trigger_without_github_token(client: TestClient, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "test-admin-key")

    response = client.post(
        "/trigger",
        headers={"Authorization": "Bearer test-admin-key"},
        json={"data_url": "https://example.com/data.csv"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "GitHub workflow trigger is not configured"}


def test_trigger_when_github_api_fails(client: TestClient, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test")
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "test-admin-key")

    with patch("app_model.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError()

        response = client.post(
            "/trigger",
            headers={"Authorization": "Bearer test-admin-key"},
            json={"data_url": "https://example.com/data.csv"},
        )

        assert response.status_code == 502
        assert response.json() == {
            "detail": "Failed to trigger GitHub training workflow"
        }


def test_trigger_without_admin_api_key(client: TestClient, monkeypatch):
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "test-admin-key")

    response = client.post(
        "/trigger",
        json={"data_url": "https://example.com/data.csv"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_reload_model_without_admin_api_key(client: TestClient, monkeypatch):
    monkeypatch.setenv("MODEL_ADMIN_API_KEY", "test-admin-key")

    response = client.post("/reload-model")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
