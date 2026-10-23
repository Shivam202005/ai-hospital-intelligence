
import os

import requests

API_BASE_URL = os.getenv(
    "READMISSION_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

TIMEOUT_SECONDS = 60


class APIClientError(Exception):
    """Raised when the prediction API request fails."""


def check_backend():
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIClientError(
            f"Could not connect to the backend: {exc}"
        ) from exc


def predict_patient(patient, model="xgboost", threshold=0.15):
    payload = {
        "patient": patient,
        "model": model,
        "threshold": threshold,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text

            raise APIClientError(
                f"Prediction request failed "
                f"(HTTP {response.status_code}): {detail}"
            )

        return response.json()

    except requests.Timeout as exc:
        raise APIClientError(
            "The backend request timed out. Please try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise APIClientError(
            "Cannot reach FastAPI. Check that the backend is running."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError(
            f"API request failed: {exc}"
        ) from exc
