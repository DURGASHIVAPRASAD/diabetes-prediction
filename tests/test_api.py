
from fastapi.testclient import TestClient

from app1 import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()


def test_prediction_endpoint():
    payload = {
        "pregnancies": 2,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 80,
        "bmi": 28,
        "diabetes_pedigree_function": 0.5,
        "age": 35,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] in [0, 1]
    assert 0 <= result[
        "estimated_positive_class_probability"
    ] <= 1
