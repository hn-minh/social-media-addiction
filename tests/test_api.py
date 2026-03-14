import pytest
from fastapi.testclient import TestClient
from api.main import app

sample_payload = {
    "Age": 22,
    "Gender": "Male",
    "Academic_Level": "Undergraduate",
    "Country": "Vietnam",
    "Avg_Daily_Usage_Hours": 6.5,
    "Most_Used_Platform": "TikTok",
    "Affects_Academic_Performance": True,
    "Sleep_Hours_Per_Night": 5.0,
    "Mental_Health_Score": 3,
    "Relationship_Status": "Single",
    "Conflicts_Over_Social_Media": 2
}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_predict_addiction_level(client):
    response = client.post("/predict/", json=sample_payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "predicted_addiction_score" in data
    assert "better_than_percentage" in data
    assert isinstance(data["predicted_addiction_score"], int)

def test_collect_user_data(client):
    collect_payload = sample_payload.copy()
    collect_payload["Addicted_Score"] = 4 

    response = client.post("/collect/", json=collect_payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "student_id" in data
    
def test_predict_invalid_data(client):
    bad_payload = sample_payload.copy()
    bad_payload["Age"] = "Hai mươi tuổi"
    
    response = client.post("/predict/", json=bad_payload)
    assert response.status_code == 422