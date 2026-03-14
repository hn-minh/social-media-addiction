import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd

with patch("mlflow.pyfunc.load_model"), \
     patch("mlflow.artifacts.download_artifacts"), \
     patch("mlflow.tracking.MlflowClient"):
    
    from api.main import app
    client = TestClient(app)

sample_payload = {
    "Age": 22, 
    "Gender": "Male", 
    "Academic_Level": "Undergraduate",
    "Country": "Vietnam", 
    "Avg_Daily_Usage_Hours": 6.0,
    "Most_Used_Platform": "TikTok", 
    "Affects_Academic_Performance": True,
    "Sleep_Hours_Per_Night": 7.0, 
    "Mental_Health_Score": 5,
    "Relationship_Status": "Single", 
    "Conflicts_Over_Social_Media": 1
}

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

@patch("pandas.DataFrame.to_sql")
@patch("api.routers.collect.create_engine")
def test_collect_user_data_mock(mock_engine, mock_to_sql):
    collect_payload = sample_payload.copy()
    collect_payload["Addicted_Score"] = 3
    
    response = client.post("/collect/", json=collect_payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert mock_to_sql.called 

@patch("api.routers.predict.preprocess_for_prediction")
def test_predict_addiction_mock(mock_preprocess):
    
    mock_preprocess.return_value = MagicMock()
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [3]
    
    mock_target_encoder = MagicMock()
    mock_target_encoder.inverse_transform.return_value = [3]
    
    app.state.ml_components = {
        "model": mock_model,
        "target_encoder": mock_target_encoder,
        "score_percentiles": {3: 75.0}
    }
    
    response = client.post("/predict/", json=sample_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_addiction_score"] == 3
    assert data["better_than_percentage"] == 75.0

def test_predict_invalid_schema():
    bad_payload = sample_payload.copy()
    bad_payload["Age"] = "ba"
    
    response = client.post("/predict/", json=bad_payload)
    assert response.status_code == 422