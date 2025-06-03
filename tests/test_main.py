# test_main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_quote_success(monkeypatch):
    def mock_predict_price(user_id, payload):
        return 199.99, 0.1, 0.2, 1
    monkeypatch.setattr("main.predict_price", mock_predict_price)

    request_data = {
        "Alloy": "Aluminium",
        "Finish": "Powder coated",
        "Length_m": 23.8,
        "Weight_kg_m": 1.342,
        "Tolerances": 0.1,
        "GD_T": "medium",
        "Order_Quantity": 85000,
        "LME_Price_EUR": 3.4,
        "Customer_Category": "medium",
        "Lead_Time_weeks": 6,
        "Profile_Name": "Karmlist"
    }

    response = client.post("/users/test_user/quote", json=request_data)
    assert response.status_code == 200
    assert response.json() == {
        "predicted_price": 199.99,
        "PI_lower": 0.1,
        "PI_upper": 0.2,
        "model_version": 1
    }

def test_get_quote_failure(monkeypatch):
    def mock_predict_price(user_id, payload):
        raise ValueError("No model found")
    monkeypatch.setattr("main.predict_price", mock_predict_price)

    request_data = {
        "Alloy": "Aluminium",
        "Finish": "Powder coated",
        "Length_m": 23.8,
        "Weight_kg_m": 1.342,
        "Tolerances": 0.1,
        "GD_T": "medium",
        "Order_Quantity": 85000,
        "LME_Price_EUR": 3.4,
        "Customer_Category": "medium",
        "Lead_Time_weeks": 6,
        "Profile_Name": "Karmlist"
    }

    response = client.post("/users/test_user/quote", json=request_data)
    assert response.status_code == 404
    assert response.json() == {"detail": "No model found"}

def test_post_feedback_success(monkeypatch):
    def mock_add_feedback(user_id, payload):
        return 2
    monkeypatch.setattr("main.add_feedback", mock_add_feedback)

    feedback_data = {
        "Alloy": "Aluminium",
        "Finish": "Powder coated",
        "Length_m": 23.8,
        "Weight_kg_m": 1.342,
        "Tolerances": 0.1,
        "GD_T": "medium",
        "Order_Quantity": 85000,
        "LME_Price_EUR": 3.4,
        "Customer_Category": "medium",
        "Lead_Time_weeks": 6,
        "Profile_Name": "Karmlist",
        "Quote_Price_SEK": 200.0,
        "Quote_Date": "2023-10-01"
    }

    response = client.post("/users/test_user/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json() == {
        "message": "Feedback added and model retrained",
        "new_version": 2
    }
