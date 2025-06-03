import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestRegressor

from pricing_engine.model_training import train_model

@pytest.fixture
def mock_user_data():
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        "Quote_Date": dates,
        "Quote_Price_SEK": np.random.uniform(200, 400, size=100),
        "Alloy": np.random.randint(0, 5, size=100),
        "Finish": np.random.randint(0, 3, size=100),
        "Length_m": np.random.uniform(1.0, 3.0, size=100),
        "Weight_kg_m": np.random.uniform(1.0, 2.0, size=100),
        "Profile_Name": np.random.uniform(0, 1, size=100),
        "Tolerances": np.random.uniform(0.05, 0.2, size=100),
        "GD_T": np.random.randint(0, 3, size=100),
        "Order_Quantity": np.random.randint(500, 5000, size=100),
        "LME_Price_EUR": np.random.uniform(2.0, 4.0, size=100),
        "Customer_Category": np.random.randint(0, 2, size=100),
        "Lead_Time_weeks": np.random.uniform(2, 6, size=100),
        "Profile_Complexity": np.random.uniform(0.01, 0.2, size=100),
        "Manufacturing_Difficulty": np.random.uniform(0.1, 1.0, size=100),
        "LME_MA_7": np.random.uniform(2.5, 3.5, size=100),
        "LME_Lag_1": np.random.uniform(2.5, 3.5, size=100),
    })
    return df

def test_train_model(monkeypatch, mock_user_data):
    # Mock data preprocessing
    from pricing_engine import data_preprocessing
    monkeypatch.setattr(data_preprocessing, "load_and_preprocess", lambda user_id: mock_user_data)

    # Mock model saving
    from pricing_engine import model_storage
    monkeypatch.setattr(model_storage, "save_model", lambda user_id, model, component: 1)

    # Mock logger to silence output
    from pricing_engine import utils
    monkeypatch.setattr(utils, "get_user_logger", lambda user_id: MagicMock())

    # Run training
    model, version = train_model("test_user")

    assert isinstance(model, RandomForestRegressor)
    assert version == 1
    assert hasattr(model, "predict")
