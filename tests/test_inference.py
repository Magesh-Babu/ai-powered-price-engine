import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestRegressor

from pricing_engine.inference import predict_price

def test_predict_price_with_existing_model(monkeypatch):
    # Create dummy estimator (tree)
    dummy_tree = MagicMock()
    dummy_tree.predict.return_value = [123.45]

    # Create dummy model with estimators_
    dummy_model = MagicMock(spec=RandomForestRegressor)
    dummy_model.predict.return_value = [123.45]
    dummy_model.estimators_ = [dummy_tree, dummy_tree, dummy_tree]

    # Mock model_storage.load_model to return model and version
    monkeypatch.setattr("pricing_engine.model_storage.load_model", lambda user_id, component: (dummy_model, 2))

    # Mock transform_user_input to return test DataFrame
    monkeypatch.setattr("pricing_engine.data_preprocessing.transform_user_input", lambda uid, feats: np.zeros((1, 10)))

    # Mock logger
    monkeypatch.setattr("pricing_engine.utils.get_user_logger", lambda uid: MagicMock())

    user_input = {
        "Alloy": "Aluminium",
        "Finish": "Gloss",
        "Length_m": 2.5,
        "Weight_kg_m": 1.2,
        "Profile_Name": "P1",
        "Tolerances": 0.1,
        "GD_T": 1,
        "Order_Quantity": 1000,
        "LME_Price_EUR": 3.4,
        "Customer_Category": "small",
        "Lead_Time_weeks": 2
    }

    predicted_price, lower, upper, version = predict_price("test_user", user_input)

    assert isinstance(predicted_price, float)
    assert predicted_price == 123.45
    assert lower == predicted_price
    assert upper == predicted_price
    assert version == 2

def test_predict_price_triggers_training(monkeypatch):
    # Create dummy estimator (tree)
    dummy_tree = MagicMock()
    dummy_tree.predict.return_value = [123.45]

    # Create dummy model with estimators_
    dummy_model = MagicMock(spec=RandomForestRegressor)
    dummy_model.predict.return_value = [123.45]
    dummy_model.estimators_ = [dummy_tree, dummy_tree, dummy_tree]

    # Simulate load_model raising FileNotFoundError (no saved model)
    monkeypatch.setattr("pricing_engine.model_storage.load_model", lambda uid, component: (_ for _ in ()).throw(FileNotFoundError))

    # Now simulate model_training.train_model returning model
    monkeypatch.setattr("pricing_engine.model_training.train_model", lambda uid: (dummy_model, 3))

    # Mock transform
    monkeypatch.setattr("pricing_engine.data_preprocessing.transform_user_input", lambda uid, feats: np.zeros((1, 10)))

    # Mock logger
    monkeypatch.setattr("pricing_engine.utils.get_user_logger", lambda uid: MagicMock())

    user_input = {
        "Alloy": "Aluminium",
        "Finish": "Gloss",
        "Length_m": 2.5,
        "Weight_kg_m": 1.2,
        "Profile_Name": "P1",
        "Tolerances": 0.1,
        "GD_T": 1,
        "Order_Quantity": 1000,
        "LME_Price_EUR": 3.4,
        "Customer_Category": "small",
        "Lead_Time_weeks": 2
    }

    predicted_price, lower, upper, version = predict_price("test_user", user_input)

    assert predicted_price == 123.45
    assert lower == predicted_price
    assert upper == predicted_price    
    assert version == 3
