import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import pytest
from pricing_engine.config import config

from pricing_engine.data_preprocessing import (
    apply_imputation,
    treat_outlier,
    encode_cat_features,
    feature_engineering,
    transform_user_input
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Alloy": ["Aluminium", "Copper", "Iron"],
        "Finish": ["Gloss", "Matte", "Gloss"],
        "Length_m": [2.5, 3.0, 2.7],
        "Weight_kg_m": [1.2, 1.5, 200],
        "Profile_Name": ["A1", "A2", "A1"],
        "Tolerances": [0.1, 0.2, 0.1],
        "GD_T": [1, 2, 1],
        "Order_Quantity": [1000, 1500, 1200],
        "LME_Price_EUR": [3.4, 3.5, 3.6],
        "Customer_Category": ["small", "large", "small"],
        "Quote_Price_SEK": [250, 270, 260],
        "Lead_Time_weeks": [2, None, 3],
        "Quote_Date": ["2023-01-01", "2023-01-02", "2023-01-03"]
    })

def test_apply_imputation(sample_df):
    df = apply_imputation(sample_df.copy())
    assert df["Lead_Time_weeks"].isnull().sum() == 0

def test_treat_outlier(sample_df):
    df = sample_df.copy()
    df.loc[1, "Weight_kg_m"] = 999  # Insert extreme outlier
    cleaned_df = treat_outlier(df)
    assert not cleaned_df["Weight_kg_m"].isnull().any()
    assert cleaned_df["Weight_kg_m"].max() < 999  # Outlier removed

def test_encode_cat_features(sample_df):
    df = sample_df.copy()
    df["Lead_Time_weeks"] = df["Lead_Time_weeks"].fillna(2)  # Just to make pipeline stable
    encoded_df = encode_cat_features(df, user_id="test_user")
    assert "Profile_Name" in encoded_df.columns
    assert not any(encoded_df[["Alloy", "Finish", "GD_T", "Customer_Category"]].dtypes == object)

def test_feature_engineering(sample_df):
    df = sample_df.copy()
    df["Lead_Time_weeks"] = df["Lead_Time_weeks"].fillna(2)
    df["Profile_Name"] = "P1"
    engineered_df = feature_engineering(df, user_id="test_user")
    assert "Profile_Complexity" in engineered_df.columns
    assert "Manufacturing_Difficulty" in engineered_df.columns
    assert "LME_MA_7" in engineered_df.columns
    assert "LME_Lag_1" in engineered_df.columns

def test_transform_user_input(monkeypatch):
    from pricing_engine import model_storage

    # Create a mock encoder that returns proper shaped arrays
    dummy_encoder = MagicMock()

    def mock_transform(x):
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(np.zeros_like(x, dtype=float), columns=x.columns)
        elif isinstance(x, pd.Series):
            return pd.Series(np.zeros(len(x)))
        return x
    
    dummy_encoder.transform.side_effect = mock_transform

    monkeypatch.setattr(model_storage, "load_model", lambda user_id, component: {
        "freq_encoder": (dummy_encoder, None),
        "ordinal_encoder": (dummy_encoder, None),
        "profile_dict": ({"P1": 0.5}, None),
        "lme": ({"LME_MA_7": 3.5, "LME_Lag_1": 3.4}, None)
    }[component])

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

    df = transform_user_input("test_user", user_input)
    assert "Profile_Complexity" in df.columns
    assert "Manufacturing_Difficulty" in df.columns
    assert "LME_MA_7" in df.columns
    assert "LME_Lag_1" in df.columns
