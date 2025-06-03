import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import tempfile
import shutil
import pytest
import csv
from unittest.mock import MagicMock
from pricing_engine.config import config

from pricing_engine.continuous_learning import add_feedback

@pytest.fixture
def temp_config_and_env(monkeypatch):
    # Create a temp directory to act as storage
    temp_dir = tempfile.mkdtemp()

    # Patch the config dict directly
    monkeypatch.setitem(config['storage'], 'base_dir', temp_dir)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)

def test_add_feedback_creates_file_and_trains(monkeypatch, temp_config_and_env):
    user_id = "test_user"
    expected_version = 5

    # Simulate config fields
    from pricing_engine.config import config
    config['features'] = {
        'indep_var': [
            "Alloy", "Finish", "Length_m", "Weight_kg_m", "Profile_Name",
            "Tolerances", "GD_T", "Order_Quantity", "LME_Price_EUR", "Customer_Category", "Lead_Time_weeks"
        ],
        'target': "Quote_Price_SEK",
        'date': "Quote_Date"
    }

    feedback_data = {
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
        "Lead_Time_weeks": 2,
        "Quote_Price_SEK": 250.00,
        "Quote_Date": "2023-10-01"
    }

    # Mock logger
    monkeypatch.setattr("pricing_engine.utils.get_user_logger", lambda uid: MagicMock())

    # Mock train_model
    monkeypatch.setattr("pricing_engine.model_training.train_model", lambda uid: ("model_obj", expected_version))

    version = add_feedback(user_id, feedback_data)

    # Assert returned version is as mocked
    assert version == expected_version

    # Check that the file was created and contains expected data
    file_path = os.path.join(config['storage']['base_dir'], user_id, "simulated_dataset.csv")
    assert os.path.isfile(file_path)

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    assert rows[0] == config['features']['indep_var'] + [config['features']['target']] + [config['features']['date']]  # header
    assert rows[1] == [str(feedback_data[col]) for col in rows[0]]  # data row
