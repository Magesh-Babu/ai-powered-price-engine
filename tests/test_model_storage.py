import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import tempfile
import shutil
import pickle
import pytest

from pricing_engine import model_storage
from pricing_engine.config import config


@pytest.fixture
def temp_user_env(monkeypatch):
    # Set up temporary directory
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setitem(config['storage'], 'base_dir', temp_dir)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def test_ensure_user_dirs(temp_user_env):
    user_id = "test_user"
    model_storage.ensure_user_dirs(user_id)

    base_path = os.path.join(temp_user_env, user_id)
    assert os.path.isdir(os.path.join(base_path, "models"))
    assert os.path.isdir(os.path.join(base_path, "logs"))


def test_save_and_load_model(temp_user_env):
    user_id = "test_user"
    dummy_object = {"a": 1, "b": 2}

    # Save
    version = model_storage.save_model(user_id, dummy_object, component="test_model")
    assert version == 1

    # Load
    loaded_object, loaded_version = model_storage.load_model(user_id, component="test_model", version=1)
    assert loaded_object == dummy_object
    assert loaded_version == 1


def test_save_model_versioning(temp_user_env):
    user_id = "versioned_user"
    for i in range(3):
        model_storage.save_model(user_id, {"v": i}, component="test_model")

    # Check if version increments correctly
    _, version = model_storage.load_model(user_id, component="test_model")
    assert version == 3


def test_get_latest_model_path(temp_user_env):
    user_id = "latest_model_user"
    model_storage.ensure_user_dirs(user_id)

    model_dir = os.path.join(config['storage']['base_dir'], user_id, "models")
    for version in [1, 3, 2]:
        with open(os.path.join(model_dir, f"model_v{version}.pkl"), 'wb') as f:
            pickle.dump({"v": version}, f)

    path, ver = model_storage.get_latest_model_path(user_id)
    assert ver == 3
    assert os.path.exists(path)
    assert path.endswith("model_v3.pkl")


def test_load_model_not_found(temp_user_env):
    user_id = "missing_model_user"
    model_storage.ensure_user_dirs(user_id)

    with pytest.raises(FileNotFoundError):
        model_storage.load_model(user_id, component="nonexistent_model")
