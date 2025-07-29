import os
import shutil
import tempfile
import pytest
from unittest import mock

import download_data

def test_download_kaggle_dataset_success():
    # Mock kagglehub.dataset_download to return a temp dir with dummy files
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as target_dir:
        dummy_file = os.path.join(src_dir, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("test")
        with mock.patch("kagglehub.dataset_download", return_value=src_dir):
            # Run function
            result = download_data.download_kaggle_dataset(target_dir=target_dir, kaggle_data="dummy/dataset")
            # Check that dummy file was copied
            assert os.path.exists(os.path.join(target_dir, "dummy.txt"))
            assert result == target_dir

def test_download_kaggle_dataset_failure():
    # Mock kagglehub.dataset_download to raise an exception
    with mock.patch("kagglehub.dataset_download", side_effect=Exception("Download error")):
        result = download_data.download_kaggle_dataset(target_dir="not_a_real_dir", kaggle_data="dummy/dataset")
        assert result is None

def test_download_kaggle_dataset_creates_target_dir():
    # Mock kagglehub.dataset_download to return a temp dir
    with tempfile.TemporaryDirectory() as src_dir:
        with mock.patch("kagglehub.dataset_download", return_value=src_dir):
            temp_dir = tempfile.mkdtemp()
            shutil.rmtree(temp_dir)  # Remove so function must create it
            result = download_data.download_kaggle_dataset(target_dir=temp_dir, kaggle_data="dummy/dataset")
            assert os.path.exists(temp_dir)
            shutil.rmtree(temp_dir)
