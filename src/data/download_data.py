# Code to download data from Kaggle

import kagglehub
import os
import shutil

def download_eye_disease_dataset(
		target_dir = "../data/external",
		kaggle_data = "abhinav099802/eye-disease-image-dataset"):
	
	"""
	Downloads the Eye Disease Image Dataset from Kaggle and copies it to the target directory.

	Args:
		target_dir (str): The directory where the dataset will be copied.
		kaggle_data (str): The Kaggle dataset identifier.

	Returns:
		str: The path to the dataset files in the target directory.
	"""
	
	os.makedirs(target_dir, exist_ok=True)
	
	try:
		# Download latest version
		path = kagglehub.dataset_download(kaggle_data)
		# Copy contents
		shutil.copytree(path, target_dir, dirs_exist_ok=True)
		print("Path to dataset files:", target_dir)
		return target_dir
	
	except Exception as e:
		print(f"Error downloading or copying dataset: {e}")
		return None