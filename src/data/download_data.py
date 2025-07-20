# Code to download data from Kaggle

import kagglehub
import os
import shutil

def download_kaggle_dataset(
		target_dir="../data/external", 
		kaggle_dataset="abhinav099802/eye-disease-image-dataset"):
	
	"""
	Downloads a dataset from Kaggle and copies it to the target directory.

	Args:
		target_dir (str): The directory where the dataset will be copied.
		kaggle_dataset (str): The Kaggle dataset identifier (e.g., "owner/dataset-name").

	Returns:
		str: The path to the dataset files in the target directory.
	"""
	
	os.makedirs(target_dir, exist_ok=True)
	
		# Download latest version
		path = kagglehub.dataset_download(kaggle_dataset)
		# Copy contents
		shutil.copytree(path, target_dir, dirs_exist_ok=True)
		print("Path to dataset files:", target_dir)
		return target_dir
		return target_dir
	
	except Exception as e:
		print(f"Error downloading or copying dataset: {e}")
		return None