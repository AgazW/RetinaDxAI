# Code to download data from Kaggle

import kagglehub
import os
import shutil

# Specify your desired directory
target_dir = "../data/external"  # or any absolute path like "/home/user/data"

# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("abhinav099802/eye-disease-image-dataset")

# Copy contents
shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("Path to dataset files:", path)