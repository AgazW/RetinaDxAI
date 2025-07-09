# Code to download data from Kaggle

import kagglehub
import os

# Specify your desired directory
my_download_dir = "data/external"  # or any absolute path like "/home/user/data"

# Create the directory if it doesn't exist
os.makedirs(my_download_dir, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("abhinav099802/eye-disease-image-dataset")

print("Path to dataset files:", path)