# Code to download data from Kaggle

import kagglehub

# Download latest version
path = kagglehub.dataset_download("abhinav099802/eye-disease-image-dataset")

print("Path to dataset files:", path)