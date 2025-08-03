# RetinaDxAI

🧑‍⚕️ **RetinaDxAI** is a deep learning-powered web application for retina image classification.  
It uses PyTorch and Streamlit to provide instant predictions for various retinal diseases.

---

## Features

- Upload retina images (JPG, JPEG, PNG) and get instant predictions.
- Classifies images into 10 retinal conditions, including:
  - Central Serous Chorioretinopathy
  - Diabetic Retinopathy
  - Disc Edema
  - Glaucoma
  - Healthy
  - Macular Scar
  - Myopia
  - Pterygium
  - Retinal Detachment
  - Retinitis Pigmentosa
- Interactive web interface built with Streamlit.
- Model training and evaluation utilities.
- MLflow integration for experiment tracking.

---

## Project Structure

```
RetinaDxAI/
├── app/
│   ├── app.py                # Streamlit web app
│   ├── requirements.txt      # App dependencies
├── models/
│   ├── Resnet_model_weights.pth  # Trained model weights
├── src/
│   ├── data/
│   │   ├── download_data.py          # Data downloading utility
│   │   └── test_download_data.py     # Tests for download_data.py
│   ├── models/
│   │   ├── evaluate.py               # Model loading, preprocessing, prediction
│   │   ├── train.py                  # Model training script
│   │   ├── test_evaluate.py          # Tests for evaluate.py
│   │   └── test_train.py             # Tests for train.py
│   ├── preprocess/
│   │   ├── preprocessing.py          # Preprocessing utilities
│   │   ├── subset_data.py            # Data subsetting utility
│   │   ├── test_preprocessing.py     # Tests for preprocessing.py
│   │   └── test_subset_data.py       # Tests for subset_data.py
│   └── visualization/
│       ├── plot_performance.py       # Plotting functions
│       ├── visualize_images.py       # Image visualization utility
│       ├── test_plot_performance.py  # Tests for plot_performance.py
│       └── test_visualize_images.py  # Tests for visualize_images.py
├── notebooks/
│   ├── train.ipynb           # Model training notebook
│   └── evaluate.ipynb        # Model evaluation notebook
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AgazW/RetinaDxAI.git
   cd RetinaDxAI/app
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download or place the trained model weights:**
   - Ensure `Resnet_model_weights.pth` is in `app/models/`.

---

## Usage

### Run the Streamlit App

```sh
streamlit run app.py
```

- Open your browser at the provided local URL.
- Upload a retina image to get predictions.

---

## Model Training

- Use `notebooks/train.ipynb` to train models and log experiments with MLflow.
- Save trained weights to `app/models/Resnet_model_weights.pth` for use in the app.

---

## Testing

Run all tests from the project root:
```sh
pytest
```

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- streamlit
- torch
- torchvision
- numpy
- pandas
- pillow
- matplotlib
- seaborn
- scikit-learn
- mlflow

---

## Contributing

Pull requests and issues are welcome!  
Please add tests for new features and follow best practices.

---

## License

MIT License (or specify your license here)

---

## Acknowledgements

- PyTorch
- Streamlit
- MLflow
- scikit-learn
