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
│   └── models/
│       └── Resnet_model_weights.pth  # Trained model weights
├── src/
│   ├── models/
│   │   └── evaluate.py       # Model loading, preprocessing, prediction
│   ├── preprocess/           # Preprocessing utilities
│   └── visualization/
│       └── plot_performance.py # Plotting functions
├── notebooks/
│   ├── train.ipynb           # Model training notebook
│   └── evaluate.ipynb        # Model evaluation notebook
├── tests/
│   └── test_evaluate.py      # Unit tests for evaluate.py
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
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
