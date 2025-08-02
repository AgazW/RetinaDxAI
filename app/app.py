import streamlit as st
import torch
from PIL import Image
from evaluate import load_model, preprocess_image, predict, class_names

st.title("RetinaDxAI Classifier")

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model (cache for performance)
@st.cache_resource
def get_model():
    num_classes = len(class_names)
    return load_model("models/Resnet_model_weights.pth", num_classes)

model = get_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img.save("temp_uploaded_image.jpg")  # Save temporarily for preprocessing
    img_tensor = preprocess_image("temp_uploaded_image.jpg")
    pred_class = predict(model, img_tensor)
    st.success(f"Predicted class: {pred_class}")