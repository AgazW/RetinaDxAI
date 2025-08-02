import streamlit as st
import torch
from PIL import Image
import sys

sys.path.append("src")
from models.evaluate import load_model, preprocess_image, predict, class_names

st.set_page_config(page_title="RetinaDxAI Classifier", page_icon="🧑‍⚕️", layout="centered")
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSuccess {background-color: #d4edda;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("https://images.unsplash.com/photo-1517841905240-472988babdf9", use_container_width=True)
st.sidebar.title("RetinaDxAI")
st.sidebar.markdown("Upload a retina image to classify.")

st.title("🧑‍⚕️ RetinaDxAI Classifier")
st.markdown(
    """
    <div style="background-color:#e3f2fd;padding:10px;border-radius:10px;">
    <h4>Welcome to RetinaDxAI!</h4>
    <p>This app uses a deep learning model to classify retina images. Upload a JPG, JPEG, or PNG image and get instant predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        # Preprocess directly from PIL Image
        img_tensor = preprocess_image(img)
        pred_class = predict(model, img_tensor)
        st.success(f"The image is most likely **{pred_class}**")
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload a retina image to get started.")