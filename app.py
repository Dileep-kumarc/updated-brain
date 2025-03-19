import os
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import gdown

# Download models at startup
def download_file(url, filename):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        try:
            gdown.download(url, filename, quiet=False)
            # Verify file size to ensure it’s not an HTML page
            file_size = os.path.getsize(filename)
            if file_size < 1024:  # If file is too small (<1KB), it’s likely not the model
                st.error(f"Downloaded {filename} is too small ({file_size} bytes). Expected a large model file. Check the Google Drive link.")
                raise ValueError("Invalid file size")
            st.success(f"Downloaded {filename} successfully!")
        except Exception as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            raise

# Download models at app startup
download_file("https://drive.google.com/file/d/1asvDh7lSvkL7yW6rhtLAzz7BHhI9Dzwv/view?usp=sharing", "best_mri_classifier.pth")
download_file("https://drive.google.com/file/d/1jey7rlkoK4qgIpBFiXG9RtwwzLjEYnTp/view?usp=sharing", "brain_tumor_classifier.h5")

@st.cache_resource
def load_models():
    def load_custom_model():
        class CustomCNN(nn.Module):
            def __init__(self):
                super(CustomCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, 2)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * 28 * 28)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = CustomCNN()
        model.load_state_dict(torch.load("best_mri_classifier.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    custom_cnn_model = load_custom_model()
    classifier_model = tf.keras.models.load_model("brain_tumor_classifier.h5")
    return custom_cnn_model, classifier_model

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array

# MRI Validation
def validate_mri(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor_image = transform(image).unsqueeze(0)
    output = model(tensor_image)
    pred = torch.argmax(output, dim=1).item()
    return ("MRI", True) if pred == 0 else ("Non-MRI", False)

# Tumor Classification
def classify_tumor(image, model):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return classes[np.argmax(predictions)], np.max(predictions)

# Streamlit interface
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f7f7f7;
    }
    .title {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #34495e;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
    }
    .btn {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
    }
    .btn:hover {
        background-color: #0056b3;
    }
    .input-area {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .navigation {
        margin-bottom: 1.5rem;
    }
    .upload-container {
        background-color: #f9f9f9;
        border: 1px dashed #ccc;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Detection")

uploaded_file = st.sidebar.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
custom_cnn_model, classifier_model = load_models()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.header("Step 1: MRI Validation")
    image_type, is_mri = validate_mri(image, custom_cnn_model)
    if not is_mri:
        st.error(f"Detected image type: {image_type}. Please upload a valid MRI image.")
    else:
        st.success("Image validated as MRI. Proceeding to classification...")

        st.header("Step 2: Tumor Classification")
        tumor_type, confidence = classify_tumor(image, classifier_model)
        st.write(f"**Tumor Type Detected:** {tumor_type} (Confidence: {confidence:.2f})")

        if tumor_type == "No Tumor":
            st.info("No tumor detected in the image.")
        else:
            st.warning("Tumor detected in the image!")
