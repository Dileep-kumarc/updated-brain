import os
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Step 1: Define model downloading and loading functions
def download_file(url, filename):
    try:
        # Check if URL is a valid Google Drive link
        if '/d/' not in url:
            st.error(f"Invalid Google Drive URL: {url}. Please provide a valid sharing link.")
            raise ValueError("URL must contain '/d/' (Google Drive sharing link format)")
        
        # Extract file ID
        file_id = url.split('/d/')[1].split('/')[0]
        base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Use a session for persistent cookies
        session = requests.Session()
        response = session.get(base_url, stream=True)
        
        # Handle large file confirmation token
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            download_url = f"{base_url}&confirm={token}"
            response = session.get(download_url, stream=True)
        
        # Verify the response
        response.raise_for_status()
        
        # Download with progress bar
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    f.write(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
        
        progress_bar.empty()
        st.success(f"Downloaded {filename} successfully!")
    except IndexError:
        st.error(f"Could not extract file ID from URL: {url}. Ensure it’s a valid Google Drive link (e.g., https://drive.google.com/file/d/FILE_ID/view).")
        raise
    except requests.RequestException as e:
        st.error(f"Download failed for {filename}: {str(e)}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred while downloading {filename}: {str(e)}")
        raise

@st.cache_resource
def load_models():
    # Load Custom CNN for MRI Validation
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

        model_path = "best_mri_classifier.pth"
        if not os.path.exists(model_path):
            st.info("Downloading MRI validation model...")
            download_file("https://drive.google.com/file/d/1asvDh7lSvkL7yW6rhtLAzz7BHhI9Dzwv/view?usp=sharing", model_path)
        
        model = CustomCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    custom_cnn_model = load_custom_model()

    # Load Tumor Classification Model (Keras/TensorFlow)
    classifier_path = "brain_tumor_classifier.h5"
    if not os.path.exists(classifier_path):
        st.info("Downloading tumor classification model...")
        download_file("https://drive.google.com/file/d/1jey7rlkoK4qgIpBFiXG9RtwwzLjEYnTp/view?usp=sharing", classifier_path)
    
    classifier_model = tf.keras.models.load_model(classifier_path)

    return custom_cnn_model, classifier_model

# Step 2: Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array

# Step 3: MRI Validation
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

# Step 4: Tumor Classification
def classify_tumor(image, model):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return classes[np.argmax(predictions)], np.max(predictions)

# Step 5: Streamlit interface
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
