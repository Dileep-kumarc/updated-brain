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

def download_file(url, filename):
    try:
        if '/d/' not in url:
            st.error(f"Invalid Google Drive URL: {url}. Please provide a valid sharing link.")
            raise ValueError("URL must contain '/d/' (Google Drive sharing link format)")
        
        file_id = url.split('/d/')[1].split('/')[0]
        base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        
        response = session.get(base_url, stream=True, allow_redirects=True)
        
        if "text/html" in response.headers.get("Content-Type", ""):
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                download_url = f"{base_url}&confirm={token}"
                response = session.get(download_url, stream=True, allow_redirects=True)
            else:
                st.error(f"Failed to bypass Google Drive confirmation for {filename}. Check link permissions.")
                raise ValueError("No confirmation token found in response")
        
        response.raise_for_status()
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
        
        with open(filename, 'rb') as f:
            header = f.read(4)
            if header.startswith(b'<'):
                st.error(f"Downloaded file {filename} is still an HTML page. Verify the Google Drive link and permissions.")
                raise ValueError("Invalid file content: HTML detected")
        
        st.success(f"Downloaded {filename} successfully!")
    except IndexError:
        st.error(f"Could not extract file ID from URL: {url}. Ensure it’s a valid Google Drive link.")
        raise
    except requests.RequestException as e:
        st.error(f"Download failed for {filename}: {str(e)}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred while downloading {filename}: {str(e)}")
        raise

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

        model_path = "best_mri_classifier.pth"
        if not os.path.exists(model_path):
            st.info("Downloading MRI validation model...")
            download_file("https://drive.google.com/file/d/1asvDh7lSvkL7yW6rhtLAzz7BHhI9Dzwv/view?usp=sharing", model_path)
        
        model = CustomCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    custom_cnn_model = load_custom_model()

    classifier_path = "brain_tumor_classifier.h5"
    if not os.path.exists(classifier_path):
        st.info("Downloading tumor classification model...")
        download_file("https://drive.google.com/file/d/1jey7rlkoK4qgIpBFiXG9RtwwzLjEYnTp/view?usp=sharing", classifier_path)
    
    classifier_model = tf.keras.models.load_model(classifier_path)

    return custom_cnn_model, classifier_model

# Rest of your code...
