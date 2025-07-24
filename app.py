import os

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from src.dataset import get_load_data
from src.model import ViT
from src.predict import predict
from torchvision import transforms

# --- Configs ---
MODEL_PATH = "model_weights/model_weights.pt"
# CLASS_NAMES = [f"class_{i}" for i in range(102)]  # Replace with actual names if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Load Model ---
@st.cache_resource
def load_model():
    vit_config = {
        'img_dim': (3, 224, 224),
        'patch_size': 16,
        'num_classes': 102,
        'hidden_dim': 768,
        'num_heads': 12,
        'num_transformers': 12
    }
    model = ViT(**vit_config).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def plot_image(image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)


# --- Streamlit UI ---
st.title("🌸 ViT Flower Classifier")

model = load_model()

_, test_set = get_load_data(root="data", dataset="Flowers102", download=True)
idx = st.slider("Choose test image index", 0, len(test_set)-1, 0)
image, label = test_set[idx]
plot_image(image.permute(1, 2, 0))

pred_idx, prob = predict(model, image)
st.write(f"**Prediction:** {pred_idx} ({prob * 100:.2f}%)")
st.write(f"**Ground Truth:** {label}")