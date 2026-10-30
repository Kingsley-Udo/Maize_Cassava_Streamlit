import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CASSAVA_LABELS = {
    "cbb": "Cassava Bacterial Blight",
    "cbsd": "Cassava Brown Streak Disease",
    "cgm": "Cassava Green Mite Damage",
    "cmd": "Cassava Mosaic Disease",
    "healthy": "Healthy",
}

MAIZE_LABELS = {
    "Blight": "Northern Leaf Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy",
}

def get_display_name(crop, class_name):
    mapping = CASSAVA_LABELS if crop == "cassava" else MAIZE_LABELS
    return mapping.get(class_name, class_name)

st.set_page_config(page_title="Cassava/Maize Leaf Disease Detector", layout="centered")
st.title("🌿 Cassava/Maize Leaf Disease Detector")



@st.cache_resource(show_spinner="Loading AI disease detection model...")
def load_model_and_classes(crop):

    model_filename = f"{crop}_model.keras"
    classes_filename = f"{crop}_classes.json"

    # Download trained model
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=model_filename,
        repo_type="model"
    )

    # Download class names
    classes_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=classes_filename,
        repo_type="model"
    )

    # Load TensorFlow model
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load class names
    with open(classes_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    # Validate class names
    if not isinstance(class_names, list):
        raise ValueError(
            f"{classes_filename} must contain a JSON list of class names."
        )

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "The number of class names does not match "
            "the model output classes."
        )

    return model, class_names