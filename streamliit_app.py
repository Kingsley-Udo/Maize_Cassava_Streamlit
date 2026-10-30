
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

IMG_SIZE = (224, 224)

HF_REPO_ID = "kiddix/cassava_mazia_disease"

st.set_page_config(
    page_title="Cassava/Maize Leaf Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ==========================================
# DISEASE CLASS LABELS
# ==========================================

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
    mapping = (
        CASSAVA_LABELS
        if crop == "cassava"
        else MAIZE_LABELS
    )

    return mapping.get(class_name, class_name)


# ==========================================
# LOAD MODEL AND CLASS NAMES FROM HUGGING FACE
# ==========================================

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


# ==========================================
# APPLICATION INTERFACE
# ==========================================

st.title("🌿 Cassava/Maize Leaf Disease Detector")

st.markdown(
    """
    Upload an image of a cassava or maize leaf to identify
    possible diseases using a trained artificial intelligence model.
    """
)

st.divider()

# Select crop
crop = st.selectbox(
    "Which crop is this leaf from?",
    ["cassava", "maize"]
)

# ==========================================
# LOAD SELECTED MODEL
# ==========================================

try:

    model, class_names = load_model_and_classes(crop)

    st.success(
        f"✅ {crop.capitalize()} disease detection model loaded successfully!"
    )

except Exception as error:

    st.error(
        f"Unable to load the {crop} disease detection model."
    )

    with st.expander("View technical error details"):
        st.exception(error)

    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# IMAGE PROCESSING AND PREDICTION
# ==========================================

if uploaded_file is not None:

    try:

        # Read uploaded image
        pil_img = Image.open(uploaded_file).convert("RGB")

        st.image(
            pil_img,
            caption="Uploaded leaf image",
            width="stretch"
        )

        # Resize image
        img_resized = pil_img.resize(IMG_SIZE)

        # Convert image to NumPy array
        img_array = np.array(img_resized).astype(np.float32)

        # Create batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Generate prediction
        predictions = model.predict(
            img_batch,
            verbose=0
        )[0]

        # Identify predicted class
        pred_idx = int(np.argmax(predictions))

        pred_class = class_names[pred_idx]

        confidence = float(predictions[pred_idx]) * 100

        # ==================================
        # DISPLAY RESULTS
        # ==================================

        st.divider()

        st.subheader("🔍 Disease Detection Results")

        st.markdown(
            f"### Prediction: {get_display_name(crop, pred_class)}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.1f}%"
        )

        st.divider()

        # ==================================
        # CLASS PROBABILITIES
        # ==================================

        st.subheader("📊 Class Probabilities")

        sorted_predictions = sorted(
            zip(class_names, predictions),
            key=lambda x: -x[1]
        )

        for class_name, probability in sorted_predictions:

            display_name = get_display_name(
                crop,
                class_name
            )

            percentage = float(probability) * 100

            st.write(
                f"**{display_name}: {percentage:.1f}%**"
            )

            st.progress(
                min(max(float(probability), 0.0), 1.0)
            )

        st.divider()

        st.caption(
            "AI predictions are estimates and should be "
            "confirmed by an agricultural expert when necessary."
        )

    except Exception as error:

        st.error("An error occurred while processing the image.")

        with st.expander("View technical error details"):
            st.exception(error)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Cassava/Maize Leaf Disease Detector | "
    "Powered by TensorFlow and Streamlit"
)