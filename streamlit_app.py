
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

IMG_SIZE = (224, 224)

HF_REPO_ID = "kiddix/cassava_mazia_disease"

st.set_page_config(
    page_title="Cassava and Maize Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# =====================================================
# 2. DISEASE CLASS LABELS
# =====================================================

CASSAVA_LABELS = {
    "cbb": "Cassava Bacterial Blight",
    "cbsd": "Cassava Brown Streak Disease",
    "cgm": "Cassava Green Mite Damage",
    "cmd": "Cassava Mosaic Disease",
    "healthy": "Healthy"
}

MAIZE_LABELS = {
    "Blight": "Northern Leaf Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy"
}


# =====================================================
# 3. DISPLAY DISEASE NAMES
# =====================================================

def get_display_name(crop, class_name):

    if crop == "cassava":
        mapping = CASSAVA_LABELS
    else:
        mapping = MAIZE_LABELS

    return mapping.get(class_name, class_name)


# =====================================================
# 4. LOAD MODELS FROM HUGGING FACE
# =====================================================

@st.cache_resource(show_spinner=False)
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

    # Load trained TensorFlow model
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load class names
    with open(
        classes_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    # Validate class names
    if not isinstance(class_names, list):

        raise ValueError(
            f"{classes_filename} must contain a JSON list."
        )

    if not all(isinstance(name, str) for name in class_names):

        raise ValueError(
            "Every class name must be a string."
        )

    # Validate model output
    output_shape = model.output_shape

    if isinstance(output_shape, list):

        raise ValueError(
            "This application expects a model with one output."
        )

    if output_shape[-1] != len(class_names):

        raise ValueError(
            "The number of class names does not match "
            "the model output classes."
        )

    return model, class_names


# =====================================================
# 5. APPLICATION HEADER
# =====================================================

st.title("🌿 Cassava and Maize Leaf Disease Detector")

st.markdown(
    """
    Welcome to the Cassava and Maize Leaf Disease Detector.

    Upload an image of a cassava or maize leaf to identify
    possible diseases using artificial intelligence.
    """
)

st.divider()


# =====================================================
# 6. CROP SELECTION
# =====================================================

crop = st.selectbox(
    "Select the crop:",
    ["cassava", "maize"]
)


# =====================================================
# 7. LOAD SELECTED MODEL
# =====================================================

try:

    with st.spinner(
        f"Loading {crop} disease detection model..."
    ):

        model, class_names = load_model_and_classes(crop)

    st.success(
        f"✅ {crop.capitalize()} model loaded successfully!"
    )

except Exception as error:

    st.error(
        f"Unable to load the {crop} disease detection model."
    )

    with st.expander("Technical error details"):

        st.exception(error)

    st.stop()


# =====================================================
# 8. IMAGE UPLOAD
# =====================================================

st.subheader("📷 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a cassava or maize leaf image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# 9. IMAGE PROCESSING AND PREDICTION
# =====================================================

if uploaded_file is not None:

    try:

        # Read image
        pil_img = Image.open(
            uploaded_file
        ).convert("RGB")

        # Display uploaded image
        st.image(
            pil_img,
            caption="Uploaded Leaf Image",
            width="stretch"
        )

        # Resize image
        img_resized = pil_img.resize(IMG_SIZE)

        # Convert to NumPy array
        # Keep pixel values between 0 and 255
        img_array = np.array(
            img_resized
        ).astype(np.float32)

        # Add batch dimension
        img_batch = np.expand_dims(
            img_array,
            axis=0
        )

        # Generate prediction
        predictions = model.predict(
            img_batch,
            verbose=0
        )

        predictions = np.asarray(predictions)

        if predictions.ndim != 2 or predictions.shape[0] != 1:
            raise ValueError(
                "Unexpected prediction output shape."
            )

        predictions = predictions[0]

        if len(predictions) != len(class_names):
            raise ValueError(
                "Prediction output does not match class names."
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                "Model returned invalid prediction values."
            )

        # Identify predicted class
        pred_idx = int(
            np.argmax(predictions)
        )

        pred_class = class_names[pred_idx]

        confidence = float(
            predictions[pred_idx]
        ) * 100

        # =================================================
        # 10. DISPLAY PREDICTION RESULTS
        # =================================================

        st.divider()

        st.subheader("🔍 Disease Detection Results")

        disease_name = get_display_name(
            crop,
            pred_class
        )

        st.markdown(
            f"### Prediction: {disease_name}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # =================================================
        # 11. CLASS PROBABILITIES
        # =================================================

        st.divider()

        st.subheader("📊 Class Probabilities")

        sorted_predictions = sorted(
            zip(class_names, predictions),
            key=lambda item: float(item[1]),
            reverse=True
        )

        for class_name, probability in sorted_predictions:

            display_name = get_display_name(
                crop,
                class_name
            )

            percentage = float(probability) * 100

            st.write(
                f"**{display_name}: {percentage:.2f}%**"
            )

            st.progress(
                min(
                    max(float(probability), 0.0),
                    1.0
                )
            )

        # =================================================
        # 12. ADDITIONAL INFORMATION
        # =================================================

        st.divider()

        if pred_class.lower() == "healthy":

            st.success(
                "The model classified this leaf as healthy."
            )

        else:

            st.warning(
                "A possible disease was detected. "
                "Consider consulting an agricultural expert "
                "for confirmation."
            )

        st.caption(
            "Note: AI predictions are estimates and should "
            "not replace professional agricultural diagnosis."
        )

    except Exception as error:

        st.error(
            "An error occurred while processing the uploaded image."
        )

        with st.expander("Technical error details"):

            st.exception(error)


# =====================================================
# 13. APPLICATION FOOTER
# =====================================================

st.divider()

st.caption(
    "🌿 Cassava and Maize Leaf Disease Detector"
)

st.caption(
    "Powered by TensorFlow, Hugging Face and Streamlit."
)
