import streamlit as st
import xgboost as xgb
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog

# -------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to classify the tumor type")


# -------------------------------------
# LOAD XGBOOST MODEL
# -------------------------------------
@st.cache_resource
def load_model_xgb():
    model = xgb.XGBClassifier()
    model.load_model(r"D:\my project\archive\saved_models\xgboost_brain_tumor_20251209-105115.json")
    return model

model = load_model_xgb()

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224


# -------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------
def preprocess_image(image: Image.Image):
    img = np.array(image)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to 224 × 224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply blur + histogram equalization
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0

    return img


# -------------------------------------
# HOG FEATURE EXTRACTION
# -------------------------------------
def extract_hog_features(img):
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features.reshape(1, -1)   # (1, 8100)


# -------------------------------------
# FILE UPLOADER
# -------------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess (same pipeline as training)
    gray = preprocess_image(image)
    features = extract_hog_features(gray)

    # Predict
    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    pred_label = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    st.success(f"### Prediction: **{pred_label.upper()}**")
    st.write(f"**Confidence:** {confidence:.2f}%")