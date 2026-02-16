import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# IMPORTANT: Must match label order used during training
class_names = ["airplane", "cat", "tree"]

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🐱🌳✈️ Image Classification App")
st.write("Upload an image of **Airplane, Cat, or Tree**")

uploaded_file = st.file_uploader("Drag and Drop Image Here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess (must match training)
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, -1)  # Flatten for sklearn model

    # Predict
    prediction = model.predict(img)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(img)[0]
        confidence = np.max(probabilities) * 100
    else:
        confidence = None

    predicted_class = class_names[prediction[0]]

    st.success(f"Prediction: {predicted_class}")

    if confidence:
        st.info(f"Confidence: {confidence:.2f}%")
