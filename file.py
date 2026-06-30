import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("brain_tumor_efficientnetb0.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classifier With Explainable AI")
st.write("🚀 NEW VERSION LOADED")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    image = Image.open(uploaded_file).convert("RGB") 
    img = np.array(image)
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    

    # Prediction
    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")