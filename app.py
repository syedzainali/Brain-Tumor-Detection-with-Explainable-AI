import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_efficientnetb0.keras")

model = load_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Map model output → UI labels (IMPORTANT FIX)
label_map = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

st.title("🧠 Brain Tumor MRI Detection With Explainable AI")

# -------------------------
# MODE SELECTION
# -------------------------
mode = st.radio("Select Mode", ["🎯 Demo Mode" , "🔼 Upload Mode" ])

# -------------------------
# DEMO IMAGES
# -------------------------
DEMO_PATHS = {
    "Glioma": "assets/demo/glioma.jpg",
    "Meningioma": "assets/demo/meningioma.jpg",
    "No Tumor": "assets/demo/notumor.jpg",
    "Pituitary": "assets/demo/pituitary.jpg"
}

@st.cache_data
def load_demo_images():
    return {k: Image.open(v) for k, v in DEMO_PATHS.items()}

demo_images = load_demo_images()

uploaded_file = None
image = None

# -------------------------
# INPUT HANDLING (FIXED)
# -------------------------
if mode == "🔼 Upload Mode":
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif mode == "🎯 Demo Mode":
    demo_choice = st.selectbox(
        "Select Demo Case",
        ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    )

    image = demo_images[demo_choice]

# -------------------------
# GRAD-CAM
# -------------------------
def get_gradcam_heatmap(model, img_array, layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), class_idx.numpy()


def overlay_heatmap(img, heatmap):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if img.shape != heatmap.shape:
        img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# RUN ONLY IF IMAGE EXISTS
# -------------------------
if image is not None:

    tab1, tab2 = st.tabs(["🔎 Prediction", "🔬 Explainability"])

    img = preprocess(image)

    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    predicted_label = label_map[predicted_class]

    # =======================
    # TAB 1 - PREDICTION
    # =======================
    with tab1:
        st.image(image, caption="Input MRI" , width=300) #, use_container_width=True

        st.subheader("Diagnosis Result")
        st.write(f"**Class:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")

    # =======================
    # TAB 2 - EXPLAINABILITY
    # =======================
    with tab2:

        st.subheader("📊 Class Probabilities")

        fig, ax = plt.subplots()
        ax.bar(class_names, pred[0])
        ax.set_ylim([0, 1])
        st.pyplot(fig)

        st.subheader("📝 Model Attention Summary")

        if predicted_class == "glioma":
            st.write("Model focuses on abnormal high-intensity regions.")
        elif predicted_class == "meningioma":
            st.write("Model detects boundary mass near meninges.")
        elif predicted_class == "pituitary":
            st.write("Model focuses on central brain region.")
        else:
            st.write("No abnormal tumor-like patterns detected.")

        st.subheader("📖 Explanation")

        st.write(f"""
        The model predicts **{predicted_label}** with confidence **{confidence:.2f}**.

        Grad-CAM highlights regions that contributed most to this decision.
        The model uses texture, intensity, and spatial patterns in MRI scans.
        """)

        st.subheader("Grad-CAM Heatmap")

        heatmap, _ = get_gradcam_heatmap(model, img, "top_conv")

        orig = np.array(image.resize((160, 160)))
        overlay = overlay_heatmap(orig, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(orig, caption="Original")
        with col2:
            st.image(overlay, caption="Heatmap")

        # -------------------------
        # PDF GENERATION
        # -------------------------
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Brain Tumor Report", ln=True)
            pdf.ln(10)

            pdf.cell(200, 10, txt=f"Diagnosis: {predicted_label}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)

            return pdf.output(dest="S").encode("latin-1")

        pdf_data = generate_pdf()

        st.download_button(
            "📄 Download Report",
            data=pdf_data,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf"
        )