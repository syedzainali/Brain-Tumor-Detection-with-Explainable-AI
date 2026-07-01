# Brain Tumor MRI Classification with Explainable AI

## Overview
This project uses EfficientNetB0 and Grad-CAM to classify brain MRI scans into:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Features
- MRI classification
- Grad-CAM visualization
- Explainable AI (XAI)
- Streamlit web application
- PDF report generation

## Technologies
- TensorFlow
- EfficientNetB0
- Streamlit
- OpenCV
- Grad-CAM

## Results
Test Accuracy: 96%

## How to Run

```bash
streamlit run app.py

#New
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
.env
.venv/
venv/


## **NOTE**:
This project was implemented and extended based on a publicly available Kaggle notebook. I adapted the implementation, built a Streamlit application for interactive inference, resolved deployment issues, and organized the project for reproducibility.
