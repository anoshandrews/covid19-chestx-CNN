import streamlit as st
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("COVID-19 Xray Detection.h5")

st.set_page_config(page_title="COVID-19 X-ray Detection")

st.title("🩺 COVID-19 X-ray Detection")

st.write("""
Upload a chest X-ray image and the model will predict whether it shows signs of pneumonia (which could be COVID-19 or other types).
""")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess the image
    image_np = tf.keras.utils.img_to_array(image_pil)
    img_gray = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
    img_resized = cv.resize(img_gray, (224, 224))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, 224, 224, 1)

    # Predict
    predict = model.predict(img_input)[0][0]

    if predict >= 0.5:
        pred_label = "PNEUMONIA"
        confidence = predict * 100
    else:
        pred_label = "NORMAL"
        confidence = (1 - predict) * 100

    st.subheader("Prediction Result:")
    st.markdown(f"**{pred_label}**")
    st.caption(f"Confidence: {confidence:.2f}%")
else:
    st.info("Please upload an X-ray image to get a prediction.")