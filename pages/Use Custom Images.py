import streamlit as st
from PIL import Image
import torch
from going_modular.predictions import predict_single_image

# Page Configuration
st.set_page_config(page_title="🔥 Wildfire Classification System", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        color: #333333;
        background-color: #eef2f3;
    }
    .stTitle {
        color: #d9534f;
        font-weight: bold;
        font-size: 36px;
    }
    .stSidebar {
        background-color: #6c757d;
        color: white;
    }
    .stSuccess {
        color: #5cb85c;
        font-weight: bold;
    }
    .stWarning {
        color: #f0ad4e;
        font-weight: bold;
    }
    .stError {
        color: #d9534f;
        font-weight: bold;
    }
    .stFooter {
        color: #ffffff;
        background-color: #343a40;
        text-align: center;
        padding: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown("<div class='stTitle'>🔥 Wildfire Detection Using Satellite Imagery</div>", unsafe_allow_html=True)

# Sidebar Instructions
st.sidebar.header("📌 User Guide")
st.sidebar.markdown(
    """
    ### 🛠️ Steps to Use This Application:
    1. **🛰️ Obtain a Satellite Image**:
       - Utilize **Bing Maps** to capture a relevant satellite image.
       - Select **satellite terrain mode** for optimal accuracy.
       - Ensure the image covers approximately **one acre** with a **100m elevation**.
    2. **📤 Upload Your Image**:
       - Upload the satellite image for wildfire detection.
       - Supported formats: **JPG, PNG, JPEG**.

    👉 You can access **Bing Maps** [here](https://www.bing.com/maps?cp=47.431688%7E-53.948823&lvl=8.4&style=a).
    """
)

# Device Information
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"⚡ Prediction will be performed using: **{device.upper()}**")

# Image Upload Section
st.header("📸 Upload Satellite Image for Wildfire Detection")

uploaded_image = st.file_uploader("🖼️ Select an image file (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

# Image Processing and Prediction
if uploaded_image:
    try:
        with st.spinner("🔍 Analyzing the image..."):
            # Load and display the uploaded image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
            
            # Perform wildfire prediction
            prediction_result = predict_single_image(image)  # Expected to return a classification label
            
            # Display the result
            st.markdown(f"<div class='stSuccess'>✅ Prediction Outcome: <strong>{prediction_result}</strong></div>", unsafe_allow_html=True)
    except Exception as error:
        st.markdown(f"<div class='stError'>❌ An error occurred during processing: {error}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='stWarning'>⚠️ Please upload a valid satellite image to proceed with prediction.</div>", unsafe_allow_html=True)

# Footer Section
st.markdown("""<div class='stFooter'>🌍 Wildfire Classification System - Developed for Environmental Safety 🚒</div>""", unsafe_allow_html=True)
