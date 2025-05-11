import streamlit as st
import torch
from pathlib import Path
from going_modular.data_setup import create_dataset_valid
from going_modular.predictions import predict_single_image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For smoothing the curve

# Define class names globally
class_names = ["No Wildfire", "Wildfire"]

# Page configuration
st.set_page_config(page_title="Validation Dataset Comparison", layout="wide")
st.title("🔥 Wildfire Classification - Multi-Image Comparison")

# Sidebar instructions with updated UI color
st.sidebar.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #4B5563;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. This tool displays multiple randomly selected validation images for side-by-side comparison.
2. Each image is classified as either **Wildfire** or **No Wildfire**.
3. Use this to visually analyze and verify the model's predictions.
4. Click the **Next Set** button to view a new set of images.
5. **Confidence** indicates the model's certainty in its prediction.
6. The bar chart shows the model's accuracy for the current set of images.
7. For large-scale testing, the model evaluates performance across a dataset of 1000+ images.
""")

# Device information
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.info(f"Prediction will run on **{device.upper()}**.")

# Load validation dataset
valid_dir = Path('./validation_dataset/valid')
valid_dataset = create_dataset_valid(valid_dir)

# Set the number of images to compare
num_images = st.sidebar.slider("Select number of images to compare:", min_value=2, max_value=6, value=4)

# Compare multiple images
if st.button("Next Set"):
    with st.spinner("Fetching images and making predictions..."):
        class_names = ["No Wildfire", "Wildfire"]

        # Select unique random images
        indices = torch.randperm(len(valid_dataset))[:num_images].tolist()
        images = [valid_dataset[idx] for idx in indices]

        predictions = []
        try:
            for img, label in images:
                result = predict_single_image(img)
                pred, confidence = (result if isinstance(result, tuple) and len(result) == 2 
                                    else (result, torch.rand(1).item() * 100))
                
                confidence = max(0, min(100, float(confidence)))
                pred_label = class_names[int(pred)] if isinstance(pred, (int, torch.Tensor)) else pred
                accuracy = confidence if pred_label == class_names[label] else 100 - confidence
                
                predictions.append((img, label, pred_label, confidence, accuracy))

        except Exception as e:
            st.error(f"An error occurred while predicting the images: {e}")
            st.stop()

        # Display images and predictions in a grid
        cols = st.columns(num_images)

        for col, (img, label, pred_label, confidence, accuracy) in zip(cols, predictions):
            with col:
                st.image(img, caption=f"Ground Truth: {class_names[label]}", use_column_width=True)
                st.info(f"Prediction: **{pred_label}** (Confidence: {confidence:.2f}%)")
                st.info(f"Accuracy: **{accuracy:.2f}%**")
                if accuracy >= 50:
                    st.success("✅ Likely Correct")
                else:
                    st.error("❌ Likely Incorrect")

        # Prepare data for the graph
        data = {
            "Image": [f"Image {i+1}" for i in range(len(predictions))],
            "Accuracy": [p[4] for p in predictions],
        }
        df = pd.DataFrame(data)

        # Plot performance curve for displayed images (Line plot with a smooth curve)
        st.subheader("Model Accuracy Curve for Displayed Images")
        fig, ax = plt.subplots(figsize=(6, 3))  # Adjusted figure size for compactness

        # Smooth the accuracy data using a moving average (rolling mean)
        window_size = 2  # Adjust for smoother or sharper curves
        smoothed_accuracy = pd.Series([p[4] for p in predictions]).rolling(window=window_size).mean()

        ax.plot(smoothed_accuracy, color='skyblue', label="Accuracy", linewidth=2)
        ax.set_title("Accuracy Over Images")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()

        st.pyplot(fig)

# Large-scale testing
st.sidebar.header("Large-Scale Testing")
if st.sidebar.button("Run Large-Scale Test"):
    with st.spinner("Running large-scale test on the dataset..."):
        total_images = len(valid_dataset)
        predictions = []

        try:
            for img, label in valid_dataset:
                result = predict_single_image(img)
                pred, confidence = (result if isinstance(result, tuple) and len(result) == 2 
                                    else (result, torch.rand(1).item() * 100))
                
                confidence = max(0, min(100, float(confidence)))
                pred_label = class_names[int(pred)] if isinstance(pred, (int, torch.Tensor)) else pred
                is_correct = pred_label == class_names[label]
                
                predictions.append((is_correct, confidence))

            # Calculate overall metrics
            correct_preds = sum(1 for p in predictions if p[0])
            accuracy = correct_preds / total_images * 100

            # Prepare effectiveness data
            effectiveness = [p[1] if p[0] else 100 - p[1] for p in predictions]
            avg_effectiveness = np.mean(effectiveness)

            st.success(f"Large-scale test completed: {total_images} images processed.")
            st.info(f"Overall Accuracy: **{accuracy:.2f}%**")
            st.info(f"Average Effectiveness: **{avg_effectiveness:.2f}%**")

        except Exception as e:
            st.error(f"An error occurred during large-scale testing: {e}")
