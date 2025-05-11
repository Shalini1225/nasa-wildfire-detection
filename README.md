🔥 Wildfire Classification Using Satellite Images
AI for Earth Safety — Deep Learning meets Disaster Prevention
Detecting wildfires with satellite & drone imagery, powered by EfficientNet & Streamlit.

📌 Overview
This project aims to develop a deep learning model to classify satellite images into two categories: "Wildfire" and "No Wildfire".

Key Features:

🌍 Uses EfficientNet_B0 architecture for powerful yet efficient image classification.

🧠 Deep learning pipeline from preprocessing to prediction.

💡 Web interface built with Streamlit for easy interaction.

🗂️ Project Structure
markdown
Copy
Edit
1.py
app.py
demo.ipynb
going_modular/
    __pycache__/
    data_setup.py
    engine.py
    model_builder.py
    predictions.py
    README.md
    train.py
    utils.py
images/
model/
    EfficientNet_b0-Wildfire_Classifier.pt
pages/
    Use Custom Images.py
    Use Validation Images.py
Read.me
readme.md
requirements.txt
validation_dataset/
    README.txt
    valid/
        nowildfire/
        wildfire/
wildfire-classification.ipynb
WildFire.pptx
🧠 Key Components
✅ Model Selection & Preprocessing
Pretrained EfficientNet_B0 from torchvision.

Transforms images (resize, normalize) using pretrained weights.

📦 Data Loading
Custom DataLoader for batch training & validation.

🏋️ Model Training
Trained on labeled wildfire satellite images.

Optimizer and loss function tune weights over epochs.

📈 Evaluation
Accuracy measured on a separate validation set.

💾 Model Saving/Loading
Trained model is saved as:

Copy
Edit
model/EfficientNet_b0-Wildfire_Classifier.pt
🌐 Web Interface (Streamlit)
Features:
Upload an image

Classify it in real time

See results instantly

To run:

bash
Copy
Edit
streamlit run app.py
📊 Performance
✅ 99.17% Test Accuracy

⚠️ 2% False Positive Rate

🔍 Handles large, high-resolution image datasets efficiently

🧪 Notable Scripts
wildfire-classification.ipynb: Full development notebook

going_modular/:

data_setup.py: Dataset prep

engine.py: Training & eval loop

model_builder.py: Model constructor

predictions.py: Predict with the model

app.py: Runs the Streamlit web app

🧾 Dataset
📂 Located in: validation_dataset/valid/

🔗 Source: Wildfire Prediction Dataset on Kaggle

Contains:

wildfire/

nowildfire/

⚙️ Requirements
Install dependencies:

txt
Copy
Edit
torch==1.12.1
torchvision==0.13.1
torchinfo
pathlib
tqdm
streamlit==1.27.0
pillow==9.4.0
matplotlib
Install with:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Getting Started
🔧 Train the Model
bash
Copy
Edit
python going_modular/train.py
🌐 Run the App
bash
Copy
Edit
streamlit run app.py
Then visit the local URL in your browser.

📌 Conclusion
This project demonstrates how deep learning + satellite imagery can be harnessed for wildfire detection.
It enables faster, scalable, and smarter fire identification to support early response and environmental safety.

🌲🔥 Protecting forests, one pixel at a time.

🙌 Acknowledgements
Developed by: Shalini Jada, Xingguo Xiong, Ahmed El-Sayed, Navarun Gupta
Presented at the ASEE Northeast Section Conference 2025
With support from University of Bridgeport

