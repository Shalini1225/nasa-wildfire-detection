import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(page_title='🔥 Wildfire Classification', layout='wide')

# Custom CSS for better UI aesthetics
st.markdown("""
    <style>
        body { background-color: #1e1e1e; color: #dcdcdc; }
        .title { text-align: center; font-size: 36px; font-weight: bold; color: #ff7043; }
        .description { text-align: justify; font-size: 18px; line-height: 1.6; }
        .emoji { font-size: 22px; }
        .caption { font-size: 16px; font-style: italic; text-align: center; }
        .image-container { display: flex; justify-content: center; }
        .image-container img { border-radius: 10px; box-shadow: 0px 4px 10px rgba(255, 112, 67, 0.5); }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>🔥 Wildfire Classification and Analysis Using Satellite Images</div>", unsafe_allow_html=True)

# Description with better readability
st.markdown("""
    <div class='description'>
        🌍 Wildfires are a devastating natural disaster that pose significant threats to both human life and the environment. 
        These catastrophic events, often driven by climate change and human activity, have been increasing in frequency and intensity.
        Therefore, innovative approaches are needed to monitor, predict, and mitigate wildfires.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='description'>
        📡 One such approach is the use of satellite images for wildfire classification. This project utilizes satellite imagery to 
        classify and detect wildfires, offering numerous benefits:
    </div>
""", unsafe_allow_html=True)

# Key benefits with emojis
st.markdown("""
- 🚀 **Early Detection**: Detect wildfires quickly, minimizing damage.
- 🔥 **Improved Prediction**: Forecast fire spread for proactive firefighting.
- 🌎 **Global Coverage**: Monitor wildfires anywhere on Earth with satellite imagery.
- 💰 **Cost Reduction**: Efficiently allocate resources, reducing firefighting costs.
- 🌱 **Environmental Monitoring**: Assess the aftermath and long-term impact of wildfires.
""")

# Conclusion with improved typography
st.markdown("""
    <div class='description'>
        ✅ The use of satellite images for wildfire classification is crucial in enhancing early detection, prediction, and monitoring of wildfires. 
        This project contributes to addressing the challenges posed by climate change and its impact on wildfire activity. 
        By leveraging satellite technology, we can take significant steps toward protecting communities and preserving ecosystems. 🌿
    </div>
""", unsafe_allow_html=True)

# Adding space for better layout
st.write('')

# Displaying the image with better cropping
st.subheader("🛰️ Satellite Image of Wildfire Detection")
img = Image.open(r'./images/forest_fires.jpg')
cropped_img = img.crop((50, 50, img.width - 50, img.height - 50))  # Adjust crop for better framing
st.markdown("<div class='image-container'>", unsafe_allow_html=True)
st.image(cropped_img, caption="Satellite Image of a Wildfire", use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Optional File Selector for future use
# def file_selector(folder_path='./images'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('📂 You selected `%s`' % filename)
