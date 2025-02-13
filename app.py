import streamlit as st
import tensorflow as tf
import numpy as np
import gdown 
import os
from tqdm import tqdm

def download_file_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    if total_size == 0:
        st.error("Failed to fetch file size. Download may not work correctly.")

    # Create a progress bar in Streamlit
    progress_bar = st.progress(0)
    downloaded_size = 0
    
    with open(output_path, "wb") as file:
        for data in tqdm(response.iter_content(1024), total=total_size // 1024, unit="KB"):
            file.write(data)
            downloaded_size += len(data)
            progress_bar.progress(min(downloaded_size / total_size, 1.0))  # Update progress bar

    st.success("✅ Model Downloaded Successfully!")

# Check if model exists, otherwise download it
if not os.path.exists(model_path):
    st.warning("Downloading Model from Google Drive... Please wait.")
    download_file_with_progress(download_url, model_path)

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128)) 
    input_arr = tf.keras.preprocessing.image.img_to_array(image) 
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

class_labels = {
    0: "Potato__Early_Blight",
    1: "Potato__Late_Blight",
    2: "Potato__Healthy"
}

st.sidebar.title("🥔 Plant Disease Detection System")

home_button = st.sidebar.button("🏠 Home", key="home_btn")
disease_button = st.sidebar.button("🔍 Disease Recognition", key="disease_btn")

if "page" not in st.session_state:
    st.session_state.page = "Home"

if home_button:
    st.session_state.page = "Home"
if disease_button:
    st.session_state.page = "Disease Recognition"

# Home Page
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center;'>🌿 Potato Disease Detection 🌿</h1>", unsafe_allow_html=True)
    st.image("potato_disease_banner.jpg", use_container_width=True) 
    st.write(
        "### 🌍 AI-powered detection for potato plant diseases."
        "\n📌 Upload an image of a potato leaf to detect diseases."
        "\n📌 Get real-time predictions using deep learning."
    )

# Disease Recognition Page
elif st.session_state.page == "Disease Recognition":
    st.markdown("## 🔍 Potato Disease Recognition")
    st.write("Upload an image of a potato leaf to detect diseases.")

    uploaded_image = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        show_image = st.button("📷 Show Image")

        if show_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        predict_button = st.button("🔍 Predict Disease")

        if predict_button:
            analyzing_text = st.empty()
            analyzing_text.write("🕵️ Analyzing...")

            predicted_class = model_prediction(uploaded_image)
            disease_name = class_labels.get(predicted_class, "Unknown Disease")

            analyzing_text.empty()
            
            st.success(f"✅ Prediction: **{disease_name}**")
