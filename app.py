import streamlit as st
import tensorflow as tf
import numpy as np
import gdown 
import os

url = "https://drive.google.com/uc?id=1Cuo6gXgG3fLoIO3S4Rr1a56PYiFSaKOX"
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    with st.status("Downloading Model from Google Drive..."):
        gdown.download(url, model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

model_path = "trained_plant_disease_model.keras"

def model_prediction(test_image,confidence_threshold=0.25, margin_threshold=0.05):
    model = tf.keras.models.load_model(model_path)
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128)) 
    input_arr = tf.keras.preprocessing.image.img_to_array(image) 
    input_arr = np.array([input_arr])
    
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)

    # Calculate the margin between the highest and second highest probabilities
      sorted_confidences = np.sort(predictions[0])
    second_max_confidence = sorted_confidences[-2] if len(sorted_confidences) > 1 else 0
    margin = confidence - second_max_confidence

    st.write("Predictions:", predictions)
    st.write("Confidence:", confidence)
    st.write("Margin:", margin)

    if confidence < confidence_threshold or margin < margin_threshold:
        return None, confidence  # Image might not be a potato leaf
    else:
        return predicted_class, confidence

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
