# 🥔 Potato Leaf Disease Detection

## 📌 Overview
This is an AI-powered web application built using **Streamlit** and **TensorFlow** that detects diseases in potato leaves. The app allows users to upload an image of a potato leaf, and the model predicts whether the leaf is:

- **Potato Early Blight**
- **Potato Late Blight**
- **Healthy**

The deep learning model is trained on an image dataset and is loaded dynamically in the application.

---

## 🚀 Features
✅ **User-friendly UI:** Built with Streamlit for a simple and intuitive interface.  
✅ **Real-time Predictions:** Upload an image and get instant results.  
✅ **Deep Learning Model:** Uses a trained **TensorFlow/Keras** model.  
✅ **Automatic Model Download:** If the model is missing, it downloads from Google Drive.  

---

## 🏗️ Tech Stack
- **Frontend & UI:** Streamlit
- **Backend:** Python, TensorFlow, NumPy
- **Model Handling:** TensorFlow/Keras

---

## 🔧 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Faizan-313/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```sh
streamlit run app.py
or
python -m streamlit run app.py
```

---

## 📥 Model Downloading
- The trained model (~90MB) is stored on **Google Drive**.
- If not present locally, the app **automatically downloads** it.

---

## 🖼️ Usage
1️⃣ **Open the app** using `streamlit run app.py or python -m streamlit run app.py`.  
2️⃣ **Upload an image** of a potato leaf.  
3️⃣ Click **Predict Disease**.  
4️⃣ Get a **disease classification**.  

---

## 🛠️ Troubleshooting
**Model not loading?**
- Ensure `trained_plant_disease_model.keras` is present.
- If not, the app will attempt to download it from Google Drive.

**Memory Issues?**
- If hosting on **Streamlit Cloud**, check available memory.
- If needed, avoid caching large models (`@st.cache_resource`).

---

## 📜 License
This project is licensed under the **MIT License**.

