import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 1. Page Configuration & Professional Styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LungsAI - Medical Diagnosis System",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical Theme (Clean & Trustworthy Look)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #0b5394;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #0b5394; /* Professional Blue */
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #073763; /* Darker Blue on Hover */
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #888;
        padding: 10px;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. Sidebar Section (Project Info)
# ---------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("🫁 LungsAI")
    st.caption("Advanced Pneumonia Detection System")
    st.write("---")
    st.write("### ℹ️ About Project")
    st.info(
        """
        This AI system uses Deep Learning (CNN) to detect Pneumonia from Chest X-Ray images.
        
        **Accuracy:** ~95.3%
        **Model:** Custom CNN
        **Dataset:** Kershaw Hospital X-Ray Data
        """
    )
    st.write("### 📞 Contact Developer")
    st.write("**Name:** Shahed Rahman")
    st.write("**Email:** shahedrahmanltd@gmail.com")
    st.write("---")
    st.warning("⚠️ **Note:** For research purpose only.")

# ---------------------------------------------------------------------
# 3. Main Header
# ---------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🫁 LungsAI – Deep Learning Pneumonia Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Artificial Intelligence Powered Pulmonary Screening</p>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------
# 4. Model Loading Logic
# ---------------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    model = load_model('pneumonia_model.h5')
    return model

try:
    model = load_ai_model()
except:
    st.error("🚨 Critical Error: Model file 'pneumonia_model.h5' not found! Please check directory.")

# ---------------------------------------------------------------------
# 5. Image Upload & Processing Section
# ---------------------------------------------------------------------
st.write("### 📤 Upload Patient X-Ray")
uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Centering Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Convert & Display Image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Patient X-Ray Preview', use_container_width=True)
        
        # Centered Button Logic
        left, mid, right = st.columns([0.5, 2, 0.5])
        with mid:
            analyze_button = st.button('🔍 RUN DIAGNOSIS')

    # Analysis Logic
    if analyze_button:
        with st.spinner('🧬 AI is analyzing lung patterns...'):
            # Preprocessing
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediction
            prediction = model.predict(img_array)
            result_score = prediction[0][0]
            
            # --- Result Display Section ---
            st.write("---")
            st.subheader("📋 Diagnosis Report")
            
            # Layout for results
            res_col1, res_col2 = st.columns(2)
            
            if result_score > 0.5:
                # PNEUMONIA CASE
                with res_col1:
                    st.error("⚠️ **DETECTION: POSITIVE**")
                    st.write("The AI has detected signs of **Pneumonia**.")
                with res_col2:
                    st.metric(label="AI Confidence Score", value=f"{result_score*100:.2f}%", delta="High Risk", delta_color="inverse")
                
                st.warning("🩺 **Medical Advice:** Please consult a Pulmonologist immediately for clinical verification.")
                
            else:
                # NORMAL CASE
                with res_col1:
                    st.success("✅ **DETECTION: NEGATIVE**")
                    st.write("The lungs appear **Normal** and healthy.")
                with res_col2:
                    st.metric(label="AI Confidence Score", value=f"{(1-result_score)*100:.2f}%", delta="Low Risk")
                
                st.info("ℹ️ **Note:** No significant abnormalities detected by the AI system.")

# ---------------------------------------------------------------------
# 6. Terms, Conditions & Footer
# ---------------------------------------------------------------------
st.write("")
st.write("")

# Terms and Conditions (Expandable)
with st.expander("⚖️ Terms of Service & Medical Disclaimer (Read Carefully)"):
    st.markdown("""
    ### 1. Intended Use
    This application (**LungsAI**) is a prototype designed for **educational and research purposes only** (Science Fair Project 2025). It is NOT a certified medical device.

    ### 2. No Medical Advice
    The results provided by this AI system should **never** be treated as a final medical diagnosis. 
    - Always consult a certified doctor or radiologist for official diagnosis.
    - Do not ignore professional medical advice based on this app's results.

    ### 3. Accuracy & Liability
    While the model has a high accuracy rate (~95%), errors (False Positives/Negatives) can occur. The developer (**Shahed Rahman**) is not liable for any decisions made based on this software.

    ### 4. Data Privacy
    This application runs locally on your device. No patient data or images are uploaded to any external server or cloud storage.
    """)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #888; font-size: 12px;'>
        <hr>
        <p>Developed by <b>Shahed Rahman</b> | Diploma in Engineering Student</p>
        <p>© 2025 LungsAI Research Project. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)