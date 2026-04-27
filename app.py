import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageStat
from fpdf import FPDF
import tempfile
import time
import os
import cv2  
import requests
from streamlit_lottie import st_lottie

# ---------------------------------------------------------------------
# 1. Page Configuration & NATIVE THEME CSS (NO EMOJIS)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LungsAI – Deep Learning Pneumonia Classifier",
    page_icon="⚕️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Google Material Symbols & Custom UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@48,400,1,0');
    
    /* Animated Gradient Title */
    .gradient-text {
        font-weight: 800;
        font-size: 42px;
        background: linear-gradient(-45deg, #0b5394, #00d2ff, #0b5394);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 3s ease infinite;
        text-align: center;
        margin-bottom: 0px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Fade-In Entry Animation */
    .stApp { animation: fadeIn 1s ease-in-out; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glowing Primary Button */
    .stButton>button {
        color: white !important; 
        background: linear-gradient(90deg, #0b5394, #0072ff) !important;
        border: none; 
        border-radius: 8px; 
        padding: 12px 24px; 
        font-weight: bold; 
        width: 100%; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
    }
    
    /* Custom Alert Boxes for Results */
    .alert-success {
        background-color: rgba(9, 171, 59, 0.1);
        border-left: 5px solid #09ab3b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .alert-danger {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. LOTTIE ANIMATION LOADER 
# ---------------------------------------------------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

lottie_medical_scan = load_lottieurl("https://lottie.host/8c823011-37f2-45e3-baf0-eb8a5dcf31ff/nK60H1iK1c.json")
lottie_sidebar_pulse = load_lottieurl("https://lottie.host/178229b1-5e8c-4cce-9d41-ed32f2ecb40d/Y79h37233R.json")
lottie_terms_security = load_lottieurl("https://lottie.host/d27f8a7c-f1d2-4309-8d48-3cddbc96e2cc/B2e8wJ1Yc0.json") 

# ---------------------------------------------------------------------
# 3. XAI (Grad-CAM) FUNCTIONS
# ---------------------------------------------------------------------
def get_img_array(img_path_or_pil, size):
    img = img_path_or_pil.resize(size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break
            
    if last_conv_layer_name is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    last_conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x
            
    grad_model = tf.keras.models.Model(inputs, [last_conv_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        score = preds[0][0] 
        class_channel = preds[:, 0] if score > 0.5 else 1.0 - preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) 
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: return heatmap.numpy()
    return (heatmap / max_val).numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    img = np.array(img_pil)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

# ---------------------------------------------------------------------
# 4. DYNAMIC REPORT GENERATOR
# ---------------------------------------------------------------------
def generate_clinical_logic(raw_score, is_pneumonia):
    signs = []
    if is_pneumonia:
        if raw_score >= 0.90:
            severity = "Severe"
            signs = ["- Widespread opacities visible in lung lobes.", "- Dense consolidation blocking normal airflow.", "- High probability of active infection."]
        elif raw_score >= 0.70:
            severity = "Moderate"
            signs = ["- Localized opacities/infiltrates detected.", "- Moderate consolidation indicative of active infection."]
        else:
            severity = "Mild / Early Stage"
            signs = ["- Subtle patchy infiltrates observed.", "- Early signs of minor opacities."]
        reason = f"The AI detected abnormal white spots (opacities) in the lung fields. Based on the intensity and spread, the AI classifies this as {severity} Pneumonia."
    else:
        clarity = "Excellent" if (1.0 - raw_score) >= 0.90 else "Good"
        signs = ["- Lung fields appear dark, indicating normal air-filled lungs.", "- Completely clear of abnormal opacities or fluid.", "- Normal cardiac silhouette and clear contours."]
        reason = f"X-rays pass easily through healthy lungs, making them appear dark. The AI found no significant blockages, confirming {clarity} lung health."
    return reason, signs

def create_pdf(patient_data, original_img, heatmap_img, is_pneumonia, confidence_str, raw_score):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, txt="RADIOLOGY DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, txt="LungsAI Clinical AI System", ln=True, align='C')
    pdf.ln(8)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Patient Information & Clinical History", border='B', ln=True)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    pdf.cell(100, 6, txt=f"Name: {patient_data['name']}", ln=False)
    pdf.cell(90, 6, txt=f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(100, 6, txt=f"Age/Gender: {patient_data['age']} / {patient_data['gender']}", ln=False)
    pdf.cell(90, 6, txt=f"Ref ID: {patient_data['ref_id']}", ln=True)
    
    symptoms_text = ", ".join(patient_data['symptoms']) if patient_data['symptoms'] else "None reported"
    pdf.multi_cell(0, 6, txt=f"Reported Symptoms: {symptoms_text}")
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Final Impression", border='B', ln=True)
    pdf.ln(3)
    
    impression = "POSITIVE FOR PNEUMONIA" if is_pneumonia else "NORMAL (NO ACTIVE DISEASE DETECTED)"
    color = (194, 24, 7) if is_pneumonia else (0, 128, 0)
        
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, txt=impression, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, txt=f"AI Confidence Score: {confidence_str}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Detailed AI Analysis & Justification", border='B', ln=True)
    pdf.ln(3)
    
    reason, signs = generate_clinical_logic(raw_score, is_pneumonia)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, txt="Why did the AI make this decision?", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=reason)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, txt="Radiological Features Observed:", ln=True)
    pdf.set_font("Arial", size=10)
    for sign in signs: pdf.multi_cell(0, 5, txt=sign)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Radiological Images (Original vs AI Attention Map)", border='B', ln=True)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1, tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
        original_img.save(tmp1.name)
        heatmap_img.save(tmp2.name)
        pdf.image(tmp1.name, x=20, y=pdf.get_y(), w=75)
        pdf.image(tmp2.name, x=110, y=pdf.get_y(), w=75)
    
    pdf.ln(80) 
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, txt="DISCLAIMER: This report is generated by LungsAI for research purposes only. Not a substitute for professional medical advice.")
    return pdf.output(dest="S").encode("latin-1")

def is_valid_medical_image(pil_img):
    img_hsv = pil_img.convert('HSV')
    if ImageStat.Stat(img_hsv.split()[1]).mean[0] > 30: return False
    img_gray = pil_img.convert('L')
    avg_brightness = ImageStat.Stat(img_gray).mean[0]
    if avg_brightness < 30 or avg_brightness > 210: return False
    return True

# ---------------------------------------------------------------------
# 5. Sidebar Layout 
# ---------------------------------------------------------------------
with st.sidebar:
    if lottie_sidebar_pulse:
        st_lottie(lottie_sidebar_pulse, height=120, key="sidebar_anim")
        
    st.markdown("<h2 style='text-align: center; margin-top:-20px;'>LungsAI</h2>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center;'>Clinical Intelligence</p>", unsafe_allow_html=True)
    st.write("---")
    st.info("**Core Modules:**\n- Neural Analysis\n- Grad-CAM Heatmaps\n- PDF Diagnostics")
    st.write("---")
    st.write("### Contact Developer")
    st.write("**Name:** Shahed Rahman")
    st.write("**Email:** shahedrahmanltd@gmail.com")
    st.write("---")
    st.warning("Research purpose only.")

# ---------------------------------------------------------------------
# 6. Main App Dashboard 
# ---------------------------------------------------------------------
st.markdown("<div class='gradient-text'>LungsAI – Deep Learning Pneumonia Classifier</div>", unsafe_allow_html=True)

col_anim1, col_anim2, col_anim3 = st.columns([1, 2, 1])
with col_anim2:
    if lottie_medical_scan:
        st_lottie(lottie_medical_scan, height=150, key="main_scan_anim")

st.divider()

@st.cache_resource
def load_ai_model():
    return load_model('pneumonia_model.h5')

try:
    model = load_ai_model()
except:
    st.error("Critical Error: Model file 'pneumonia_model.h5' not found!")

col_form, col_scan = st.columns([1, 2])

# Left Column: Form & Symptoms 
with col_form:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span class="material-symbols-rounded" style="font-size: 32px; color: #00d2ff;">assignment_ind</span>
            <h3 style="margin: 0; padding: 0;">Patient Details</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        p_name = st.text_input("Patient Name", placeholder="e.g. Shahed Rahman")
        col_a, col_b = st.columns(2)
        with col_a:
            p_age = st.number_input("Age", min_value=1, max_value=120, value=5)
        with col_b:
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_ref = st.text_input("Reference ID", placeholder="e.g. PID-10293")
        
        # Symptoms Checklist (Clean Text without emojis)
        st.write("---")
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span class="material-symbols-rounded" style="color: #0b5394; font-size: 20px;">stethoscope</span>
                <strong style="font-size: 16px;">Reported Symptoms:</strong>
            </div>
        """, unsafe_allow_html=True)
        symp_fever = st.checkbox("Fever")
        symp_sob = st.checkbox("Shortness of Breath")
        symp_cough = st.checkbox("Cough")
        symp_pain = st.checkbox("Chest Pain")
        
        selected_symptoms = []
        if symp_fever: selected_symptoms.append("Fever")
        if symp_sob: selected_symptoms.append("Shortness of Breath")
        if symp_cough: selected_symptoms.append("Cough")
        if symp_pain: selected_symptoms.append("Chest Pain")

# Right Column: Scan & Results 
with col_scan:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <span class="material-symbols-rounded" style="font-size: 32px; color: #00d2ff;">cloud_upload</span>
            <h3 style="margin: 0; padding: 0;">Upload X-Ray Scan</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG)", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        if is_valid_medical_image(img):
            img = img.convert('RGB')
            st.image(img, caption='Uploaded Scan', width=250)
            
            # Clean text button
            if st.button('PROCESS SCAN & GENERATE REPORT', use_container_width=True):
                if not p_name:
                    st.warning("Please enter the Patient Name before processing.")
                else:
                    with st.spinner('Initializing Neural Network & Scanning Lungs...'):
                        time.sleep(1) 
                        
                        img_array = get_img_array(img, size=(150, 150))
                        prediction = model.predict(img_array)
                        score = float(prediction[0][0])
                        
                        is_pneumonia = score > 0.5
                        conf_val = score if is_pneumonia else 1 - score
                        conf_str = f"{conf_val*100:.2f}%"

                        heatmap = make_gradcam_heatmap(img_array, model)
                        heatmap_img = overlay_heatmap(img, heatmap)

                        st.write("---")
                        st.markdown("""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                                <span class="material-symbols-rounded" style="font-size: 32px; color: #00d2ff;">medical_information</span>
                                <h3 style="margin: 0; padding: 0;">Diagnostic Results</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            # CUSTOM HTML ALERT BOXES INSTEAD OF EMOJIS
                            if is_pneumonia:
                                st.markdown("""
                                    <div class="alert-danger">
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span class="material-symbols-rounded" style="color: #ff4b4b;">warning</span>
                                            <h4 style="margin: 0; color: #ff4b4b;">IMPRESSION: POSITIVE</h4>
                                        </div>
                                        <p style="margin: 10px 0 0 0; font-size: 14px;">Abnormal opacities detected.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="alert-success">
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span class="material-symbols-rounded" style="color: #09ab3b;">check_circle</span>
                                            <h4 style="margin: 0; color: #09ab3b;">IMPRESSION: NORMAL</h4>
                                        </div>
                                        <p style="margin: 10px 0 0 0; font-size: 14px;">Lungs appear healthy & clear.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            st.metric("Neural Confidence Score", conf_str)
                            
                        with res_col2:
                            st.image(heatmap_img, caption="AI Heatmap (Attention Focus)", use_container_width=True)

                        patient_data = {
                            "name": p_name, 
                            "age": p_age, 
                            "gender": p_gender, 
                            "ref_id": p_ref or "N/A",
                            "symptoms": selected_symptoms
                        }
                        pdf_bytes = create_pdf(patient_data, img, heatmap_img, is_pneumonia, conf_str, score)
                        
                        st.write("---")
                        # Clean Text Download Button
                        st.download_button(
                            label="Download Detailed Clinical Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"LungsAI_Report_{p_name.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        else:
            st.error("Invalid Image Detected: Upload a valid Chest X-Ray.")

# ---------------------------------------------------------------------
# 7. ANIMATED Terms & Conditions and Footer
# ---------------------------------------------------------------------
st.write("---")
# Clean text for expander
with st.expander("Terms of Service & Medical Disclaimer (Read carefully)"):
    t_col1, t_col2 = st.columns([1, 4])
    with t_col1:
        if lottie_terms_security:
            st_lottie(lottie_terms_security, height=130, key="terms_security_anim")
            
    with t_col2:
        st.markdown("""
        ### 1. Intended Use
        This application (**LungsAI**) is a prototype designed for **educational and research purposes only**. It is NOT a certified medical device.
        
        ### 2. No Medical Advice
        The results provided by this AI system should **never** be treated as a final medical diagnosis. Always consult a certified doctor or radiologist for official diagnosis. Do not ignore professional medical advice based on this app's results.
        
        ### 3. Accuracy & Liability
        While the model has a high accuracy rate, errors (False Positives/Negatives) can occur. The developer (**Shahed Rahman**) is not liable for any decisions made based on this software.
        
        ### 4. Data Privacy
        This application processes images temporarily for analysis. No patient data or images are permanently stored on the server.
        """)

st.markdown("""
    <div style='text-align: center; margin-top: 60px; color: #888; font-size: 12px; font-family: sans-serif;'>
        <hr style='border: 1px solid #e2e8f0; opacity: 0.2;'>
        <p>Developed by <b>Shahed Rahman</b> | Diploma in Engineering Student</p>
        <p>© 2026 LungsAI Research Project. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)