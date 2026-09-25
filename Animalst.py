import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# 1. Page Configuration and Custom CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="SymptoScan - Diagnostic System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for headers, prediction buttons, and UI components
st.markdown("""
    <style>
    /* Adjust top padding for the sidebar main title */
    [data-testid="stSidebar"] h2 {
        padding-top: 20px !important;
    }

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        display: none;
    }
    [data-testid="stSidebar"] > div:first-child {
        -ms-overflow-style: none;  
        scrollbar-width: none;  
    }

    /* Reduce vertical spacing between fields so they appear neat and compact */
    [data-testid="stSidebar"] .stNumberInput, 
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stRadio {
        margin-bottom: -8px !important;
    }

    /* Arrange and reduce margins */
    [data-testid="stSidebar"] h3 {
        padding-top: 10px !important;
        padding-bottom: 5px !important;
        font-size: 1.1rem !important;
    }
    
    /* Main header styling with gradient background */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #e0e0e0;
        margin-top: 8px;
        font-size: 1.05rem;
    }

    /* Primary prediction button styling */
    div.stButton > button:first-child {
        background-color: #2a5298;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1e3c72;
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.3);
    }
    
    /* Checkbox margin adjustment */
    .stCheckbox {
        padding: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Load Model Pipeline, Label Encoder, and Metadata
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    # Load the 3 saved model files exported from Google Colab
    model = joblib.load('symptoscan_best_model.pkl')
    target_encoder = joblib.load('symptoscan_label_encoder.pkl')
    metadata = joblib.load('symptoscan_metadata.pkl')
    return model, target_encoder, metadata

try:
    model, target_encoder, metadata = load_pipeline()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}")
    st.stop()

# -------------------------------------------------------------
# 3. Application Main Header
# -------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <h1>🐾 SymptoScan</h1>
        <p>Early Veterinary Diagnostic & Clinical Support System</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Sidebar Inputs: Demographic and Vital Health Metrics
# -------------------------------------------------------------
st.sidebar.markdown("## 📋 Animal Vitals & Profile")
st.sidebar.markdown("---")

animal_type = st.sidebar.selectbox("Animal Type 🐶🐱🐄 : ", ['Dog', 'Cat', 'Cow', 'Horse', 'Sheep', 'Goat', 'Pig', 'Rabbit'])
gender = st.sidebar.radio("Gender : ", ['Male', 'Female'], horizontal=True)

st.sidebar.markdown("### Physical Metrics")
age = st.sidebar.number_input("Age (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.1, max_value=1000.0, value=55.0, step=0.5)
heart_rate = st.sidebar.number_input("Heart Rate (BPM)", min_value=20, max_value=250, value=88, step=1)
body_temp = st.sidebar.number_input("Body Temp (°C)", min_value=30.0, max_value=45.0, value=39.2, step=0.1)
duration = st.sidebar.number_input("Duration (Days)", min_value=1, max_value=60, value=12, step=1)

# -------------------------------------------------------------
# 5. Main Body: Symptoms Selection Section
# -------------------------------------------------------------
st.markdown("### Select Observed Symptoms")
st.caption("Check all clinical symptoms currently exhibited by the animal.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Gastrointestinal & General**")
    appetite_loss = st.checkbox("Appetite Loss")
    vomiting = st.checkbox("Vomiting")
    diarrhea = st.checkbox("Diarrhea")
    weight_loss = st.checkbox("Weight Loss")
    dehydration = st.checkbox("Dehydration")

with col2:
    st.markdown("**Respiratory**")
    coughing = st.checkbox("Coughing")
    labored_breathing = st.checkbox("Labored Breathing")
    sneezing = st.checkbox("Sneezing")
    nasal_discharge = st.checkbox("Nasal Discharge")

with col3:
    st.markdown("**Systemic & Physical**")
    fever = st.checkbox("Fever")
    lethargy = st.checkbox("Lethargy")
    swelling = st.checkbox("Swelling / Inflammation")
    lameness = st.checkbox("Lameness")

with col4:
    st.markdown("**Dermatological & Other**")
    skin_lesions = st.checkbox("Skin Lesions")
    eye_discharge = st.checkbox("Eye Discharge")
    reduced_milk = st.checkbox("Reduced Milk")
    reduced_wool = st.checkbox("Reduced Wool")

st.markdown("<br>", unsafe_allow_html=True)

# Check if at least one symptom is checked across all columns
symptoms_selected = any([
    appetite_loss, vomiting, diarrhea, weight_loss, dehydration,
    coughing, labored_breathing, sneezing, nasal_discharge,
    fever, lethargy, swelling, lameness,
    skin_lesions, eye_discharge, reduced_milk, reduced_wool
])

# -------------------------------------------------------------
# 6. Prediction Logic and Diagnostic Results Display
# -------------------------------------------------------------
run_button = st.button("Run Diagnostic Prediction", use_container_width=True)

if run_button:
    # Trigger warning if user clicks the button without selecting any symptoms
    if not symptoms_selected:
        st.warning("⚠️ Please select at least one clinical symptom.")
    else:
        # Construct input feature dictionary matching model schema
        input_data = {
            'Animal_Type': animal_type,
            'Gender': gender,
            'Age': float(age),
            'Weight': float(weight),
            'Heart_Rate': int(heart_rate),
            'Body_Temperature': float(body_temp),
            'Duration_Days': int(duration),
            'Appetite_Loss': int(appetite_loss),
            'Vomiting': int(vomiting),
            'Diarrhea': int(diarrhea),
            'Coughing': int(coughing),
            'Labored_Breathing': int(labored_breathing),
            'Lameness': int(lameness),
            'Skin_Lesions': int(skin_lesions),
            'Nasal_Discharge': int(nasal_discharge),
            'Eye_Discharge': int(eye_discharge),
            'Reduced_Milk': int(reduced_milk),
            'Reduced_Wool': int(reduced_wool),
            'Swelling': int(swelling),
            'Fever': int(fever),
            'Lethargy': int(lethargy),
            'Weight_Loss': int(weight_loss),
            'Dehydration': int(dehydration),
            'Sneezing': int(sneezing)
        }
        
        # Convert input payload into pandas DataFrame
        df_input = pd.DataFrame([input_data])
        
        try:
            # Align column order with feature order expected during training if available
            if hasattr(model, 'feature_names_in_'):
                df_input = df_input[model.feature_names_in_]

            # Compute prediction probabilities via full sklearn Pipeline
            probs = model.predict_proba(df_input)[0]

            # Sort top 3 prediction probabilities in descending order
            top_indices = np.argsort(probs)[::-1][:3]
            top_1_disease = target_encoder.inverse_transform([top_indices[0]])[0]
            top_1_confidence = probs[top_indices[0]] * 100

            # Retrieve disease family category from metadata dictionary
            disease_family_map = metadata.get('disease_family_map', {})
            top_1_family = disease_family_map.get(top_1_disease, "General Pathology")

            # Display output section
            st.markdown("---")
            st.subheader("Diagnostic Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success("✅ **Primary Diagnosis Identified**")
                st.metric(label="Predicted Condition", value=str(top_1_disease))
                st.caption(f"Category / Family: **{top_1_family}**")
                
            with res_col2:
                st.info("📊 **Model Confidence**")
                st.metric(label="Top Match Confidence", value=f"{top_1_confidence:.1f}%")

            # Display Top Differential Diagnoses with animated progress bars
            st.markdown("### 📋 Top Differential Diagnoses")
            for idx in top_indices:
                disease_name = target_encoder.inverse_transform([idx])[0]
                prob_val = probs[idx] * 100
                st.progress(int(prob_val), text=f"**{disease_name}**: {prob_val:.1f}%")
                
        except Exception as e:
            st.error(f"Error during prediction execution: {str(e)}")
