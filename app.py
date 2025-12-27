import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px

# Page Config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_cardio.csv')
    return df

@st.cache_resource
def load_model():
    return joblib.load('cardio_prediction_model.pkl')

df = load_data()
pipeline = load_model()

# --- 2. Horizontal Input Section ---
st.title("Cardiovascular Disease Prediction")
st.markdown("Enter patient details below to generate a risk assessment.")

def user_input_features():
    # Create a container with a border to group inputs visually
    with st.container(border=True):
        st.subheader("Patient Health Data")
        
        # ROW 1: Basic Vitals (4 Columns)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (Years)", min_value=10, max_value=100, value=50, step=1)
        with c2:
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        with c3:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165, step=1)
        with c4:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            
        # ROW 2: Clinical Metrics (4 Columns)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            ap_hi = st.number_input("Systolic BP", min_value=50, max_value=250, value=120, step=1)
        with c6:
            ap_lo = st.number_input("Diastolic BP", min_value=30, max_value=180, value=80, step=1)
        with c7:
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        with c8:
            gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
            
        # ROW 3: Lifestyle & Action (3 Columns + Submit Button)
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            smoke = st.radio("Smoker?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        with c10:
            alco = st.radio("Alcohol Intake?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        with c11:
            active = st.radio("Physical Activity?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        with c12:
            st.write("") # Spacer to align button
            st.write("") 
            run_btn = st.button("Analyze Risk", type="primary", use_container_width=True)

    data = {
        'age': age, 'gender': gender, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
    }
    return pd.DataFrame(data, index=[0]), run_btn

input_df, run_btn = user_input_features()

# --- 3. Analysis & Dashboard ---
# Only show results when button is pressed
if run_btn:
    st.divider() # Horizontal line separator
    
    # Prediction Logic
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    # Create Tabs
    tab1, tab2 = st.tabs(["🏥 Prediction Result", "📊 Model Performance"])

    with tab1:
        st.subheader("Diagnosis Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability * 100:.2f}%")
        
        with col2:
            if prediction == 1:
                st.error("⚠️ HIGH RISK DETECTED")
                st.write(f"The model predicts a **{probability*100:.1f}%** chance of cardiovascular disease based on the provided clinical data.")
            else:
                st.success("✅ LOW RISK")
                st.write(f"The model predicts a **{probability*100:.1f}%** chance. The patient is likely healthy regarding cardiovascular issues.")

    with tab2:
        st.header("Model Performance & Validation")
        st.write("This section visualizes how well the model performs on unseen test data.")

        # Re-generate Test Split
        X = df.drop('cardio', axis=1)
        y = df['cardio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        # --- ROW 1: Confusion Matrix ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Healthy', 'Disease'], y=['Healthy', 'Disease'])
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC = {roc_auc:.2f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
            )
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)

        # --- ROW 2: Feature Importance ---
        st.subheader("Feature Importance")
        
        model_step = pipeline.named_steps['model']
        preprocessor_step = pipeline.named_steps['preprocessor']
        
        num_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        cat_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        cat_names = preprocessor_step.named_transformers_['cat'].get_feature_names_out(cat_features)
        feature_names = num_features + list(cat_names)
        
        importances = model_step.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        
        fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig_feat, use_container_width=True)