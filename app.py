import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px

# Page Config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# --- 1. Load Data & Model (Cached for Speed) ---
@st.cache_data
def load_data():
    # Load the cleaned dataset to generate performance metrics on the fly
    df = pd.read_csv('cleaned_cardio.csv')
    df = df[df['cardio'].isin([0, 1])] # Ensure no ghost rows
    return df

@st.cache_resource
def load_model():
    return joblib.load('cardio_prediction_model.pkl')

df = load_data()
pipeline = load_model()

# --- 2. Sidebar: Patient Input ---
st.sidebar.header("Patient Health Data")

def user_input_features():
    age_years = st.sidebar.slider("Age (Years)", 30, 100, 50)
    # Convert age to days because the training data likely used days (common in this dataset)
    # If your model used years, remove the * 365
    age = age_years # Assuming you fixed this in cleaning, otherwise use age_years * 365
    
    gender = st.sidebar.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
    height = st.sidebar.slider("Height (cm)", 140, 200, 165)
    weight = st.sidebar.slider("Weight (kg)", 40.0, 150.0, 70.0)
    ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 90, 200, 120)
    ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 60, 130, 80)
    cholesterol = st.sidebar.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    gluc = st.sidebar.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    smoke = st.sidebar.radio("Smoker?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    alco = st.sidebar.radio("Alcohol Intake?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    active = st.sidebar.radio("Physical Activity?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    data = {
        'age': age, 'gender': gender, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. Main Dashboard ---
st.title("❤️ Cardiovascular Disease Prediction Dashboard")

# Create Tabs
tab1, tab2 = st.tabs(["🏥 Prediction & Diagnosis", "📊 Model Performance Metrics"])

with tab1:
    st.subheader("Patient Diagnosis")
    
    # Display Input Data Summary
    st.write("Patient Data Summary:")
    st.dataframe(input_df)

    if st.button("Analyze Risk"):
        # Prediction
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability * 100:.2f}%")
        
        with col2:
            if prediction == 1:
                st.error("⚠️ HIGH RISK DETECTED")
                st.write("The model suggests a high probability of cardiovascular disease.")
            else:
                st.success("✅ LOW RISK")
                st.write("The model suggests a low probability of cardiovascular disease.")

with tab2:
    st.header("Model Performance & Validation")
    st.write("This section visualizes how well the model performs on unseen test data.")

    # Re-generate Test Split to Calculate Metrics Live
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get Predictions on Test Set
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    # --- ROW 1: Confusion Matrix ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        st.write("Shows where the model gets confused (False Positives vs False Negatives).")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Healthy', 'Disease'], y=['Healthy', 'Disease'])
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        st.write("Measures the trade-off between sensitivity and specificity. Higher AUC is better.")
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
    st.write("Which health factors contribute most to the risk?")
    
    # Extract Model and Feature Names
    model_step = pipeline.named_steps['model']
    preprocessor_step = pipeline.named_steps['preprocessor']
    
    # Get feature names (Numerical + Categorical)
    num_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    cat_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    cat_names = preprocessor_step.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_names)
    
    # Plot
    importances = model_step.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True) # Ascending for horizontal bar
    
    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_feat, use_container_width=True)