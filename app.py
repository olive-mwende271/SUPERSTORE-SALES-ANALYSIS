import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Superstore Predictor", layout="wide")

st.title("📊 Superstore Sales Prediction App")

@st.cache_resource
def load_model():
    # This loads the entire Pipeline (Preprocessor + Regressor)
    return joblib.load('superstore_model.pkl')

try:
    model_pipeline = load_model()
    
    st.success("✅ Prediction Engine Ready")

    # Create UI Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Order Details")
        year = st.number_input("Year", 2024, 2030, 2024)
        month = st.slider("Month", 1, 12, 1)
        quantity = st.number_input("Quantity", 1, 100, 1)
        discount = st.slider("Discount", 0.0, 0.8, 0.0, 0.05)

    with col2:
        st.subheader("Selection")
        # These MUST match the strings used during model.fit()
        segment = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'])
        region = st.selectbox("Region", ['West', 'East', 'Central', 'South'])
        category = st.selectbox("Category", ['Technology', 'Office Supplies', 'Furniture'])
        
        predict_btn = st.button("Calculate Forecast", type="primary", use_container_width=True)

    if predict_btn:
        # Create a dataframe with the EXACT column names used in training
        # Training features were: ['Year', 'Month', 'Segment', 'Region', 'Category', 'Quantity', 'Discount']
        input_dict = {
            'Year': year,
            'Month': month,
            'Segment': segment,
            'Region': region,
            'Category': category,
            'Quantity': quantity,
            'Discount': discount
        }
        
        X_input = pd.DataFrame([input_dict])
        
        # Pass the raw dataframe to the pipeline
        # The pipeline's 'preprocessor' will apply the OneHotEncoder automatically
        prediction_log = model_pipeline.predict(X_input)
        
        # Reverse the log transformation (np.log1p -> np.expm1)
        final_sales = np.expm1(prediction_log)[0]

        st.divider()
        st.metric("Predicted Sales Value", f"${final_sales:,.2f}")

except Exception as e:
    st.error("🚨 App Error Detected")
    st.write(f"Details: {e}")
    st.info("Check if 'superstore_model.pkl' is actually a Pipeline and not just a model.")
