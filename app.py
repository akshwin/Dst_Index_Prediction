import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model (Replace 'model.h5' with your actual model file)
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()

# Define input features
features = ['dst_t-1', 'dst_t-2', 'dst_t-3', 'bz_gsm', 'theta_gsm', 'bz_gse', 'density', 'theta_gse', 'bx_gsm', 'bx_gse']

st.set_page_config(page_title="Dst Prediction App", layout="wide")
st.title("🌍 Dst Prediction App")
st.markdown("**Predict the Disturbance Storm Time (Dst) Index based on solar wind parameters.**")

# Arrange input fields in a two-column layout
col1, col2 = st.columns(2)

inputs = []
for i, feature in enumerate(features):
    if i % 2 == 0:
        value = col1.number_input(f"{feature.replace('_', ' ').title()}:", value=0.0, format="%.2f")
    else:
        value = col2.number_input(f"{feature.replace('_', ' ').title()}:", value=0.0, format="%.2f")
    inputs.append(value)

# Add some spacing before the prediction button
st.markdown("---")

# Predict Dst
if st.button("Predict Dst", use_container_width=True):
    inputs_array = np.array(inputs).reshape(1, -1)  # Reshape for model
    prediction = model.predict(inputs_array)[0][0]  # Make prediction
    st.success(f"📊 Predicted Dst Value: {prediction:.2f}")
