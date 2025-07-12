import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# -------------------------------
# Define Custom Quadratic Layer
# -------------------------------
class QuadraticLayer(Layer):
    def __init__(self, units, **kwargs):
        super(QuadraticLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(input_shape[-1], self.units), initializer="he_normal", trainable=True)
        self.W2 = self.add_weight(shape=(input_shape[-1], self.units), initializer="he_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        quadratic_output = K.dot(inputs, self.W1) + K.dot(K.square(inputs), self.W2) + self.b
        return K.clip(quadratic_output, -1e3, 1e3)

# -------------------------------
# Build Model
# -------------------------------
model = Sequential([
    Dense(10, activation=None, input_shape=(10,)),
    QuadraticLayer(128),
    Dropout(0.2),
    QuadraticLayer(64),
    Dropout(0.2),
    QuadraticLayer(32),
    Dropout(0.2),
    QuadraticLayer(16),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse', metrics=['mae'])

# -------------------------------
# Classification Function
# -------------------------------
def classify_dst(dst_value):
    if dst_value > -20:
        return "Quiet"
    elif -20 >= dst_value > -50:
        return "Unsettled"
    elif -50 >= dst_value > -100:
        return "Moderate Storm"
    elif -100 >= dst_value > -200:
        return "Intense Storm"
    else:
        return "Extreme Geomagnetic Storm"

# -------------------------------
# Prediction Function
# -------------------------------
def predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm, bz_gse, density, theta_gse, bx_gsm, bx_gse):
    try:
        input_data = np.array([[float(dst_t_1), float(dst_t_2), float(dst_t_3), float(bz_gsm),
                                float(theta_gsm), float(bz_gse), float(density), float(theta_gse),
                                float(bx_gsm), float(bx_gse)]])
        prediction = model.predict(input_data)[0][0]
        classification = classify_dst(prediction)
        return prediction, classification
    except ValueError:
        return "Error: Invalid input", None

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="DST Predictor", layout="wide", page_icon="🌌")

    # Theme-compatible minimal styling
    st.markdown("""
        <style>
            .stButton>button {
                font-size: 16px;
                border-radius: 8px;
                padding: 6px 16px;
            }
            .stTextInput>div>input {
                border-radius: 5px;
            }
            h1, h5 {
                text-align: center;
            }
            .center-button {
                display: flex;
                justify-content: center;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Info
    with st.sidebar:
        st.markdown("## 🌌 **DST Predictor Info**")
        st.markdown("""
        - Forecasts **Disturbance Storm Time (DST)** index using AI.  
        - Useful for satellite ops, power grids & space weather tracking.
        """)
        st.markdown("---")

        with st.expander("📊 **Classification Guide**"):
            classification_data = pd.DataFrame({
                "DST Range (nT)": ["> -20", "-20 to -50", "-50 to -100", "-100 to -200", "< -200"],
                "Classification": [
                    "🟢 Quiet",
                    "🟢 Unsettled",
                    "🟠 Moderate Storm",
                    "🔴 Intense Storm",
                    "🔴 Extreme Storm"
                ]
            })
            st.table(classification_data)

        with st.expander("⚙️ **Model Details**"):
            st.markdown("""
            - **Model**: DNN + Quadratic Neurons  
            - **Optimizer**: Adam (LR: 0.0005)  
            - **Loss**: Mean Squared Error  
            - **Metric**: Mean Absolute Error  
            - **Inputs**: 10 space weather features  
            """)

        with st.expander("📚 **About DST**"):
            st.markdown("""
            - DST quantifies Earth's magnetic field changes.  
            - Lower values indicate stronger geomagnetic storms.
            """)

        st.markdown("---")
        st.markdown("👨‍💻 Built by **Akshwin T, Ravin D, Vinay Deep Jaiswal**")
        st.markdown("📬 [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")
        st.markdown("📬 [ravind.2003@gmail.com](mailto:ravind.2003@gmail.com)")
        st.markdown("📬 [vinaydeepjaiswal@gmail.com](mailto:vinaydeepjaiswal@gmail.com)")

    # Header
    st.markdown("<h1>🌌 DST Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h5>AI meets Space Weather | Powered by Quadratic Neurons</h5>", unsafe_allow_html=True)

    # Input Form
    with st.form(key="dst_form"):
        col1, col2 = st.columns(2)
        with col1:
            dst_t_1 = st.text_input("DST_t-1 (nT)", "0.0")
            dst_t_2 = st.text_input("DST_t-2 (nT)", "0.0")
            dst_t_3 = st.text_input("DST_t-3 (nT)", "0.0")
            bz_gsm = st.text_input("Bz_GSM (nT)", "0.0")
            theta_gsm = st.text_input("Theta_GSM (°)", "0.0")
        with col2:
            bz_gse = st.text_input("Bz_GSE (nT)", "0.0")
            density = st.text_input("Density (cm⁻³)", "0.0")
            theta_gse = st.text_input("Theta_GSE (°)", "0.0")
            bx_gsm = st.text_input("Bx_GSM (nT)", "0.0")
            bx_gse = st.text_input("Bx_GSE (nT)", "0.0")

        # Centered Predict Button
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🌠 Predict DST")
        st.markdown('</div>', unsafe_allow_html=True)

    # Output Section
    if submit_button:
        predicted_dst, classification = predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm,
                                                    bz_gse, density, theta_gse, bx_gsm, bx_gse)
        st.markdown("### 📈 Prediction Result")
        if classification:
            st.success(f"**Predicted DST:** `{predicted_dst:.2f} nT`")
            st.info(f"**Classification:** `{classification}`")
        else:
            st.error(predicted_dst)

if __name__ == "__main__":
    main()
