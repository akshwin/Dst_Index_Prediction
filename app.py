import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Compile model
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
    st.set_page_config(page_title="DST Predictor", layout="centered", page_icon="🌌")

    # Sidebar with expandable info sections
    with st.sidebar:
        st.markdown("## 🌌 About the App")
        st.markdown("""
        This AI-powered application predicts the **Disturbance Storm Time (DST)** index using space weather data.  
        DST is crucial for understanding geomagnetic storms that can impact satellites and power grids.

        **Model:** Custom Deep Neural Network with Quadratic Neurons  
        **Metric:** Mean Squared Error (MSE)
        """)
        
        with st.expander("📊 DST Classification Guide"):
            st.table(pd.DataFrame({
                "DST Range (nT)": ["> -20", "-20 to -50", "-50 to -100", "-100 to -200", "< -200"],
                "Classification": ["Quiet", "Unsettled", "Moderate Storm", "Intense Storm", "Extreme Storm"]
            }))

        with st.expander("⚙️ Model Details"):
            st.markdown("""
            - **Network Architecture**: Custom model with **Quadratic Neurons**  
            - **Optimizer**: Adam (learning rate: 0.0005)  
            - **Loss Function**: Mean Squared Error (MSE)  
            - **Metrics**: Mean Absolute Error (MAE)  
            - **Training Dataset**: Based on space weather parameters for geomagnetic storm prediction.
            """)
        
        with st.expander("📚 About the DST Index"):
            st.markdown("""
            The **Disturbance Storm Time (DST)** index measures the strength of geomagnetic storms.  
            - **Quiet**: No geomagnetic storm activity.  
            - **Unsettled**: Some geomagnetic activity.  
            - **Moderate Storm**: More intense storm activity.  
            - **Intense Storm**: Severe storm conditions.  
            - **Extreme Storm**: Very severe geomagnetic storm, with potential for major disruptions.
            """)

        st.markdown("---")
        st.markdown("👨‍💻 Developed by Akshwin T")
        st.markdown("📬 Contact: [akshwint.2003@gmail.com](mailto:youremail@example.com)")

    # Input form
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🌌 DST Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Powered by AI & Quadratic Neurons</h5>", unsafe_allow_html=True)

    with st.form(key="dst_form"):
        st.markdown("### 🔢 Input Parameters")
        col1, col2 = st.columns(2)
        with col1:
            dst_t_1 = st.text_input("DST_t-1 (nT)", "0.0")
            dst_t_2 = st.text_input("DST_t-2 (nT)", "0.0")
            dst_t_3 = st.text_input("DST_t-3 (nT)", "0.0")
            bz_gsm = st.text_input("Bz_GSM (nT)", "0.0")
            theta_gsm = st.text_input("Theta_GSM (deg)", "0.0")
        with col2:
            bz_gse = st.text_input("Bz_GSE (nT)", "0.0")
            density = st.text_input("Density (cm⁻³)", "0.0")
            theta_gse = st.text_input("Theta_GSE (deg)", "0.0")
            bx_gsm = st.text_input("Bx_GSM (nT)", "0.0")
            bx_gse = st.text_input("Bx_GSE (nT)", "0.0")

        submit_button = st.form_submit_button(label="🌠 Predict DST")

    # Prediction result
    if submit_button:
        predicted_dst, classification = predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm,
                                                    bz_gse, density, theta_gse, bx_gsm, bx_gse)
        if classification:
            st.success(f"🌟 **Predicted DST:** {predicted_dst:.2f} nT")
            st.info(f"📊 **Classification:** {classification}")
        else:
            st.error(predicted_dst)

if __name__ == "__main__":
    main()
