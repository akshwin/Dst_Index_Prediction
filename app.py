import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

class QuadraticLayer(Layer):
    def __init__(self, units, **kwargs):
        super(QuadraticLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(input_shape[-1], self.units), 
                                  initializer="he_normal",  # More stable than Glorot
                                  trainable=True)
        self.W2 = self.add_weight(shape=(input_shape[-1], self.units), 
                                  initializer="he_normal",  # Helps prevent exploding values
                                  trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        quadratic_output = K.dot(inputs, self.W1) + K.dot(K.square(inputs), self.W2) + self.b
        return K.clip(quadratic_output, -1e3, 1e3)  # Clip extreme values to prevent NaNs

# Define input using Functional API
model = Sequential([
    Dense(10, activation=None, input_shape=(10,)),  # Replacing Input() layer

    QuadraticLayer(128),
    Dropout(0.2),

    QuadraticLayer(64),
    Dropout(0.2),

    QuadraticLayer(32),
    Dropout(0.2),

    QuadraticLayer(16),

    Dense(1)  # Output layer
])

# Define an adaptive learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Compile the model with Adam optimizer and learning rate decay
model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0),  # Gradient clipping prevents explosion
              loss='mse',
              metrics=['mae'])

print("Model loaded successfully!")
    
def classify_dst(dst_value):
    """Classify the DST value into categories."""
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

def predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm, bz_gse, density, theta_gse, bx_gsm, bx_gse):
    """Predicts the DST value using the trained model and classifies it."""
    try:
        input_data = np.array([[float(dst_t_1), float(dst_t_2), float(dst_t_3), float(bz_gsm),
                                float(theta_gsm), float(bz_gse), float(density), float(theta_gse),
                                float(bx_gsm), float(bx_gse)]])
        
        prediction = model.predict(input_data)[0][0]  # Extract scalar value
        classification = classify_dst(prediction)
        
        return prediction, classification
    except ValueError:
        return "Error: Please enter valid numerical values", None

# Streamlit UI
def main():
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">DST Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Sidebar Section
    st.sidebar.header("About")
    st.sidebar.info("The DST Predictor is an advanced AI application designed to predict the Disturbance Storm Time (DST) index.")
    st.sidebar.info("This application utilizes Quadratic Neurons to generate accurate DST predictions.")
    
    # Adding classification table in sidebar
    st.sidebar.markdown("""
    | **DST Value (nT)**   | **Classification**            |
    |----------------------|------------------------------|
    | **> -20**           | Quiet                        |
    | **-20 to -50**      | Unsettled                    |
    | **-50 to -100**     | Moderate Storm               |
    | **-100 to -200**    | Intense Storm                |
    | **< -200**          | Extreme Geomagnetic Storm    |
    """, unsafe_allow_html=True)
    
    # Input fields
    
    dst_t_1 = float(st.text_input("DST_t-1", "0.0"))
    dst_t_2 = float(st.text_input("DST_t-2", "0.0"))
    dst_t_3 = float(st.text_input("DST_t-3", "0.0"))
    bz_gsm = float(st.text_input("Bz_GSM", "0.0"))
    theta_gsm = float(st.text_input("Theta_GSM", "0.0"))
    bz_gse = float(st.text_input("Bz_GSE", "0.0"))
    density = float(st.text_input("Density", "0.0"))
    theta_gse = float(st.text_input("Theta_GSE", "0.0"))
    bx_gsm = float(st.text_input("Bx_GSM", "0.0"))
    bx_gse = float(st.text_input("Bx_GSE", "0.0"))


    result = ""

    # if st.button("Predict"):
    #     result = predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm, bz_gse, density, theta_gse, bx_gsm, bx_gse)
    #     st.success(f"The output is: {result}")
    
    if st.button("Predict"):
        predicted_dst, category = predict_dst(dst_t_1, dst_t_2, dst_t_3, bz_gsm, theta_gsm, bz_gse, density, theta_gse, bx_gsm, bx_gse)
    
        if category:
            st.success(f"The predicted DST value is: {predicted_dst:.2f}")
            st.info(f"Classification: {category}")
            
if __name__ == '__main__':
    main()
