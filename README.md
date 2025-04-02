# DST Prediction

## Overview
This project aims to predict the **Dst (Disturbance Storm Time) index**, which measures geomagnetic activity, using an Artificial Neural Network (ANN) with quadratic neurons. The model leverages historical and real-time data to forecast geomagnetic disturbances that impact satellite communications, power grids, and GPS systems.

## Features
- **Data Preprocessing**: Handling missing values, feature selection, and correlation analysis.
- **Lag Features**: Incorporating previous time step values (`dst_t-1`, `dst_t-2`, `dst_t-3`) to capture temporal dependencies.
- **Feature Scaling**: Standardization using `StandardScaler` to improve model performance.
- **Quadratic Layer**: A custom deep learning layer that enhances feature representation.
- **ANN Model**: A deep learning model with multiple quadratic layers for robust learning.
- **Training Enhancements**: Adaptive learning rate scheduler, early stopping, and dropout layers.
- **Evaluation Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Dataset
The dataset includes multiple features related to space weather and geomagnetic activity, including:
- **Solar Wind Parameters**: Velocity, density, IMF components
- **Interplanetary Magnetic Field (IMF) Data**
- **Historical Dst Index Values**

## Installation
Ensure you have Python 3.x installed along with the required dependencies. Install the necessary libraries using:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage
1. **Load the Dataset**: Import and preprocess the dataset, including handling missing values.
2. **Feature Engineering**: Add lag features, drop irrelevant columns, and select the most correlated features.
3. **Data Visualization**: Generate a heatmap to analyze feature correlation.
4. **Model Training**:
   - Use an ANN with custom `QuadraticLayer`.
   - Apply dropout to prevent overfitting.
   - Use early stopping and learning rate decay.
5. **Evaluation**:
   - Assess model performance using MSE, MAE, and R² metrics.
   - Compare predictions with actual DST values.

## Results
- **Model Performance**:
  - The ANN model with quadratic layers effectively captures complex dependencies.
  - Lag features significantly improve prediction accuracy.
- **Future Improvements**:
  - Hyperparameter tuning to optimize model performance.
  - Exploring LSTMs or Transformers for better temporal sequence modeling.

## License
This project is licensed under the MIT License.

---
**Contributors:** Akshwin T

Feel free to contribute by improving the model or adding new features!

