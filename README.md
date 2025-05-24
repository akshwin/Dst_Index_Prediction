# 🌌 DST Predictor – Geomagnetic Storm Forecasting App

An AI-powered web application that predicts the **Disturbance Storm Time (DST)** index using 10 space weather parameters. Built with **Streamlit**, powered by a custom **Quadratic Neural Network**, and optimized for real-time inference.

🌐 **Live Demo:**  
[![Live Demo](https://img.shields.io/badge/Visit%20App-Click%20Here-blue?style=for-the-badge)](https://dst-predictor.streamlit.app/)

---

## 🚀 Features

- Predicts DST index based on past values and solar wind inputs.
- Classifies space weather conditions into Quiet, Unsettled, or Storm categories.
- Simple and interactive interface using Streamlit.
- Visual display of input features and prediction results.
- Backend powered by a custom deep learning model with quadratic neurons.

---

## 🧠 Model Architecture

- **Input:** 10 space weather parameters  
- **Hidden Layers:**  
  - Dense (128) → Dropout  
  - Quadratic Layer (64) → Dropout  
  - Quadratic Layer (32) → Dropout  
  - Dense (16) → Dropout  
- **Output:** 1 neuron for DST prediction  
- **Optimizer:** Adam (lr=0.0005)  
- **Loss:** Mean Squared Error (MSE)  
- **Metrics:** Mean Absolute Error (MAE)

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- TensorFlow / Keras
- NumPy, Pandas, Pillow
- Matplotlib (optional)
- Deployed on Streamlit Cloud

---

## 📁 Project Structure

```

dst-predictor/
├── app.py
├── model/
│   └── dst\_model.h5
├── utils/
│   └── preprocess.py (optional)
├── requirements.txt
├── README.md
└── ...

````

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dst-predictor.git
cd dst-predictor
````

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App Locally

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧾 Input Features

| Feature    | Description               |
| ---------- | ------------------------- |
| DST\_t-1   | DST index 1 hour ago      |
| DST\_t-2   | DST index 2 hours ago     |
| DST\_t-3   | DST index 3 hours ago     |
| Bz\_GSM    | IMF Bz in GSM coordinates |
| Theta\_GSM | IMF clock angle in GSM    |
| Bz\_GSE    | IMF Bz in GSE coordinates |
| Density    | Solar wind density        |
| Theta\_GSE | IMF clock angle in GSE    |
| Bx\_GSM    | IMF Bx in GSM coordinates |
| Bx\_GSE    | IMF Bx in GSE coordinates |

---

## 📊 Output Classification

| DST Range (nT) | Classification |
| -------------- | -------------- |
| > -20          | Quiet          |
| -20 to -50     | Unsettled      |
| -50 to -100    | Moderate Storm |
| -100 to -200   | Intense Storm  |
| < -200         | Extreme Storm  |

---

## 📦 Requirements

`requirements.txt` should include:

```
numpy
pandas
matplotlib
keras
tensorflow
Pillow
ipython
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Made with 🌌 by **Akshwin T**
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/) | [GitHub](https://github.com/akshwin)

---

