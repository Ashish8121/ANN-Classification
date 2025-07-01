# Churn Prediction using ANN

This project implements an Artificial Neural Network (ANN) to predict customer churn based on bank customer data.

---

## 🚀 **Project Overview**

- **Objective:** Classify whether a customer will leave the bank (churn) or stay.
- **Dataset:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers) with features like Credit Score, Geography, Gender, Age, Balance, and more.
- **Model:** Sequential ANN built using TensorFlow and Keras.

---

## 🧠 **Model Architecture**

- **Input Layer:** Features after encoding (e.g., Geography OneHot, Gender Label)
- **Hidden Layers:** 
  - Dense (64 units, ReLU)
  - Dense (32 units, ReLU)
- **Output Layer:** 
  - Dense (1 unit, Sigmoid) for binary classification

---

## ⚙️ **Technologies Used**

- Python
- Pandas & NumPy
- Scikit-Learn (Preprocessing)
- TensorFlow & Keras (Model building)
- Streamlit (Optional for deployment)

---

## 💻 **How to Run**
https://ann-classification-gzsmu7y5sstxlsxuyvxmzq.streamlit.app/)
