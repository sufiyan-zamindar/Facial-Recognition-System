# Facial Recognition System using Machine Learning



## 📌 Project Overview

This project implements a robust **Facial Recognition System** using **Machine Learning** models trained on numerical facial feature vectors. Instead of working with raw image files, we use high-dimensional facial embeddings for identity classification and verification.

The project was developed as a part of my internship at **Evoastra Pvt. Ltd.**, focusing on real-time biometric verification through a user-friendly interface.

---

## 🎯 Objectives

- Build a reliable ML model to recognize faces using feature vectors.
- Preprocess, normalize, and visualize high-dimensional facial data.
- Train multiple classification models and evaluate their performance.
- Deploy the model using **Streamlit** for real-time identity prediction.

---

## 📂 Dataset

- Input Data: Facial features represented as numerical vectors (CSV format).
- Labels: Identity of individuals (used for supervised classification).
- Preprocessing: 
  - Null value checks
  - Min-Max normalization
  - Dimensionality reduction using **PCA**

> Note: Due to privacy, the dataset is not included in this repo.

---

## 🧠 Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Random Forest Classifier
- Multi-Layer Perceptron (MLP)

Each model was evaluated using:
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

---

## 🚀 Future Enhancement

The best-performing model can be deploy using **Streamlit**, allowing users to:
- Upload face vector input (CSV/NumPy array)
- Predict the identity of the person
- View confidence scores and prediction results

---

## Conclusion

This project demonstrates a lightweight yet effective way to implement facial recognition using feature vectors and classical ML models. The system is extendable for real-world use cases in security, authentication, and attendance systems.

---

## Acknowledgment

Special thanks to Evoastra Pvt. Ltd. for the guidance and opportunity to work on this project during my internship.

---

## Connect With Me

- Feel free to connect with me on https://www.linkedin.com/in/sufiyan012/
- For queries or collaborations: sufiyanzamindar012@gmail.com

---
