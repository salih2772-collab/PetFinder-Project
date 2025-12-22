#  PetFinder Adoption Speed Predictor

## Project Overview
This project aims to predict the adoption speed of pets based on their profiles from PetFinder.my. By analyzing tabular data, text descriptions, and images, the model helps shelters identify pets that might wait longer for adoption, allowing them to improve profiles and strategies.

**Author:** Seyedali Hosseini 
**Date:** December 2025

##  Key Features
* **Multi-Modal Analysis:** Combines metadata (Tabular), text descriptions (TF-IDF), and image features (MobileNetV2 CNN & Hand-crafted features).
* **Advanced Modeling:** Utilizes a **Stacking Ensemble** (XGBoost + Random Forest + SVM) achieving a QWK score of **0.34**.
* **Explainable AI (XAI):** Interprets model decisions using **SHAP** (Global Feature Importance).
* **Web Application:** A containerized Streamlit app for real-time predictions.

## 📂 Project Structure
```text
PetFinder-Project/
├── Petfinder2.ipynb       # Main Jupyter Notebook (Training, EDA, & XAI)
├── app.py                 # Streamlit Web Application
├── Dockerfile             # Docker configuration
├── requirements.txt       # Dependencies
├── *.pkl                  # Trained models (Stacking, Scaler, PCA, etc.)
└── README.md              # Project documentation


