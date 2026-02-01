**🩺 Breast Cancer Diagnostic Assistant using ANN**

**Live Demo :"" https://breast-cancer-detection-by-tejas.streamlit.app/

An AI-powered clinical decision support system that predicts whether a breast tumor is Benign or Malignant using an Artificial Neural Network (ANN).
The application features a doctor-friendly Streamlit UI designed for clarity, simplicity, and real-world usability.


**📌 Project Overview**
  Early detection of breast cancer is critical for effective treatment.
This project uses a supervised deep learning approach to classify tumors based on clinical measurements derived from biopsy images.

**system:**

Takes patient tumor measurements as input

Applies the same preprocessing used during training

Uses a trained ANN model to predict diagnosis

Displays results in a clear, medical-friendly UI


**🛠️ Tech Stack**

Python

TensorFlow / Keras

Scikit-learn

Pandas & NumPy

Streamlit


**🚀 Features**

🧠 ANN-based binary classification (Benign / Malignant)

🩺 Doctor-friendly, clean Streamlit interface

📊 Probability-based prediction output

🔄 Proper preprocessing with saved scaler

💾 Model & scaler persistence (production-ready)

⚠️ Medical disclaimer for ethical usage



**📊 Dataset Information**


Source: Kaggle – Breast Cancer Wisconsin (Diagnostic) Dataset

Type: Binary Classification

Samples: 569

Features: 30 numerical clinical features

Target Variable:
                0 → Benign
                1 → Malignant


**🧠 Model Details**

Model Type: Artificial Neural Network (ANN)
Framework: TensorFlow / Keras

Architecture:
Input Layer (30 features)
Hidden Layers with ReLU activation
Output Layer with Sigmoid activation
Loss Function: Binary Cross-Entropy
Optimizer: Adam
Evaluation Metric: Accuracy


**🧪 Machine Learning Pipeline**

Data loading and exploration

Label encoding of target variable

Feature scaling using StandardScaler

Train-test split

ANN model training

Model evaluation

Saving trained model & scaler

Deployment using Streamlit


**🩺 Application UI (Doctor-Friendly)**

The UI is designed to:
Use clinical terminology, not ML jargon
Clearly separate input and output sections
Highlight diagnosis using color-coded results
Show confidence level for decision support

**⚠️ Note:**
This application is intended for decision support only and must not replace professional medical diagnosis.


**📈 Model Performance**

Achieved very high accuracy on test data
Performance validated using:
Training vs validation metrics
Confusion matrix
Probability-based predictions

High accuracy is expected due to the clean and well-structured nature of the dataset, but additional validation steps were performed to avoid overfitting.

**🧾 Disclaimer**

This software is an AI-assisted diagnostic support tool.
It is not a substitute for professional medical diagnosis, clinical tests, or expert judgment.


**🚀 Future Enhancements**

CSV upload for bulk patient screening
ROC-AUC and Confusion Matrix visualization
Model explainability (SHAP / feature importance)
Cloud deployment (Streamlit Cloud / AWS)
Role-based access (Doctor / Admin)

**👨‍💻 Author**
**Tejas Nikam
Aspiring Data Scientist | Machine Learning Enthusiast**
