# ❤️ Heart Disease Prediction System

A Machine Learning based web application that predicts the risk of heart disease using patient health parameters.

This project uses a Random Forest classifier trained on a heart disease dataset and is deployed using Streamlit.

---

## 🚀 Live Demo
()

---

## 📌 Features

- User-friendly Streamlit interface
- Random Forest ML model
- One-hot encoded categorical handling
- Real-time risk prediction
- Probability-based output
- Clean and professional UI

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- NumPy

---

## 📊 Model Details

- Algorithm: Random Forest Classifier
- Preprocessing:
  - One-hot encoding
  - Feature scaling using StandardScaler
- Evaluation: Train-test split (70-30)

---

## 🏗️ Project Structure


heart-disease-prediction-ml/
│
├── app.py
├── train_model.py
├── heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
├── requirements.txt
└── README.md


---

## ⚙️ Machine Learning Pipeline

1. Load dataset  
2. Separate features and target  
3. Apply one-hot encoding  
4. Train-test split (70/30)  
5. Feature scaling  
6. Train Random Forest model  
7. Save model artifacts using Joblib  
8. Deploy via Streamlit  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository

---
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction-ml.git

cd heart-disease-prediction-ml

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Train the Model (First Time Only)
python train_model.py


### 4️⃣ Run the Application


streamlit run app.py


---

## 📈 Model Information

- Algorithm: Random Forest Classifier
- Evaluation Method: Train-Test Split
- Output: Binary Classification (High Risk / Low Risk)
- Includes probability-based prediction score

---

## ⚠️ Disclaimer

This project is built for educational and demonstration purposes only.  
It should not be used as a substitute for professional medical diagnosis.

---

## 👩‍💻 Author

Vidya  
Machine Learning & Data Science Enthusiast

---

## 🌟 Future Improvements

- Add model comparison (Logistic Regression, XGBoost, SVM)
- Add SHAP explainability
- Deploy on AWS or Azure
- Add authentication system
- Add downloadable PDF medical report


