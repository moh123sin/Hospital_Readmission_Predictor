# 🏥 Hospital Readmission Predictor

A machine learning web app that predicts 30-day hospital readmission risk for patients using clinical features.

## 🔗 Live Demo
[Open the Hospital Readmission Predictor](https://hospitalreadmissionpredictor-teyrx8aebzjqrukfjn2opx.streamlit.app/)

## 🎯 Business Problem
Hospital readmissions within 30 days cost the US healthcare system $26 billion annually. Early identification of high-risk patients allows hospitals to intervene with targeted discharge planning and follow-up care.

## 🛠️ Tech Stack
- **Python** · Pandas · NumPy
- **Scikit-learn** — Random Forest, cross-validation, ROC-AUC
- **Imbalanced-learn** — SMOTE for class imbalance handling
- **Streamlit** — interactive web app
- **Matplotlib / Seaborn** — visualizations

## 📊 Features
- **Dashboard** — readmission rates by age group, discharge type, prior history
- **Live Predictor** — enter patient details and get real-time risk score
- **Model Performance** — ROC curve, confusion matrix, classification report
- **Data Explorer** — filter and download the dataset

## 🧠 ML Approach
1. Feature engineering on clinical data (age, diagnoses, medications, HbA1c)
2. SMOTE oversampling to handle class imbalance
3. Random Forest classifier (200 trees, balanced class weights)
4. 5-fold cross-validation for robust evaluation
5. ROC-AUC as primary metric (more meaningful than accuracy for imbalanced data)

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
hospital_readmission/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md
```

## 📈 Results
| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.78 |
| Cross-val AUC | ~0.77 |
| Precision | ~0.72 |
| Recall | ~0.68 |
