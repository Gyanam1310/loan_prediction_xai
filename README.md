# Loan Approval Prediction using Explainable AI (SHAP & LIME)

## 📌 Overview
This project presents a machine learning pipeline for predicting loan approval decisions, enhanced with Explainable AI (XAI) techniques to improve transparency and interpretability.

The model leverages classification algorithms along with SMOTE for handling class imbalance and uses SHAP and LIME to explain model predictions.

---

## 🎯 Objectives
- Predict whether a loan application will be approved or not  
- Handle class imbalance using SMOTE  
- Improve model performance on minority class  
- Provide interpretability using SHAP and LIME  

---

## 📊 Dataset
The dataset contains applicant information such as:
- Age  
- Income  
- Credit Score  
- Number of Dependents  
- Home Ownership Status  

Target variable:
- `loan_approved` (0 = Not Approved, 1 = Approved)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights:
- Dataset is imbalanced (~78% approved, ~22% not approved)  
- Credit score has the strongest relationship with loan approval  
- Home ownership positively influences approval  
- Higher number of dependents reduces approval likelihood  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handling missing values  
- Encoding features  
- Train-test split  

### 2. Handling Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
- Improved minority class representation  

### 3. Model Training
- Logistic Regression  
- Random Forest (tuned)

### 4. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 📈 Results

| Model | Data | Accuracy | Recall (Minority Class) |
|------|------|---------|------------------------|
| Logistic Regression | Original | 0.905 | 0.66 |
| Random Forest | Original | 0.88 | 0.63 |
| Logistic Regression | SMOTE | 0.915 | 0.93 |
| Random Forest (Tuned) | SMOTE | **0.93** | **0.95** |

### 🔥 Key Insight:
SMOTE significantly improved the model’s ability to detect minority cases, making predictions more reliable and fair.

---

## 🤖 Explainable AI (XAI)

### 🔹 SHAP (Global + Local Interpretability)
- Identified **credit score** as the most important feature  
- Showed how features impact predictions globally  
- Waterfall plots explain individual predictions  

### 🔹 LIME (Local Interpretability)
- Explained individual predictions  
- Confirmed consistency with SHAP results  

---

## 📊 Key Findings
- Credit score is the most influential feature  
- Home ownership indicates financial stability  
- Dependents negatively impact approval probability  
- Income has moderate influence  
- Age has minimal impact  

---

## 🧠 Conclusion
The project demonstrates that combining machine learning with explainable AI techniques results in models that are both accurate and interpretable. Handling class imbalance using SMOTE significantly improves minority class prediction, making the system more reliable for real-world applications.

---

## 🚀 Future Work
- Use real-world financial datasets  
- Explore advanced models (XGBoost, Neural Networks)  
- Incorporate fairness and bias analysis  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- SHAP  
- LIME  
- Matplotlib, Seaborn  

---

## 📁 Project Structure
├── notebook.ipynb
├── shap_beeswarm.png
├── shap_bar.png
├── shap_waterfall.png
├── lime_plot.png
└── README.md


---

## ⭐ Highlights
- End-to-end ML pipeline  
- Strong focus on interpretability  
- Real-world financial relevance  
- Suitable for research and production use  

---