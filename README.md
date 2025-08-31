# Customer Churn Prediction

## 🚀 Project Overview
Customer churn is one of the biggest challenges for subscription-based businesses. Predicting which customers are likely to leave allows companies to take proactive measures to retain them.  

This project implements a **machine learning solution** to predict customer churn using a Random Forest classifier, with an **interactive Streamlit web app** for real-time predictions.

---

## 📊 Dataset
- **Source:** [Your dataset source / synthetic]  
- **Number of records:** 1,000+ customers  
- **Columns:**
| Feature                    | Type        | Description                          |
|----------------------------|-------------|--------------------------------------|
| ContractType               | Categorical | Month-to-month / One year / Two year |
| TechSupport                | Categorical | Yes / No                             |
| Gender                     | Categorical | Male / Female                        |
| Tenure                     | Numerical   | Number of months customer stayed     |
| MonthlyCharges             | Numerical   | Monthly subscription fee             |
| TotalCharges               | Numerical   | Total billed amount                  |
| InternetService_FiberOptic | Categorical | Yes / No                             |
| Churn                      | Target      | Customer churn (Yes / No)            |


- **Class distribution:**  
    Yes: 883
    No: 117

> Note: The dataset is imbalanced, which is handled during modeling.

---

## 🛠 Data Preprocessing
- Handled **categorical variables** with label encoding.  
- Created **feature engineering columns**:
  - Age Groups
  - Tenure Groups
  - Support Call Rate
  - Long Contract Indicator
- Checked feature correlation with **Churn** using ANOVA / Chi-square.  
- Applied **SMOTE** to handle class imbalance in the training set.

---

## 📈 Modeling
### Models Tried:
- Logistic Regression
- KNN
- Decision tree
- Random Forest Classifier
- Support Vector Classifier
- XGBoost
- LightGBM

### Final Model:
- **Random Forest Classifier** with tuned hyperparameters using `RandomizedSearchCV`  
- **Class imbalance handled** using `class_weight='balanced'` + SMOTE  
- Threshold adjustment applied for better “No Churn” prediction  

### Model Performance:
| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 1.0 | 0.99 |
| F1 Score | 1.0 | 0.9943 |
| Mean CV Accuracy | 0.995 | - |

---

## 🎯 Deployment
- Built an **interactive Streamlit app**  
- Users can input customer details using dropdowns and sliders  
- Predicts **Churn probability** with visual feedback (probability bar, emojis)  
- Integrated **trained Random Forest model** with scaling and preprocessed features  
- Deployed on **[Streamlit Cloud](Your App Link)**

---

## ⚙️ How to Run Locally
```bash
# Clone the repo
git clone https://github.com/AdeenaRamzan/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

DEMO: https://customer-churn-prediction-ml-model.streamlit.app/
