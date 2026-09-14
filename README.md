# 📊 Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer is likely to churn, built with **Python**, **Scikit-learn**, and **Streamlit**.

## 🔍 Overview

Customer churn — when a customer stops using a company's service — is a critical metric for subscription-based businesses. This project uses a **Random Forest Classifier** trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict churn based on customer demographics, account information, and service usage.

The trained model is deployed through an interactive **Streamlit** web app, allowing users to input customer details and instantly see a churn prediction with a confidence score.

## ✨ Features

- Predicts customer churn in real time based on 19 input features
- Clean, card-based UI organized into Personal Info, Services, and Billing sections
- Displays prediction confidence percentage
- Random Forest model trained with Scikit-learn
- Simple, reusable model training pipeline

## 🛠️ Tech Stack

- **Python** – core programming language
- **Pandas / NumPy** – data manipulation and preprocessing
- **Scikit-learn** – model training (Random Forest Classifier) and evaluation
- **Streamlit** – interactive web app / UI
- **Joblib** – model and encoder serialization

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (from Kaggle)
├── train_model.py                          # Script to train and save the model
├── app.py                                  # Streamlit web application
├── churn_model.pkl                         # Saved trained model (generated)
├── encoders.pkl                            # Saved label encoders (generated)
├── columns.pkl                             # Saved column order (generated)
└── README.md                               # Project documentation
```

## 📦 Dataset

**Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customer records with 21 attributes including demographics (gender, senior citizen status), account details (tenure, contract type, payment method), and subscribed services (internet, phone, streaming), along with the target label `Churn` (Yes/No).

## ⚙️ Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

3. **Download the dataset**
   Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root folder.

## 🚀 Usage

1. **Train the model**
   ```bash
   python train_model.py
   ```
   This will clean the data, train a Random Forest model, print the accuracy, and save `churn_model.pkl`, `encoders.pkl`, and `columns.pkl`.

2. **Run the web app**
   ```bash
   streamlit run app.py
   ```

3. Open the local URL shown in the terminal (usually `http://localhost:8501`), fill in the customer details, and click **Predict Churn** to see the result.

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier
- **Preprocessing:** Label Encoding for categorical features, missing value handling for `TotalCharges`
- **Train/Test Split:** 80/20
- **Evaluation Metric:** Accuracy Score

## 📈 Future Improvements

- Add more advanced models (XGBoost, Logistic Regression comparison)
- Hyperparameter tuning with GridSearchCV
- Add feature importance visualization
- Deploy the app on Streamlit Community Cloud

## 👤 Author

**Aryan Khirasariya**
Data Science Intern Project

## 📄 License

This project is for educational purposes as part of an internship/academic project.