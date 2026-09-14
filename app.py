import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        font-weight: 800;
        text-align: center;
        padding-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #c7d2fe;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
       [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border-radius: 15px !important;
    padding: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
[data-testid="stVerticalBlockBorderWrapper"] * {
    color: #1e293b !important;
}
    }
    .card h4, .card label, .card p {
        color: #1e293b !important;
    }

    /* Force label text visible */
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Selectbox / input boxes inside cards */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    .stNumberInput input {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border-radius: 8px;
    }

    /* Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white !important;
        font-weight: 700;
        font-size: 18px;
        padding: 12px 0;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.5);
    }

    /* Result boxes */
    .result-churn {
        background: linear-gradient(90deg, #fee2e2, #fecaca);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border-left: 6px solid #ef4444;
        color: #7f1d1d !important;
    }
    .result-stay {
        background: linear-gradient(90deg, #dcfce7, #bbf7d0);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border-left: 6px solid #22c55e;
        color: #14532d !important;
    }
    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

# Header
st.markdown("<h1>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a customer is likely to churn using Machine Learning</div>", unsafe_allow_html=True)

input_data = {}

# Layout in columns/cards
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 👤 Personal Info")
        input_data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        input_data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
        input_data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
        input_data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        input_data["tenure"] = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    with st.container(border=True):
        st.markdown("#### 📞 Services")
        input_data["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
        input_data["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        input_data["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        input_data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        input_data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        input_data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

with col3:
    with st.container(border=True):
        st.markdown("#### 💳 Billing")
        input_data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        input_data["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        input_data["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        input_data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
        input_data["PaymentMethod"] = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        input_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        input_data["TotalCharges"] = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.write("")

# Predict button
if st.button("🔮 Predict Churn"):
    df_input = pd.DataFrame([input_data])
    for col in df_input.select_dtypes(include="object").columns:
        df_input[col] = encoders[col].transform(df_input[col])
    df_input = df_input[columns]

    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    if prediction == 1:
        st.markdown(f"""
            <div class='result-churn'>
                <div class='result-title'>⚠️ Likely to Churn</div>
                <div>Confidence: {proba[1]*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-stay'>
                <div class='result-title'>✅ Likely to Stay</div>
                <div>Confidence: {proba[0]*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("This app predicts customer churn using a Random Forest model trained on the Telco Customer Churn dataset.")
    st.markdown("### 🛠️ Tech Stack")
    st.write("- Python\n- Scikit-learn\n- Streamlit\n- Pandas")