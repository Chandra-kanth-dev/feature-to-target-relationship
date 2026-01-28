import streamlit as st
# =========================================
# IMPORTS
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# =========================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# CUSTOM CSS (AFTER set_page_config)
# =========================================
st.markdown("""
<style>

/* ---- GLOBAL BACKGROUND ---- */
.stApp {
    background: radial-gradient(circle at top, #1a0000 0%, #000000 45%, #000000 100%);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* ---- MAIN TITLE ---- */
h1 {
    color: #ff2c2c !important;
    text-align: center;
    font-size: 3.2rem !important;
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 0 0 15px rgba(255,44,44,0.8);
}

/* ---- SUBHEADINGS ---- */
h2, h3 {
    color: #ff5555 !important;
    text-shadow: 0 0 8px rgba(255,85,85,0.7);
}

/* ---- PARAGRAPH TEXT ---- */
p, label, span {
    color: #f2f2f2 !important;
    font-size: 1.05rem;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0000, #120000, #000000);
    border-right: 2px solid #ff2c2c;
}

/* ---- SIDEBAR TITLE ---- */
section[data-testid="stSidebar"] h2 {
    color: #ff2c2c !important;
    font-size: 1.6rem;
    text-shadow: 0 0 10px rgba(255,44,44,0.8);
}

/* ---- INPUT FIELDS ---- */
input, select {
    background-color: #0f0f0f !important;
    color: #ffffff !important;
    border: 1.5px solid #ff2c2c !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

/* ---- INPUT LABELS ---- */
label {
    color: #ff8a8a !important;
    font-weight: 600;
}

/* ---- BUTTON ---- */
div.stButton > button {
    background: linear-gradient(90deg, #ff0000, #ff4d4d);
    color: #ffffff;
    font-size: 1.2rem;
    font-weight: bold;
    border-radius: 14px;
    padding: 14px 26px;
    border: none;
    box-shadow: 0 0 25px rgba(255,0,0,0.9);
    transition: all 0.3s ease-in-out;
}

/* ---- BUTTON HOVER ---- */
div.stButton > button:hover {
    background: linear-gradient(90deg, #ff4d4d, #ff0000);
    box-shadow: 0 0 40px rgba(255,0,0,1);
    transform: scale(1.06);
}

/* ---- SUCCESS BOX ---- */
.stSuccess {
    background: linear-gradient(90deg, #0a0a0a, #1f0000);
    border-left: 6px solid #00ff88;
    box-shadow: 0 0 20px rgba(0,255,136,0.7);
}

/* ---- ERROR BOX ---- */
.stError {
    background: linear-gradient(90deg, #0a0a0a, #1f0000);
    border-left: 6px solid #ff0000;
    box-shadow: 0 0 25px rgba(255,0,0,0.9);
}

/* ---- METRIC BOX ---- */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0d0d0d, #1a0000);
    border-radius: 15px;
    padding: 20px;
    border: 1.5px solid #ff2c2c;
    box-shadow: 0 0 18px rgba(255,44,44,0.6);
}

/* ---- METRIC VALUE ---- */
div[data-testid="metric-container"] > div {
    color: #ff4444 !important;
    font-weight: 800;
}

/* ---- DATAFRAME ---- */
.dataframe {
    background-color: #0b0b0b;
    color: white;
}

/* ---- ANIMATED GLOW LINE ---- */
.glow-line {
    height: 4px;
    background: linear-gradient(90deg, red, black, red);
    animation: glowmove 3s infinite linear;
}

@keyframes glowmove {
    0% { background-position: 0%; }
    100% { background-position: 200%; }
}

</style>
""", unsafe_allow_html=True)

# =========================================
# APP CONTENT STARTS HERE
# =========================================


# ================== ADVANCED UI STYLING ==================



st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** "
    "to predict whether a loan will be approved by combining multiple ML models."
)

# =========================================
# LOAD DATA
# =========================================


@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🔍 HARD DEBUG — shows EXACT files Streamlit sees
    st.error("DEBUG: Files in app directory")
    st.write(os.listdir(BASE_DIR))

    DATA_PATH = os.path.join(BASE_DIR, "loan_prediction.csv")  # 👈 exact name

    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================================
# PREPROCESSING
# =========================================
df = df.drop(['Loan_ID', 'Gender'], axis=1)

le = LabelEncoder()
for col in ['Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
df['Dependents'].fillna(df['Dependents'].mean(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)

# =========================================
# FEATURES & TARGET
# =========================================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# MODELS
# =========================================
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
dt = DecisionTreeClassifier(class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced')
knn = KNeighborsClassifier()
svm = SVC(probability=True, class_weight='balanced')

base_models = [
    ('Logistic Regression', lr),
    ('Decision Tree', dt),
    ('Random Forest', rf)
]

meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

stacking_model = StackingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
    final_estimator=meta_model
)

# Train models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)

# =========================================
# SIDEBAR – USER INPUT
# =========================================
st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", value=360)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_history = 1 if credit_history == "Yes" else 0
employment = 0 if employment == "Salaried" else 1
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

user_data = np.array([[
    1,  # Married (assumed)
    0,  # Dependents
    1,  # Education (Graduate)
    employment,
    app_income,
    coapp_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
]])

user_data_scaled = scaler.transform(user_data)

# =========================================
# MODEL ARCHITECTURE DISPLAY
# =========================================
st.subheader("🧠 Model Architecture (Stacking)")
st.markdown("""
**Base Models Used**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression  
""")

# =========================================
# PREDICTION
# =========================================
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    lr_pred = lr.predict(user_data_scaled)[0]
    dt_pred = dt.predict(user_data_scaled)[0]
    rf_pred = rf.predict(user_data_scaled)[0]

    stack_pred = stacking_model.predict(user_data_scaled)[0]
    stack_prob = stacking_model.predict_proba(user_data_scaled)[0][1]

    st.subheader("📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred==1 else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred==1 else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred==1 else 'Rejected'}")

    st.subheader("🧠 Final Stacking Decision")

    if stack_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("📈 Confidence Score", f"{round(stack_prob*100,2)}%")

    # =========================================
    # BUSINESS EXPLANATION
    # =========================================
    st.subheader("💼 Business Explanation")
    st.write(
        f"""
        Based on the applicant's income, credit history, and combined predictions
        from multiple machine learning models, the system estimates the likelihood
        of loan repayment.

        **Final Decision:** {'Loan Approved' if stack_pred==1 else 'Loan Rejected'}
        """
    )

# =========================================
# MODEL COMPARISON
# =========================================
st.subheader("📊 Model Performance Comparison")

lr_acc = accuracy_score(y_test, lr.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))
stack_acc = accuracy_score(y_test, stacking_model.predict(X_test))

st.write(f"Logistic Regression Accuracy: {round(lr_acc,3)}")
st.write(f"Decision Tree Accuracy: {round(dt_acc,3)}")
st.write(f"Random Forest Accuracy: {round(rf_acc,3)}")
st.write(f"Stacking Model Accuracy: {round(stack_acc,3)}")

st.markdown("""
### 📌 Is stacking always better?
No. Stacking improves performance **when base models are diverse** and
make different types of errors. If base models are weak or similar,
stacking may not improve results.
""")


