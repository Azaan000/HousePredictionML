import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM UI (BLACK + RED) ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1a1a1a, #000000);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    color: #ff3b3b;
    font-size: 48px;
    font-weight: 800;
}
h2, h3 {
    color: #ff5c5c;
}

[data-testid="stSidebar"] {
    background: rgba(20,20,20,0.95);
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 0 30px rgba(255,60,60,0.15);
}

.result-card {
    background: linear-gradient(135deg, #ff3b3b, #8b0000);
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    box-shadow: 0 0 45px rgba(255,0,0,0.45);
}

.stButton>button {
    background: linear-gradient(90deg, #ff3b3b, #b30000);
    color: white;
    border-radius: 14px;
    padding: 12px 26px;
    font-size: 18px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff5c5c, #ff0000);
    transform: scale(1.05);
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "data", "house_prediction_model.pkl")
model = joblib.load(model_path)

# ---------------- FEATURE ENGINEERING ----------------
def preprocess_input(df):
    df = df.copy()
    df["total_rooms"] = np.log(df["total_rooms"] + 1)
    df["total_bedrooms"] = np.log(df["total_bedrooms"] + 1)
    df["population"] = np.log(df["population"] + 1)
    df["households"] = np.log(df["households"] + 1)
    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["household_rooms"] = df["total_rooms"] / df["households"]
    return df

# ---------------- TITLE ----------------
st.markdown("<h1>🏠 AI House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:18px; color:#ccc;'>Predict median house values using a trained Random Forest model</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📊 Property Details")

longitude = st.sidebar.number_input("Longitude", value=-119.5)
latitude = st.sidebar.number_input("Latitude", value=35.3)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 60, 30)
total_rooms = st.sidebar.number_input("Total Rooms", value=2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=400)
population = st.sidebar.number_input("Population", value=1200)
households = st.sidebar.number_input("Households", value=350)
median_income = st.sidebar.slider("Median Income", 0.5, 15.0, 4.5)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# ---------------- INPUT VALIDATION ----------------
error = False
if total_bedrooms > total_rooms:
    st.error("❌ Total bedrooms cannot exceed total rooms")
    error = True

if households <= 0:
    st.error("❌ Households must be greater than 0")
    error = True

# ---------------- INPUT DATAFRAME ----------------
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

input_data = preprocess_input(input_data)

# One-hot encoding
ocean_encoded = pd.get_dummies(input_data["ocean_proximity"])
input_data = input_data.drop("ocean_proximity", axis=1)
input_data = input_data.join(ocean_encoded)

# Align columns
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict House Price") and not error:
    with st.spinner("Running AI model..."):
        prediction = model.predict(input_data)[0]

        tree_preds = np.array([
            tree.predict(input_data)[0]
            for tree in model.estimators_
        ])
        lower = np.percentile(tree_preds, 10)
        upper = np.percentile(tree_preds, 90)

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h2>Estimated House Value</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1>${prediction:,.0f}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:18px;'>Expected Range: ${lower:,.0f} – ${upper:,.0f}</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 🔍 Feature Importance")

fi_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

st.bar_chart(fi_df.set_index("Feature"))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 📈 Model Performance")
st.write("""
- **Model:** Random Forest Regressor  
- **R² Score:** ~0.81  
- **MAE:** ~32,000  
- **RMSE:** ~48,000  
""")
st.caption("Metrics calculated during offline evaluation on test data.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>House Prices Prediction Using Machine Learning</div>", unsafe_allow_html=True)
