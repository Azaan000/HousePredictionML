🚀 Project Overview
This project uses the California Housing Dataset to build a regression model capable of estimating house prices based on location, population statistics, housing features, and proximity to the ocean.

The trained model is deployed as a Streamlit web application, allowing users to input property details and receive:

A predicted house price
A confidence range
Feature importance explanation
🧠 Machine Learning Model
Algorithm: Random Forest Regressor
Target Variable: median_house_value
Evaluation Metrics:
R² ≈ 0.81
MAE ≈ 32,000
RMSE ≈ 48,000
The Random Forest model was selected due to its strong performance on nonlinear relationships in housing data.

🛠️ Features
🔢 User-friendly input form
🧮 Automated feature engineering (log transforms & ratios)
🌊 One-hot encoding for categorical features
📊 Feature importance visualization
📈 Prediction confidence range (10th–90th percentile)
🎨 Modern black & red UI with glassmorphism
⚠️ Input validation to prevent unrealistic data
📂 Project Structure

HousePrediction/ 
│── app.py 
│── data/ │ 
└── house_prediction_model.pkl
│── requirements.txt 
│── README.md

📦 Requirements

Python 3.8+

Streamlit 

Pandas NumPy 

Scikit-learn 

Joblib 

Matplotlib 

Seaborn
