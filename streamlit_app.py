import streamlit as st
import numpy as np
import joblib

# Load trained models
gb_model = joblib.load("gb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")

st.set_page_config(
    page_title="Temperature Prediction App",
    page_icon="⛅",
    layout="centered"
)

st.title("⛅ Daily Mean Temperature Prediction")
st.write(
    "This app uses trained machine learning models to predict the **mean temperature** "
    "based on daily weather conditions."
)

# Sidebar: model selection
model_choice = st.sidebar.selectbox(
    "Select model to use",
    ("Gradient Boosting (Best)", "Random Forest", "Linear Regression")
)

st.subheader("Enter Weather Inputs")

cloud_cover = st.number_input("Cloud Cover (0–8)", min_value=0, max_value=8, value=5)
humidity = st.number_input("Humidity (0–1)", min_value=0.0, max_value=1.0, value=0.6)
pressure = st.number_input("Pressure (e.g. 1.0–1.1 kPa)", min_value=0.0, value=1.02)
global_radiation = st.number_input("Global Radiation", value=0.50)
precipitation = st.number_input("Precipitation", value=0.00)
sunshine = st.number_input("Sunshine (hours)", value=4.0)
temp_min = st.number_input("Minimum Temperature (°C)", value=3.0)
temp_max = st.number_input("Maximum Temperature (°C)", value=8.0)

# IMPORTANT: the feature order here MUST match the order you used when training in Python
features = np.array([[cloud_cover, humidity, pressure,
                      global_radiation, precipitation,
                      sunshine, temp_min, temp_max]])

if st.button("Predict Mean Temperature"):
    if model_choice == "Gradient Boosting (Best)":
        pred = gb_model.predict(features)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(features)[0]
    else:
        pred = lr_model.predict(features)[0]

    st.success(f"Predicted Mean Temperature: {pred:.2f} °C")
