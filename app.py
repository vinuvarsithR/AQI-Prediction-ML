import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("aqi_model.pkl", "rb"))

st.title("Air Quality Index Prediction")

st.write("Enter pollutant levels to predict AQI")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")
city = st.number_input("City Code")
month = st.number_input("Month")
day = st.number_input("Day of Week")

if st.button("Predict AQI"):

    features = np.array([[pm25, pm10, no2, so2, co, o3, city, month, day]])

    prediction = model.predict(features)

    st.success(f"Predicted AQI: {prediction[0]}")