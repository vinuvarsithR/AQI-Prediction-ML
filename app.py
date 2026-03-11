import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "aqi_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.set_page_config(page_title="AQI Prediction", layout="wide")

st.title("🌍 Air Quality Index Prediction System")
st.write("Predict AQI using environmental pollutant levels")

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("Pollution Input")

    pm25 = st.slider("PM2.5", 0, 500, 50)
    pm10 = st.slider("PM10", 0, 500, 20)
    no2 = st.slider("NO2", 0, 200, 30)
    so2 = st.slider("SO2", 0, 200, 4)
    co = st.slider("CO", 0.0, 10.0, 0.8)
    o3 = st.slider("O3", 0, 200, 3)

    city = st.slider("City Code", 0, 500, 333)
    month = st.slider("Month", 1, 12, 3)
    day = st.slider("Day of Week", 0, 6, 2)

if st.button("Predict AQI"):

    features = np.array([[pm25, pm10, no2, so2, co, o3, city, month, day]])

    prediction = model.predict(features)[0]

    with col2:

        st.subheader("Prediction Result")

        st.metric("Predicted AQI", round(prediction,2))

        # AQI Category
        if prediction <= 50:
            category = "Good 🟢"
        elif prediction <= 100:
            category = "Satisfactory 🟡"
        elif prediction <= 200:
            category = "Moderate 🟠"
        elif prediction <= 300:
            category = "Poor 🔴"
        elif prediction <= 400:
            category = "Very Poor 🟣"
        else:
            category = "Severe ⚫"

        st.write("### AQI Category:", category)

        st.divider()

        # Pollution bar chart
        pollutants = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        }

        df = pd.DataFrame(
            pollutants.items(),
            columns=["Pollutant", "Value"]
        )

        fig, ax = plt.subplots()

        ax.bar(df["Pollutant"], df["Value"])

        ax.set_title("Pollution Levels")

        st.pyplot(fig)

st.divider()

st.write("Built with Machine Learning + Streamlit")
