import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Solar Power Prediction",
    layout="wide"
)

st.title("SOLAR POWER GENERATION PREDICTION")
st.write("Application of predicting solar power and generation across different countries using Machine Learning.")

model=joblib.load("models/solar_power_model.pkl")

st.header("Enter Solar Panel Details")
country=st.selectbox("Select Country",options=["Norway","Brazil","India","Australia"])
ambient_temp=st.number_input("Ambient Temperature",min_value=-12.0,max_value=40.0,value=25.0)
module_temp=st.number_input("Module Temperature",min_value=-8.0,max_value=47.0,value=30.0)
irradiation=st.number_input("Irradiation",min_value=0.0,max_value=1.3,value=0.5)
day=st.number_input("Day",min_value=1,max_value=31,value=15)
month=st.selectbox("Month",range(1,13))
hour=st.number_input("Hour",min_value=0,max_value=23,value=12)

temperature_difference=module_temp-ambient_temp