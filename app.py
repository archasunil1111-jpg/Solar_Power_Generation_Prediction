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