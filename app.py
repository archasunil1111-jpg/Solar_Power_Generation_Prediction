import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Solar Power Prediction",
    layout="wide")

model=joblib.load("models/solar_power_model.pkl")

st.title("SOLAR POWER GENERATION PREDICTION")

st.write( "An interactive application for predicting solar power "
    "generation across different countries.")