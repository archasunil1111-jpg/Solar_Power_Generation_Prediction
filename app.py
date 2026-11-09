import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Solar Power Generation Prediction",
    layout="wide"
)

st.sidebar.title("Solar Dashboard")
st.sidebar.caption("Power Generation Analysis")
with st.sidebar.container(border=True):
  page=st.radio("Navigation",["Dashboard","Power Prediction","Country Analysis","Model Performance","SHAP Explanation","About Project"])
if page=="Dashboard":
    st.markdown("# :green[SOLAR POWER GENERATION PREDICTION AND PERFORMANCE ANALYSIS]")
    st.markdown("Explore solar power generation and performance across "
        "Norway, Brazil, India and Australia using Machine Learning.")
    st.divider()
    with st.container(border=True):
      st.markdown("## :green[Dataset Overview]")
      col1,col2,col3=st.columns(3)
      col1.metric("Total Records","35,136")
      col2.metric("Countries","4")
      col3.metric("Prediction Target","AC Power")
    with st.container(border=True):
      st.markdown("## :green[Best Model Performance]")
      st.caption("Tuned Random Forest Regressor")
      col1,col2,col3=st.columns(3)
      col1.metric("MAE","0.5930")
      col2.metric("RMSE","1.0983")
      col3.metric("R² Score","0.9972")
    st.divider()
    st.markdown("## :green[Key Project Features]")
    col1, col2 = st.columns(2)
    with col1:
      with st.container(border=True):
        st.markdown("**Power Prediction**")
        st.caption("Predict AC power using solar and environmental conditions.")
      with st.container(border=True):
        st.markdown("**Country Analysis**")
        st.caption("Compare solar power generation across four countries.")
    with col2:
      with st.container(border=True):
        st.markdown("**Model Evaluation**")
        st.caption("Compare machine learning models and their performance.")
      with st.container(border=True):
        st.markdown("**SHAP Explainability**")
        st.caption("Understand which features influence model predictions.")




model=joblib.load("models/solar_power_model.pkl")
dataset=pd.read_csv("data/Solar data.csv")


input_data=None
if page in ["Power Prediction","SHAP Explanation"]:
  st.header("Enter Solar Panel Details")
  country=st.selectbox("Select Country",options=["Norway","Brazil","India","Australia"])
  ambient_temp=st.number_input("Ambient Temperature",min_value=-12.0,max_value=40.0,value=25.0)
  module_temp=st.number_input("Module Temperature",min_value=-8.0,max_value=47.0,value=30.0)
  irradiation=st.number_input("Irradiation",min_value=0.0,max_value=1.3,value=0.5)
  day=st.number_input("Day",min_value=1,max_value=31,value=15)
  month=st.selectbox("Month",range(1,13))
  hour=st.number_input("Hour",min_value=0,max_value=23,value=12)

  temperature_difference=module_temp-ambient_temp

  input_data=pd.DataFrame({
    "Country":[country],
    "Ambient_Temperature":[ambient_temp],
    "Module_Temperature":[module_temp],
    "Irradiation":[irradiation],
    "Day":[day],
    "Month":[month],
    "Hour":[hour],
    "Temperature_Difference":[temperature_difference]
  })
if page=="Power Prediction":
 st.header("Power Prediction")
 if st.button("Predict Solar Power"):
    prediction=model.predict(input_data)

    st.success(f"Predicted AC Power:{prediction[0]:.3f}")



if page=="Model Performance":
  st.header("Actual vs Predicted Power Analysis")
  X=dataset[["Country","Ambient_Temperature","Module_Temperature",
     "Irradiation", "Day", "Month", "Hour"]].copy()
  X["Temperature_Difference"] = (X["Module_Temperature"]- X["Ambient_Temperature"])
  y=dataset["AC_Power"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
  y_pred=model.predict(X_test)


  st.subheader("Actual vs Predicted AC Power")
  fig,ax=plt.subplots(figsize=(7,3.5))
  ax.scatter(y_test, y_pred,alpha=0.5)
  ax.set_xlabel("Actual AC Power")
  ax.set_ylabel("Predicted AC Power")
  ax.set_title("Actual vs Predicted AC Power")
  st.pyplot(fig,width=600)
  plt.close(fig)

if page=="Country Analysis":
  st.header("Country Performance Analysis")
  selected_country=st.selectbox("Select Country",
        ["All Countries", "Norway", "Brazil", "India", "Australia"])
  if selected_country=="All Countries":
    country_data=dataset.copy()
  else:
    country_data=dataset[dataset["Country"]==selected_country]
  st.subheader("Average AC Power by Country")
  fig,ax=plt.subplots(figsize=(7,3.5))
  sns.barplot(x="Country",y="AC_Power",data=dataset,ax=ax)
  ax.set_title("Average AC Power by Country",fontsize=12)
  ax.set_xlabel("Country",fontsize=10)
  ax.set_ylabel("Average AC Power",fontsize=10)
  ax.tick_params(axis="x",rotation=0)
  ax.grid(axis="y",linestyle="--",alpha=0.3)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  st.pyplot(fig,width=600)
  plt.close(fig)

if page=="Model Performance":
  st.header("Model Performance Comparison")
  metrics=pd.DataFrame({
    "Model":["Linear Regression","Decision Tree","Random Forest","Gradient Boosting","Voting Regressor","Tuned Random Forest"],
    "MAE":[2.0695,0.8139,0.5977,0.6686,0.8792,0.5930],
    "RMSE":[2.7795,1.5103,1.1191,1.1566,1.2918,1.0983],
    "R2 Score":[0.981866,0.994646,0.997060,0.996860,0.996083,0.997169]
  })
  st.subheader("Test Set Performance")
  st.dataframe(metrics)

if page=="Model Performance":
  st.header("Feature Importance Analysis")
  importance = pd.DataFrame({
    "Feature":model.named_steps["preprocessor"].get_feature_names_out(),
    "Importance":model.named_steps["regressor"].feature_importances_
  })
  st.bar_chart(importance.set_index("Feature"))

if page=="Country Analysis":
  st.header("Monthly Solar Power Analysis")
  st.subheader("Average AC Power by Month")
  monthly_data=country_data.groupby("Month")["AC_Power"].mean().reset_index()
  fig,ax=plt.subplots(figsize=(10, 5))
  sns.lineplot(data=monthly_data, x="Month", y="AC_Power", ax=ax)
  ax.set_xlabel("Month")
  ax.set_ylabel("Average AC Power")
  ax.set_title("Monthly Average AC Power")
  ax.tick_params(labelsize=9)
  ax.grid(axis="y",linestyle="--",alpha=0.3)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  st.pyplot(fig,width=600)
  plt.close(fig)

if page=="Country Analysis":
  st.header("Hourly Solar Power Analysis")
  st.subheader("Average AC Power by Hour")
  hourly_data=country_data.groupby("Hour")["AC_Power"].mean().reset_index()
  fig,ax=plt.subplots(figsize=(10, 5))
  sns.lineplot(data=hourly_data, x="Hour", y="AC_Power", ax=ax)
  ax.set_xlabel("Hour")
  ax.set_ylabel("Average AC Power")
  ax.set_title("Hourly Average AC Power")
  ax.tick_params(labelsize=9)
  ax.grid(axis="y",linestyle="--",alpha=0.3)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  st.pyplot(fig,width=600)
  plt.close(fig)

if page=="About Project":
  st.header("About the Project")
  st.write("""This project predicts solar power generation across four different countries
using Machine Learning.
The application uses a Tuned Random Forest model to predict AC Power
based on country, temperature, irradiation, day, month and hour.
""")
  st.subheader("Dataset Information")
  st.write("Total Records:",len(dataset))
  st.write("Countries:Norway,Brazil,India,Australia")
  st.subheader("Technologies Used")
  st.write("""
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib""")

if page=="SHAP Explanation":
  st.header("SHAP Feature Explanation")
  if st.button("Explain Prediction"):
    preprocessor=model.named_steps["preprocessor"]
    regressor=model.named_steps["regressor"]
    transformed_data=preprocessor.transform(input_data)
    explainer=shap.TreeExplainer(regressor)
    shap_values=explainer.shap_values(transformed_data)
    explanation=shap.Explanation(values=shap_values[0],
            base_values=explainer.expected_value[0],
            data=transformed_data[0],
            feature_names=preprocessor.get_feature_names_out())
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    st.pyplot(fig,width=900)
    plt.close(fig)

