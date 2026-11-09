import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

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

if st.button("Predict Solar Power"):
    prediction=model.predict(input_data)

    st.success(f"Predicted AC Power:{prediction[0]:.3f}")


    st.header("SHAP Feature Explanation")
    preprocessor=model.named_steps["preprocessor"]
    regressor=model.named_steps["regressor"]
    transformed_data=preprocessor.transform(input_data)
    explainer=shap.TreeExplainer(regressor)
    shap_values=explainer.shap_values(transformed_data)
    explanation=shap.Explanation(values=shap_values[0],
        base_values=explainer.expected_value[0],
        data=transformed_data[0],
        feature_names=preprocessor.get_feature_names_out()
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)


st.header("Interactive Country Comparison")
if st.button("Compare All Countries"):
    countries=["Norway","Brazil","India","Australia"]
    results=[]
    for i in countries:
        comparison_input=input_data.copy()
        comparison_input["Country"]=i
        prediction=model.predict(comparison_input)
        results.append({
            "Country":i,
            "Predicted AC Power":prediction[0]
        })
    comparison_df=pd.DataFrame(results)
    st.subheader("Predicted AC Power by Country")
    st.dataframe(comparison_df)
    st.bar_chart(comparison_df.set_index("Country"))


st.header("Prediction Reliability Dashboard")
st.subheader("Tuned Random Forest Performance")
col1,col2,col3=st.columns(3)
col1.metric("MAE","0.5930")
col2.metric("RMSE","1.0983")
col3.metric("R2 Score",0.9972)


st.header("Country Performance Analysis")
dataset=pd.read_csv("data/Solar data.csv")
st.subheader("Average AC Power by Country")
fig,ax=plt.subplots(figsize=(8,5))
sns.barplot(x="Country",y="AC_Power",data=dataset,ax=ax)
ax.set_title("Average AC Power by Country")
ax.set_xlabel("Country")
ax.set_ylabel("Average AC Power")
ax.tick_params(axis="x",rotation=0)
st.pyplot(fig)
plt.close(fig)

st.header("Model Performance Comparison")
metrics=pd.DataFrame({
    "Model":["Linear Regression","Decision Tree","Random Forest","Gradient Boosting","Voting Regressor","Tuned Random Forest"],
    "MAE":[2.0695,0.8139,0.5977,0.6686,0.8792,0.5930],
    "RMSE":[2.7795,1.5103,1.1191,1.1566,1.2918,1.0983],
    "R2 Score":[0.981866,0.994646,0.997060,0.996860,0.996083,0.997169]
})
st.subheader("Test Set Performance")
st.dataframe(metrics)

st.header("Feature Importance Analysis")
importance = pd.DataFrame({
    "Feature":model.named_steps["preprocessor"].get_feature_names_out(),
    "Importance":model.named_steps["regressor"].feature_importances_
})
st.bar_chart(importance.set_index("Feature"))

st.header("Monthly Solar Power Analysis")
st.subheader("Average AC Power by Month")
monthly_data=dataset.groupby("Month")["AC_Power"].mean().reset_index()
fig,ax=plt.subplots(figsize=(10, 5))
sns.lineplot(data=monthly_data, x="Month", y="AC_Power", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Average AC Power")
ax.set_title("Monthly Average AC Power")
st.pyplot(fig)
plt.close(fig)

st.header("Hourly Solar Power Analysis")
st.subheader("Average AC Power by Hour")
monthly_data=dataset.groupby("Hour")["AC_Power"].mean().reset_index()
fig,ax=plt.subplots(figsize=(10, 5))
sns.lineplot(data=monthly_data, x="Hour", y="AC_Power", ax=ax)
ax.set_xlabel("Hour")
ax.set_ylabel("Average AC Power")
ax.set_title("Hourly Average AC Power")
st.pyplot(fig)
plt.close(fig)

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


