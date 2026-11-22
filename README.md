# Solar Power Generation Prediction and Performance Analysis

An end-to-end Machine Learning project for predicting solar power generation and analyzing system performance across different geographical regions.

## Project Overview

Solar power generation is influenced by several environmental and operational factors, including irradiation, ambient temperature, module temperature, time of day and geographical location.

This project uses Machine Learning regression algorithms to predict AC power generation and evaluate their performance. It also provides an interactive Streamlit dashboard for exploring predictions, comparing models and understanding the factors influencing model outputs.

## Project Objectives

- Analyze solar power generation data through Exploratory Data Analysis (EDA).
- Identify patterns and relationships between environmental variables and power generation.
- Perform data preprocessing and feature engineering.
- Train and compare multiple Machine Learning regression algorithms.
- Optimize model performance using hyperparameter tuning.
- Explain model predictions using SHAP.
- Develop an interactive dashboard for prediction and performance analysis.

## Dataset Description

The dataset contains 35,136 records representing solar power generation observations across four countries.

| Attribute | Description |
|---|---|
| Total records | 35,136 |
| Countries | Norway, Brazil, India and Australia |
| Target variable | AC_Power |
| Data period | 2024 |
| Problem type | Supervised Machine Learning |
| Learning task | Regression |

### Features Used

- Country
- Ambient_Temperature
- Module_Temperature
- Irradiation
- Day
- Month
- Hour
- Temperature_Difference

The target variable is `AC_Power`, representing the actual AC power generation recorded in the dataset.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed detailed data analysis to understand the dataset and identify important patterns.

- Dataset inspection and statistical analysis
- Data quality and missing-value checks
- Outlier detection using the IQR method
- Univariate and bivariate analysis
- Correlation analysis
- Pairplot visualization
- Time-based analysis
- Country-wise performance analysis

### 2. Data Preprocessing

- Separated input features and target variable.
- Checked for missing values.
- Identified numerical and categorical features.
- Applied categorical encoding using OneHotEncoder.
- Applied numerical feature scaling using StandardScaler.
- Created the Temperature_Difference feature.
- Split the dataset chronologically into training and testing sets.

### 3. Model Development

The following regression algorithms were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor
- Tuned Random Forest Regressor
- Tuned Gradient Boosting Regressor

TimeSeriesSplit cross-validation and GridSearchCV were used for model evaluation and hyperparameter optimization.

### 4. Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

#### Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 2.0447 | 2.6580 | 0.9833 |
| Decision Tree | 0.6697 | 1.2930 | 0.9961 |
| Random Forest | 0.5962 | 1.1344 | 0.9970 |
| Gradient Boosting | 0.6356 | 1.1184 | 0.9970 |
| Voting Regressor | 0.8846 | 1.2580 | 0.9963 |
| Tuned Random Forest | 0.5791 | 1.1037 | 0.9971 |
| Tuned Gradient Boosting | 0.6000 | 1.0986 | 0.9971 |

**Selected Model: Tuned Gradient Boosting Regressor**

The Tuned Gradient Boosting Regressor was selected based on its lowest MSE and RMSE and highest R² score among the evaluated models. Tuned Random Forest achieved a slightly lower MAE.

The reported metrics are based on the held-out test dataset.

### 5. Model Explainability

SHAP (SHapley Additive exPlanations) was used to understand model predictions and analyze feature contributions.

The project includes:

- Feature importance analysis
- SHAP explainability analysis
- Actual vs. predicted comparison
- Residual analysis

## Interactive Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit and Plotly.

### Dashboard Pages

- **Overview:** Project summary, dataset information and model performance.
- **Power Prediction:** Interactive solar power generation prediction.
- **Country Insights:** Country-wise data exploration and analysis.
- **Time Analysis:** Monthly and hourly generation patterns.
- **Model Performance:** Evaluation metrics and model comparisons.
- **Model Explainability:** Feature importance and SHAP-based analysis.
- **About Project:** Project information, workflow and technologies.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Explainability | SHAP |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Cloud Deployment | AWS EC2, Streamlit Community Cloud |

## Project Structure

```text
Solar_Power_Generation_Prediction/
│
├── .devcontainer/
│
├── Screenshots/
│
├── data/
│   └── Solar data.csv
│
├── models/
│   └── solar_power_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/archasunil1111-jpg/Solar_Power_Generation_Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Solar_Power_Generation_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The dashboard will open in your default browser.

## Live Demo

**Streamlit Application:** [Solar Power Prediction Dashboard](https://solar-power-prediction-shap.streamlit.app/)

The interactive application is deployed using Streamlit Community Cloud.

## AWS EC2 Deployment

The application was also deployed on Amazon Web Services (AWS) using an EC2 Linux instance.

- **Cloud Provider:** Amazon Web Services (AWS)
- **Service:** Amazon EC2
- **Instance Type:** t3.micro
- **Region:** Asia Pacific (Mumbai)
- **Application Framework:** Streamlit
- **Deployment Status:** Temporarily stopped to avoid ongoing compute costs.

**Note:** The AWS EC2 deployment is currently inactive because the instance is stopped. The Streamlit Community Cloud link above is available for accessing the application demo.

## Future Enhancements

- Explore additional cloud deployment options and improve deployment scalability.
- Explore additional Machine Learning algorithms.
- Improve model generalization using additional datasets.
- Extend the analysis with more solar power generation variables.
- Integrate real-time solar generation data where available.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for solar power generation prediction and performance analysis, covering data exploration, preprocessing, model development, evaluation, explainability and interactive dashboard development.
