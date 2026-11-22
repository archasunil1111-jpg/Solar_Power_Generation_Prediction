
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar Power Generation | Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Solar data.csv"
MODEL_PATH = BASE_DIR / "models" / "solar_power_model.pkl"

# ============================================================
# 3. CUSTOM DARK THEME
# ============================================================

st.markdown("""
<style>
/* Application background */
.stApp {
    background: #0B1120;
    color: #E2E8F0;
}

.main .block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1rem;
}

section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    min-height: 45px;
    text-align: left;
    border-radius: 8px;
    border: 1px solid #263449;
    background: #172235;
    color: #CBD5E1;
    font-weight: 600;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: #164E63;
    border-color: #14B8A6;
    color: white;
}

section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: #0F766E;
    border: 1px solid #2DD4BF;
    color: white;
}

/* Headings and text */
h1, h2, h3 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #CBD5E1;
}

.stCaption, small {
    color: #94A3B8 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #172235;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 105px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #2DD4BF !important;
    font-weight: 750;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #172235;
    border-color: #334155;
    color: #E2E8F0;
}

.stNumberInput input {
    color: #F1F5F9;
}

.stSelectbox label,
.stNumberInput label {
    color: #CBD5E1 !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    border-radius: 8px;
    font-weight: 650;
    transition: all 0.2s ease;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #263449;
    border-radius: 10px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: #263449;
    margin-top: 1.3rem;
    margin-bottom: 1.3rem;
}

/* Forms */
div[data-testid="stForm"] {
    background: #111827;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 22px;
}

/* About page technology cards */
.tech-card {
    background: linear-gradient(145deg, #172235, #111827);
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 20px;
    min-height: 170px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
}

.tech-card:hover {
    border-color: #14B8A6;
}

.tech-card h4 {
    color: #2DD4BF;
    font-size: 16px;
    margin: 0 0 12px 0;
}

.tech-card p {
    color: #94A3B8;
    font-size: 13px;
    line-height: 1.6;
    margin: 0 0 12px 0;
}

.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.tech-tag {
    display: inline-block;
    background: #203449;
    color: #99F6E4;
    border: 1px solid #28566A;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 600;
}

/* Project header */
.project-header {
    background: linear-gradient(120deg, #102A3A, #123D45);
    border: 1px solid #23616A;
    border-radius: 14px;
    padding: 26px;
    margin-bottom: 20px;
}

.project-header h1 {
    color: #F0FDFA !important;
    font-size: 30px;
    margin-bottom: 8px;
}

.project-header p {
    color: #A7F3D0;
    font-size: 14px;
    margin-bottom: 0;
}

/* Selection and focus */
::selection {
    background: #0F766E;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. CHECK PROJECT FILES
# ============================================================

if not DATA_PATH.exists():
    st.error("Dataset file not found.")
    st.code(str(DATA_PATH))
    st.info("Make sure Solar data.csv is inside the data folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.code(str(MODEL_PATH))
    st.info("Make sure solar_power_model.pkl is inside the models folder.")
    st.stop()

# ============================================================
# 5. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    if (
        "Temperature_Difference" not in data.columns
        and "Module_Temperature" in data.columns
        and "Ambient_Temperature" in data.columns
    ):
        data["Temperature_Difference"] = (
            data["Module_Temperature"] -
            data["Ambient_Temperature"]
        )

    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    analysis_data = load_data()
    model = load_model()

except Exception as error:
    st.error(f"Could not load the project data or model: {error}")
    st.stop()

# ============================================================
# 6. MODEL FEATURES AND DATA VALIDATION
# ============================================================

FEATURES = [
    "Country",
    "Ambient_Temperature",
    "Module_Temperature",
    "Irradiation",
    "Day",
    "Month",
    "Hour",
    "Temperature_Difference"
]

TARGET = "AC_Power"

missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in analysis_data.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if analysis_data.empty:
    st.error("The dataset is empty.")
    st.stop()

# ============================================================
# 7. CHART SETTINGS
# ============================================================

CHART_COLORS = [
    "#14B8A6",
    "#2DD4BF",
    "#0F766E",
    "#38BDF8",
    "#818CF8",
    "#A78BFA",
    "#F59E0B"
]


def style_chart(fig, title=None):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#F1F5F9")
        ) if title else None,
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#CBD5E1", size=12),
        margin=dict(l=20, r=20, t=65, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1")
        ),
        hoverlabel=dict(
            bgcolor="#172235",
            font_color="#F1F5F9"
        )
    )

    fig.update_xaxes(
        gridcolor="#263449",
        zerolinecolor="#263449"
    )

    fig.update_yaxes(
        gridcolor="#263449",
        zerolinecolor="#263449"
    )

    return fig


def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.divider()


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    "Overview",
    "Power Prediction",
    "Country Insights",
    "Time Analysis",
    "Model Performance",
    "Model Explainability",
    "About Project"
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"

with st.sidebar:
    for page_name in PAGES:

        is_selected = (
            st.session_state.selected_page == page_name
        )

        if st.button(
            page_name,
            key=f"nav_{page_name}",
            width="stretch",
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_page = page_name

page = st.session_state.selected_page

# ============================================================
# 9. OVERVIEW DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown("""
    <div class="project-header">
        <h1>SOLAR POWER GENERATION PREDICTION AND PERFORMANCE ANALYSIS</h1>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Explore solar power generation across Norway, Brazil, India "
        "and Australia using Machine Learning."
    )

    st.divider()

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(analysis_data):,}")

    with col2:
        st.metric("Countries", analysis_data["Country"].nunique())

    with col3:
        st.metric("Prediction Target", "AC Power")

    st.divider()

    st.subheader("Best Model Performance")
    st.markdown("### Tuned Gradient Boosting Regressor")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "0.6000")

    with col2:
        st.metric("RMSE", "1.0986")

    with col3:
        st.metric("R² Score", "0.9971")


# ============================================================
# 10. POWER PREDICTION
# ============================================================

elif page == "Power Prediction":

    page_header(
        "Power Prediction",
        "Enter environmental conditions to estimate solar power generation."
    )

    country_options = sorted(
        analysis_data["Country"].dropna().unique().tolist()
    )

    if not country_options:
        st.warning("No country values are available in the dataset.")
        st.stop()

    with st.form("prediction_form"):

        st.subheader("Environmental Conditions")

        col1, col2 = st.columns(2)

        with col1:

            country = st.selectbox(
                "Country",
                country_options
            )

            ambient_temperature = st.number_input(
                "Ambient Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=25.0,
                step=0.5
            )

            module_temperature = st.number_input(
                "Module Temperature (°C)",
                min_value=-20.0,
                max_value=100.0,
                value=35.0,
                step=0.5
            )

            irradiation = st.number_input(
                "Irradiation",
                min_value=0.0,
                max_value=1.5,
                value=0.5,
                step=0.01
            )

        with col2:

            day = st.number_input(
                "Day",
                min_value=1,
                max_value=31,
                value=15
            )

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=6
            )

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12
            )

        predict_button = st.form_submit_button(
            "Predict AC Power",
            width="stretch",
            type="primary"
        )

    if predict_button:

        temperature_difference = (
            module_temperature - ambient_temperature
        )

        input_data = pd.DataFrame(
            [[
                country,
                ambient_temperature,
                module_temperature,
                irradiation,
                day,
                month,
                hour,
                temperature_difference
            ]],
            columns=FEATURES
        )

        try:

            prediction = max(
                0.0,
                float(model.predict(input_data)[0])
            )

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                "Predicted AC Power",
                f"{prediction:.4f}"
            )

        except Exception as error:
            st.error(
                f"Prediction could not be completed: {error}"
            )


# ============================================================
# 11. COUNTRY INSIGHTS
# ============================================================

elif page == "Country Insights":

    page_header(
        "Country Insights",
        "Explore and compare solar power generation patterns across countries."
    )

    country_summary = (
        analysis_data
        .groupby("Country")["AC_Power"]
        .agg(["mean", "max", "min"])
        .reset_index()
    )

    country_summary.columns = [
        "Country",
        "Average AC Power",
        "Maximum AC Power",
        "Minimum AC Power"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            country_summary,
            x="Country",
            y="Average AC Power",
            color="Country",
            color_discrete_sequence=CHART_COLORS,
            title="Average AC Power by Country"
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    with col2:

        fig = px.box(
            analysis_data,
            x="Country",
            y="AC_Power",
            color="Country",
            color_discrete_sequence=CHART_COLORS,
            title="AC Power Distribution by Country"
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    st.subheader("Country Performance Summary")

    st.dataframe(
        country_summary.round(4),
        width="stretch",
        hide_index=True
    )


# ============================================================
# 12. TIME ANALYSIS
# ============================================================

elif page == "Time Analysis":

    page_header(
        "Time Analysis",
        "Understand how solar power generation changes throughout the day and year."
    )

    col1, col2 = st.columns(2)

    with col1:

        monthly_data = (
            analysis_data
            .groupby("Month")["AC_Power"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            monthly_data,
            x="Month",
            y="AC_Power",
            markers=True,
            title="Average AC Power by Month"
        )

        fig.update_traces(
            line=dict(color="#14B8A6", width=3),
            marker=dict(size=8)
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    with col2:

        hourly_data = (
            analysis_data
            .groupby("Hour")["AC_Power"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly_data,
            x="Hour",
            y="AC_Power",
            markers=True,
            title="Average AC Power by Hour"
        )

        fig.update_traces(
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=8)
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    st.subheader("Monthly Generation Analysis")

    fig = px.bar(
        monthly_data,
        x="Month",
        y="AC_Power",
        color="AC_Power",
        color_continuous_scale="Teal",
        title="Monthly Average AC Power"
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )


# ============================================================
# 13. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Compare the performance of the Machine Learning models evaluated in this project."
    )

    performance_data = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Voting Regressor",
            "Tuned Random Forest",
            "Tuned Gradient Boosting"
        ],

        "MAE": [
            2.044676,
            0.669705,
            0.596212,
            0.635560,
            0.884564,
            0.579120,
            0.600025
        ],

        "MSE": [
            7.064760,
            1.671890,
            1.286755,
            1.250799,
            1.582589,
            1.218049,
            1.206885
        ],

        "RMSE": [
            2.657962,
            1.293016,
            1.134352,
            1.118391,
            1.258010,
            1.103653,
            1.098583
        ],

        "R2 Score": [
            0.983311,
            0.996051,
            0.996960,
            0.997045,
            0.996262,
            0.997123,
            0.997149
        ]
    })

    st.subheader("Model Comparison")

    st.dataframe(
        performance_data.round(4),
        width="stretch",
        hide_index=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            performance_data,
            x="Model",
            y="RMSE",
            color="Model",
            color_discrete_sequence=CHART_COLORS,
            title="RMSE Comparison"
        )

        fig.update_layout(xaxis_tickangle=-35)

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    with col2:

        fig = px.bar(
            performance_data,
            x="Model",
            y="R2 Score",
            color="Model",
            color_discrete_sequence=CHART_COLORS,
            title="R² Score Comparison"
        )

        fig.update_layout(xaxis_tickangle=-35)

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    st.divider()

    st.subheader("Selected Model")
    st.markdown("### Tuned Gradient Boosting Regressor")

    st.write(
        "The Tuned Gradient Boosting Regressor was selected "
        "based on its overall test performance. It achieved "
        "the lowest MSE and RMSE and the highest R² score "
        "among the evaluated models. Tuned Random Forest "
        "achieved a slightly lower MAE."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "0.6000")

    with col2:
        st.metric("RMSE", "1.0986")

    with col3:
        st.metric("R² Score", "0.9971")


# ============================================================
# 14. MODEL EXPLAINABILITY
# ============================================================

elif page == "Model Explainability":

    page_header(
        "Model Explainability",
        "Understand how different features influence the selected model's predictions using SHAP."
    )

    try:

        import shap

        preprocessor = model.named_steps["columntransformer"]

        regressor = model.named_steps[
            "gradientboostingregressor"
        ]

        sample_size = min(
            500,
            len(analysis_data)
        )

        X_sample = analysis_data[FEATURES].sample(
            n=sample_size,
            random_state=42
        )

        transformed_sample = preprocessor.transform(
            X_sample
        )

        if hasattr(transformed_sample, "toarray"):
            transformed_sample = transformed_sample.toarray()

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        explainer = shap.TreeExplainer(regressor)

        shap_values = explainer.shap_values(
            transformed_sample
        )

        importance_values = np.abs(
            shap_values
        ).mean(axis=0)

        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute SHAP Value": importance_values
        })

        importance_data = importance_data.sort_values(
            by="Mean Absolute SHAP Value",
            ascending=True
        )

        st.subheader("Feature Importance")

        fig = px.bar(
            importance_data,
            x="Mean Absolute SHAP Value",
            y="Feature",
            orientation="h",
            color="Mean Absolute SHAP Value",
            color_continuous_scale="Teal",
            title="SHAP Feature Importance"
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

        st.subheader("Feature Importance Values")

        st.dataframe(
            importance_data.sort_values(
                by="Mean Absolute SHAP Value",
                ascending=False
            ).round(5),
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Higher mean absolute SHAP values indicate a stronger "
            "average influence on model predictions. These values "
            "represent influence magnitude, not whether a feature "
            "increases or decreases the prediction."
        )

    except ImportError:

        st.warning(
            "SHAP is not installed. Install it using: "
            "python -m pip install shap"
        )

    except Exception as error:

        st.error(
            f"Model explainability could not be displayed: {error}"
        )


# ============================================================
# 15. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    page_header(
        "About Project",
        "Solar Power Generation Prediction and Performance Analysis using Machine Learning."
    )

    # Project overview
    st.subheader("Project Overview")

    st.write(
        "This project focuses on analysing solar power generation "
        "and predicting AC Power using Machine Learning. The dataset "
        "contains solar generation and environmental information from "
        "Norway, Brazil, India and Australia. The application allows "
        "users to explore country-wise generation patterns, analyse "
        "time-based trends, compare Machine Learning models and predict "
        "AC Power using environmental input values."
    )

    st.divider()

    # Dataset information
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Size",
            f"{len(analysis_data):,}"
        )

    with col2:
        st.metric(
            "Number of Countries",
            analysis_data["Country"].nunique()
        )

    with col3:
        st.metric(
            "Prediction Target",
            "AC Power"
        )

    st.divider()

    # Technologies used
    st.subheader("Technologies Used")

    st.caption(
        "Tools, libraries and frameworks used to develop the project."
    )

    tech_sections = [

        {
            "title": "Programming Language",
            "description": (
                "Core programming language used to build the "
                "application and machine learning workflow."
            ),
            "items": ["Python"]
        },

        {
            "title": "Data Analysis",
            "description": (
                "Libraries used for data loading, manipulation, "
                "numerical operations and analysis."
            ),
            "items": ["Pandas", "NumPy"]
        },

        {
            "title": "Data Visualization",
            "description": (
                "Tools used to create interactive charts and "
                "explore solar generation patterns."
            ),
            "items": ["Plotly", "Matplotlib", "Seaborn"]
        },

        {
            "title": "Machine Learning",
            "description": (
                "Algorithms and libraries used for model training, "
                "evaluation and hyperparameter tuning."
            ),
            "items": [
                "Scikit-learn",
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Voting Regressor",
                "GridSearchCV"
            ]
        },

        {
            "title": "Model Explainability",
            "description": (
                "Explainable AI technique used to understand "
                "feature contributions to model predictions."
            ),
            "items": ["SHAP"]
        },

        {
            "title": "Web Application",
            "description": (
                "Framework used to build the interactive "
                "solar analytics dashboard."
            ),
            "items": ["Streamlit"]
        },

        {
            "title": "Model Storage",
            "description": (
                "Library used to save and load the trained "
                "machine learning pipeline."
            ),
            "items": ["Joblib"]
        },

        {
            "title": "Development Tools",
            "description": (
                "Environment and tools used during development "
                "and project version management."
            ),
            "items": [
                "Jupyter Notebook",
                "VS Code",
                "Git",
                "GitHub"
            ]
        }
    ]

    tech_columns = st.columns(2)

    for index, tech in enumerate(tech_sections):

        tags_html = "".join(
            f'<span class="tech-tag">{item}</span>'
            for item in tech["items"]
        )

        card_html = f"""
        <div class="tech-card">
            <h4>{tech["title"]}</h4>
            <p>{tech["description"]}</p>
            <div class="tech-tags">{tags_html}</div>
        </div>
        """

        with tech_columns[index % 2]:
            st.markdown(
                card_html,
                unsafe_allow_html=True
            )

    st.divider()

    # Selected model
    st.subheader("Selected Machine Learning Model")

    st.markdown("### Tuned Gradient Boosting Regressor")

    st.write(
        "The Tuned Gradient Boosting Regressor was selected as "
        "the final model based on its overall test performance. "
        "It achieved the lowest RMSE and highest R² score among "
        "the evaluated models."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "0.6000")

    with col2:
        st.metric("RMSE", "1.0986")

    with col3:
        st.metric("R² Score", "0.9971")

    st.divider()

    st.caption(
        "Solar Power Generation Prediction and Performance Analysis | "
        "Machine Learning Project"
    )
