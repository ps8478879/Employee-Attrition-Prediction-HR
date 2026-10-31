
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px

warnings.filterwarnings("ignore")

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = BASE_DIR / "models" / "best_attrition_model.joblib"
META_PATH = BASE_DIR / "models" / "model_metadata.json"

OUTPUT_DIR = BASE_DIR / "outputs"

RISK_PATH = OUTPUT_DIR / "at_risk_employees.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

DROP_COLUMNS = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Employee Attrition HR Analytics",
    page_icon="🏢",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_optional_csv(path_string):
    path = Path(path_string)

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_data
def load_metadata(path_string):
    path = Path(path_string)

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

try:
    df = load_dataset()
    model = load_model()
    metadata = load_metadata(str(META_PATH))

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()


if "Attrition" not in df.columns:
    st.error("Attrition column is missing from dataset.")
    st.stop()


# ==========================================
# PREPARE DATA
# ==========================================

df = df.copy()

df["Attrition"] = (
    df["Attrition"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

total_employees = len(df)
employees_left = int((df["Attrition"] == "Yes").sum())
employees_stayed = int((df["Attrition"] == "No").sum())

attrition_rate = (
    employees_left / total_employees * 100
    if total_employees else 0
)

feature_columns = [
    column for column in df.columns
    if column not in DROP_COLUMNS
]

selected_model = metadata.get(
    "selected_model",
    "Logistic Regression"
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏢 HR Analytics")
st.sidebar.caption("Employee Attrition Prediction")

pages = [
    "Overview",
    "Department Analysis",
    "Employee Prediction",
    "Attrition Factors",
    "At-Risk Employees",
    "Model Performance"
]

page = st.sidebar.selectbox(
    "Navigate Dashboard",
    pages
)

st.sidebar.divider()

st.sidebar.info(
    f"Selected Model: {selected_model}"
)

st.sidebar.divider()
st.sidebar.caption(
    "Employee Attrition Prediction | HR Analytics"
)

# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.title("📊 Employee Attrition Dashboard")

    st.write(
        "HR analytics and employee attrition monitoring system."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        f"{total_employees:,}"
    )

    col2.metric(
        "Employees Left",
        f"{employees_left:,}"
    )

    col3.metric(
        "Employees Stayed",
        f"{employees_stayed:,}"
    )

    col4.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Employee Attrition Distribution")

        attrition_counts = (
            df["Attrition"]
            .value_counts()
            .rename_axis("Attrition")
            .reset_index(name="Employees")
        )

        fig = px.pie(
            attrition_counts,
            names="Attrition",
            values="Employees",
            hole=0.45,
            title="Attrition: Yes vs No"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Attrition by Department")

        if "Department" in df.columns:

            dept_chart = (
                df.groupby(["Department", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                dept_chart,
                x="Department",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Department-wise Employee Attrition"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==========================================
# DEPARTMENT ANALYSIS
# ==========================================

elif page == "Department Analysis":

    st.title("🏢 Department Analysis")

    if "Department" not in df.columns:

        st.warning("Department column is not available.")

    else:

        # Calculate department statistics
        dept = (
            df.assign(
                Attrition_Flag=df["Attrition"].eq("Yes").astype(int)
            )
            .groupby("Department")
            .agg(
                Employees=("Attrition_Flag", "size"),
                Attrition_Count=("Attrition_Flag", "sum"),
                Attrition_Rate=("Attrition_Flag", "mean")
            )
            .reset_index()
        )

        dept["Attrition_Rate"] = (
            dept["Attrition_Rate"] * 100
        ).round(2)

        dept = dept.rename(columns={
            "Employees": "Total Employees",
            "Attrition_Count": "Employees Left",
            "Attrition_Rate": "Attrition Rate (%)"
        })

        st.subheader("Department-wise Employee Details")

        st.dataframe(
            dept,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Department-wise Attrition Rate (%)")

        fig = px.bar(
            dept,
            x="Department",
            y="Attrition Rate (%)",
            color="Department",
            text="Attrition Rate (%)",
            title="Department-wise Attrition Rate"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Employees by Department")

        fig = px.bar(
            dept,
            x="Department",
            y=["Total Employees", "Employees Left"],
            barmode="group",
            title="Total Employees vs Employees Left"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# EMPLOYEE PREDICTION
# ==========================================

elif page == "Employee Prediction":

    st.title("🔍 Employee Attrition Prediction")

    st.write(
        "Enter employee details to estimate attrition risk."
    )

    input_data = {}

    with st.form("employee_prediction_form"):

        st.subheader("Employee Information")

        col1, col2 = st.columns(2)

        for index, feature in enumerate(feature_columns):

            target_col = col1 if index % 2 == 0 else col2

            with target_col:

                if pd.api.types.is_numeric_dtype(df[feature]):

                    minimum = int(df[feature].min())
                    maximum = int(df[feature].max())
                    default = int(df[feature].median())

                    input_data[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

                else:

                    options = sorted(
                        df[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    input_data[feature] = st.selectbox(
                        feature,
                        options
                    )

        submitted = st.form_submit_button(
            "Predict Attrition",
            use_container_width=True
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]

            classes = list(model.classes_)

            # Training target mapping: No = 0, Yes = 1
            positive_index = None

            for index, class_value in enumerate(classes):

                if str(class_value).strip() == "1":
                    positive_index = index
                    break

            if positive_index is None:
                st.error(
                    f"Class 1 not found. Model classes: {classes}"
                )

            else:

                probability = float(
                    probabilities[positive_index]
                )

                st.divider()
                st.subheader("Prediction Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    if int(prediction) == 1:

                        st.metric(
                            "Prediction",
                            "Likely to Leave"
                        )

                    else:

                        st.metric(
                            "Prediction",
                            "Likely to Stay"
                        )

                with result_col2:

                    st.metric(
                        "Estimated Attrition Probability",
                        f"{probability * 100:.2f}%"
                    )

                st.progress(probability)

                if probability >= 0.50:

                    st.warning(
                        "The model estimates a higher attrition risk."
                    )

                else:

                    st.success(
                        "The model estimates a lower attrition risk."
                    )

                st.caption(
                    "This is a model estimate, not certainty. "
                    "Do not use it as the sole basis for employment decisions."
                )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# ==========================================
# ATTRITION FACTORS
# ==========================================

elif page == "Attrition Factors":

    st.title("📈 Attrition Factors")

    importance = load_optional_csv(
        str(IMPORTANCE_PATH)
    )

    if importance.empty:

        st.warning(
            "Feature importance data is not available. "
            "Please check outputs/feature_importance.csv."
        )

    else:

        st.subheader("Feature Importance Data")

        st.dataframe(
            importance,
            use_container_width=True
        )

        # Find feature and importance columns automatically
        feature_col = next(
            (
                c for c in importance.columns
                if c.lower() in ["feature", "features"]
            ),
            None
        )

        importance_col = next(
            (
                c for c in importance.columns
                if "importance" in c.lower()
                and "std" not in c.lower()
            ),
            None
        )

        if feature_col and importance_col:

            chart_data = importance.copy()

            chart_data[importance_col] = pd.to_numeric(
                chart_data[importance_col],
                errors="coerce"
            )

            chart_data = (
                chart_data
                .dropna(subset=[importance_col])
                .sort_values(
                    importance_col,
                    ascending=False
                )
                .head(15)
            )

            st.subheader("Top 15 Important Features")

            fig = px.bar(
                chart_data,
                x=importance_col,
                y=feature_col,
                orientation="h",
                title="Top 15 Important Features"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Feature and importance column names could not be identified."
            )

    st.caption(
        "Feature importance shows model-related associations, "
        "not proof of causation."
    )


# ==========================================
# AT-RISK EMPLOYEES
# ==========================================

elif page == "At-Risk Employees":

    st.title("⚠️ At-Risk Employee Analysis")

    risk_data = load_optional_csv(
        str(RISK_PATH)
    )

    if risk_data.empty:

        st.warning(
            "At-risk employee data is not available."
        )

    else:

        st.metric(
            "Employee Records",
            len(risk_data)
        )

        if "Predicted_Attrition_Probability" in risk_data.columns:

            risk_data = risk_data.sort_values(
                "Predicted_Attrition_Probability",
                ascending=False
            )

            st.subheader("Highest Estimated Risk")

            st.dataframe(
                risk_data.head(20),
                use_container_width=True
            )

        else:

            st.dataframe(
                risk_data,
                use_container_width=True
            )

        st.caption(
            "Use risk estimates for supportive HR review, "
            "not punitive employee decisions."
        )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    comparison = load_optional_csv(
        str(COMPARISON_PATH)
    )

    if comparison.empty:

        st.warning(
            "Model comparison data is not available."
        )

    else:

        st.subheader("Model Comparison")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        model_column = next(
            (
                c for c in comparison.columns
                if c.lower() == "model"
            ),
            None
        )

        if model_column and numeric_columns:

            plot_data = comparison.melt(
                id_vars=[model_column],
                value_vars=numeric_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                plot_data,
                x="Metric",
                y="Score",
                color=model_column,
                barmode="group",
                title="Model Performance Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("Selected Model")

    st.info(selected_model)

    st.caption(
        "Model metrics describe performance on the evaluation data. "
        "Predictions should support human review, not replace it."
    )
