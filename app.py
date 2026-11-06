
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "accidents.csv"
MODEL_FILE = PROJECT_DIR / "models" / "random_forest_model.pkl"
METADATA_FILE = (
    PROJECT_DIR / "outputs" / "reports" / "model_metadata.json"
)
REPORT_FILE = (
    PROJECT_DIR / "outputs" / "reports" / "classification_report.json"
)

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide",
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

st.title("🚦 Road Accident Severity Prediction")
st.caption(
    "Machine Learning project using a Random Forest Classifier"
)

missing_files = [
    str(path)
    for path in [DATA_FILE, MODEL_FILE]
    if not path.is_file()
]

if missing_files:
    st.error(
        "Required project files were not found in this repository. "
        "Include the data/ and models/ folders alongside app.py:"
    )
    for file_path in missing_files:
        st.write(file_path)
    st.stop()

try:
    df = load_data()
    model = load_model()
    metadata = load_json(METADATA_FILE)
    report = load_json(REPORT_FILE)
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()

target = metadata.get("target_column", "Accident_Severity_C")
if target != "Accident_Severity_C":
    st.error(
        "Model metadata must use the original target column "
        "'Accident_Severity_C'."
    )
    st.stop()

if target not in df.columns:
    st.error(f"Severity column '{target}' was not found in the dataset.")
    st.stop()

feature_columns = metadata.get("feature_columns", [])
missing_features = [
    feature for feature in feature_columns if feature not in df.columns
]
if not feature_columns or missing_features:
    st.error(
        "Model feature metadata does not match the dataset. "
        f"Missing features: {missing_features or 'feature list is empty'}"
    )
    st.stop()

model_features = getattr(model, "feature_names_in_", None)
if model_features is not None and list(model_features) != feature_columns:
    st.error(
        "The saved model feature order does not match model_metadata.json. "
        "Use the matching trained model and metadata files."
    )
    st.stop()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Severity Prediction",
        "Model Performance",
        "Dataset Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Project 136**")
st.sidebar.write("Algorithm: Random Forest")

# --------------------------------------------------
# HOME DASHBOARD
# --------------------------------------------------
if page == "Home Dashboard":
    st.header("Accident Analytics Dashboard")

    severity_data = df[target].dropna()
    total_records = len(df)
    labelled_records = len(severity_data)
    missing_severity = total_records - labelled_records

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Records with Severity", f"{labelled_records:,}")
    col3.metric("Missing Severity", f"{missing_severity:,}")
    col4.metric("Severity Classes", severity_data.nunique())

    st.subheader("Accident Severity Distribution")

    severity_counts = (
        severity_data.astype(str)
        .value_counts()
        .rename_axis("Severity")
        .reset_index(name="Accidents")
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            severity_counts,
            x="Severity",
            y="Accidents",
            color="Severity",
            title="Accidents by Severity",
            text="Accidents",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Accidents",
            title="Severity Share",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional charts based on available dataset columns
    chart_options = [
        ("Day_of_Week", "Accidents by Day of Week"),
        ("Road_Condition_F", "Accidents by Road Condition"),
        ("Weather_Conditions_H", "Accidents by Weather"),
        ("Causes_D", "Accidents by Cause"),
        ("Road_Feature_E", "Accidents by Road Feature"),
        ("Vehicle_Type_Involved_J_V1", "Accidents by Vehicle Type"),
    ]

    for column, title in chart_options:
        if column in df.columns:
            counts = (
                df[column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis("Category")
                .reset_index(name="Accidents")
            )

            st.subheader(title)
            fig = px.bar(
                counts,
                x="Category",
                y="Accidents",
                title=title,
                text="Accidents",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# SEVERITY PREDICTION
# --------------------------------------------------
elif page == "Severity Prediction":
    st.header("Predict Accident Severity")
    st.write(
        "Enter the accident details below. "
        "The trained Random Forest model will predict the severity class."
    )

    inputs = {}

    with st.form("prediction_form"):
        st.subheader("Accident Details")
        columns = st.columns(2)

        for index, feature in enumerate(feature_columns):
            with columns[index % 2]:
                if feature not in df.columns:
                    st.warning(f"Missing feature in dataset: {feature}")
                    continue

                series = df[feature]

                if pd.api.types.is_numeric_dtype(series):
                    numeric = pd.to_numeric(series, errors="coerce")
                    median = numeric.median()
                    default = float(median) if pd.notna(median) else 0.0

                    inputs[feature] = st.number_input(
                        feature.replace("_", " "),
                        value=default,
                    )
                else:
                    options = (
                        series.dropna()
                        .astype(str)
                        .value_counts()
                        .head(100)
                        .index.tolist()
                    )

                    options = sorted(options)

                    if not options:
                        inputs[feature] = st.text_input(
                            feature.replace("_", " ")
                        )
                    else:
                        inputs[feature] = st.selectbox(
                            feature.replace("_", " "),
                            options=options,
                        )

        submitted = st.form_submit_button(
            "Predict Severity",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if len(inputs) != len(feature_columns):
            st.error(
                "Some model inputs are missing. "
                "Check the dataset and model metadata."
            )
        else:
            try:
                input_df = pd.DataFrame(
                    [[inputs[column] for column in feature_columns]],
                    columns=feature_columns,
                )

                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Severity Class: {prediction}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_

                    probability_df = pd.DataFrame({
                        "Severity": [str(value) for value in classes],
                        "Probability (%)": probabilities * 100,
                    })

                    st.subheader("Prediction Probabilities")
                    fig = px.bar(
                        probability_df,
                        x="Severity",
                        y="Probability (%)",
                        text_auto=".1f",
                        title="Model Confidence by Severity Class",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "This is a model prediction, not a guarantee "
                    "of real-world accident severity."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Random Forest Model Performance")

    accuracy = metadata.get("accuracy")

    if accuracy is not None:
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    else:
        st.info(
            "Accuracy was not found in model metadata. "
            "Retrain the model to generate evaluation results."
        )

    st.write("Training rows:", metadata.get("training_rows", "Not available"))
    st.write("Testing rows:", metadata.get("testing_rows", "Not available"))
    st.write("Target column:", metadata.get("target_column", "Not available"))

    if report:
        report_rows = []

        for label, values in report.items():
            if isinstance(values, dict) and "f1-score" in values:
                report_rows.append({
                    "Class": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-score": values.get("f1-score"),
                    "Samples": values.get("support"),
                })

        if report_rows:
            report_df = pd.DataFrame(report_rows)
            st.subheader("Classification Report")
            st.dataframe(report_df, use_container_width=True)

            metric_df = report_df[
                report_df["Class"].isin(
                    [str(value) for value in model.classes_]
                )
            ]

            if not metric_df.empty:
                plot_df = metric_df.melt(
                    id_vars="Class",
                    value_vars=["Precision", "Recall", "F1-score"],
                    var_name="Metric",
                    value_name="Score",
                )

                fig = px.bar(
                    plot_df,
                    x="Class",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Model Metrics by Class",
                    range_y=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Evaluation report not found. Run train_model.py again."
        )

    st.info(
        "Accuracy alone does not describe the full model performance. "
        "Review precision, recall, F1-score, and class balance too."
    )

# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    st.write("Dataset shape:", df.shape)

    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Missing Values")
    missing_df = (
        df.isna()
        .sum()
        .rename("Missing Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Download Dataset")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="accidents.csv",
        mime="text/csv",
    )
