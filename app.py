
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Default Prediction")

st.write(
    "Compare classification models for predicting whether a "
    "credit-card client will default on the next payment."
)

# Load preprocessing and models
preprocessor = joblib.load("model/preprocessor.pkl")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl")
}

# Upload test data
st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload the test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully.")

    st.write("Dataset shape:", data.shape)

    required_features = [
        "X1", "X2", "X3", "X4", "X5",
        "X6", "X7", "X8", "X9", "X10",
        "X11", "X12", "X13", "X14", "X15",
        "X16", "X17", "X18", "X19", "X20",
        "X21", "X22", "X23"
    ]

    required_columns = required_features + ["Y"]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )

    else:

        X_uploaded = data[required_features]
        y_uploaded = data["Y"]

        # Model selection
        st.header("2. Select Model")

        selected_model_name = st.selectbox(
            "Choose a classification model:",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        # Preprocess
        X_processed = preprocessor.transform(X_uploaded)

        # Prediction
        y_pred = selected_model.predict(X_processed)
        y_prob = selected_model.predict_proba(X_processed)[:, 1]

        # Evaluation metrics
        st.header("3. Evaluation Metrics")

        accuracy = accuracy_score(y_uploaded, y_pred)
        auc = roc_auc_score(y_uploaded, y_prob)
        precision = precision_score(y_uploaded, y_pred)
        recall = recall_score(y_uploaded, y_pred)
        f1 = f1_score(y_uploaded, y_pred)
        mcc = matthews_corrcoef(y_uploaded, y_pred)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_uploaded,
            y_pred
        )

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Default", "Default"]
        )

        display.plot(
            ax=ax,
            colorbar=False
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        st.pyplot(fig)

        # Classification Report
        st.header("5. Classification Report")

        report = classification_report(
            y_uploaded,
            y_pred,
            target_names=["No Default", "Default"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )
