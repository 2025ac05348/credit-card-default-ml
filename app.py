
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


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# Sidebar - Project Information
# ============================================================

with st.sidebar:

    st.title("💳 Credit Default ML")

    st.markdown(
        "### Project Overview"
    )

    st.write(
        "A machine learning application for predicting "
        "whether a credit-card client will default on "
        "the next payment."
    )

    st.divider()

    st.markdown("### Dataset")

    st.metric("Total Instances", "30,000")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training", "24,000")

    with col2:
        st.metric("Testing", "6,000")

    st.write("**Input Features:** 23")

    st.divider()

    st.markdown("### Models Evaluated")

    st.write(
        """
        • Logistic Regression  
        • Decision Tree  
        • KNN  
        • Naive Bayes  
        • Random Forest
        """
    )

    st.divider()

    st.caption(
        "BITS Pilani M.Tech AI/ML Project"
    )


# ============================================================
# Main Application
# ============================================================

st.title("💳 Credit Card Default Prediction")

st.markdown(
    "### Machine Learning Model Comparison"
)

st.write(
    "Compare five classification models for predicting "
    "whether a credit-card client will default on the "
    "next payment."
)

st.info(
    "Upload the test dataset, select a model, and explore "
    "its performance using multiple evaluation metrics."
)


# ============================================================
# Load Preprocessor and Models
# ============================================================

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)

models = {
    "Logistic Regression": joblib.load(
        "model/logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        "model/decision_tree.pkl"
    ),
    "KNN": joblib.load(
        "model/knn.pkl"
    ),
    "Naive Bayes": joblib.load(
        "model/naive_bayes.pkl"
    ),
    "Random Forest": joblib.load(
        "model/random_forest.pkl"
    )
}


# ============================================================
# Model Descriptions
# ============================================================

model_descriptions = {

    "Logistic Regression":
        "A linear classification model that provides a "
        "simple and interpretable baseline.",

    "Decision Tree":
        "A tree-based model that makes predictions using "
        "a sequence of decision rules.",

    "KNN":
        "A distance-based model that predicts a class using "
        "the nearest training examples.",

    "Naive Bayes":
        "A probabilistic classifier based on Bayes' theorem "
        "with a conditional independence assumption.",

    "Random Forest":
        "An ensemble of decision trees that combines "
        "multiple trees to improve predictive performance."
}


# ============================================================
# Section 1 - Upload Test Data
# ============================================================

st.header("1. Upload Test Data")

st.write(
    "Upload the CSV file containing the test observations "
    "and their default outcomes."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success(
        f"Dataset uploaded successfully — "
        f"{len(data):,} observations loaded."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Instances",
        f"{len(data):,}"
    )

    col2.metric(
        "Columns",
        data.shape[1]
    )

    col3.metric(
        "Expected Features",
        "23"
    )

    required_features = [
        "X1", "X2", "X3", "X4", "X5",
        "X6", "X7", "X8", "X9", "X10",
        "X11", "X12", "X13", "X14", "X15",
        "X16", "X17", "X18", "X19", "X20",
        "X21", "X22", "X23"
    ]

    required_columns = required_features + ["Y"]

    missing_columns = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:

        st.error(
            f"Missing required columns: {missing_columns}"
        )

    else:

        X_uploaded = data[required_features]
        y_uploaded = data["Y"]

        # ====================================================
        # Section 2 - Model Selection
        # ====================================================

        st.divider()

        st.header("2. Select Model")

        selected_model_name = st.selectbox(
            "Choose a classification model:",
            list(models.keys())
        )

        selected_model = models[
            selected_model_name
        ]

        st.caption(
            model_descriptions[
                selected_model_name
            ]
        )

        # ====================================================
        # Preprocessing
        # ====================================================

        X_processed = preprocessor.transform(
            X_uploaded
        )


        # ====================================================
        # Evaluate All Models
        # ====================================================

        all_results = {}

        for model_name, model in models.items():

            predictions = model.predict(
                X_processed
            )

            probabilities = model.predict_proba(
                X_processed
            )[:, 1]

            all_results[model_name] = {

                "Accuracy": accuracy_score(
                    y_uploaded,
                    predictions
                ),

                "AUC": roc_auc_score(
                    y_uploaded,
                    probabilities
                ),

                "Precision": precision_score(
                    y_uploaded,
                    predictions
                ),

                "Recall": recall_score(
                    y_uploaded,
                    predictions
                ),

                "F1": f1_score(
                    y_uploaded,
                    predictions
                ),

                "MCC": matthews_corrcoef(
                    y_uploaded,
                    predictions
                )
            }


        # ====================================================
        # Selected Model Prediction
        # ====================================================

        y_pred = selected_model.predict(
            X_processed
        )

        y_prob = selected_model.predict_proba(
            X_processed
        )[:, 1]


        # ====================================================
        # Section 3 - Evaluation Metrics
        # ====================================================

        st.header("3. Evaluation Metrics")

        selected_results = all_results[
            selected_model_name
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{selected_results['Accuracy']:.4f}"
        )

        col2.metric(
            "AUC",
            f"{selected_results['AUC']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{selected_results['Precision']:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{selected_results['Recall']:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{selected_results['F1']:.4f}"
        )

        col6.metric(
            "MCC",
            f"{selected_results['MCC']:.4f}"
        )


        # ====================================================
        # Model Interpretation
        # ====================================================

        st.subheader(
            f"Model Interpretation — {selected_model_name}"
        )

        if selected_model_name == "Random Forest":

            st.success(
                "Random Forest provides the strongest overall "
                "performance among the evaluated models, "
                "achieving the highest Accuracy, AUC, F1 Score "
                "and MCC on the test dataset."
            )

        elif selected_model_name == "Naive Bayes":

            st.warning(
                "Naive Bayes achieves very high recall, meaning "
                "it identifies most default cases. However, "
                "its low precision and accuracy indicate a high "
                "number of false positive predictions."
            )

        else:

            st.info(
                f"{selected_model_name} provides a different "
                "trade-off between identifying default cases "
                "and avoiding false positive predictions."
            )


        # ====================================================
        # Section 4 - Confusion Matrix
        # ====================================================

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_uploaded,
            y_pred
        )

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No Default",
                "Default"
            ]
        )

        display.plot(
            ax=ax,
            colorbar=False
        )

        ax.set_title(
            f"Confusion Matrix — {selected_model_name}"
        )

        st.pyplot(fig)


        # ====================================================
        # Section 5 - Classification Report
        # ====================================================

        st.header("5. Classification Report")

        report = classification_report(
            y_uploaded,
            y_pred,
            target_names=[
                "No Default",
                "Default"
            ],
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


        # ====================================================
        # Section 6 - Model Comparison
        # ====================================================

        st.divider()

        st.header(
            "6. Model Comparison"
        )

        comparison_df = pd.DataFrame(
            all_results
        ).T

        comparison_df = comparison_df[
            [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "MCC"
            ]
        ]

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

        best_model = comparison_df[
            "F1"
        ].idxmax()

        st.info(
            f"Based on F1 Score, the best-performing model "
            f"on this test dataset is **{best_model}**."
        )
