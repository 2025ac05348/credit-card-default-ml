
# Credit Card Default Prediction

## Project Overview

This project implements and compares multiple machine learning classification models to predict whether a credit-card client will default on the next month's payment.

The project also includes a Streamlit web application that allows users to upload test data, select a machine learning model, and view its evaluation results.

## Dataset

The **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository is used.

- Number of instances: 30,000
- Number of features: 23
- Target variable: `Y`
- Problem type: Binary classification
- Class 0 (No Default): 23,364
- Class 1 (Default): 6,636

The features include demographic information, credit limit, payment status, bill amounts, and previous payment amounts.

### Train-Test Split

The dataset was divided into:

- Training set: 24,000 instances (80%)
- Test set: 6,000 instances (20%)

## Data Preprocessing

The dataset was preprocessed before model training and evaluation. 
The same fitted preprocessing pipeline was saved as `preprocessor.pkl`
and reused by the Streamlit application when processing test data.

## Machine Learning Models

Five classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8088 | 0.7100 | 0.6923 | 0.2442 | 0.3610 | 0.3302 |
| Decision Tree | 0.7185 | 0.6055 | 0.3731 | 0.4009 | 0.3865 | 0.2043 |
| KNN | 0.7948 | 0.7040 | 0.5561 | 0.3587 | 0.4361 | 0.3292 |
| Naive Bayes | 0.2903 | 0.7240 | 0.2340 | 0.9714 | 0.3771 | 0.1034 |
| Random Forest | 0.8105 | 0.7592 | 0.6243 | 0.3595 | 0.4562 | 0.3711 |

## Model Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.8088 and the highest precision of 0.6923. However, its recall of 0.2442 indicates that it missed a significant number of actual defaulters.

### Decision Tree

Decision Tree achieved an accuracy of 0.7185 and an AUC of 0.6055. Its overall performance was lower than the other conventional classification models.

### KNN

KNN achieved an accuracy of 0.7948 and an F1 score of 0.4361. It provided moderate performance but remained below Random Forest overall.

### Naive Bayes

Naive Bayes achieved the highest recall of 0.9714, identifying almost all actual defaulters. However, its precision of 0.2340 and accuracy of 0.2903 were very low because it generated a large number of false-positive predictions.

### Random Forest

Random Forest achieved the highest accuracy (0.8105), AUC (0.7592), F1 score (0.4562), and MCC (0.3711). It provided the best overall balance among the evaluated models.

## Overall Winner

**Random Forest** was selected as the overall best-performing model because it achieved the highest values for four of the six evaluation metrics: Accuracy, AUC, F1 Score, and MCC.

Although Logistic Regression achieved higher precision and Naive Bayes achieved higher recall, Random Forest provided the strongest overall balance.

## Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Machine learning model selection
- Accuracy, AUC, Precision, Recall, F1 Score and MCC
- Confusion matrix
- Classification report

## Running the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```
Run the Streamlit application:

```bash
streamlit run app.py
```

## Live Application

[Open the Streamlit application](https://credit-card-default-ml-2025ac05348.streamlit.app/)

## Conclusion

Among the five evaluated classification models, Random Forest
provided the strongest overall performance. It achieved the highest
Accuracy (0.8105), AUC (0.7592), F1 Score (0.4562), and MCC (0.3711).

Logistic Regression achieved the highest Precision (0.6923), while
Naive Bayes achieved the highest Recall (0.9714). However, Naive Bayes
also produced substantially lower Accuracy and Precision due to a large
number of false-positive predictions.

Therefore, Random Forest was selected as the overall best-performing
model based on the balance across the evaluation metrics.
