# Telecom Customer Churn Prediction and Profit Optimization

## Overview

This team project uses machine learning to predict customer churn in the telecommunications industry and optimize retention campaign decisions based on expected financial outcomes.

Rather than focusing solely on predictive accuracy, we evaluated how classification thresholds affect both model performance and expected incremental profit. The goal was to connect machine learning predictions with business-oriented decision-making.

## Dataset and Preprocessing

We used the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), containing 7,043 customer records and 21 original columns, including customer demographics, service subscriptions, contract information, and billing history.

Data preprocessing included:

- Handling missing values and converting `TotalCharges` to a numeric format.
- Encoding categorical variables using binary mapping and one-hot encoding.
- Creating additional features, including `AvgCharges` and `HasInternet`.
- Standardizing numerical features.
- Applying random undersampling to address class imbalance.
- Removing customer identifiers before model training.

## Modeling and Evaluation

We compared four machine learning algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Each algorithm was tuned using both **GridSearchCV** and **Optuna**, resulting in eight model configurations.

Performance was evaluated using accuracy, precision, recall, F1-score, ROC AUC, confusion matrices, feature importance analysis, and stratified 5-fold cross-validation.

## Profit-Based Threshold Optimization

To incorporate business considerations into model evaluation, we defined an incremental profit function based on retention campaign costs, assumed customer value, and the probability of successful retention.

The function assigns an expected benefit to correctly targeted churners (true positives), subtracts unnecessary campaign costs for false positives, and uses a no-campaign baseline for other outcomes.

We then examined different classification thresholds to identify the decision rule that maximized expected incremental profit.

## Key Results

Under the project's simulated business assumptions, reducing the classification threshold from **0.50 to 0.12** produced the following results:

| Metric | Baseline (0.50) | Profit-Optimized (0.12) |
|---|---:|---:|
| Accuracy | 0.77 | 0.65 |
| Recall | 0.80 | 0.99 |
| Expected Incremental Profit | KRW 373.90M | KRW 439.06M |

**Key finding:** The profit-optimized threshold increased simulated incremental profit by approximately **17%**, while improving churn recall from 0.80 to 0.99.

Although classification accuracy decreased, the lower threshold identified more customers at risk of churning, demonstrating the importance of aligning predictive models with business objectives.

## Technologies

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Hyperparameter Optimization:** GridSearchCV, Optuna
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## Project Files

- [`Customer_Churn_Prediction.ipynb`](Customer_Churn_Prediction.ipynb) — Data preprocessing, model training, evaluation, and profit optimization
- [`Telco_Customer_Churn.csv`](Telco_Customer_Churn.csv) — Customer churn dataset
- [`report.pdf`](report.pdf) — Detailed project report
- [`Presentation.pdf`](Presentation.pdf) — Project presentation

## Limitations

The financial results are based on simulated campaign assumptions rather than observed business outcomes. Customer value and retention success probabilities were simplified using fixed values.

Future work could incorporate customer-specific lifetime value estimates, predicted campaign response probabilities, and budget-constrained retention strategies.
