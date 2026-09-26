# Telecom Customer Churn Prediction

---

## 📌 Project Overview

Customer attrition (churn) poses a major challenge for telecommunications companies. This project builds a complete end-to-end Machine Learning pipeline to predict whether a telecom customer is likely to churn. By identifying high-risk churn customers in advance, telecom companies can take proactive retention steps, offer tailored promotions, and improve overall customer lifetime value.

---

## 🎯 Business Objectives

1. **Predictive Modeling:** Train supervised classification algorithms to predict customer churn probability.
2. **Data Cleaning & Handling Imbalance:** Handle missing continuous values (e.g., `TotalCharges`), encode categorical features, and address class imbalance.
3. **Key Factor Analysis:** Identify primary triggers leading to customer departure (e.g., contract types, tenure, payment methods).

---

## 🏗️ System Architecture & Workflow

Below is the end-to-end system architecture pipeline for this project:

```
+-----------------------------------------------------------------------+
|                                                                       |
|                          Project Pipeline                             |
|                                                                       |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  Stage 1: Problem Definition & Data Collection                        |
|  - Load Dataset (`Customer-Churn.csv`)                                 |
|  - Define Target Variable (`Churn`)                                   |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  Stage 2: Exploratory Data Analysis & Preprocessing                   |
|  - Type casting (Convert `TotalCharges` to numeric)                  |
|  - Drop Null / Missing values                                         |
|  - One-Hot Encoding (`pd.get_dummies`)                                |
|  - Feature Scaling & Train-Test Split                                 |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  Stage 3: Model Building & Training                                   |
|  - Decision Tree Classifier                                           |
|  - Random Forest Classifier                                           |
|  - AdaBoost / Ensemble Methods                                        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  Stage 4: Model Evaluation                                            |
|  - Classification Report (Precision, Recall, F1-score)                |
|  - Confusion Matrix & Accuracy Analysis                               |
+-----------------------------------------------------------------------+

```

---

## 📊 Data Insights & Initial Findings

* **Baseline Churn Rate:** Approximately **26.54%** of customers in the dataset churned.
* **Dataset Structure:** Contains 7,043 rows and 21 columns covering demographic info, subscribed services, account details, and charges.
* **Key Cleaning Steps:**
* `TotalCharges` contained whitespace strings for new customers (`tenure = 0`), which were converted to numeric NaN values and dropped.
* Implemented `pd.get_dummies()` to transform categorical variables into binary indicator features.



---

## 🛠️ Tools & Technologies

* **Language:** Python
* **Environment:** Jupyter Notebook
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Decision Tree, Random Forest, AdaBoost, Train-Test Split, Metrics)
* **Data Visualization:** Matplotlib, Seaborn

---

## 🧑‍💻 Author

**Monish Lalani**








# classification--telecom-churn-customer-analysis

## Tools and Technology Used
<a href="https://www.javatpoint.com/classification-algorithm-in-machine-learning" rel="nofollow"><img alt="Classification" src="https://img.shields.io/badge/-Classification-f5841f?style=for-the-badge" style="max-width: 100%;"/></a>
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) [![Jupyter](https://img.shields.io/badge/-Jupyter-f5841f?style=for-the-badge)](https://img.shields.io/badge/-Jupyter-f5841f?style=for-the-badge) [![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) [![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) [![MATPLOTLIB](https://img.shields.io/badge/-MATPLOTLIB-007aa6?style=for-the-badge)](https://img.shields.io/badge/-MATPLOTLIB-007aa6?style=for-the-badge) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)

---

## Machine Learning Lifecycle
Below is the end-to-end architecture and workflow for the Telecom Customer Churn Prediction project:


```

+-------------------------------------------------------------------------------+
|                        TELECOM CHURN PREDICTION PIPELINE                       |
+-------------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------------+
| STAGE 1: DATA INGESTION & EXPLORATION                                         |
|  * Load raw telecom dataset (7,043 rows x 21 features)                        |
|  * Analyze class distribution (Overall Churn Rate: ~26.53%)                   |
+-------------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------------+
| STAGE 2: DATA PREPROCESSING & CLEANING                                         |
|  * Convert 'TotalCharges' from object/string to float                        |
|  * Handle missing values (drop nulls resulting from missing charge values)    |
|  * Feature Encoding: One-Hot Encoding categorical features                    |
|  * Target Encoding: Map target binary variable ('No': 0, 'Yes': 1)            |
+-------------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------------+
| STAGE 3: MODEL TRAINING & EVALUATION                                          |
|  * Train-Test Split (80% Train / 20% Test)                                    |
|  * Train baseline classifiers (e.g., DecisionTree, RandomForest, AdaBoost)    |
|  * Model evaluation using Precision, Recall, F1-score, & Confusion Matrix     |
+-------------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------------+
| STAGE 4: MODEL DEPLOYMENT & STREAMLIT WEB APP                                 |
|  * Save & export trained model pipelines (e.g., via pickle/joblib)            |
|  * Build interactive Streamlit GUI for real-time customer feature input       |
|  * Display immediate churn prediction probability and risk scoring            |
+-------------------------------------------------------------------------------+

```

---

## Business Objective
1. **Predictive Modeling:** Build a predictive machine learning pipeline to identify telecom subscribers who are likely to churn (cancel their service).
2. **Probability Scoring:** Estimate the churn risk percentage for each individual subscriber record to enable targeted customer retention strategies.
3. **Interactive Web Interface:** Deploy an intuitive Streamlit web application allowing business stakeholders to test custom input values and obtain instant predictions.

---

## Telecom Customer Retention Key Insights

- **Baseline Exploration:** Overall churn rate in the dataset is **26.53%**, highlighting a significant class imbalance that requires evaluation metrics beyond standard accuracy.
- **Data Preprocessing Insights:** Identified data type mismatches where numerical features (such as `TotalCharges`) were imported as strings with missing blank values, necessitating type conversion and missing value cleaning.
- **Model Performance:** Initial Decision Tree baseline achieved **~77% accuracy**, with detailed performance evaluated across Precision, Recall, and F1-score for the positive churn class.
- **Streamlit Web Application:** Features an interactive web dashboard built with Streamlit to enable real-time risk scoring for new customers.
- **Actionable Business Recommendations:**
  - Focus retention efforts on month-to-month subscribers and customers with high monthly charges without long-term contracts.
  - Implement proactive engagement for short-tenure subscribers during initial onboarding phases.
  - Offer incentives or discounted upgrades for higher-value tech support and security add-ons to increase customer stickiness.

---

## Author

**Monish Lalani**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/monish-lalani/) 
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:monishlalani12@gmail.com)

```
