Built classification models using Random Forest, AdaBoost and SMOTE to identify suspicious viewers, reducing manual ground-vigilance costs by ~30%.


Developed market estimation and growth projection models using CatBoost, Decision Trees and K-Means, enabling data-driven estimation across 12 markets with limited survey data.

Built an automated anomaly detection model using Isolation Forest to identify unusual viewers, reducing outlier-identification processing time by ~50%.


Optimized Python data-processing and statistical weighting workflows, achieving 12x faster production processing.


Used Python, SQL and statistical analysis to solve complex business problems, analyze large datasets and deliver data-driven solutions.


__________

Built a Telecom Customer Churn Propensity Model using XGBoost, Random Forest and SMOTE, achieving X% recall and X% F1-score for churn prediction.

Developed a Telecom Customer Revenue Prediction Model using XGBoost and Random Forest to identify and target high value customers.





# Telecom Customer Churn Prediction


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
