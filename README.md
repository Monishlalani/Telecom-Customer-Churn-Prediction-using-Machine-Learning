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
