#!/usr/bin/env python
# coding: utf-8

# In[104]:


## Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[105]:


## Reading the file
df = pd.read_csv('/content/Customer-Churn.csv')


# In[106]:


df.head(5)


# In[107]:


df.Churn.value_counts()/len(df)*100


# ## **Churn Rate**: 26.53%
# 
# - Which means, 26.53% of the customers churn out of this telecom company

# In[108]:


## Divide data into X and y - X (Independent features), y(Dependent variable)

# Define y (target variable)
y = df['Churn']

# Define X (features) by dropping 'customerID' and 'Churn'
X = df.drop(columns=['customerID', 'Churn'])

# Optional: Verify the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# ## Train Test Split

# In[109]:


X = pd.get_dummies(X, drop_first=True)
y = df['Churn'].map({'No': 0, 'Yes': 1})


# In[110]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Model Building

# In[111]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)


# In[112]:


y_pred_dt = model_dt.predict(X_test)


# In[113]:


print(classification_report(y_test, y_pred_dt))


# In[114]:


df.info()


# ## Initial Insights
# 
# - Base model has a accuracy of 76% which is not reliable because of the imbalanced dataset
# - TotalCharges needs to be a float/int type (Data Cleaning)
# - Need to perform Feature Scaling

# ## Data Cleaning

# In[115]:


telco_data = df.copy()


# In[116]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')


# In[117]:


telco_data.info()


# In[118]:


telco_data.loc[telco_data['TotalCharges'].isnull() == True]


# In[119]:


telco_data.dropna(how='any', inplace=True)


# In[120]:


telco_data.info()


# In[121]:


telco_data.head(5)


# In[122]:


telco_data.tenure.max()


# In[123]:


import pandas as pd

# Define the bin edges (0 to 72, stepping by 12 months)
bins = [0, 12, 24, 36, 48, 60, 72]

# Define the labels for each bin
labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']

# Create the new binned column
telco_data['tenure_bin'] = pd.cut(telco_data['tenure'], bins=bins, labels=labels, include_lowest=True)

# Optional: Check the result
print(telco_data[['tenure', 'tenure_bin']].head(10))
print(telco_data['tenure_bin'].value_counts().sort_index())


# In[124]:


telco_data.head(5)


# In[125]:


## Divide data into X and y - X (Independent features), y(Dependent variable)

# Define y (target variable)
y = telco_data['Churn']

# Define X (features) by dropping 'customerID' and 'Churn'
X = telco_data.drop(columns=['customerID', 'Churn', 'tenure'])

# Optional: Verify the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# In[90]:


X


# In[126]:


X = pd.get_dummies(X, drop_first=True)
y = telco_data['Churn'].map({'No': 0, 'Yes': 1})


# In[127]:


X


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Feature Scaling

# In[129]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[130]:


model_dt2 = DecisionTreeClassifier()
model_dt2.fit(X_train, y_train)

y_pred_dt2 = model_dt2.predict(X_test)


# In[132]:


print(classification_report(y_test, y_pred_dt2))


# ## Feature Scaling - MinMaxScaler

# In[133]:


mms = MinMaxScaler()
X_train_mms = mms.fit_transform(X_train)
X_test_mms = mms.transform(X_test)


# In[134]:


model_dt3 = DecisionTreeClassifier()
model_dt3.fit(X_train_mms, y_train)

y_pred_dt3 = model_dt3.predict(X_test_mms)


# In[135]:


print(classification_report(y_test, y_pred_dt3))


# ## SMOTEENN () [UpSampling + ENN)

# In[136]:


from imblearn.combine import SMOTEENN


# In[137]:


sm = SMOTEENN()
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)


# In[138]:


model_dt_smoteenn = DecisionTreeClassifier()
model_dt_smoteenn.fit(X_train_resampled, y_train_resampled)

y_pred_dt_smoteenn = model_dt_smoteenn.predict(X_test)


# In[139]:


print(classification_report(y_test, y_pred_dt_smoteenn))


# In[140]:


model_rf_smoteenn = RandomForestClassifier(n_estimators=500)
model_rf_smoteenn.fit(X_train_resampled, y_train_resampled)

y_pred_rf_smoteenn = model_rf_smoteenn.predict(X_test)


# In[141]:


print(classification_report(y_test, y_pred_rf_smoteenn))


# ## XGBoost without SMOTEENN

# In[142]:


from xgboost import XGBClassifier

# Initialize and train the XGBoost model
model_xgb = XGBClassifier(random_state=42)  # random_state for reproducibility
model_xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = model_xgb.predict(X_test)


# In[143]:


print(classification_report(y_test, y_pred_xgb))


# ## XGBoost with SMOTEENN

# In[144]:


from xgboost import XGBClassifier

# Initialize and train the XGBoost model
model_xgb_smoteenn = XGBClassifier(random_state=42)  # random_state for reproducibility
model_xgb_smoteenn.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_xgb_smoteenn = model_xgb_smoteenn.predict(X_test)


# In[145]:


print(classification_report(y_test, y_pred_xgb_smoteenn))


# ## XGBoost with SMOTE

# In[146]:


from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

# Check class distribution before SMOTE
print("Before SMOTE:", Counter(y_train))

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("After SMOTE:", Counter(y_train_smote))

# Initialize and train the XGBoost model on resampled data
model_xgb_smote = XGBClassifier(random_state=42)
model_xgb_smote.fit(X_train_smote, y_train_smote)

# Make predictions on the original test set (unchanged)
y_pred_xgb_smote = model_xgb_smote.predict(X_test)


# In[147]:


print(classification_report(y_test, y_pred_xgb_smote))


# In[150]:


from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from collections import Counter

# Check original distribution
print("Before ADASYN:", Counter(y_train))

# Apply ADASYN to training data only
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Check new distribution
print("After ADASYN:", Counter(y_train_adasyn))

# Train XGBoost on ADASYN-resampled data
model_xgb_adasyn = XGBClassifier(random_state=42)
model_xgb_adasyn.fit(X_train_adasyn, y_train_adasyn)

# Predict on the original test set
y_pred_xgb_adasyn = model_xgb_adasyn.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_xgb_adasyn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb_adasyn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb_adasyn))


# In[149]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate scale_pos_weight (ratio of negative to positive class)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost with scale_pos_weight
model_xgb_weighted = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'  # avoids warnings
)

model_xgb_weighted.fit(X_train, y_train)
y_pred_weighted = model_xgb_weighted.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred_weighted))
print("\nClassification Report:\n", classification_report(y_test, y_pred_weighted))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_weighted))


# In[151]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Calculate the weight for the positive (minority) class
# Common approach: weight_positive = negative_count / positive_count
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
weight_positive = negative_count / positive_count if positive_count > 0 else 1
print(f"Negative class count: {negative_count}")
print(f"Positive class count: {positive_count}")
print(f"Weight for positive class: {weight_positive:.2f}")

# Create sample weights: higher weight for minority class
sample_weight = np.where(y_train == 1, weight_positive, 1.0)

# Train AdaBoost with sample weights
model_ada_weighted = AdaBoostClassifier(
    n_estimators=50,  # default, you can tune
    random_state=42
)

model_ada_weighted.fit(X_train, y_train, sample_weight=sample_weight)

# Predict and evaluate
y_pred_ada_weighted = model_ada_weighted.predict(X_test)

print("AdaBoost with sample weighting - Accuracy:", accuracy_score(y_test, y_pred_ada_weighted))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada_weighted))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_ada_weighted))


# ## **Hyper Parameter Optimization**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, scale_pos_weight]  # try both balanced and weighted
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# Use 'f1' or 'recall' scoring since churn is imbalanced
search = RandomizedSearchCV(
    xgb,
    param_grid,
    n_iter=50,
    cv=5,
    scoring='f1',  # or 'recall' if catching churn is priority
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best CV F1 score:", search.best_score_)

# Use best model
best_model = search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nTest Classification Report:\n", classification_report(y_test, y_pred_best))


# In[ ]:


# Save this model

import joblib

# Save the best model to a .pkl file
joblib.dump(best_model, 'best_xgboost_churn_model.pkl')

print("Best model successfully saved as 'best_xgboost_churn_model.pkl'")


# ## Trying out few more techniques

# In[152]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=300,
    max_depth=6
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))


# In[155]:


from catboost import CatBoostClassifier

model_cat = CatBoostClassifier(
    auto_class_weights='Balanced',  # or 'SqrtBalanced'
    verbose=0,
    random_state=42
)
model_cat.fit(X_train, y_train)
y_pred_cat = model_cat.predict(X_test)
print(classification_report(y_test, y_pred_cat))


# In[156]:


from lightgbm import LGBMClassifier

model_lgb = LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
model_lgb.fit(X_train, y_train)
y_pred_lgb = model_lgb.predict(X_test)
print(classification_report(y_test, y_pred_lgb))


# ## **Optuna fine tuning**

# In[ ]:


import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2), # L2 regularization
        'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, scale_pos_weight]),
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)

    # Use F1 for minority class as scorer
    f1_scorer = make_scorer(f1_score, average='binary', pos_label=1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer, n_jobs=-1).mean()

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Increase trials for better results

print("Best parameters:", study.best_params)
print("Best CV F1 score:", study.best_value)

# Train final model with best params
best_model_optuna = XGBClassifier(**study.best_params)
best_model_optuna.fit(X_train, y_train)
y_pred_optuna = best_model_optuna.predict(X_test)
print(classification_report(y_test, y_pred_optuna))


# In[ ]:


# Save this model

import joblib

# Save the best model to a .pkl file
joblib.dump(best_model_optuna, 'best_optuna_churn_model.pkl')

print("Best model successfully saved as 'best_optuna_churn_model.pkl'")


# In[157]:


# Save this model

import joblib

# Save the best model to a .pkl file
joblib.dump(model_ada_weighted, 'ada_boost_churn_model.pkl')

print("Best model successfully saved as 'ada_boost_churn_model.pkl'")


# In[ ]:




