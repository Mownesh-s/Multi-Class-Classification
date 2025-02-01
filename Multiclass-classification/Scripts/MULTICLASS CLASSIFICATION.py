#!/usr/bin/env python
# coding: utf-8

# In[408]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
"""import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")"""


# # IMPORTING DATASET

# In[409]:


data=pd.read_csv('diabetesML.csv')


# # OVERVIEW OF DATASET

# In[410]:


data.head()


# In[411]:


data.tail()


# In[412]:


data.shape


# # TAKING SAMPLE DATA FROM THE DATASET FOR WORKING 

# In[413]:


df = data.sample(frac=0.1, random_state=42)


# In[414]:


df.shape


# In[415]:


df.head()


# In[416]:


df.tail()


# In[417]:


df.columns=df.columns.str.strip()


# In[418]:


df.columns.tolist()


# In[419]:


for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"{col} is Numeric")
    elif pd.api.types.is_cetogorical_dtype(df[col]) or df[col].dtype=='object':
        print(f'{col} is cetogorical')


# In[420]:


df.nunique()


# In[421]:


df.info()


# In[422]:


df.describe()


# In[423]:


print("Count of duplicated entries:", df.duplicated().sum())


# In[424]:


df = df.drop_duplicates()
print("Count of duplicated entries:", df.duplicated().sum())


# In[425]:


df.shape


#   # ANALYSING THE DATA AND VISUALIZING IT :

# In[426]:


numerical_columns=df.select_dtypes(include=['int64']).columns
plt.figure(figsize=(10,5))
df[numerical_columns].boxplot(rot=90,fontsize=10)
plt.title("Box Plot for Numerical Features",fontsize=10)
plt.show()


# In[427]:


df.corr(numeric_only=True)


# In[428]:


plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features', fontsize=14)
plt.show()


# In[429]:


num_columns = len(numerical_columns)
n_cols = 4 
n_rows = (num_columns // n_cols) + (num_columns % n_cols > 0)
plt.figure(figsize=(20, n_rows * 5))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[430]:


class_distribution = df['Diabetes_012'].value_counts()

plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Class Distribution of Diabetes_012', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # DATA PREPROCESSING 

# In[431]:


from sklearn.preprocessing import MinMaxScaler,label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[432]:


df.drop(columns=['Education', 'Income'])
# THESE COLUMNS WERE DORPED BECAUSE THESE TWO ARE NOT GIVING TO MUCH INFORMATION REGARDING THE TARGET 


# In[433]:


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# In[434]:


df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Diabetes_012'])
numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
df_train = remove_outliers(df_train, numerical_columns)


# In[435]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
data[numerical_columns].boxplot(ax=axes[0])
axes[0].set_title("Before Outlier Removal")
df_train[numerical_columns].boxplot(ax=axes[1])
axes[1].set_title("After Outlier Removal")
plt.show()


# In[436]:


scaler = MinMaxScaler()
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])


# In[437]:


df_train


# In[438]:


df_test


# In[439]:


plt.figure(figsize=(14, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df_train, x=column, kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[440]:


X_train, y_train = df_train.drop(columns=['Diabetes_012']), df_train['Diabetes_012']
X_test, y_test = df_test.drop(columns=['Diabetes_012']), df_test['Diabetes_012']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[441]:


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette='viridis')
plt.title('Class Distribution (Before SMOTE)')
plt.xlabel('Diabetes_012')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, palette='viridis')
plt.title('Class Distribution (After SMOTE)')
plt.xlabel('Diabetes_012')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[442]:


X_resampled


# In[443]:


y_resampled 


# # MODEL BUILDING

# In[444]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,accuracy_score


# In[445]:


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    print(f"{model_name} Model\n")
    print('Accuracy :',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {i}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()
    
    return {"Model": model_name,"Accuracy": accuracy_score(y_test, y_pred), "AUC ROC": auc_score}


# # RANDOM FOREST CLASSIFIER

# In[446]:


rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)


# In[447]:


rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")


# # XGBClassifier 

# In[448]:


xgb = XGBClassifier(random_state=42)
xgb.fit(X_resampled, y_resampled)


# In[449]:


xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost")


# #  GradientBoostingClassifier

# In[450]:


gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_resampled, y_resampled)


# In[451]:


gbc_results = evaluate_model(gbc, X_test, y_test, "Gradient Boosting")


# # LGBMClassifier

# In[452]:


lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_resampled, y_resampled)


# In[453]:


lgbm_results = evaluate_model(lgbm, X_test, y_test, "LightGBM")


# In[454]:


results_df = pd.DataFrame([rf_results, xgb_results, gbc_results, lgbm_results])
results_df


# # HYPER PARAMETER TUNNING 

# In[455]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


# In[456]:


def evaluate_model_tunned(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted'),
        "AUC ROC": roc_auc_score(y_test, y_proba, multi_class='ovr')
    }


# # Hyper Tunning for Random Forest

# In[457]:


rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_result = evaluate_model_tunned(rf_search.best_estimator_, X_test, y_test)
rf_result["Model"] = "Random Forest"
rf_result["Best Params"] = rf_search.best_params_


# # Hyper Tunning for XGBClassifier 

# In[458]:


xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

xgb_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb_result = evaluate_model_tunned(xgb_search.best_estimator_, X_test, y_test)
xgb_result["Model"] = "XGBoost"
xgb_result["Best Params"] = xgb_search.best_params_


# # Hyper Tunning for LGBMClassifier

# In[459]:


lgbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [-1, 10, 20]
}

lgbm_search = RandomizedSearchCV(LGBMClassifier(random_state=42), lgbm_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
lgbm_search.fit(X_train, y_train)
lgbm_result = evaluate_model_tunned(lgbm_search.best_estimator_, X_test, y_test)
lgbm_result["Model"] = "LightGBM"
lgbm_result["Best Params"] = lgbm_search.best_params_


# # Hyper Tunning for Gradient Boosting

# In[460]:


gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
gb_search.fit(X_train, y_train)
gb_result = evaluate_model_tunned(gb_search.best_estimator_, X_test, y_test)
gb_result["Model"] = "Gradient Boosting"
gb_result["Best Params"] = gb_search.best_params_


# In[461]:


result = pd.DataFrame([rf_result, gb_result, xgb_result, lgbm_result])
result


# In[462]:


results_df


# # After Model Building 
# # Without Hyper Tunning LightGBM Performs Good With An Accuracy of (0.837761):-83%
# # With Hyper Tunning All the Model's Accuracy are Similar (0.84):-84%
# 

# # This is ok for the models if we increase the perameters it may overfit 

# In[ ]:




