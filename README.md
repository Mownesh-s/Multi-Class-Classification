# Multiclass Classification on Diabetes Dataset

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Key Steps](#key-steps)
- [Models Implemented](#models-implemented)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## Project Description
This project focuses on **multiclass classification** to predict the severity of diabetes using machine learning models. Various classification models were applied and evaluated using metrics such as **accuracy, precision, recall, F1-score, and AUC-ROC**.

---

## Dataset
- The dataset used is `diabetesML.csv`, containing patient health information.
- Features include **BMI, Mental Health, Physical Health, Age, and more**.
- The target variable is `Diabetes_012`, which has three classes: `0 (No Diabetes)`, `1 (Prediabetes)`, and `2 (Diabetes)`.
- Data preprocessing steps include:
  - Handling missing values
  - Removing duplicates
  - Removing outliers using the IQR method
  - Scaling numerical features using MinMaxScaler
  - Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas` and `numpy` for data manipulation
  - `matplotlib` and `seaborn` for data visualization
  - `scikit-learn` for preprocessing, modeling, and evaluation
  - `imbalanced-learn` for handling class imbalance (SMOTE)
  - `XGBoost` and `LightGBM` for boosting algorithms

---

## Key Steps
1. **Data Exploration**:
   - Identified numerical and categorical variables.
   - Visualized feature distributions and class imbalances.
2. **Feature Engineering**:
   - Handled missing values and removed redundant features.
   - Applied MinMax scaling for numerical features.
3. **Class Balancing**:
   - Applied **SMOTE** to balance the dataset.
4. **Model Training**:
   - Trained multiple classification models.
5. **Evaluation**:
   - Used metrics like accuracy, precision, recall, and ROC curves.
6. **Hyperparameter Tuning**:
   - Applied `RandomizedSearchCV` for parameter optimization.

---

## Models Implemented
1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Gradient Boosting Classifier**
4. **LightGBM Classifier**

---

## Hyperparameter Tuning
- Used `RandomizedSearchCV` to optimize model performance.
- Tuned parameters included:
  - `n_estimators`, `max_depth`, `learning_rate` (for boosting models)
  - `min_samples_split`, `criterion`, `max_features` (for Random Forest)
- Models achieved **similar accuracy (~84%) after tuning**.

---

## Results
### **Without Hyperparameter Tuning:**
- **LightGBM performed the best** with an accuracy of **83.7%**.

### **With Hyperparameter Tuning:**
- All models achieved similar accuracy of **~84%**.
- Increasing hyperparameters further may lead to overfitting.

---

## Conclusion
- **LightGBM** is the best-performing model without tuning.
- **Hyperparameter tuning** improves all models to ~84% accuracy.
- Feature selection, data balancing, and normalization significantly impact results.

---

## How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Multiclass-Classification-Project.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the classification script**:
   ```bash
   python MULTICLASS_CLASSIFICATION.py
   ```
4. **Check results and visualizations** in the console.
```
