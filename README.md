# Multiclass Classification on Diabetes Dataset

## 📌 Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## 📖 Project Description
This project performs **Multiclass Classification** to predict diabetes status (`Diabetes_012`) based on various health indicators. The dataset contains three classes:
- **0**: No Diabetes
- **1**: Pre-Diabetes
- **2**: Diabetes

### Key Features:
- Performed **Data Preprocessing**, **Feature Engineering**, and **Class Balancing**.
- Applied **SMOTE** to handle class imbalance.
- Used **Multiple Machine Learning Models** including:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - LightGBM
- **Hyperparameter tuning** was performed to improve accuracy.

---

## 📊 Dataset
- **Dataset Name:** `diabetesML.csv`
- **Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Features:**
  - Various health factors like **BMI, Age, Physical Activity, Mental Health**.
  - **Target Variable (`Diabetes_012`)**:
    - 0 → No Diabetes
    - 1 → Pre-Diabetes
    - 2 → Diabetes
- **Preprocessing:**
  - Removed **irrelevant columns** (Education, Income).
  - **Handled outliers** using IQR-based filtering.
  - **Normalized numerical features** using **MinMax Scaling**.

---

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries Used:**
  - 📊 `pandas`, `numpy` → Data manipulation
  - 🎨 `matplotlib`, `seaborn` → Data visualization
  - 🤖 `scikit-learn` → Machine learning models & preprocessing
  - 🔄 `imbalanced-learn (SMOTE)` → Class balancing
  - ⚡ `XGBoost`, `LightGBM` → Boosting algorithms

---

## 🔄 Project Workflow
### **1️⃣ Data Exploration**
- Checked class distribution and feature statistics.
- Visualized correlation among features.

### **2️⃣ Data Preprocessing**
- Removed duplicates.
- Dropped **Education & Income** columns (low correlation with target).
- **Handled Outliers** using IQR.
- **Normalized Features** with **MinMax Scaling**.

### **3️⃣ Handling Class Imbalance**
- Applied **SMOTE** to balance class distribution.

### **4️⃣ Model Development**
- Implemented **Random Forest, XGBoost, Gradient Boosting, and LightGBM**.

### **5️⃣ Model Evaluation**
- Used **Accuracy, Precision, Recall, F1-Score, AUC-ROC**.

### **6️⃣ Hyperparameter Tuning**
- Optimized each model using **RandomizedSearchCV**.

---

## 🤖 Models Implemented
| Model | Accuracy (Before Tuning) |
|--------|--------------------------|
| **Random Forest** | 83% |
| **XGBoost** | 82% |
| **Gradient Boosting** | 83% |
| **LightGBM** | **84%** ✅ |

✅ **LightGBM performed the best before tuning.**

---

## 🎯 Hyperparameter Tuning
Used **RandomizedSearchCV** to optimize models.

### **Best Tuned Parameters**
- **Random Forest:**
  - `n_estimators`: 100
  - `max_depth`: 20
  - `min_samples_split`: 5
- **XGBoost:**
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `n_estimators`: 100
- **Gradient Boosting:**
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `n_estimators`: 100
- **LightGBM:**
  - `learning_rate`: 0.1
  - `max_depth`: 10
  - `n_estimators`: 100

---

## 📈 Results
| Model | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|--------|--------------------------|--------------------------|
| **Random Forest** | 83% | 84% |
| **XGBoost** | 82% | 84% |
| **Gradient Boosting** | 83% | 84% |
| **LightGBM** | **84%** ✅ | **84%** ✅ |

✅ **After tuning, all models performed similarly (~84%)**.  
⚠ **Increasing hyperparameters further may cause overfitting**.

---

## 📊 Visualization
📌 **Feature Correlation Heatmap**
![Correlation Matrix](![image](https://github.com/user-attachments/assets/4052addb-b509-4be9-b575-c7e1f47b9064)

)

📌 **Class Distribution (Before & After SMOTE)**
![Class Distribution](![image](https://github.com/user-attachments/assets/cd42d10f-090f-4bc7-82ec-ed45f6029364)

)

📌 **ROC Curves for Each Model**
![ROC Curve](![image](https://github.com/user-attachments/assets/f16256eb-6d4f-4f0a-bce9-dbdc4e33c00b)![image](https://github.com/user-attachments/assets/a2454077-e460-45c3-9ce6-7fb9e73e49db)![image](https://github.com/user-attachments/assets/45b3a7f9-e7a0-4c04-9a3d-73aef11347d8)![image](https://github.com/user-attachments/assets/7a6c6e05-3ff5-42c2-90b2-e66142a84241)

 )
---

## 🔍 Conclusion
- **LightGBM** performed the best **before tuning**.
- **All models converged to ~84% accuracy after tuning**.
- **Feature Engineering (Outlier Removal, Normalization, SMOTE) significantly improved results.**
- Further tuning may **overfit the model**.

---

## 🚀 How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Multiclass-Classification.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the classification script**:
   ```bash
   python MULTICLASS CLASSIFICATION.PY
   ```
4. **Check results and visualizations** in the console

