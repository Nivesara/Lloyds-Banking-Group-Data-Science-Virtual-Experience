# 📈 Customer Retention Enhancement through Predictive Analytics

## 🏦 Project Context
This project was developed as part of a predictive analytics initiative for **SmartBank (Lloyds Banking Group)** to proactively identify customers who are likely to churn. The goal was to support business decision-making by deploying a machine learning model that predicts customer attrition with high accuracy, allowing the bank to implement targeted retention strategies.

---

## 🔍 Problem Statement
Customer churn significantly affects a bank's profitability. Retaining existing customers is more cost-effective than acquiring new ones. This project aims to:

- Analyze customer demographic and behavioral data
- Build a predictive model to identify customers likely to churn
- Provide actionable insights to improve retention strategies

---

## 🧠 Solution Approach

### 1. Data Understanding & EDA
- Imported a structured Excel dataset containing customer demographics, service usage, and churn labels
- Explored patterns using visualization tools (Seaborn, Matplotlib)
- Detected class imbalance and feature distributions

### 2. Data Preprocessing
- Handled missing values and encoded categorical variables using `ColumnTransformer`
- Scaled numerical features using `StandardScaler`
- Applied `SMOTE` to address class imbalance in the training set

### 3. Model Building & Evaluation
Built and evaluated multiple classification models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- XGBoost

### 📘 Models Evaluated:
| Model              | Accuracy | Recall | F1 Score | AUC   |
|-------------------|----------|--------|----------|--------|
| Logistic Regression | 79.9%   | 0.0%   | 0.0%     | 60.1% |
| Random Forest       | 97.9%   | 89.3%  | 94.3%    | 99.8% |
| Gradient Boosting   | 85.8%   | 31.0%  | 46.4%    | 88.0% |
| **XGBoost**         | **99.3%** | **96.7%** | **98.3%** | **99.8%** |


Used:
- `GridSearchCV` for hyperparameter tuning
- Cross-validation to assess generalization
- ROC-AUC, F1 Score, and confusion matrices for evaluation

---

## 🏆 Results

- **Best Model**: XGBoost with tuned hyperparameters
- **ROC-AUC Score**: *~0.89* (exact metric shown in notebook)
- **Business Insight**: The model effectively identifies ~89% of churn risk customers, enabling targeted retention offers.

---

## 📊 Key Visualizations

- Churn vs. non-churn distributions across features
- Correlation heatmaps
- Feature importances from tree-based models
- ROC curve comparisons for model selection

---

## 🚀 Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Preprocessing, modeling, evaluation |
| XGBoost | Advanced boosting model |
| SMOTE | Synthetic minority oversampling |
| Jupyter Notebook | Interactive development environment |

---

## 📁 Project Structure

📦 Lloyds_Customer_Churn/
│
├── Lloyds.ipynb                # Main Jupyter notebook
├── Customer_Churn_Data_Large.xlsx  # Source dataset
├── Customer Churn Analysis Report.pdf # Final business report
├── README.md                   # Project documentation
└── models/                     # (Optional) Saved model files
