# Customer Churn Prediction 📉

This project uses machine learning techniques to predict customer churn — whether a customer is likely to stop using a service — based on historical customer behavior data. It includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and performance comparison between multiple classification algorithms.

## 🔍 Problem Statement

Customer churn is a critical problem in subscription-based services. The goal of this project is to build a predictive model that can classify whether a customer will churn (leave the company) or not, using customer demographic and behavioral features.

---

## 📊 Dataset

- File: `customer_churn_data.csv`
- Size: ~4,000 entries
- Key features:
  - `gender`, `age`, `tenure`, `MonthlyCharges`, etc.
  - Target variable: `Churn` (Yes/No)

## 🧪 Exploratory Data Analysis (EDA)

- Handled missing values and data types
- Visualized distribution of churn across features
- Explored correlation and feature importance
- Applied label encoding and one-hot encoding where needed

## 🧠 Model Building & Evaluation

Performed in `model_building.ipynb`:
- Trained multiple classifiers:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - SVC
    
- Evaluation Metrics:
  - Accuracy
| Model                | Accuracy 
|---------------------|----------|
| Logistic Regression | 85.2%    | 
| Random Forest       | 85.0%    | 
| Decision Tree       | 86.5%    | 
| KNN                 | 88.3%    | 
| SVC                 | 89.3     |  

> 📌 *SVC* performed the best among all the models evaluated.



 

