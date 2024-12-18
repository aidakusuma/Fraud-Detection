# RISTEK Datathon 2024: Fraud Detection on Fintech Platforms

## Project Overview
This project aims to develop a machine learning model to detect fraudulent activities in fintech platform users. The objective is to identify users who have borrowed financial products but have not made payments by the predetermined deadline.

## Dataset
The dataset consists of five CSV files:
- `train.csv`: User training data with anonymized identity features and fraud labels
- `loan_activities.csv`: Records of financial product loans
- `non_borrower_user.csv`: Data on users with infrequent loan activities
- `test.csv`: User data for prediction
- `sample_submission.csv`: Sample submission file
link datasets: https://www.kaggle.com/competitions/ristek-datathon-2024/data 

## Key Findings and Methodology

### Data Preprocessing
- Handled outliers using Interquartile Range (IQR) method
- Converted data types and normalized features
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

### Feature Engineering
1. Calculated median transaction time for each user
2. Computed loan count per user
3. Identified most common loan type for each user
4. One-hot encoded loan types

### Model Development
- Algorithm: XGBoost Classifier
- Key Parameters:
  - Binary logistic objective
  - AUC evaluation metric
  - 1000 estimators
  - Learning rate: 0.1
  - Max depth: 10

### Performance Metrics
- Average Precision Score: 0.8315
- Confusion Matrix Highlights:
  - Precision (Non-Fraud): 0.98
  - Precision (Fraud): 0.84
  - Recall (Non-Fraud): 0.81
  - Recall (Fraud): 0.99
- Overall Accuracy: 0.90

### Feature Importance
Top features influencing fraud detection:
1. `loan_type_3.0`
2. `loan_type_6.0`
3. `loan_type_1.0`
4. `loan_count`

## Conclusions
- The XGBoost model demonstrates strong performance in detecting fraudulent activities
- Loan type and loan count are crucial indicators of potential fraud
- The model successfully addresses class imbalance using SMOTE

## Recommendations
- Continue refining feature engineering techniques
- Explore ensemble methods to potentially improve model performance
- Conduct regular model retraining to adapt to changing patterns

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SMOTE
- Matplotlib
- Seaborn

## Acknowledgments
RISTEK Datathon 2024 Fraud Detection Challenge
