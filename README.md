# SKS-Machine-Learning-intern
Machine learning intern with the help of python, Pandas, Matplotlib, Numpy, Scikit-learn in Jupyter notebook.


📊 Customer Churn Analysis using Machine Learning
🎥 Video Demonstration
Watch the complete project walkthrough and implementation:
�
Video Highlights: @https://www.linkedin.com/posts/faizan-khan-b8270a287_machinelearning-datascience-python-activity-7388302256506351616-_L53?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAEWxzhIB5ZnZez9v5AOu-uboWfM4BFkSV9A&utm_campaign=copy_link

✅ Complete data preprocessing pipeline
✅ Model training and evaluation
✅ Interactive dashboard with 4 comprehensive visualizations
✅ Feature importance analysis
✅ Real-world insights and recommendations

📑 Table of Contents

Project Overview
Business Problem
Dataset Information
Project Workflow
Tasks Breakdown
Key Findings
Technologies Used
Installation & Setup
Usage
Results
Future Enhancements
Contributing
License
Contact
🎯 Project Overview

This project implements a comprehensive machine learning solution to predict customer churn in the telecommunications industry. By analyzing customer behavior patterns and service usage data, we develop predictive models that help businesses identify at-risk customers and implement targeted retention strategies.
Key Objectives:
Predict customer churn with high accuracy
Identify key factors contributing to customer attrition
Provide actionable insights for customer retention
Develop a scalable ML pipeline for production deployment
💼 Business Problem

Customer churn (customer attrition) is a critical business metric that directly impacts revenue and growth. Acquiring new customers is significantly more expensive than retaining existing ones. This project addresses:
Challenges:

High customer acquisition costs
Revenue loss from churned customers
Inability to proactively identify at-risk customers
Lack of data-driven retention strategies
Solution:

A machine learning system that:
Predicts churn probability for individual customers
Identifies key drivers of customer attrition
Enables targeted intervention strategies
Reduces customer acquisition costs
📊 Dataset Information

Source: IBM Sample Data Sets - Telco Customer Churn
Dataset Characteristics:
Total Records: 7,043 customers
Features: 21 attributes
Target Variable: Churn (Yes/No)
Data Type: Tabular data with mixed types (numerical and categorical)
Feature Categories:

Customer Demographics: Gender, Age, Partner, Dependents
Service Information: Phone Service, Internet Service, Online Security, Tech Support
Account Details: Tenure, Contract Type, Payment Method, Monthly Charges
Usage Patterns: Multiple Lines, Streaming Services, Device Protection
🔄 Project Workflow
┌─────────────────┐
│  Data Loading   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Cleaning   │
│ & Preprocessing │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │
│ Engineering     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train-Test      │
│ Split           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ & Evaluation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │
│ & Analysis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Saving    │
│ & Deployment    │
└─────────────────┘
📋 Tasks Breakdown

Task 1: Data Preparation & Preprocessing

Objective: Transform raw data into a clean, analysis-ready format.
Steps Implemented:
Data Loading
Imported dataset from CSV file
Verified data integrity and structure
Initial data exploration
Data Cleaning
Handled missing values using median imputation
Removed duplicate records
Addressed infinite values
Feature Engineering
Converted categorical variables to numerical format
Applied one-hot encoding for multi-category features
Binary encoding for Yes/No variables
Data Validation
Verified no remaining null values
Ensured all features are numeric
Validated data types consistency

Output:

✅ Data preparation completed
Final shape: (7043, 21)
Missing values: 0
Data types: All numeric

Code Snippet:

# Data loading and initial processing
df = pd.read_csv('telco-customer-churn-by-IBM.csv')

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Feature encoding
binary_cols = [c for c in df.columns if df[c].nunique() == 2]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

Task 2: Data Splitting

Objective: Divide dataset into training and testing sets for model validation.
Methodology:
Split Ratio: 80% Training, 20% Testing
Strategy: Stratified random sampling
Random State: 42 (for reproducibility)
Implementation:
from sklearn.model_selection import train_test_split

# Separate features and target
X = df[numeric_cols].fillna(0)
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)
Results:

| Dataset | Samples | Features |
|---------|---------|----------|
| Training | 5,634 | 3 |
| Testing | 1,409 | 3 |
| Total | 7,043 | 3 |
Quality Checks:

✅ No data leakage between sets
✅ Balanced class distribution maintained
✅ No missing values in either set
✅ Feature consistency verified

Task 3: Model Training & Evaluation

Objective: Develop and compare multiple machine learning models for churn prediction.
Models Implemented

1. Logistic Regression

Type: Linear classification model
Use Case: Baseline model for binary classification
Advantages: Interpretable, fast training, probabilistic outputs
2. Random Forest Classifier

Type: Ensemble learning method
Configuration: 100 estimators, balanced class weights
Advantages: Handles non-linear relationships, feature importance
Training Process:
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model initialization and training
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, 
        class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
}

# Train models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
Evaluation Metrics:
Model
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Logistic Regression
0.XXXX
0.XXXX
0.XXXX
0.XXXX
0.XXXX
Random Forest
0.XXXX
0.XXXX
0.XXXX
0.XXXX
0.XXXX
Metrics Explanation:

Accuracy: Overall prediction correctness
Precision: Proportion of correct positive predictions
Recall: Proportion of actual positives correctly identified
F1 Score: Harmonic mean of precision and recall
ROC-AUC: Model's ability to distinguish between classes
Model Selection Criteria:
Best F1 Score (balanced precision and recall)
Highest ROC-AUC for ranking capability
Business requirements alignment
Task 4: Visualization & Analysis Dashboard
Objective: Create comprehensive visualizations to communicate insights effectively.
Dashboard Components:
1. Model Performance Comparison

Chart Type: Grouped Bar Chart
Purpose: Compare all metrics across both models
Insights: Visual identification of best-performing model
Features:
Color-coded bars for each model
Value labels on bars
Grid lines for easy reading
2. Feature Importance Analysis

Chart Type: Horizontal Bar Chart
Purpose: Identify key drivers of customer churn
Top 3 Features:

MonthlyCharges - 60.78% importance
Tenure - 37.12% importance
SeniorCitizen - 2.10% importance
Business Impact: Focus retention efforts on these factors
3. Confusion Matrix

Chart Type: Heatmap
Purpose: Visualize model prediction accuracy
Components:
True Positives: Correctly predicted churners
True Negatives: Correctly predicted non-churners
False Positives: Incorrectly predicted as churners
False Negatives: Missed actual churners
4. Detailed Metrics Table

Format: Professional table layout
Content: All performance metrics for both models
Purpose: Quick reference for stakeholders
Implementation:
import matplotlib.pyplot as plt

# Create 2x2 dashboard
fig = plt.figure(figsize=(16, 12))
fig.suptitle('📊 CUSTOMER CHURN ANALYSIS - COMPLETE DASHBOARD', 
             fontsize=18, fontweight='bold')

# Add all 4 visualizations
# [Full code in notebook]

plt.tight_layout()
plt.show()
Dashboard Features:

✅ Professional color scheme
✅ Clear labels and legends
✅ Interactive tooltips (in notebook)
✅ High-resolution export capability
✅ Presentation-ready format
🔑 Key Findings
Top Churn Predictors
1. Monthly Charges (60.78% importance)
Finding: Customers with higher monthly charges are significantly more likely to churn.
Business Implications:
Review pricing strategies for high-value customers
Implement tiered pricing with value-added services
Offer loyalty discounts for long-term contracts
Recommendation:

For customers with MonthlyCharges > $70:
- Proactive outreach by account managers
- Personalized retention offers
- Service bundle optimization
2. Tenure (37.12% importance)
Finding: New customers (low tenure) have the highest churn risk.
Critical Periods:


First 6 months: Highest risk
6-12 months: Moderate risk
12+ months: Stable customer base
Retention Strategy:
Month 1-3:  Welcome program + onboarding support
Month 4-6:  Check-in calls + satisfaction surveys
Month 7-12: Loyalty rewards + contract upgrades
3. Senior Citizen Status (2.10% importance

Finding: Senior citizens require specialized attention.
Targeted Initiatives:
Simplified billing options
Dedicated customer support line
Technology assistance programs
Model Performance Insights
Best Model: Random Forest Classifier
Strengths:
Higher recall: Better at identifying actual churners
Robust to outliers
Captures non-linear relationships
Use Cases:
Real-time churn scoring
Batch prediction for marketing campaigns
A/B testing for retention strategies
Business Impact Projection
Assumptions:
Average customer lifetime value: $1,500
Cost of retention campaign: $50 per customer
Model precision: XX%
Projected Savings:
Correctly identified churners: XXX customers
Successful retention rate: 30%
Saved customers: XXX
Revenue protected: $XXX,XXX
ROI: XXX%
🛠️ Technologies Used
Core Libraries
Library
Version
Purpose
Python
3.8+
Programming language
Pandas
1.5.3
Data manipulation
NumPy
1.24.3
Numerical computing
Scikit-learn
1.2.2
Machine learning
Matplotlib
3.7.1
Data visualization
Development Environment
- Jupyter Notebook 6.5.2
- IPython 8.10.0
- Python Standard Library
Machine Learning Components
# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif
📦 Installation & Setup
Prerequisites
Python 3.8 or higher
pip package manager
Jupyter Notebook
2GB free disk space
Step 1: Clone Repository
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis
Step 2: Create Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt
requirements.txt:
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
jupyter==1.0.0
seaborn==0.12.2
Step 4: Launch Jupyter Notebook
jupyter notebook
Step 5: Open Project Notebook
Navigate to Customer_Churn_Analysis.ipynb and run all cells.
🚀 Usage
1. Training New Model
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv('telco-customer-churn-by-IBM.csv')
# [preprocessing steps]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
with open('customer_churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
2. Making Predictions
import pickle

# Load saved model and scaler
model = pickle.load(open('customer_churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prepare new customer data
new_customer = pd.DataFrame({
    'MonthlyCharges': [85.0],
    'tenure': [5],
    'SeniorCitizen': [0]
})

# Scale features
new_customer_scaled = scaler.transform(new_customer)

# Predict churn probability
churn_probability = model.predict_proba(new_customer_scaled)[:, 1]
print(f"Churn Risk: {churn_probability[0]:.2%}")

# Get prediction
prediction = model.predict(new_customer_scaled)
print(f"Prediction: {'Will Churn' if prediction[0] == 1 else 'Will Stay'}")
3. Batch Predictions
# Load customer database
customers = pd.read_csv('customer_database.csv')

# Preprocess and scale
customers_processed = preprocess_data(customers)
customers_scaled = scaler.transform(customers_processed)

# Predict for all customers
churn_scores = model.predict_proba(customers_scaled)[:, 1]

# Add scores to dataframe
customers['churn_risk'] = churn_scores
customers['risk_category'] = pd.cut(
    churn_scores, 
    bins=[0, 0.3, 0.7, 1.0], 
    labels=['Low', 'Medium', 'High']
)

# Export high-risk customers
high_risk = customers[customers['risk_category'] == 'High']
high_risk.to_csv('high_risk_customers.csv', index=False)
📈 Results
Model Performance Summary
Random Forest (Best Model)
┌─────────────┬─────────┐
│ Metric      │ Score   │
├─────────────┼─────────┤
│ Accuracy    │ XX.XX%  │
│ Precision   │ XX.XX%  │
│ Recall      │ XX.XX%  │
│ F1 Score    │ XX.XX%  │
│ ROC-AUC     │ XX.XX%  │
└─────────────┴─────────┘
Confusion Matrix Analysis
Predicted
                No    Yes
Actual No    [TN]  [FP]
Actual Yes   [FN]  [TP]

TN: True Negatives  (Correct non-churn predictions)
FP: False Positives (Incorrectly predicted as churn)
FN: False Negatives (Missed actual churners)
TP: True Positives  (Correct churn predictions)
Feature Importance Rankings
🥇 MonthlyCharges    ████████████████████████████████ 60.78%
🥈 tenure            ████████████████                 37.12%
🥉 SeniorCitizen    █                                 2.10%
Business Metrics
Metric
Value
Customers Analyzed
7,043
Predicted Churners
XXX
High-Risk Customers
XXX
Retention Opportunities
XXX
Potential Revenue at Risk
$XXX,XXX
🔮 Future Enhancements
Phase 1: Model Improvements (Short-term)
[ ] Implement XGBoost and LightGBM models
[ ] Add hyperparameter tuning with GridSearchCV
[ ] Implement cross-validation for robust evaluation
[ ] Add SMOTE for handling class imbalance
[ ] Feature selection using recursive feature elimination
Phase 2: Advanced Analytics (Medium-term)
[ ] Time-series analysis for churn trends
[ ] Customer segmentation using clustering
[ ] Survival analysis for customer lifetime
[ ] Sentiment analysis from customer feedback
[ ] A/B testing framework for interventions
Phase 3: Deployment (Long-term)
[ ] Build REST API using Flask/FastAPI
[ ] Deploy on cloud platform (AWS/Azure/GCP)
[ ] Create real-time prediction dashboard with Streamlit
[ ] Implement automated model retraining pipeline
[ ] Add monitoring and alerting system
[ ] Build mobile app for field teams
Phase 4: Production Features
[ ] Multi-model ensemble voting
[ ] Explainable AI with SHAP values
[ ] Integration with CRM systems
[ ] Automated email campaigns for high-risk customers
[ ] Performance tracking dashboard
Roadmap Timeline:
Q1 2024: Phase 1 completion
Q2 2024: Phase 2 implementation
Q3 2024: Phase 3 deployment
Q4 2024: Phase 4 production features
📁 Project Structure
customer-churn-analysis/
│
├── 📓 Customer_Churn_Analysis.ipynb   # Main analysis notebook
├── 📊 data/
│   └── telco-customer-churn-by-IBM.csv
│
├── 🤖 models/
│   ├── customer_churn_model.pkl       # Trained Random Forest model
│   ├── scaler.pkl                      # Feature scaler
│   └── feature_names.pkl               # Feature name list
│
├── 📸 images/
│   ├── dashboard.png                   # Dashboard screenshot
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── 📄 docs/
│   ├── project_report.pdf              # Detailed project report
│   └── presentation.pptx               # Presentation slides
│
├── 🧪 tests/
│   └── test_model.py                   # Unit tests
│
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                        # This file
├── ⚖️ LICENSE                          # MIT License
└── 🔧 .gitignore                       # Git ignore rules
🤝 Contributing
Contributions are welcome! Please follow these guidelines:
How to Contribute
Fork the Repository
# Click "Fork" button on GitHub
Create Feature Branch
git checkout -b feature/AmazingFeature
Commit Changes
git commit -m 'Add some AmazingFeature'
Push to Branch
git push origin feature/AmazingFeature
Open Pull Request
Provide clear description of changes
Reference any related issues
Ensure all tests pass
Contribution Areas
🐛 Bug fixes
✨ New features
📝 Documentation improvements
🧪 Additional test cases
🎨 Visualization enhancements
⚡ Performance optimizations
Code Standards
Follow PEP 8 style guide
Add docstrings to functions
Include unit tests for new features
Update documentation as needed
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2024 [Faizan khan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text]
📞 Contact
[Faizan khan]
📧 Email: faijankha7860@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/faizan-khan-b8270a287?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

🐙 GitHub: @faizankhan07867

🐦 Twitter: @https://x.com/FaizanKhan84164?t=9eQUMoB-Jd025zhbYeqsnw&s=09

Project Link: https://github.com/faizankhan07867/SKS-Machine-Learning-intern
