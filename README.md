# MSC-project

The Predictive Retention System project aims to reduce telecom customer churn by developing an accurate machine learning model to predict churn using a dataset of ~7,043 records, leveraging Logistic Regression, Random Forest, and XGBoost, with the latter achieving the highest performance (82% accuracy, 0.66 F1-score, 0.87 ROC-AUC). The objectives include conducting exploratory data analysis to uncover churn drivers (e.g., month-to-month contracts, short tenure), preprocessing data through encoding and scaling, evaluating models with metrics like Precision and Recall, and identifying key features via importance analysis.

![Screenshot 2025-05-13 184144](https://github.com/user-attachments/assets/d7295b30-1062-4d67-96f6-d9b2a0877725)

A user-friendly Streamlit dashboard enables support agents to input customer data, view churn probabilities, visualize influential factors, and receive personalized intervention recommendations, such as offering contract upgrades or discounts. The system ensures robustness with error handling, supports scalability for cloud deployment, and translates insights into business strategies like proactive retention campaigns, while laying the groundwork for future enhancements like real-time CRM integration and customer segmentation.

![Screenshot 2025-05-13 184158](https://github.com/user-attachments/assets/7b656aae-b659-4d21-9435-a22b316d89b5)

# Project Objectives
Perform Exploratory Data Analysis (EDA) to understand data distributions and churn patterns.

Preprocess the data for machine learning (handling missing values, encoding, scaling).

Train and evaluate multiple classification models (Logistic Regression, Random Forest, XGBoost).

Analyze feature importance to identify key churn drivers.

Provide actionable business recommendations to reduce churn.

![image](https://github.com/user-attachments/assets/33f73894-96fd-4ebf-b923-e095a97a8ae6)

# Model Details

The project uses three machine learning models to predict customer churn:

* Logistic Regression:

Algorithm: Linear model for binary classification.
Parameters: max_iter=1000 for convergence.
Strengths: Interpretable, performs well on linearly separable data.
Usage: Predicts churn probability and provides a baseline model.

* Random Forest:

Algorithm: Ensemble of decision trees using bagging.
Parameters: random_state=42 for reproducibility.
Strengths: Handles non-linear relationships, robust to overfitting.
Usage: Provides robust predictions and feature importance.

* XGBoost:

Algorithm: Gradient boosting with decision trees.
Parameters: use_label_encoder=False, eval_metric='logloss'.
Strengths: High accuracy, handles complex patterns, effective feature importance.
Usage: Primary model for predictions and feature importance visualization.

# Project Structure

The project is organized as follows: 

![image](https://github.com/user-attachments/assets/91892bd0-5313-4bae-a554-ce8ef5834c9e)

# Features

The Streamlit app offers the following features:

* User-Friendly Interface:
Form-based input for customer demographics, account information, services, and billing details.
Organized into sections with sliders, dropdowns, and number inputs for ease of use.

* Churn Predictions:
Displays predictions from Logistic Regression, Random Forest, and XGBoost.
Shows churn probability as a percentage for each model.
Visualized using Streamlit metrics for quick interpretation.

* Feature Importance:
Displays a bar chart of the top 5 features influencing the XGBoost prediction.
Helps identify key factors driving churn (e.g., tenure, contract type).

* Personalized Interventions:
Generates tailored retention recommendations based on input data and churn probability.

* Examples:
Offer discounts for month-to-month contract customers.
Provide onboarding support for new customers (<12 months tenure).
Escalate high-risk customers (>70% churn probability) to retention specialists.

* Customer Summary:
Summarizes key customer details (ID, tenure, monthly charges, contract, services).
Aids in quick review and decision-making.

* Debugging Support:
Outputs processed data columns and expected feature names for troubleshooting feature mismatches.
Comprehensive error handling for file loading, preprocessing, and predictions.

![image](https://github.com/user-attachments/assets/e32ee714-d6e4-4012-9e21-c964898225a2)

# Conclusion
Best Model: XGBoost achieved 85% accuracy and 0.84 AUC-ROC.

Key Drivers: Tenure, MonthlyCharges, and Contract type most influence churn.

Impact: Implementing these insights can reduce churn by ~20% and improve revenue retention.

![image](https://github.com/user-attachments/assets/f9d3b78a-276f-4afa-8d0d-56f4821ff72c)


