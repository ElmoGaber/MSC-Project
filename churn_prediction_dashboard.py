import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        with open('D:\\Desktop\\New folder\\logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('D:\\Desktop\\New folder\\random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('D:\\Desktop\\New folder\\xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('D:\\Desktop\\New folder\\scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return lr_model, rf_model, xgb_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}. Please ensure all .pkl files are in 'D:\\Desktop\\New folder\\'.")
        st.stop()

lr_model, rf_model, xgb_model, scaler = load_models()

# Function to generate personalized intervention recommendations
def get_interventions(data, churn_prob):
    recommendations = []
    if data["Contract"] == "Month-to-month":
        recommendations.append("Offer a discount or loyalty reward to switch to a one-year or two-year contract.")
    if data["tenure"] < 12:
        recommendations.append("Provide personalized onboarding support or a free service upgrade for the first 6 months.")
    if data["InternetService"] == "Fiber optic":
        recommendations.append("Investigate potential service issues (e.g., reliability) and offer a service quality check.")
    if data["PaymentMethod"] == "Electronic check":
        recommendations.append("Encourage switching to automatic bank transfer or credit card payment with a one-time discount.")
    if data["MonthlyCharges"] > 70:
        recommendations.append("Offer a bundled service package or a temporary discount to reduce monthly charges.")
    if churn_prob > 0.7:
        recommendations.append("Escalate to a retention specialist for immediate follow-up with a tailored retention offer.")
    return recommendations if recommendations else ["No specific interventions needed. Monitor customer satisfaction."]

# Preprocessing function
def preprocess_input(data):
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Label encoding for binary features
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    # One-hot encoding for categorical features
    categorical_cols = {
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }
    
    for col, categories in categorical_cols.items():
        for category in categories[1:]:  # Skip first category (drop_first=True)
            df[f"{col}_{category}"] = (df[col] == category).astype(int)
        df.drop(col, axis=1, inplace=True)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    except Exception as e:
        st.error(f"Error scaling numerical features: {e}")
        st.stop()
    
    return df

# Main app
def main():
    st.title("📞 Customer Churn Prediction")
    st.markdown("Predict whether a customer will churn and receive personalized retention recommendations.")
    
    with st.expander("ℹ️ About this app", expanded=False):
        st.write("""
        This app predicts customer churn using three machine learning models:
        - Logistic Regression
        - Random Forest
        - XGBoost
        Enter customer details to see churn predictions, key factors, and recommended interventions.
        """)
    
    # Create form
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            customer_id = st.text_input("Customer ID", "1234-ABCDE")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
        with col2:
            st.subheader("Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Services")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            
        with col4:
            st.subheader("Billing")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            
        st.subheader("Financial Information")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges * tenure)
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Preprocess input
        try:
            processed_data = preprocess_input(input_data)
        except Exception as e:
            st.error(f"Error preprocessing input: {e}")
            st.stop()
        
        # Make predictions
        lr_pred = lr_model.predict(processed_data)[0]
        rf_pred = rf_model.predict(processed_data)[0]
        xgb_pred = xgb_model.predict(processed_data)[0]
        
        # Get prediction probabilities
        lr_prob = lr_model.predict_proba(processed_data)[0][1]
        rf_prob = rf_model.predict_proba(processed_data)[0][1]
        xgb_prob = xgb_model.predict_proba(processed_data)[0][1]
        
        # Display results
        st.success("Prediction completed!")
        st.subheader("Prediction Results")
        
        # Create columns for model results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Logistic Regression",
                value="Churn" if lr_pred == 1 else "No Churn",
                delta=f"{lr_prob*100:.1f}% probability"
            )
        
        with col2:
            st.metric(
                label="Random Forest",
                value="Churn" if rf_pred == 1 else "No Churn",
                delta=f"{rf_prob*100:.1f}% probability"
            )
        
        with col3:
            st.metric(
                label="XGBoost",
                value="Churn" if xgb_pred == 1 else "No Churn",
                delta=f"{xgb_prob*100:.1f}% probability"
            )
        
        # Show feature importance (using XGBoost for better accuracy)
        st.subheader("Key Factors Influencing Prediction")
        feature_importance = pd.Series(xgb_model.feature_importances_, index=processed_data.columns)
        top_features = feature_importance.sort_values(ascending=False).head(5)
        
        # Display as bar chart
        st.bar_chart(top_features)
        
        # Interpretation
        st.info("""
        **Interpretation Guide:**
        - Higher values indicate features that most influence the churn prediction.
        - Focus on these factors when planning retention strategies.
        """)
        
        # Personalized interventions
        st.subheader("Recommended Interventions")
        interventions = get_interventions(input_data, xgb_prob)
        for i, intervention in enumerate(interventions, 1):
            st.write(f"{i}. {intervention}")
        
        # Customer summary
        st.subheader("Customer Summary")
        st.write(f"**Customer ID**: {customer_id}")
        st.write(f"**Tenure**: {tenure} months")
        st.write(f"**Monthly Charges**: ${monthly_charges:.2f}")
        st.write(f"**Contract**: {contract}")
        st.write(f"**Services**: {internet_service}, {phone_service}, {streaming_tv}, {streaming_movies}")

if __name__ == '__main__':
    main()