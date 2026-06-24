import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Credit Risk Assessment System", layout="centered")

st.title("🛡️ Credit Risk Assessment & Loan Approval System")
st.markdown("### Final Year Project Stacking Ensemble Framework")


@st.cache_resource
def load_pipeline():
    with open('stacked_ensemble_risk_pipeline.pkl', 'rb') as f:
        return pickle.load(f)


artifacts = load_pipeline()

# ----------------------------------------------------
# NEW FEATURE: DETECT AND LOAD TEST DATA SAMPLES
# ----------------------------------------------------
test_data_available = os.path.exists('app_test_samples.csv')
if test_data_available:
    test_samples_df = pd.read_csv('app_test_samples.csv', index_col=0)
    st.sidebar.markdown("## 🛠️ Data Input Mode")
    mode = st.sidebar.radio("Select how to input data:",
                            ["Use Real Test Set Sample", "Manual Sliders Input"])
else:
    mode = "Manual Sliders Input"

# Set default values based on chosen mode
if mode == "Use Real Test Set Sample":
    st.sidebar.markdown("### 🗂️ Select Test Record")
    selected_row_idx = st.sidebar.selectbox("Choose Row Number from Holdout Split:",
                                            test_samples_df.index.tolist())

    # Extract the true specific row's values
    row_data = test_samples_df.loc[selected_row_idx]

    # Display the ground truth label right in the app for verification
    st.info(
        f"💡 **Ground Truth:** According to the original dataset, this specific applicant was historically labeled as: **{row_data['True_Approved_Flag']}**")

    # Pre-populate base parameters automatically
    default_score = int(row_data['Credit_Score']) if 'Credit_Score' in row_data else 650
    default_age = int(row_data['AGE']) if 'AGE' in row_data else 35
    default_income = float(
        row_data['NETMONTHLYINCOME']) if 'NETMONTHLYINCOME' in row_data else 5000.0
    default_enq = int(
        row_data['time_since_recent_enq']) if 'time_since_recent_enq' in row_data else 6
    default_empr = int(row_data['Time_With_Curr_Empr']) if 'Time_With_Curr_Empr' in row_data else 24
else:
    default_score, default_age, default_income, default_enq, default_empr = 650, 35, 5000.0, 6, 24

# ----------------------------------------------------
# SIDEBAR UI RENDERING
# ----------------------------------------------------
st.sidebar.markdown("### 📋 Applicant Profile Metrics")
credit_score = st.sidebar.slider("Credit Score Assessment", 300, 900, default_score)
age = st.sidebar.slider("Applicant Age (Years)", 18, 80, default_age)
net_income = st.sidebar.number_input("Net Monthly Income ($)", min_value=0.0, value=default_income)
recent_enq_time = st.sidebar.slider("Months Since Recent Enquiry", 0, 36, default_enq)
time_with_employer = st.sidebar.slider("Time with Current Employer (Months)", 0, 240, default_empr)

# ----------------------------------------------------
# PREDICTION ENGINE PIPELINE
# ----------------------------------------------------
if st.button("🚀 Execute Risk Diagnosis Evaluation"):
    scaler_features = artifacts['scaler'].feature_names_in_
    scaler_dummy_dict = {col: 0.0 for col in scaler_features}

    # Update active inputs using UI states
    if 'Credit_Score' in scaler_dummy_dict: scaler_dummy_dict['Credit_Score'] = credit_score
    if 'AGE' in scaler_dummy_dict: scaler_dummy_dict['AGE'] = age
    if 'NETMONTHLYINCOME' in scaler_dummy_dict: scaler_dummy_dict['NETMONTHLYINCOME'] = net_income
    if 'time_since_recent_enq' in scaler_dummy_dict: scaler_dummy_dict[
        'time_since_recent_enq'] = recent_enq_time
    if 'Time_With_Curr_Empr' in scaler_dummy_dict: scaler_dummy_dict[
        'Time_With_Curr_Empr'] = time_with_employer

    # If using test set, fill back all other exact historical columns too for precision!
    if mode == "Use Real Test Set Sample":
        for col in scaler_features:
            if col in row_data and col not in ['Credit_Score', 'AGE', 'NETMONTHLYINCOME',
                                               'time_since_recent_enq', 'Time_With_Curr_Empr']:
                scaler_dummy_dict[col] = row_data[col]

    scaler_df = pd.DataFrame([scaler_dummy_dict])
    scaled_array = artifacts['scaler'].transform(scaler_df)
    scaled_full_df = pd.DataFrame(scaled_array, columns=scaler_features)

    input_data = {}
    for col in artifacts['selected_features']:
        if col in scaled_full_df.columns:
            input_data[col] = scaled_full_df.loc[0, col]
        else:
            input_data[col] = 0.0

    input_df = pd.DataFrame([input_data])

    # Multi-class stacking predictions
    prob_lr = artifacts['base_model_lr'].predict_proba(input_df)
    prob_rf = artifacts['base_model_rf'].predict_proba(input_df)
    prob_lgb = artifacts['base_model_lgb'].predict_proba(input_df)

    meta_input = np.hstack([prob_lr, prob_rf, prob_lgb])
    final_probabilities = artifacts['meta_learner'].predict_proba(meta_input)[0]
    final_class = np.argmax(final_probabilities)

    priority_classes = {
        0: 'P1 (Highest Priority / Safe Candidate)',
        1: 'P2 (Standard Priority / Modest Risk)',
        2: 'P3 (Elevated Monitoring Required / High Risk)',
        3: 'P4 (Immediate Rejection / Default Pattern Warning)'
    }

    st.markdown("---")
    st.subheader("📊 System Diagnostic Assessment Result")

    if final_class in [0, 1]:
        st.success(f"**RECOMMENDED ALLOCATION:** {priority_classes[final_class]}")
    else:
        st.error(f"**RECOMMENDED ALLOCATION:** {priority_classes[final_class]}")

    prob_metrics_df = pd.DataFrame({
        'Risk Evaluation Tier': ['P1', 'P2', 'P3', 'P4'],
        'Ensemble Model Probability Confidence': final_probabilities
    })
    st.bar_chart(prob_metrics_df.set_index('Risk Evaluation Tier'))