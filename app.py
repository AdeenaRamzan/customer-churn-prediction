import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Centered Heading
st.markdown("<h1 style='text-align: center; font-size:3.5rem; color: #ffff;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.divider()
# Centered Sub-heading / Instruction
st.markdown("<h4 style='text-align: center; color: #ffff;'>Enter customer details to predict churn:</h4>", unsafe_allow_html=True)


# st.write("Enter customer details to predict churn:")
st.divider()


ContractType = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
Tenure = st.number_input("Enter Tenure (months)", min_value=0, max_value=130, value=10)
MonthlyCharges = st.number_input("Enter Monthly Charges", min_value=0.0, max_value=130.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=13000.0, value=1200.0)
InternetService_Fiber_Optic = st.selectbox("Internet Service Fiber Optic", ['No', 'Yes'])

ContractType_map = {'Month-to-month':0, 'One year':1, 'Two year':2}
TechSupport_map = {'No':0, 'Yes':1}
Gender_map = {'Male':0, 'Female':1}
InternetService_map = {'No':0, 'Yes':1}

ContractType = ContractType_map[ContractType]
TechSupport = TechSupport_map[TechSupport]
Gender = Gender_map[Gender]
InternetService_Fiber_Optic = InternetService_map[InternetService_Fiber_Optic]


if st.button("Predict Churn"):

    input_df = pd.DataFrame([{
        'ContractType': ContractType,
        'TechSupport': TechSupport,
        'Gender': Gender,
        'Tenure': Tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'InternetService_Fiber Optic': InternetService_Fiber_Optic
    }])

    input_df = input_df[columns]

    input_df_scaled = scaler.transform(input_df)

    pred = rf_model.predict(input_df_scaled)[0]
    prob = rf_model.predict_proba(input_df_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write("Churn" if pred==1 else "No Churn")
    st.write(f"Churn Probability: {prob:.2f}")
    if pred == 1:
        st.success("⚠️ Churn likely! Take action!")
    else:
        st.info("✅ Customer is safe!")
    st.write("Churn Probability:")
    st.progress(prob)  # prob should be 0-1, e.g., 0.77
    if pred == 1:
        st.snow()

