import streamlit as st
import datetime
from collect_user_input import collect_user_input
from cross_sell_pred import cross_sell_model
from lr_pred import lr_model
import numpy as np
import pandas as pd

def main():
    
    st.header("Insurance Attributes Prediction System")
    st.write("Area of Focus: Cross Selling, Customer Churn, Loss Ratio")
    st.divider()

    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    
    if 'cross_sell_predictions' not in st.session_state:
        st.session_state.cross_sell_predictions = None
    
    if 'lr_evaluation' not in st.session_state:
        st.session_state.lr_evaluation = None
    
    if 'lr_predictions' not in st.session_state:
        st.session_state.lr_predictions = None
    
    if 'lr_pred_proba' not in st.session_state:
        st.session_state.lr_pred_proba = None
        
    tab1, tab2, tab3 = st.tabs(["Cross Sell Prediction", "Loss Ratio Prediction", "Churn Predction"])

    with tab1:
        with st.container(border=True):
            st.subheader("Customer Data Input")
            
            with st.expander("Demographics Data"):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.radio("Gender",
                                    ["male 👦🏽", "female 👧🏻"])
                with col2:
                    age = st.number_input("Age", value=0)

                col3, col4 = st.columns(2)
                with col3:
                    county = st.text_input("County")
                with col4:
                    sub_county = st.text_input("Sub-county")
                driving_license = st.radio("Driving License",
                                    ["Yes", "No"])
            # st.divider()
            
            with st.expander("Insurance Historical Data"):
                agent_name = st.text_input("Agent Name")
                col5, col6 = st.columns(2)
                with col5:
                    previously_insured = st.radio("Previously Insured",
                                                ["Yes", "No"])
                with col6:
                    vehicle_year_of_manufacture = st.date_input("Vehicle Year of Manufacture", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
            
                col7, col8 = st.columns(2)
                with col7:
                    annual_premium = st.number_input("Annual Premium(Kes)", min_value=0.0)
                
                with col8:
                    vehicle_damage = st.radio("Vehicle Damage",
                                            ["Yes", "No"])
            
                col9, col10 = st.columns(2)
                with col9:
                    life_policy_start_date = st.date_input("Life Policy Start Date", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
                with col10:
                    customer_tenure = st.number_input("Customer Tenure", min_value=0, max_value=20)
                
                col11, col12 = st.columns(2)
                with col11:
                    last_interaction = st.date_input("Last Interaction", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
                with col12:
                    renewal_count = st.number_input("Renewal Count",min_value=0)
            # st.divider()
            
            with st.expander("Claims Historical Data"):
                col13, col14 = st.columns(2)
                with col13:
                    total_claims = st.number_input("Total Claims", min_value=0)
                with col14:
                    total_claims_paid = st.number_input("Total Claims Paid", min_value=0)
                
                col15, col16 = st.columns(2)
                with col15:
                    premium_increase = st.number_input("Premium Inrease(%)", min_value=0)
            # st.divider()
            
        col17, col18 = st.columns(2)
            
        with col17:
            if st.button("Predict Cross Selling"):
                user_data = collect_user_input(gender, age, driving_license, sub_county, previously_insured, vehicle_year_of_manufacture, 
                       vehicle_damage, annual_premium, agent_name, life_policy_start_date)
                st.session_state.cross_sell_predictions, st.session_state.evaluation = cross_sell_model(user_data)
            if st.session_state.cross_sell_predictions is not None:
                st.write(f"There is a  {(st.session_state.cross_sell_predictions.ravel()[1]*100).round(2)}% probabilty that the customer will be interested")
            performance = st.checkbox(f"Show Model Performance")
            if performance:  
                st.write(st.session_state.evaluation)
        with col18:
            if st.button("Cross Sell Model Feature Importance"):
                st.image('Utils/Cross_Sell_Prediction/cross_sell_feature_importance.png', )

    with tab2:
        with st.container(border=True):
            st.subheader("Customer Data Input for Loss Ratio Prediction")

            # Collecting user input for the features in the dataset
            with st.expander("Demographics Data"):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.radio("Gender", ["Male", "Female"])
                with col2:
                    age = st.number_input("Age", min_value=0, max_value=100, value=30)

            with st.expander("Insurance Data"):
                col3, col4 = st.columns(2)
                with col3:
                    type_of_life_insurance = st.selectbox("Type of Life Insurance", 
                                                        ["Term", "Whole Life", "Endowment"])
                with col4:
                    sum_assured = st.number_input("Sum Assured", min_value=0.0, value=1000000.0)
                
                col5, col6 = st.columns(2)
                with col5:
                    policy_term = st.number_input("Policy Term (Years)", min_value=0, value=10)
                with col6:
                    rider_info = st.radio("Rider Info", ["Yes", "No"])

            with st.expander("Customer Data"):
                col7, col8 = st.columns(2)
                with col7:
                    premium_payment_frequency = st.selectbox("Premium Payment Frequency", 
                                                            ["Monthly", "Quarterly", "Annually"])
                with col8:
                    occupation = st.text_input("Occupation")

                col9, col10 = st.columns(2)
                with col9:
                    education_level = st.selectbox("Education Level", 
                                                ["High School", "Undergraduate", "Postgraduate"])
                with col10:
                    marital_status = st.selectbox("Marital Status", 
                                                ["Single", "Married", "Divorced"])

                dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

            with st.expander("Health and Lifestyle Data"):
                col11, col12 = st.columns(2)
                with col11:
                    medical_conditions = st.radio("Medical Conditions", ["Yes", "No"])
                with col12:
                    smoker_status = st.radio("Smoker Status", ["Smoker", "Non-Smoker"])

                col13, col14 = st.columns(2)
                with col13:
                    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
                with col14:
                    exercise_and_lifestyle = st.radio("Exercise and Lifestyle", ["Active", "Sedentary"])

            with st.expander("Insurance Interaction Data"):
                col15, col16 = st.columns(2)
                with col15:
                    customer_interaction_frequency = st.selectbox("Customer Interaction Frequency", 
                                                                ["Low", "Medium", "High"])
                with col16:
                    payment_history = st.radio("Payment History", ["Good", "Average", "Poor"])

                # col17, col18 = st.columns(2)
                # with col17:
                #     earned_premium = st.number_input("Earned Premium (Kes)", min_value=0.0, value=0.0)
                # with col18:
                #     total_claims_paid = st.number_input("Total Claims Paid (Kes)", min_value=0.0, value=0.0)

            with st.expander("Economic Factors"):
                inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, value=0.0)
                # loss_ratio = st.number_input("Loss Ratio (%)", min_value=0.0, value=0.0)

            col19, col20 = st.columns(2)
            with col19:
                if st.button("Predict Loss Ratio"):
                    # Collect all input data into a dictionary or dataframe for model input
                    loss_ratio_data = {
                                "Gender": [gender],  # Convert Male to 1, Female to 0
                                "Type_of_Life_Insurance": [type_of_life_insurance],  # Encode insurance type
                                "Sum_Assured": [sum_assured],
                                "Policy_Term": [policy_term],
                                "Rider_Info": [rider_info],  # Convert Yes/No to 1/0
                                "Premium_Payment_Frequency": [premium_payment_frequency],  # Encode payment frequency
                                "Occupation": [occupation],  # Keep as string, may require further encoding
                                "Education_Level": [education_level],  # Encode education level
                                "Marital_Status": [marital_status],  # Encode marital status
                                "Dependents": [dependents],
                                "Medical_Conditions": [medical_conditions],  # Convert Yes/No to 1/0
                                "Smoker_Status": [smoker_status],  # Convert Smoker/Non-Smoker to 1/0
                                "BMI": [bmi],
                                "Exercise_and_Lifestyle": [exercise_and_lifestyle],  # Active=1, Sedentary=0
                                "Payment_History": [payment_history],  # Encode payment history
                                "Customer_Interaction_Frequency": [customer_interaction_frequency],  # Encode interaction frequency
                                "Inflation_Rate (%)": [inflation_rate],
                                # "Total_Claims_Paid": [total_claims_paid],
                                # "Earned_Premium": [earned_premium],
                                # "Loss_Ratio (%)": [loss_ratio],
                                "Age": [age]}
                    lr_m_data = pd.DataFrame(loss_ratio_data)
                    st.session_state.lr_predictions, st.session_state.lr_pred_proba, st.session_state.lr_evaluation = lr_model(lr_m_data)
                labels = ["High", "Low", "Medium"]
                if st.session_state.lr_predictions is not None:
                    st.write(f'There is a {round(st.session_state.lr_pred_proba[0][int(st.session_state.lr_predictions)]*100,2)}% probability that the Loss Ratio will be {labels[int(st.session_state.lr_predictions)]}')
                lr_performance = st.checkbox(f"Show Loss Ratio Model Performance")
                if lr_performance:  
                    st.write(st.session_state.lr_evaluation)
            with col20:
                if st.button("Loss Ratio Model Feature Importance"):
                    st.image('Utils/Loss_ratio_prediction/loss_ratio_feature_importance.png', )
                
    cola, colb = st.columns(2)
    st.markdown("***This is a Proof of Concept Machine Learning Software Developed by Virtual Analytics and Advernet Africa***")
    st.markdown("© 2024 [Virtual Analytics](https://virtualanalytics.co.ke) & [Advernet Africa](https://www.advernet.africa). All rights reserved.")     
    with cola:
        st.image('static/logos/VA_Logo.png', width=150)

    with colb:
        st.image('static/logos/NewLogoNoBgDraft5.png', width=150)
        
    

if __name__ == "__main__":
    main()
    
