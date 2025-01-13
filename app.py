import streamlit as st
import datetime
from collect_user_input import collect_user_input
from cross_sell_pred import cross_sell_model
from lr_pred import lr_model
import numpy as np
import pandas as pd
import plotly.express as px

def main():
    
    st.header("Insurance Attributes Prediction System")
    st.write("Area of Focus: Cross Selling, Loss Ratio, Customer Churn")
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
            st.subheader("Cross Sell Customer Data Input")
            
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
            
            with st.expander("Operational Data"):
                agent_name = st.text_input("Agent Name")
                col5, col6 = st.columns(2)
                with col5:
                    previously_insured = st.radio("Previously Insured",
                                                ["Yes", "No"])
                with col6:
                    vehicle_year_of_manufacture = st.date_input("Vehicle Year of Manufacture", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
            
                
            
                col9, col10 = st.columns(2)
                with col9:
                    vehicle_damage = st.radio("Vehicle Damage",
                                            ["Yes", "No"])
                with col10:
                    life_policy_start_date = st.date_input("Life Policy Start Date", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
                # with col10:
                #     customer_tenure = st.number_input("Customer Tenure", min_value=0, max_value=20)
                
                # col11, col12 = st.columns(2)
                # with col11:
                #     last_interaction = st.date_input("Last Interaction", format='DD-MM-YYYY', min_value=datetime.date(1995,1,1), max_value=datetime.date(2024, 12, 12))
                # with col12:
                #     renewal_count = st.number_input("Renewal Count",min_value=0)
            # st.divider()
            
            with st.expander("Financial Data"):
                annual_premium = st.number_input("Annual Premium(Kes)", value=10000)
                # col13, col14 = st.columns(2)
                # with col13:
                #     total_claims = st.number_input("Total Claims", min_value=0)
                # with col14:
                #     total_claims_paid = st.number_input("Total Claims Paid", min_value=0)
                
                # col15, col16 = st.columns(2)
                # with col15:
                #     premium_increase = st.number_input("Premium Inrease(%)", min_value=0)
            # st.divider()
            
        col17, col18 = st.columns(2)
            
        with col17:
            if st.button("Predict Cross Selling"):
                user_data = collect_user_input(gender, age, driving_license, sub_county, previously_insured, vehicle_year_of_manufacture, 
                       vehicle_damage, annual_premium, agent_name, life_policy_start_date)
                st.session_state.pred_output, st.session_state.cross_sell_predictions, st.session_state.evaluation = cross_sell_model(user_data)
            if st.session_state.cross_sell_predictions is not None:
                if st.session_state.pred_output == 0:
                    st.write("The customer will not be interested")
                elif st.session_state.pred_output == 1:
                    st.write("The customer will be interested.")
                st.write(f"(Probability of interest: {(st.session_state.cross_sell_predictions.ravel()[1]*100).round(2)}%)")
            performance = st.checkbox(f"Show Model Performance")
            if performance:  
                st.write(st.session_state.evaluation)
        with col18:
            if st.button("Display Cross Sell Model Feature Importance"):
                st.image('Utils/Cross_Sell_Prediction/cross_sell_feature_importance.png', )
        
        #Batch Processing
        st.subheader("Batch Processing")
        with st.expander("Batch Prediction"):
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

            if uploaded_file:
                # Load the dataset
                
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower()

                # Display the uploaded dataset
                uploaded_data = st.checkbox(f"Show Uploaded Data")
                if uploaded_data:  
                    st.write("Uploaded Dataset")
                    st.dataframe(data)

                # Ensure the necessary columns are present
                required_columns = [
                    "gender", "age", "driving_license", "sub_county", 
                    "previously_insured", "vehicle_year_of_manufacture",
                    "vehicle_damage", "annual_premium", "agent_name", "life_policy_start_date"
                ]
                
                if all(col in data.columns for col in required_columns):
                    if st.button("Predict for Uploaded Data"):
                        # Collect user data for batch prediction
                        predictions = []
                        probabilities = []

                        for _, row in data.iterrows():
                            user_data = collect_user_input(
                                row["gender"], row["age"], row["driving_license"], row["sub_county"],
                                row["previously_insured"], row["vehicle_year_of_manufacture"],
                                row["vehicle_damage"], row["annual_premium"], row["agent_name"], 
                                row["life_policy_start_date"]
                            )
                            pred_output, pred_prob, st.session_state.evaluation = cross_sell_model(user_data)
                            if pred_output == 0:
                                pred_text = "Not Interested"
                            elif pred_output == 1:
                                pred_text = "Interested"
                            predictions.append(pred_text)
                            probabilities.append(pred_prob.ravel()[1] * 100)

                        # Append predictions and probabilities to the dataset
                        data["prediction"] = predictions
                        data["probability(%)"] = [round(prob, 2) for prob in probabilities]

                        # Display the updated dataset
                        st.write("Predictions:")
                        st.dataframe(data[['id','prediction', 'probability(%)']])
                        
                        # Visualize predictions                         
                        if "prediction" in data.columns:
                            # Calculate the counts for each label
                            prediction_counts = data["prediction"].value_counts()
                            labels = prediction_counts.index
                            values = prediction_counts.values

                            # Create a pie chart using Plotly
                            fig = px.pie(
                                names=labels,
                                values=values,
                                title="Prediction Summary",
                                color_discrete_sequence=["#EF553B", "#636EFA"],  
                            )
                            fig.update_layout(
                                height=350,  
                                width=350,   
                                title=dict(font=dict(size=16))
)

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please upload data and generate predictions first.")
                else:
                    st.warning(f"The uploaded dataset must contain the following columns: {', '.join(required_columns)}")

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

                col9, col10 = st.columns(2)
                with col9:
                    education_level = st.selectbox("Education Level", 
                                                ["High School", "Undergraduate", "Postgraduate"])
                with col10:
                    marital_status = st.selectbox("Marital Status", 
                                                ["Single", "Married", "Divorced"])
                
                cold, cole = st.columns(2)
                with cold:
                    occupation = st.text_input("Occupation")
                with cole:
                    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
                
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


            with st.expander("Operational Data"):
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

                premium_payment_frequency = st.selectbox("Premium Payment Frequency", 
                                                            ["Monthly", "Quarterly", "Annually"])
    

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

            with st.expander("Financial Data"):
                inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, value=0.0)
                # loss_ratio = st.number_input("Loss Ratio (%)", min_value=0.0, value=0.0)

            col19, col20 = st.columns(2)
            with col19:
                if st.button("Predict Loss Ratio"):
                    # Collect all input data into a dictionary or dataframe for model input
                    loss_ratio_data = {
                                "Gender": [gender], 
                                "Type_of_Life_Insurance": [type_of_life_insurance],  
                                "Sum_Assured": [sum_assured],
                                "Policy_Term": [policy_term],
                                "Rider_Info": [rider_info],  
                                "Premium_Payment_Frequency": [premium_payment_frequency],  
                                "Occupation": [occupation],  
                                "Education_Level": [education_level],  
                                "Marital_Status": [marital_status],  
                                "Dependents": [dependents],
                                "Medical_Conditions": [medical_conditions],  
                                "Smoker_Status": [smoker_status],  
                                "BMI": [bmi],
                                "Exercise_and_Lifestyle": [exercise_and_lifestyle],  
                                "Payment_History": [payment_history],  
                                "Customer_Interaction_Frequency": [customer_interaction_frequency],  
                                "Inflation_Rate (%)": [inflation_rate],
                                # "Total_Claims_Paid": [total_claims_paid],
                                # "Earned_Premium": [earned_premium],
                                # "Loss_Ratio (%)": [loss_ratio],
                                "Age": [age]}
                    lr_m_data = pd.DataFrame(loss_ratio_data)
                    st.session_state.lr_predictions, st.session_state.lr_pred_proba, st.session_state.lr_evaluation = lr_model(lr_m_data)
                labels = ["High", "Low", "Medium"]
                if st.session_state.lr_predictions is not None:
                    st.write(f'The Loss Ratio will be {labels[int(st.session_state.lr_predictions)]}.')
                    st.write(f'(Probability: {round(st.session_state.lr_pred_proba[0][int(st.session_state.lr_predictions)]*100,2)}%)')
                lr_performance = st.checkbox(f"Show Loss Ratio Model Performance")
                if lr_performance:  
                    st.write(st.session_state.lr_evaluation)
            with col20:
                if st.button("Display Loss Ratio Model Feature Importance"):
                    st.image('Utils/Loss_ratio_prediction/loss_ratio_feature_importance.png', )
            # Add a new section for batch processing
            st.subheader("Batch Processing")
            
            with st.expander("Loss Ratio Prediciton"):
                uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

                if uploaded_file:
                    # Load the dataset
                    data_lr = pd.read_csv(uploaded_file)
                    
                    # Display the uploaded dataset
                    uploaded_data_lr = st.checkbox(f"Show Uploaded Data")
                    if uploaded_data_lr:  
                        st.write("Uploaded Dataset")
                        st.dataframe(data_lr)

                    # Validate required columns in the dataset
                    required_columns_lr = [
                        "Gender", "Type_of_Life_Insurance", "Sum_Assured", "Policy_Term", "Rider_Info",
                        "Premium_Payment_Frequency", "Occupation", "Education_Level", "Marital_Status",
                        "Dependents", "Medical_Conditions", "Smoker_Status", "BMI", "Exercise_and_Lifestyle",
                        "Payment_History", "Customer_Interaction_Frequency", "Inflation_Rate (%)", "Age"
                    ]

                    if all(col in data_lr.columns for col in required_columns):
                        if st.button("Predict for Uploaded Data"):
                            # Make batch predictions
                            predictions = []
                            probabilities = []

                            for _, row in data_lr.iterrows():
                                loss_ratio_data = {
                                    "Gender": [row["Gender"]],
                                    "Type_of_Life_Insurance": [row["Type_of_Life_Insurance"]],
                                    "Sum_Assured": [row["Sum_Assured"]],
                                    "Policy_Term": [row["Policy_Term"]],
                                    "Rider_Info": [row["Rider_Info"]],
                                    "Premium_Payment_Frequency": [row["Premium_Payment_Frequency"]],
                                    "Occupation": [row["Occupation"]],
                                    "Education_Level": [row["Education_Level"]],
                                    "Marital_Status": [row["Marital_Status"]],
                                    "Dependents": [row["Dependents"]],
                                    "Medical_Conditions": [row["Medical_Conditions"]],
                                    "Smoker_Status": [row["Smoker_Status"]],
                                    "BMI": [row["BMI"]],
                                    "Exercise_and_Lifestyle": [row["Exercise_and_Lifestyle"]],
                                    "Payment_History": [row["Payment_History"]],
                                    "Customer_Interaction_Frequency": [row["Customer_Interaction_Frequency"]],
                                    "Inflation_Rate (%)": [row["Inflation_Rate (%)"]],
                                    "Age": [row["Age"]]
                                }

                                # Convert to DataFrame for model input
                                lr_m_data = pd.DataFrame(loss_ratio_data)

                                # Get predictions and probabilities
                                pred, proba, _ = lr_model(lr_m_data)
                                predictions.append(pred[0])
                                probabilities.append(proba[0][int(pred[0])] * 100)

                            # Append predictions and probabilities to the dataset
                            data_lr["Prediction"] = [labels[int(pred)] for pred in predictions]
                            data_lr["Probability (%)"] = [round(prob, 2) for prob in probabilities]

                            # Display the updated dataset with predictions
                            st.write("Predictions")
                            st.dataframe(data_lr)

                            # Optionally, allow users to download the results
                            csv_lr = data_lr.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv_lr,
                                file_name="loss_ratio_predictions.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning(f"The uploaded dataset must contain the following columns: {', '.join(required_columns_lr)}")
        with tab3:
            st.subheader("Coming Soon!💡 Development in progress..")      
    cola, colb = st.columns(2)
    st.markdown("***This is a Proof of Concept Machine Learning Software Developed by Virtual Analytics and Advernet Africa***")
    st.markdown("© 2024 [Virtual Analytics](https://virtualanalytics.co.ke) & [Advernet Africa](https://www.advernet.africa). All rights reserved.")     
    with cola:
        st.image('static/logos/VA_Logo.png', width=150)

    with colb:
        st.image('static/logos/NewLogoNoBgDraft5.png', width=150)
        
    

if __name__ == "__main__":
    main()
    
