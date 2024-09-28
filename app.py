import streamlit as st
import datetime
from collect_user_input import collect_user_input
from cross_churn_pred import crose_churn_model

def main():
    
    st.header("Insurance Attributes Prediction System")
    st.write("Area of Focus: Cross Selling, Customer Churn, Loss Ratio")
    st.divider()
    
    cola, colb = st.columns(2)

    
    with cola:
        st.image('static/logos/VA_Logo.png', width=150)

    with colb:
        st.image('static/logos/NewLogoNoBgDraft5.png', width=150)
    tab1, tab2, tab3 = st.tabs(["Prediction", "Interpretation", "Automation"])

    with tab1:
        with st.container(border=True):
            st.subheader("Customer Data Input")
            
            with st.expander("Demographics Data"):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.radio("Gender",
                                    ["male 👦🏽", "female 👧🏻"])
                with col2:
                    age = st.number_input("Date of Birth", value=0)

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
            
        col17, col18, col19 = st.columns(3)
            
        with col17:
            if st.button("Predict Cross Selling"):
                user_data = collect_user_input(gender, age, driving_license, sub_county, previously_insured, vehicle_year_of_manufacture, 
                       vehicle_damage, annual_premium, agent_name, life_policy_start_date)
                cross_churn_predictions, evaluation = crose_churn_model(user_data)
                st.write(f"The is a  {(cross_churn_predictions.ravel()[1]*100).round(2)}% probabilty that the customer will be interested")
                st.write(f"Model Performance")
                st.write(evaluation)
        with col18:
            if st.button("Predict Churn"):
                st.write(age) 
        with col19:
            if st.button("Predict Loss Ratio"):
                st.write(age)       
                
        
    

if __name__ == "__main__":
    main()
    
