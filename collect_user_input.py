import pandas as pd

def collect_user_input(gender, age, driving_license, sub_county, previously_insured, vehicle_year_of_manufacture, 
                       vehicle_damage, annual_premium, agent_name, life_policy_start_date):
    
    data = {
        "Gender": ["Male" if gender == "male 👦🏽" else "Female"],  # Convert gender to 1 for male, 0 for female
        "Driving_License": [1 if driving_license == "Yes" else 0],
        "Customer_Date_of_Birth": [age],  # Assuming age here, not actual DOB
        "Customer_Residence_Sub_County": [sub_county],
        "Previously_Insured": [1 if previously_insured == "Yes" else 0],  # Convert Yes/No to 1/0
        "Vehicle_Year_of_Manufacture": [vehicle_year_of_manufacture],
        "Vehicle_Damage": [vehicle_damage],  # Convert Yes/No to 1/0
        "Annual_Premium": [annual_premium],
        "Agent_name": [agent_name],
        "Life_policy_start_date": [life_policy_start_date]
    }
    
    df = pd.DataFrame(data)
    return df