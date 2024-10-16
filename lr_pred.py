import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

lr_pred_model_path = "Utils/Cross_Sell_Prediction/best_loss_ratio_model.pkl"
lr_pred_model = joblib.load(lr_pred_model_path)

def lr_model(user_input_data):
    cols = ['Sum_Assured', 'Policy_Term', 'Dependents', 'BMI', 'Inflation_Rate (%)',
       'Age', 'Gender_Male', 'Type_of_Life_Insurance_Term',
       'Type_of_Life_Insurance_Whole Life', 'Rider_Info_Yes',
       'Premium_Payment_Frequency_Monthly',
       'Premium_Payment_Frequency_Quarterly', 'Occupation_Engineer',
       'Occupation_Entrepreneur', 'Occupation_Lawyer', 'Occupation_Teacher',
       'Education_Level_Postgraduate', 'Education_Level_Undergraduate',
       'Marital_Status_Married', 'Marital_Status_Single',
       'Smoker_Status_Smoker', 'Exercise_and_Lifestyle_Sedentary',
       'Medical_Conditions_Yes', 'Payment_History_Good',
       'Customer_Interaction_Frequency_Low',
       'Customer_Interaction_Frequency_Medium']
    
    X_data = pd.get_dummies(user_input_data)
    X_data = X_data.reindex(columns=cols, fill_value=0)
    
    predictions = lr_pred_model.predict_proba(X_data)
    model_evaluation = {"accuracy": 81.167,
                        "recall": 81.167,
                        "precision": 75.848,
                        "f1_score": 78.01
                                        }   
    
    return predictions, model_evaluation
    

