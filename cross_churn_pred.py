import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

churn_pred_model_path = "Utils/Cross_Sell_Prediction/best_cross_sell_model.pkl"
churn_pred_model = joblib.load(churn_pred_model_path)

def crose_churn_model(user_input_data):
    cols = ['Customer_Date_of_Birth',
 'Driving_License',
 'Previously_Insured',
 'Vehicle_Year_of_Manufacture',
 'Annual_Premium',
 'Life_policy_start_date',
 'Gender_Male',
 'Customer_Residence_Sub_County_BONDO',
 'Customer_Residence_Sub_County_BUMULA',
 'Customer_Residence_Sub_County_BUNGOMA SOUTH',
 'Customer_Residence_Sub_County_BURETI',
 'Customer_Residence_Sub_County_DAGORETTI',
 'Customer_Residence_Sub_County_EMBAKASI',
 'Customer_Residence_Sub_County_IGEMBE CENTRAL',
 'Customer_Residence_Sub_County_IMENTI SOUTH',
 'Customer_Residence_Sub_County_ISINYA',
 'Customer_Residence_Sub_County_JUJA',
 'Customer_Residence_Sub_County_KABETE',
 'Customer_Residence_Sub_County_KAJIADO NORTH',
 'Customer_Residence_Sub_County_KAKAMEGA NORTH',
 'Customer_Residence_Sub_County_KAMUKUNJI',
 'Customer_Residence_Sub_County_KAPSERET',
 'Customer_Residence_Sub_County_KASARANI',
 'Customer_Residence_Sub_County_KIAMBAA',
 'Customer_Residence_Sub_County_KILIFI SOUTH',
 'Customer_Residence_Sub_County_KIMININI',
 'Customer_Residence_Sub_County_KISAUNI',
 'Customer_Residence_Sub_County_KISUMU EAST',
 'Customer_Residence_Sub_County_KURIA WEST',
 'Customer_Residence_Sub_County_KWANZA',
 "Customer_Residence_Sub_County_LANG'ATA",
 'Customer_Residence_Sub_County_LIKONI',
 'Customer_Residence_Sub_County_LUNGA LUNGA',
 'Customer_Residence_Sub_County_MALINDI',
 'Customer_Residence_Sub_County_MATHARE',
 'Customer_Residence_Sub_County_NAIVASHA',
 'Customer_Residence_Sub_County_NAKURU NORTH',
 'Customer_Residence_Sub_County_NAKURU WEST',
 'Customer_Residence_Sub_County_NAROK NORTH',
 'Customer_Residence_Sub_County_NAROK SOUTH',
 'Customer_Residence_Sub_County_NDHIWA',
 'Customer_Residence_Sub_County_NJIRU',
 'Customer_Residence_Sub_County_NJORO',
 'Customer_Residence_Sub_County_NYALI',
 'Customer_Residence_Sub_County_RONGAI',
 'Customer_Residence_Sub_County_RUIRU',
 'Customer_Residence_Sub_County_SAMBURU',
 'Customer_Residence_Sub_County_SIAYA',
 'Customer_Residence_Sub_County_SOTIK',
 'Customer_Residence_Sub_County_SOY',
 'Customer_Residence_Sub_County_STAREHE',
 'Customer_Residence_Sub_County_THARAKA-NITHI',
 'Customer_Residence_Sub_County_THIKA WEST',
 'Customer_Residence_Sub_County_TRANS MARA WEST',
 'Customer_Residence_Sub_County_TRANS NZOIA EAST',
 'Customer_Residence_Sub_County_TRANS NZOIA WEST',
 'Customer_Residence_Sub_County_TURBO',
 'Customer_Residence_Sub_County_TURKANA WEST',
 'Customer_Residence_Sub_County_WESTLANDS',
 'Vehicle_Damage_Yes',
 'Agent_name_Agent_Mugo',
 'Agent_name_Agent_Mwai',
 'Agent_name_Agent_Njau',
 'Agent_name_Agent_Ogot',
 'Agent_name_Agent_Ruto',
 'Agent_name_Agent_alwa',
 'Agent_name_Agent_amau',
 'Agent_name_Agent_anga',
 'Agent_name_Agent_angi',
 'Agent_name_Agent_ango',
 'Agent_name_Agent_anja',
 'Agent_name_Agent_arie',
 'Agent_name_Agent_atha',
 'Agent_name_Agent_athi',
 'Agent_name_Agent_atta',
 'Agent_name_Agent_biru',
 'Agent_name_Agent_bugu',
 'Agent_name_Agent_dera',
 'Agent_name_Agent_duta',
 'Agent_name_Agent_egwa',
 'Agent_name_Agent_enda',
 'Agent_name_Agent_eter',
 'Agent_name_Agent_guna',
 'Agent_name_Agent_guru',
 'Agent_name_Agent_haga',
 'Agent_name_Agent_hama',
 'Agent_name_Agent_hege',
 'Agent_name_Agent_hira',
 'Agent_name_Agent_hoki',
 'Agent_name_Agent_honi',
 'Agent_name_Agent_huri',
 'Agent_name_Agent_ibet',
 'Agent_name_Agent_ieng',
 'Agent_name_Agent_ieno',
 'Agent_name_Agent_iiri',
 'Agent_name_Agent_ioka',
 'Agent_name_Agent_itau',
 'Agent_name_Agent_ithi',
 'Agent_name_Agent_iuki',
 'Agent_name_Agent_jagi',
 'Agent_name_Agent_jala',
 'Agent_name_Agent_jeru',
 'Agent_name_Agent_jiru',
 'Agent_name_Agent_johi',
 'Agent_name_Agent_jugi',
 'Agent_name_Agent_kori',
 'Agent_name_Agent_nene',
 "Agent_name_Agent_ng'a",
 "Agent_name_Agent_ng'o",
 "Agent_name_Agent_ng'u",
 'Agent_name_Agent_ngau',
 'Agent_name_Agent_njui',
 'Agent_name_Agent_noti',
 'Agent_name_Agent_nyua',
 'Agent_name_Agent_onyo',
 'Agent_name_Agent_oria',
 'Agent_name_Agent_ptoo',
 'Agent_name_Agent_reri',
 'Agent_name_Agent_riga',
 'Agent_name_Agent_rima',
 'Agent_name_Agent_rimu',
 'Agent_name_Agent_ritu',
 'Agent_name_Agent_rubo',
 'Agent_name_Agent_tich',
 'Agent_name_Agent_timu',
 'Agent_name_Agent_tuku',
 'Agent_name_Agent_umbi',
 'Agent_name_Agent_ungu',
 'Agent_name_Agent_utai',
 'Agent_name_Agent_wiri',
 'Agent_name_Agent_yaga',
 'Agent_name_Agent_yoro']
    X_data = pd.get_dummies(user_input_data)
    X_data = X_data.reindex(columns=cols, fill_value=0)
    
    predictions = churn_pred_model.predict_proba(X_data)
    model_evaluation = {"accuracy": 70.966,
                        "recall": 90.795,
                        "precision": 28.5526,
                        "f1_score": 43.4434
                                        }   
    
    return predictions, model_evaluation
    

