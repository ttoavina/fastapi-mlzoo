from typing import Dict
from keras.models import Sequential
import pandas as pd
import numpy as np

from model.config import Config

def predict(config: Config, data: Dict):
    """
        return value ()
    """
    df = pd.Series(data=data)
    predicted_value = config.model.predict(np.array([df]))
    
    index = np.argmax(predicted_value)
    disease = config.idx_to_class[index]
    
    return disease, np.max(predicted_value)
    
    
    
if __name__ == "__main__":
    
    data = {
        "itching": 0,
        "skin_rash": 0,
        "nodal_skin_eruptions": 0,
        "continuous_sneezing": 0,
        "shivering": 0,
        "chills": 0,
        "joint_pain": 0,
        "stomach_pain": 0,
        "acidity": 0,
        "ulcers_on_tongue": 0,
        "muscle_wasting": 0,
        "vomiting": 0,
        "burning_micturition": 0,
        "spotting_ urination": 0,
        "fatigue": 0,
        "weight_gain": 0,
        "anxiety": 0,
        "mood_swings": 0,
        "weight_loss": 0,
        "restlessness": 0,
        "lethargy": 0,
        "irregular_sugar_level": 0,
        "cough": 0,
        "high_fever": 0,
        "sunken_eyes": 0,
        "breathlessness": 0,
        "sweating": 0,
        "indigestion": 0,
        "headache": 0,
        "yellowish_skin": 0,
        "dark_urine": 0,
        "nausea": 0,
        "loss_of_appetite": 0,
        "pain_behind_the_eyes": 0,
        "back_pain": 0,
        "constipation": 0,
        "abdominal_pain": 0,
        "diarrhoea": 0,
        "mild_fever": 0,
        "yellow_urine": 0,
        "yellowing_of_eyes": 0,
        "acute_liver_failure": 0,
        "swelling_of_stomach": 0,
        "swelled_lymph_nodes": 0,
        "malaise": 0,
        "blurred_and_distorted_vision": 0,
        "phlegm": 0,
        "throat_irritation": 0,
        "chest_pain": 0,
        "weakness_in_limbs": 0,
        "fast_heart_rate": 0,
        "pain_during_bowel_movements": 0,
        "neck_pain": 0,
        "dizziness": 0,
        "cramps": 0,
        "obesity": 0,
        "knee_pain": 0,
        "muscle_weakness": 0,
        "stiff_neck": 0,
        "swelling_joints": 0,
        "movement_stiffness": 0,
        "spinning_movements": 0,
        "loss_of_balance": 0,
        "weakness_of_one_body_side": 0,
        "bladder_discomfort": 0,
        "passage_of_gases": 0,
        "toxic_look_(typhos)": 0,
        "depression": 0,
        "irritability": 0,
        "muscle_pain": 0,
        "red_spots_over_body": 0,
        "family_history": 0,
        "mucoid_sputum": 0,
        "rusty_sputum": 0,
        "lack_of_concentration": 0,
        "visual_disturbances": 0,
        "blood_in_sputum": 0,
        "pus_filled_pimples": 0,
        "skin_peeling": 0,
        "blister": 0,
    }
    
    config = Config.load()
    print("++++++++++++++++")
    predict = predict(config, data)
    print(predict)
    