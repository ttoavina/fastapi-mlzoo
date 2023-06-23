from model.config import Config
import pandas as pd
from typing import List,Tuple
import numpy as np
from sklearn.metrics import f1_score

def process_test_data(filename: str, feature_key_order:List) -> Tuple[pd.DataFrame,pd.DataFrame] :
    df = pd.read_csv(filename)
    print(df.head())
    target_df = df["prognosis"]
    features = df[feature_key_order]
    return features, target_df

if __name__=="__main__":
    config = Config.load()
    X, Y = process_test_data("Testing.csv", config.features_key_order)
    predicted = config.model.predict(X)
    Y = np.array(list(map(lambda x : config.class_to_idx[x],Y)))
    # predicted = np.argmax(predicted, axis=1)
    
    array_confidence = []
    for i in range(len(predicted)):
        print(f"Real: {Y[i]} , Predicted: {np.argmax(predicted[i])} , confidence: {np.max(predicted[i])} ")
    
    
    
    