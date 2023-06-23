from typing import Dict
from keras.models import Sequential
import pandas as pd
import numpy as np


def predict(config, data: Dict):
    """
        return value ()
    """
    df = pd.Series(data=data)
    predicted_value = config.model.predict(np.array([df]))
    
    index = np.argmax(predicted_value)
    disease = config.idx_to_class[index]
    
    return disease, float(np.max(predicted_value))