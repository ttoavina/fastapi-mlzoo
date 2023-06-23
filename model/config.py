from dataclasses import dataclass
from typing import Dict, List
from keras.models import Sequential
import pickle

@dataclass
class Config:
    class_to_idx : Dict[str, int]
    idx_to_class : Dict[int, str]
    model : Sequential
    features_key_order: List
    target_key_order: List

    def save(self):
        with open("config.dat","wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load():
        with open("config.dat", "rb") as f:
            config = pickle.load(f)

        return config
