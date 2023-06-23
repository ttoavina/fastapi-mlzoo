from pprint import pprint
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense,Input
import numpy as np
import pandas as pd

from model.config import Config


def eda(filename: str, target:str, corr_threshold: float = 0.80) -> Tuple[pd.DataFrame,pd.DataFrame] :
    df = pd.read_csv(filename)
    df = df.drop("Unnamed: 133", axis=1)
    target_df = df[target]
    features = df.drop(target, axis=1)

    sum_data = [(col, sum(features[col])) for col in features.columns]
    sum_data.sort(key=lambda x: x[1])
    print(f"Removing {sum_data[0][0]}")
    features = features.drop(sum_data[0][0], axis=1)

    # Correlation analysis
    correlation = df.corr().abs()
    correlation = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))

    to_drop = [column for column in correlation.columns if any(correlation[column] > corr_threshold)]

    final_features = features.drop(to_drop, axis=1)

    return final_features, target_df

def make_model(input: int,output: int) -> Model :
    model = Sequential()
    model.add(Input(shape=(None, input)))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(output, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

    return model


if __name__ == "__main__":
    X , Y = eda("Training.csv","prognosis")
    features_order = X.columns
    
    target_order = list(set(Y))
    target_order.sort()
    class_to_idx = { classe : i for i, classe in enumerate(target_order)}
    idx_to_class = { i : classe for i, classe in enumerate(target_order)}

    Y = np.array(list(map(lambda x : class_to_idx[x],Y)))
    print(Y)

    model = make_model(input = X.shape[1], output=len(list(set(Y))))
    model.summary()
    history = model.fit(X, Y, epochs= 100)

    config = Config(
        class_to_idx = class_to_idx,
        idx_to_class = idx_to_class,
        target_key_order=target_order,
        features_key_order=features_order,
        model=model
    )

    config.save()
    print(config.features_key_order)