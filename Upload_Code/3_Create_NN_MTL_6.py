# ！/Programmierung/Anaconda/envs python
# @File    : 3_Create_NN
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 3_Create_NN_MTL_6
# @Software: PyCharm


import json
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping


output_parameters = [
    "number_particles",
    "power_dissipation",
    "power_impact",
    "power_shear",
    "velocity_absolute_translational",
    "velocity_impact_normal_mean"
]

def load_data(file_paths):
    features_list = []
    targets_list = [[] for _ in range(6)]

    for i, file_path in enumerate(file_paths):
        param_index = i % 6
        param_name = output_parameters[param_index]

        with open(file_path, 'r') as file:
            json_data = json.load(file)

        if isinstance(json_data, list) and len(json_data) > 0:
            df = pd.json_normalize(json_data, 'data', sep='_')
        else:
            raise ValueError("Invalid JSON format. Expected a 'data' key with list of dictionaries")

        feature_cols = [
            "static friction p-w",
            "static friction p-p",
            "rolling resistance",
            "E Modul",
            "restitution coefficient p-w",
            "restitution coefficient p-p",
            "viscosity",
            "coupling",
            "rotational speed",
            "value of flow",
            "diameter",
        ]


        if param_index == 0:
            X = df[feature_cols].to_numpy()
            timestep = float(re.search(r'_(\d.\d+)_', file_path).group(1))
            print(f"Extracted timestep: {timestep} from file: {file_path}")
            X = np.concatenate((X, np.full((X.shape[0], 1), timestep)), axis=1)
            features_list.append(X)


        y = np.array([item for sublist in df[param_name] for item in sublist]).flatten().reshape(-1, 16000)
        targets_list[param_index].append(y)


    features = np.concatenate(features_list, axis=0)
    targets = [np.concatenate(targets, axis=0).reshape(-1, 16000) for targets in targets_list]

    print(f"X shape: {features.shape}")
    for i, target in enumerate(targets):
        print(f"y[{i}] shape: {target.shape}")

    return features, targets


def load_all_data(folder_path):

    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")])
    data = [load_data(file_paths[i:i+6]) for i in range(0, len(file_paths), 6)]
    return data

class GlobalBestEpochCallback(Callback):
    def __init__(self):
        super(GlobalBestEpochCallback, self).__init__()
        self.global_best_val_loss = float('inf')
        self.global_best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.global_best_val_loss:
            self.global_best_val_loss = val_loss
            self.global_best_epoch = epoch + 1
            print(f"New overall best model found at epoch {self.global_best_epoch} with val_loss: {self.global_best_val_loss:.4f}")

    def on_train_end(self, logs=None):
        print(f"Final best model obtained at epoch {self.global_best_epoch} with val_loss: {self.global_best_val_loss:.4f}")

def generate_AutokerasModel(max_trials, epochs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_folder = 'Filtered_Target_Norm_2'
    folder_path = os.path.join(script_dir, relative_data_folder)
    data = load_all_data(folder_path)

    X_train = []
    y_train = [[] for _ in range(6)]
    X_test = []
    y_test = [[] for _ in range(6)]

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    for features, targets in train_data:
        X_train.append(features)
        for i in range(6):
            y_train[i].append(targets[i])

    for features, targets in test_data:
        X_test.append(features)
        for i in range(6):
            y_test[i].append(targets[i])

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = [np.concatenate(y, axis=0).reshape(-1, 16000) for y in y_train]
    y_test = [np.concatenate(y, axis=0).reshape(-1, 16000) for y in y_test]

    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    for i, y in enumerate(y_train):
        print(f"y_train[{i}] shape: {y.shape}, dtype: {y.dtype}")

    batch_size = 16

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, tuple(y_train))).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, tuple(y_test))).batch(batch_size)

    save_path = os.path.join(folder_path, 'Autokeras_models_MTL6')
    os.makedirs(save_path, exist_ok=True)

    input_node = ak.StructuredDataInput()
    output_nodes = [ak.RegressionHead(metrics=["mse"], loss="mse", dropout=0.2) for _ in range(6)]

    auto_model = ak.AutoModel(
        inputs=input_node,
        outputs=output_nodes,
        max_trials=max_trials,
        directory=save_path,
        project_name='structured_data_regressor1'
    )

    auto_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[GlobalBestEpochCallback()]
    )

    model = auto_model.export_model()

    model_path = os.path.join(save_path, f'ak_best_model')
    model.save(model_path, save_format='tf', include_optimizer=True)

    if os.path.exists(model_path):
        loaded_model = keras.models.load_model(model_path)
        loaded_model.summary()
    else:
        print(f"Error: Model file not found at {model_path}")

if __name__ == '__main__':
    max_trials = 50
    epochs = 200

    generate_AutokerasModel(max_trials, epochs)
