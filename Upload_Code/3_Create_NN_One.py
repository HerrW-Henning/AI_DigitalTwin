# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 3_Create_NN_One
# @Software: PyCharm

import json
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.losses import Huber

output_parameters = [
    "number_particles",
    "power_dissipation",
    "power_impact",
    "power_shear",
    "velocity_absolute_translational",
    "velocity_impact_normal_mean"
]

def load_data(file_paths, param_name):
    features_list = []
    targets_list = []

    for file_path in file_paths:
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

        X = df[feature_cols].to_numpy()
        timestep = float(re.search(r'_(\d.\d+)_', file_path).group(1))
        print(f"Extracted timestep: {timestep} from file: {file_path}")
        X = np.concatenate((X, np.full((X.shape[0], 1), timestep)), axis=1)
        features_list.append(X)

        if param_name in df.columns:
            y = np.array([item for sublist in df[param_name] for item in sublist]).flatten().reshape(-1, 16000)
            targets_list.append(y)
        else:
            raise ValueError(f"Parameter '{param_name}' not found in the data.")

    features = np.concatenate(features_list, axis=0)
    targets = np.concatenate(targets_list, axis=0).reshape(-1, 16000)

    print(f"X shape: {features.shape}")
    print(f"y shape: {targets.shape}")

    return features, targets


def load_relevant_data(folder_path, param_name):
    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json") and param_name in f])
    return load_data(file_paths, param_name)

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


def generate_single_model(X_train, y_train, X_test, y_test, output_param, max_trials, epochs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f'AK_{output_param}_Real_mse')
    os.makedirs(save_path, exist_ok=True)

    input_node = ak.StructuredDataInput()
    output_node = ak.RegressionHead(metrics=["mse"], loss="mse")
    # output_node = ak.RegressionHead(metrics=["mae", "mse"], loss=Huber(delta=delta_value))

    auto_model = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        max_trials=max_trials,
        directory=save_path,
        project_name=f'structured_data_regressor'
    )

    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


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

    param_to_train = "number_particles"
    max_trials = 50
    epochs = 200
    delta_value = 0.5

    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_folder = os.path.join(script_dir, 'Filtered_Real_Data\\Training_Dataset')
    test_folder = os.path.join(script_dir, 'Filtered_Real_Data\\Test_Dataset')
    X_train, y_train = load_relevant_data(train_folder, param_to_train)
    X_test, y_test = load_relevant_data(test_folder, param_to_train)

    generate_single_model(X_train, y_train, X_test, y_test, param_to_train, max_trials, epochs)

