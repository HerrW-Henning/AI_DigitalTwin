# ！/Programmierung/Anaconda/envs python
# @File    : 5_Prediction_and_Bewertung
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 5.1_predict_Initial
# @Software: PyCharm


import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import autokeras as ak
from datetime import datetime
import csv
from tensorflow import keras
from tensorflow.keras.models import load_model

# model = tf.keras.models.load_model('D:\\Programmierung\\Programmierung_fuer_MA\\Final\\'
#                                    '3_Create_NN\\AK_velocity_absolute_translational_Real_mse\\ak_best_model')

model = tf.keras.models.load_model('D:\\Programmierung\\Programmierung_fuer_MA\\Final\\'
                                   '4_Optimization\\Opimized_model_velocity_absolute_translational_GRL_2')
model.summary()

def display_info_panel():
    print("Info Panel:")
    print("-------------")
    print("For static friction p-w: Value should be between 0.1 and 0.9.")
    print("For static friction p-p: Value should be between 0.1 and 0.9.")
    print("For rolling resistance: Value should be between 0.001 and 0.5.")
    print("For E Modul: Value should be between 1 and 200.")
    print("For restitution coefficient p-w: Value should be between 0.1 and 0.9.")
    print("For restitution coefficient p-p: Value should be between 0.1 and 0.9.")
    print("For viscosity: Value should be between 0.0011 and 100.")
    print("For coupling: Value should be either 1 or 2.")
    print("For rotational speed: Value should be between 6 and 12.")
    print("For value of flow: Value should be between 20 and 60.")
    print("For diameter: Value should be between 1 and 5.")
    print("For timestep: Value should be lower than 0.5.")
    print("-------------")

def predict_input():
    # # get user input for input parameters
    # static_friction_pw = float(input("Enter static friction p-w: "))
    # static_friction_pp = float(input("Enter static friction p-p: "))
    # rolling_resistance = float(input("Enter rolling resistance: "))
    # E_Modul = float(input("Enter E Modul: "))
    # restitution_coefficient_pw = float(input("Enter restitution coefficient p-w: "))
    # restitution_coefficient_pp = float(input("Enter restitution coefficient p-p: "))
    # viscosity = float(input("Enter viscosity: "))
    # coupling = int(input("Enter coupling: "))
    # rotational_speed = float(input("Enter rotational speed: "))
    # value_of_flow = float(input("Enter value of flow: "))
    # diameter = float(input("Enter diameter: "))
    # timestep = float(input("Enter timestep: "))

    static_friction_pw = 0.5
    static_friction_pp = 0.5
    rolling_resistance = 0.1
    E_Modul = 50
    restitution_coefficient_pw = 0.5
    restitution_coefficient_pp = 0.5
    viscosity = 1
    coupling = 2
    rotational_speed = 9
    value_of_flow = 40
    diameter = 3
    timestep = 0.5

    input_data = np.array([[static_friction_pw, static_friction_pp, rolling_resistance, E_Modul,
                            restitution_coefficient_pw, restitution_coefficient_pp, viscosity,
                            coupling, rotational_speed, value_of_flow, diameter, timestep]])


    predictions = model.predict(input_data)
    real_predictions = np.maximum(predictions, 0)
    save_predictions_to_csv(input_data, real_predictions)

def save_predictions_to_csv(input_data, predictions):

    current_date = datetime.now().strftime("%y-%m-%d %H-%M")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f'predicted_{current_date}')
    os.makedirs(save_dir, exist_ok=True)

    output_parameters = [
        # "number_particles",
        # "power_dissipation",
        # "power_impact",
        # "power_shear",
        "velocity_absolute_translational",
        # "velocity_impact_normal_mean",
    ]

    scaling_factor = 1

    for i, prediction in enumerate(predictions):

        if output_parameters[i] == "number_particles":
            prediction = prediction * scaling_factor

        csv_file_path = os.path.join(save_dir, f'{output_parameters[i]}_{current_date}.csv')

        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(["Date of prediction:", current_date])
            csv_writer.writerow(["Input Parameters:", "static_friction_pw", "static_friction_pp", "rolling_resistance",
                                 "E_Modul", "restitution_coefficient_pw", "restitution_coefficient_pp", "viscosity",
                                 "coupling", "rotational speed", "value of flow", "diameter", "timestep"])
            csv_writer.writerow(
                [input_data[0, 0], input_data[0, 1], input_data[0, 2], input_data[0, 3], input_data[0, 4],
                 input_data[0, 5], input_data[0, 6], input_data[0, 7], input_data[0, 8], input_data[0, 9],
                 input_data[0, 10], input_data[0, 11]])

            csv_writer.writerow([])

            csv_writer.writerow([output_parameters[i]])

            csv_writer.writerows(prediction.reshape(-1, 1))

    print(f"Predicted output parameters saved to {save_dir}")

if __name__ == "__main__":
    display_info_panel()
    predict_input()

