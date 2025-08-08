# ！/Programmierung/Anaconda/envs python
# @File    : 5_Prediction_and_Bewertung
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 5.3_Bewertung_Zusammen
# @Software: PyCharm

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

real_data_folder = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\3_Create_NN\\Filtered_Real_Data\\Test_Dataset'
prediction_folder = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\5_Prediction_and_Bewertung'
prediction_suffix = 'predicted_24-11-20 16-30'
predicted_label = 'Prediction'

file_prefixes = [
    "number_particles",
    # "power_dissipation",
    # "power_impact",
    # "power_shear",
    # "velocity_absolute_translational",
    # "velocity_impact_normal_mean"
    ]

for prefix in file_prefixes:
    real_file_path = os.path.join(real_data_folder, f'Standart_0.50_{prefix}.json')
    with open(real_file_path, 'r') as f:
        real_data = json.load(f)

    real_values = pd.DataFrame(real_data[0]['data'])
    real_values_flat = np.concatenate(real_values[prefix].values)

    predicted_file = os.path.join(prediction_folder, prediction_suffix,
                                  f"{prefix}_{prediction_suffix.split('_')[-1]}.csv")
    predicted_data = pd.read_csv(predicted_file, skiprows=4)
    predicted_values = pd.to_numeric(predicted_data[predicted_data.columns[-1]], errors='coerce').dropna()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_values_flat, fill=False, label="Real Data")
    sns.kdeplot(predicted_values, fill=False, label=predicted_label)
    plt.title(f'Density Distribution of {prefix}')
    plt.xlabel(prefix)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.ecdfplot(real_values_flat, label="Real Data")
    sns.ecdfplot(predicted_values, label=predicted_label)
    plt.title(f'Cumulative Distribution of {prefix}')
    plt.xlabel(prefix)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.show()


