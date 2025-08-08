# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 5.2_Produce_heatmaps
# @Software: PyCharm


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import csv
from datetime import datetime

def plot_heatmap(reshaped_data, file_name, column_name, save_path=None):
    for z_layer, z_data in enumerate(reshaped_data):
        plt.figure()
        sns.heatmap(z_data, cmap="jet", cbar_kws={'label': column_name}, vmin=0, vmax=0.4)
        # plt.title(f'{file_name} - Column: {column_name} - Z-layer {z_layer}')
        # plt.xlabel('X-Axis')
        # plt.ylabel('Y-Axis')
        plt.title(f'Z-layer {z_layer}')
        plt.xticks([])
        plt.yticks([])

        subfolder_path = os.path.join(save_path, column_name)
        os.makedirs(subfolder_path, exist_ok=True)
        plt.savefig(os.path.join(subfolder_path, f'{column_name}_Z-layer_{z_layer}.png'))
        plt.close()

heatmaps_folder = "heatmaps\\24-08-15 18-50_3"
os.makedirs(heatmaps_folder, exist_ok=True)

input_csv_folder = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\5_Prediction_and_Bewertung'
current_date = '24-08-15 18-50'
output_parameters = [
    # "number_particles",
    # "power_dissipation",
    "power_impact",
    # "power_shear",
    # "velocity_absolute_translational",
    # "velocity_impact_normal_mean",

]

num_layers = 40
x_layer = 20
y_layer = 20


for param in output_parameters:
    # csv_file_path = os.path.join(input_csv_folder, f'predicted_{current_date}', f'{param}_{current_date}.csv')

    csv_file_path = os.path.join(input_csv_folder, f'pro_Standard_{current_date}', f'{param}_{current_date}.csv')

    df = pd.read_csv(csv_file_path, header=0)
    df = df.iloc[:, 11:]

    # df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(':', '')

    file_name = os.path.basename(csv_file_path)
    file_name_without_extension = file_name.split(".csv")[0]

    save_path = os.path.join(input_csv_folder, heatmaps_folder, file_name_without_extension)


    for column in df.columns:
        data = df[column].astype(float)
        num_values = len(data)
        val1 = data[0]
        print(f"First value of column {column} in file {file_name}: {val1}")

        if num_values == num_layers * x_layer * y_layer:
            print("The dimensions fit")
        else:
            print("The dimension of the predicted data doesn´t match the dimensions of the heatmaps")
            continue

        expected_values = num_layers * x_layer * y_layer
        if num_values != expected_values:
            print(f"Skipping file {file_name} due to dimension mismatch: expected {expected_values}, got {num_values}")
            continue

        reshaped_data = np.array(data).reshape((num_layers, x_layer, y_layer))
        plot_heatmap(reshaped_data, file_name_without_extension, column, save_path)

