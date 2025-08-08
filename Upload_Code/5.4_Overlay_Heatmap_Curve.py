# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 5.4_Overlay_Heatmap_Curve
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

normalized_heatmap_folder = "D:\\Programmierung\\Programmierung_fuer_MA\\Final\\5_Prediction_and_Bewertung\\predicted_24-11-21 12-16"
absolute_distribution_folder = "D:\\Programmierung\\Programmierung_fuer_MA\\Final\\5_Prediction_and_Bewertung\\predicted_24-11-20 16-30"
output_folder = "D:\\Programmierung\\Programmierung_fuer_MA\\Final\\5_Prediction_and_Bewertung\\Overlay\\Final_3"

parameters = [
    "number_particles",
    # "power_dissipation",
    # "power_impact",
    # "power_shear",
    # "velocity_absolute_translational",
    # "velocity_impact_normal_mean"
]

for param in parameters:

    abs_file_path = os.path.join(absolute_distribution_folder, f"{param}_24-11-20 16-30.csv")
    abs_data = pd.read_csv(abs_file_path, skiprows=5, header=None).values.flatten()
    abs_values_sorted = np.sort(abs_data)
    percentiles = np.linspace(0, 1, len(abs_values_sorted))

    cdf_to_absolute = interp1d(percentiles, abs_values_sorted, fill_value="extrapolate")
    csv_file_path = os.path.join(normalized_heatmap_folder, f"{param}_24-11-21 12-16.csv")
    df = pd.read_csv(csv_file_path, header=3)

    absolute_param_folder = os.path.join(output_folder, param)
    os.makedirs(absolute_param_folder, exist_ok=True)

    for column in df.columns:
        data = df[column].astype(float).values
        if len(data) != 40 * 20 * 20:
            print(f"Skipping {param} due to incorrect data dimensions.")
            continue

        reshaped_data = data.reshape((40, 20, 20))

        for z_layer, z_data in enumerate(reshaped_data):
            flat_z_data = z_data.flatten()
            percentiles_for_data = (flat_z_data - flat_z_data.min()) / (flat_z_data.max() - flat_z_data.min())
            absolute_heatmap = cdf_to_absolute(percentiles_for_data)

            if param == "number_particles":
                absolute_heatmap = np.round(absolute_heatmap)


            print(f"Layer {z_layer} - Original normalized data (first 5 values):", flat_z_data[:5])
            print(f"Layer {z_layer} - Mapped absolute data (first 5 values):", absolute_heatmap[:5])

            absolute_heatmap = absolute_heatmap.reshape(20, 20)

            plt.figure()
            sns.heatmap(absolute_heatmap, cmap="jet", vmin=0, vmax=6)
            plt.title(f'Z-layer {z_layer}')
            plt.axis('off')
            absolute_heatmap_path = os.path.join(absolute_param_folder, f"{param}_Absolute_Z-layer_{z_layer}.png")
            plt.savefig(absolute_heatmap_path)
            plt.close()

