# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : Check_Value
# @Software: PyCharm

import json
import os
import numpy as np

base_path = r"D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Filtered_Data"

output_parameters = [
    "number_particles",
    "power_dissipation",
    "power_impact",
    "power_shear",
    "velocity_absolute_translational",
    "velocity_impact_normal_mean"
]


def extract_min_max_with_positions(file_path, parameter_name):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if parameter_name in data[0]["data"][0]:
        param_data = np.array(data[0]["data"][0][parameter_name])
    else:
        raise ValueError(f"Parameter '{parameter_name}' not found in the data.")

    split_data = np.split(param_data, 40)

    min_values, min_positions, max_values, max_positions = [], [], [], []
    for group_index, group in enumerate(split_data):
        min_val = np.min(group)
        max_val = np.max(group)
        min_pos = np.where(group == min_val)[0][0] + group_index * 400
        max_pos = np.where(group == max_val)[0][0] + group_index * 400

        min_values.append(min_val)
        min_positions.append(min_pos)
        max_values.append(max_val)
        max_positions.append(max_pos)

    return min_values, min_positions, max_values, max_positions



for file_name in os.listdir(base_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(base_path, file_name)

        for parameter in output_parameters:
            try:
                min_values, min_positions, max_values, max_positions = extract_min_max_with_positions(file_path,
                                                                                                      parameter)

                print(f"File: {file_name}, Parameter: {parameter}")
                print(f"Min Values: {min_values}")
                print(f"Min Positions: {min_positions}")
                print(f"Max Values: {max_values}")
                print(f"Max Positions: {max_positions}\n")
            except ValueError as e:
                print(e)

