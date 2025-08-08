# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 2.1_Nomalization_Target
# @Software: PyCharm

import json
import os
import numpy as np
import csv

def min_max_normalize_non_zero(data):

    data_array = np.array(data, dtype=np.float64)
    non_zero_data = data_array[data_array != 0]

    if len(non_zero_data) == 0:
        return data_array.tolist()

    min_val = np.min(non_zero_data)
    max_val = np.max(non_zero_data)

    if max_val == min_val:
        normalized_data = np.zeros_like(non_zero_data)
    else:
        normalized_data = (non_zero_data - min_val) / (max_val - min_val)

    data_array[data_array != 0] = normalized_data

    return data_array.tolist()


def percentage_normalize(data):

    data_array = np.array(data, dtype=np.float64)
    total = np.sum(data_array)
    percentage_data = (data_array / total)

    return percentage_data.tolist()

def process_json_file(file_path, output_directory):

    with open(file_path, 'r') as f:
        json_data = json.load(f)

    for data_entry in json_data:
        for field in data_entry['schema']['fields']:
            field_name = field['name']
            if field_name != "number_particles" and isinstance(data_entry['data'][0][field_name], list):
                data_entry['data'][0][field_name] = min_max_normalize_non_zero(data_entry['data'][0][field_name])

            elif field_name == "number_particles" and isinstance(data_entry['data'][0][field_name], list):
                data_entry['data'][0][field_name] = percentage_normalize(data_entry['data'][0][field_name])

    os.makedirs(output_directory, exist_ok=True)
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_directory, file_name)

    with open(output_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Processed file saved at: {output_file_path}")


def process_all_json_files(input_directory, output_directory):

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_directory, file_name)
            process_json_file(file_path, output_directory)

input_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Filtered_Data'
output_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\2_Data_Preprocessing\Filtered_Target_Norm'

process_all_json_files(input_directory, output_directory)

