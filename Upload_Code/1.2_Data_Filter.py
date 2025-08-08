# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 1.2_Filter_Data
# @Software: PyCharm

import os
import json
import numpy as np


def replace_negative_with_zero(data):
    data = np.array(data)
    data[data < 0] = 0
    return data.tolist()


def process_max_value(data):

    layers = [data[i:i + 400] for i in range(0, len(data), 400)]
    last_but_one_layer = 38

    other_layers = [layer for i, layer in enumerate(layers) if i != last_but_one_layer]
    all_values = np.concatenate(other_layers)
    A1 = np.max(all_values)
    A2 = np.partition(all_values, -2)[-2]

    layers[last_but_one_layer] = [A2 if elem > A1 else elem for elem in layers[last_but_one_layer]]

    return [item for layer in layers for item in layer]


def process_json_file(file_path, output_directory):

    with open(file_path, 'r') as f:
        json_data = json.load(f)

    for data_entry in json_data:
        for field in data_entry['schema']['fields']:
            field_name = field['name']
            if isinstance(data_entry['data'][0][field_name], list):

                data_entry['data'][0][field_name] = replace_negative_with_zero(data_entry['data'][0][field_name])

                data_entry['data'][0][field_name] = process_max_value(data_entry['data'][0][field_name])

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))

    with open(output_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Processed file saved at: {output_file_path}")


def process_all_json_files(input_directory, output_directory):

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_directory, file_name)
            process_json_file(file_path, output_directory)

input_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Original_Data_Json'
output_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Filtered_Data'

process_all_json_files(input_directory, output_directory)





