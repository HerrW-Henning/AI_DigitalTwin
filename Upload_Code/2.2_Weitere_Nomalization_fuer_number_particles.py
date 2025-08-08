# ！/Programmierung/Anaconda/envs python
# @File    : 2_Data_Pre_Processing
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : Weitere_Nomalization_fuer_number_particles
# @Software: PyCharm


import json
import os
import numpy as np
import csv


def find_global_min_max(directory_path):
    global_min = float('inf')
    global_max = float('-inf')

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    if 'number_particles' in data[0]['data'][0]:
                        number_particles = np.array(data[0]['data'][0]['number_particles'], dtype=np.float64)
                        global_min = min(global_min, np.min(number_particles))
                        global_max = max(global_max, np.max(number_particles))

    return global_min, global_max


def normalize(data, min_val, max_val):
    if max_val != min_val:
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data


def process_json_files_with_global_norm(directory_path, global_min, global_max):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    if 'number_particles' in data[0]['data'][0]:
                        number_particles = np.array(data[0]['data'][0]['number_particles'], dtype=np.float64)
                        normalized_number_particles = normalize(number_particles, global_min, global_max)
                        data[0]['data'][0]['number_particles'] = normalized_number_particles.tolist()

                        with open(json_file_path, 'w') as json_file:
                            json.dump(data, json_file, indent=2)

                        print(f"Processed file with global normalization: {json_file_path}")


def save_global_min_max_to_csv(global_min, global_max, output_path):

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Parameter', 'Global Min', 'Global Max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Parameter': 'number_particles', 'Global Min': global_min, 'Global Max': global_max})

json_directory_path = os.path.join(os.getcwd(), 'Filtered_Target_Norm')

global_min, global_max = find_global_min_max(json_directory_path)
print(f"Global min: {global_min}, Global max: {global_max}")

process_json_files_with_global_norm(json_directory_path, global_min, global_max)

csv_output_path = os.path.join(os.getcwd(), 'global_min_max_number_particles_basic.csv')
save_global_min_max_to_csv(global_min, global_max, csv_output_path)
print(f"Global min and max saved to {csv_output_path}")
