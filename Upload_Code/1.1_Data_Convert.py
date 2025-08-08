# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 1.1_Data_Convert
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
import re
import json

def list_csvs():

    file_path = r'D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Original_Data'

    list_csvs = []

    for r, _d, f in os.walk(file_path):
        for csv in f:
            if csv.endswith(".csv"):
                csv_path = os.path.join(r, csv)
                df = pd.read_csv(csv_path)
                print(f"File: {csv_path}, Column Names: {df.columns.tolist()}")
                list_csvs.append(csv_path)

    return list_csvs

def divide_into_layers(x_layers, y_layers, file_path):

    df = pd.read_csv(file_path)
    rows = df.shape[0]

    required_columns = ['ParticleX-Coordinate', 'ParticleY-Coordinate', 'ParticleZ-Coordinate',
                        'AbsoluteTranslationalVelocity', 'Power_Dissipation', 'Power_Impact', 'Power_Shear',
                        'Velocity_Impact_Normal_Mean']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing columns in CSV file: {missing_columns}")

    print("Unique Column Names:", df.columns.unique())

    velo_abs_trans = []
    pow_dissip = []
    pow_imp = []
    pow_sh = []
    velo_imp_norm_mean = []
    num_particles = []

    total_particles = 0

    for k in range(0, 40):

        velocity_absolute_translational = [0 for _ in range(x_layers * y_layers)]
        power_dissipation = [0 for _ in range(x_layers * y_layers)]
        power_impact = [0 for _ in range(x_layers * y_layers)]
        power_shear = [0 for _ in range(x_layers * y_layers)]
        velocity_impact_normal_mean = [0 for _ in range(x_layers * y_layers)]
        number_particles = [0 for _ in range(x_layers * y_layers)]

        df_z_begin = df.loc[df['ParticleZ-Coordinate'] > -0.152 + 0.00449 * k]
        df_z_end = df_z_begin.loc[df_z_begin['ParticleZ-Coordinate'] < -0.152 + 0.00449 + 0.00449 * k]

        for i in range(0, 20):
            df_zx_begin = df_z_end.loc[df_z_end['ParticleX-Coordinate'] > -0.04315 + 0.004315 * i]
            df_zx_end = df_zx_begin.loc[df_zx_begin['ParticleX-Coordinate'] < -0.04315 + 0.004315 + 0.004315 * i]

            for j in range(0, 20):
                df_zxy_begin = df_zx_end.loc[df_zx_end['ParticleY-Coordinate'] > -0.04315 + 0.004315 * j]
                df_zxy = df_zxy_begin.loc[df_zxy_begin['ParticleY-Coordinate'] < -0.04315 + 0.004315 + 0.004315 * j]

                V_absolute_translational = df_zxy['AbsoluteTranslationalVelocity'].sum()
                velocity_absolute_translational[j + i * x_layers] = V_absolute_translational

                P_dissipation = df_zxy['Power_Dissipation'].sum()
                power_dissipation[j + i * x_layers] = P_dissipation

                P_impact = df_zxy['Power_Impact'].sum()
                power_impact[j + i * x_layers] = P_impact

                P_shear = df_zxy['Power_Shear'].sum()
                power_shear[j + i * x_layers] = P_shear

                V_impact_normal_mean = df_zxy['Velocity_Impact_Normal_Mean'].sum()
                velocity_impact_normal_mean[j + i * x_layers] = V_impact_normal_mean

                number_particles[j + i * x_layers] = df_zxy.shape[0]

                total_particles += df_zxy.shape[0]

                if df_zxy.empty:
                    print(f"Empty DataFrame at Z layer {k}, X layer {i}, Y layer {j}")
                else:
                    print(f"DataFrame Size at Z layer {k}, X layer {i}, Y layer {j}: {df_zxy.shape}")


        velocity_absolute_translational = np.array(velocity_absolute_translational, dtype=np.float64)
        power_dissipation = np.array(power_dissipation, dtype=np.float64)
        power_impact = np.array(power_impact, dtype=np.float64)
        power_shear = np.array(power_shear, dtype=np.float64)
        velocity_impact_normal_mean = np.array(velocity_impact_normal_mean, dtype=np.float64)

        velo_abs_trans += velocity_absolute_translational.tolist()
        pow_dissip += power_dissipation.tolist()
        pow_imp += power_impact.tolist()
        pow_sh += power_shear.tolist()
        velo_imp_norm_mean += velocity_impact_normal_mean.tolist()
        num_particles += number_particles

        print(velo_abs_trans)
        print(num_particles)

    num_particles = np.array(num_particles, dtype=np.float64)

    return velo_abs_trans, pow_dissip, pow_imp, pow_sh, velo_imp_norm_mean, num_particles.tolist()



def data_to_json(file_path):

    parameter_lookup = {
        'static friction p-w': {'min': 0.1, 'standard': 0.5, 'max': 0.9},
        'static friction p-p': {'min': 0.1, 'standard': 0.5, 'max': 0.9},
        'rolling resistance': {'min': 0.001, 'standard': 0.1, 'max': 0.5},
        'E Modul': {'min': 1, 'standard': 50, 'max': 200},
        'restitution coefficient p-w': {'min': 0.1, 'standard': 0.5, 'max': 0.9},
        'restitution coefficient p-p': {'min': 0.1, 'standard': 0.5, 'max': 0.9},
        'viscosity': {'min': 0.0011, 'standard': 1, 'max': 100},
        'coupling': {'min': 1, 'standard': 2},
        'rotational speed': {'min': 6, 'standard': 9, 'max': 12},
        'value of flow': {'min': 20, 'standard': 40, 'max': 60},
        'diameter': {'min': 1, 'standard': 3, 'max': 5}
    }


    parameter_info = {param_key: param_values['standard'] for param_key, param_values in parameter_lookup.items()}

    if "one way" in file_path.lower():
        parameter_info['coupling'] = parameter_lookup['coupling']['min']
    else:
        parameter_info['coupling'] = parameter_lookup['coupling']['standard']

    for param_key, param_values in parameter_lookup.items():
        pattern = r"(min|max)?\s*" + param_key.replace(" ", r"\s*")
        match = re.search(pattern, file_path, re.IGNORECASE)

        if match:
            min_max_prefix = match.group(1)
            parameter_info[param_key] = param_values[min_max_prefix] if min_max_prefix else param_values['standard']

    diameter_match = re.search(r'(\d+)mm', file_path, re.IGNORECASE)
    if diameter_match:
        parameter_info['diameter'] = float(diameter_match.group(1))

    for key in parameter_lookup.keys():
        if key not in parameter_info:
            parameter_info[key] = parameter_lookup[key]['standard']

    print("Extracted parameter info:", parameter_info)

    velo_abs_trans, pow_dissip, pow_imp, pow_sh, velo_imp_norm_mean, num_particles = divide_into_layers(20, 20,
                                                                                                        file_path)

    output_parameters = {
        "velocity_absolute_translational": velo_abs_trans,
        "power_dissipation": pow_dissip,
        "power_impact": pow_imp,
        "power_shear": pow_sh,
        "velocity_impact_normal_mean": velo_imp_norm_mean,
        "number_particles": num_particles
    }

    file_name_parts = re.split(r'[_\\/]', file_path)
    underscore_name = file_name_parts[-3]
    time_suffix = re.search(r'_(\d+\.?\d*)s', file_path).group(1)

    print("underscore name:", underscore_name, "time suffix:", time_suffix)

    for param, data in output_parameters.items():
        dict1 = {
            "schema": {
                "fields": [
                    {"name": "static friction p-w", "type": "number"},
                    {"name": "static friction p-p", "type": "number"},
                    {"name": "rolling resistance", "type": "number"},
                    {"name": "E Modul", "type": "number"},
                    {"name": "restitution coefficient p-w", "type": "number"},
                    {"name": "restitution coefficient p-p", "type": "number"},
                    {"name": "viscosity", "type": "number"},
                    {"name": "coupling", "type": "number"},
                    {"name": "rotational speed", "type": "number"},
                    {"name": "value of flow", "type": "number"},
                    {"name": "diameter", "type": "number"},
                    {"name": param, "type": "string"}
                ]
            },
            "data": [
                {
                    'static friction p-w': parameter_info['static friction p-w'],
                    'static friction p-p': parameter_info['static friction p-p'],
                    'rolling resistance': parameter_info['rolling resistance'],
                    'E Modul': parameter_info['E Modul'],
                    'restitution coefficient p-w': parameter_info['restitution coefficient p-w'],
                    'restitution coefficient p-p': parameter_info['restitution coefficient p-p'],
                    'viscosity': parameter_info['viscosity'],
                    'coupling': parameter_info['coupling'],
                    'rotational speed': parameter_info['rotational speed'],
                    'value of flow': parameter_info['value of flow'],
                    'diameter': parameter_info['diameter'],
                    param: data
                },
            ]
        }

        data = [dict1]

        json_file_name = f"{underscore_name}_{time_suffix}_{param}.json"
        json_file_name = json_file_name.replace(':', '_')

        output_dir = os.path.join(os.getcwd(), 'Original_Data_Json')
        os.makedirs(output_dir, exist_ok=True)
        json_file_path = os.path.join(output_dir, json_file_name)

        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)
        print(f"JSON file saved at: {json_file_path}")

csv_files = list_csvs()
for csv_file in csv_files:
    data_to_json(csv_file)


