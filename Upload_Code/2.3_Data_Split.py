# ！/Programmierung/Anaconda/envs python
# @File    : 2_Data_Preprocessing
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : Data_Split
# @Software: PyCharm

import os
import shutil
import random


def group_files(file_list, group_size=6):

    return [file_list[i:i + group_size] for i in range(0, len(file_list), group_size)]


def split_data_groups_random(grouped_files, train_ratio=0.8):

    random.shuffle(grouped_files)
    total_groups = len(grouped_files)
    train_size = int(total_groups * train_ratio)

    train_groups = grouped_files[:train_size]
    test_groups = grouped_files[train_size:]

    return train_groups, test_groups


def move_files(file_groups, target_directory):
    os.makedirs(target_directory, exist_ok=True)

    for group in file_groups:
        for file_path in group:
            file_name = os.path.basename(file_path)
            target_path = os.path.join(target_directory, file_name)
            shutil.copy(file_path, target_path)
    print(f"Files moved to {target_directory}")


def main(input_directory, train_directory, test_directory, group_size=6, train_ratio=0.8):

    json_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.json')]
    assert len(json_files) == 582, "总文件数量应该为582个"

    grouped_files = group_files(json_files, group_size=group_size)
    train_groups, test_groups = split_data_groups_random(grouped_files, train_ratio=train_ratio)
    move_files(train_groups, train_directory)
    move_files(test_groups, test_directory)

input_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\1_Data_Convert\Filtered_Data'
train_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\3_Create_NN\Filtered_Real_Data\Training_Dataset'
test_directory = r'D:\Programmierung\Programmierung_fuer_MA\Final\3_Create_NN\Filtered_Real_Data\Test_Dataset'

main(input_directory, train_directory, test_directory)
