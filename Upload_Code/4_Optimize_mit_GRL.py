# ！/Programmierung/Anaconda/envs python
# -*- coding: utf-8 -*-
# @Author  : Shun
# @File    : 4_Optimize_mit_GRL
# @Software: PyCharm

import json
import numpy as np
import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import re
import autokeras as ak
import gc
import copy
from tensorflow.keras.backend import clear_session

GENS = 50
POP_SIZE = 12
BATCH_SIZE = 16
CROPB = 0.8
MUTPB = 0.1
EPOCHS = 10
PERCENTILE = 0.3

output_parameters = [
    "power_dissipation",
    "power_impact",
    "power_shear",
    "velocity_absolute_translational",
    "velocity_impact_normal_mean",
]

def load_data(file_paths, param_name):
    features_list = []
    targets_list = []

    for file_path in file_paths:

        with open(file_path, 'r') as file:
            json_data = json.load(file)

        if isinstance(json_data, list) and len(json_data) > 0:
            df = pd.json_normalize(json_data, 'data', sep='_')
        else:
            raise ValueError("Invalid JSON format. Expected a 'data' key with list of dictionaries")

        feature_cols = [
            "static friction p-w",
            "static friction p-p",
            "rolling resistance",
            "E Modul",
            "restitution coefficient p-w",
            "restitution coefficient p-p",
            "viscosity",
            "coupling",
            "rotational speed",
            "value of flow",
            "diameter",
        ]

        X = df[feature_cols].to_numpy()
        timestep = float(re.search(r'_(\d.\d+)_', file_path).group(1))
        print(f"Extracted timestep: {timestep} from file: {file_path}")
        X = np.concatenate((X, np.full((X.shape[0], 1), timestep)), axis=1)
        features_list.append(X)

        if param_name in df.columns:
            y = np.array([item for sublist in df[param_name] for item in sublist]).flatten().reshape(-1, 16000)
            targets_list.append(y)
        else:
            raise ValueError(f"Parameter '{param_name}' not found in the data.")

    features = np.concatenate(features_list, axis=0)
    targets = np.concatenate(targets_list, axis=0).reshape(-1, 16000)

    print(f"X shape: {features.shape}")
    print(f"y shape: {targets.shape}")

    return features, targets

def load_relevant_data(folder_path, param_name):
    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json") and param_name in f])

    return load_data(file_paths, param_name)

train_folder = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\3_Create_NN\\Filtered_Real_Data\\Training_Dataset'
test_folder = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\3_Create_NN\\Filtered_Real_Data\\Test_Dataset'
param_to_train = "velocity_absolute_translational"

base_path = 'D:\\Programmierung\\Programmierung_fuer_MA\\Final\\4_Optimization'
save_path = os.path.join(base_path, f'Opimized_model_{param_to_train}_GRL_2')

X_train, y_train = load_relevant_data(train_folder, param_to_train)
X_test, y_test = load_relevant_data(test_folder, param_to_train)

batch_size = BATCH_SIZE
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


def load_existing_model():
    # model_path = ('D:\\Programmierung\\Programmierung_fuer_MA\\Final\\3_Create_NN'
    #               '\\AK_velocity_absolute_translational_Real_mse\\ak_best_model')

    model_path = os.path.join(base_path, f'Opimized_model_{param_to_train}_GRL')

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        model.compile(optimizer=model.optimizer, loss="mse", metrics=["mse"])
        model.summary()
        return model
    else:
        raise FileNotFoundError(f"Model not found：{model_path}")


def create_individual(model):
    weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in weights])
    return flat_weights

def create_population(pop_size, model):
    return [create_individual(model) for _ in range(pop_size)]

def layer_crossover(model, parent1, parent2, crossover_point_layer=2):
    weight_shapes = [w.shape for w in model.get_weights()]
    split_indices = np.cumsum([np.prod(shape) for shape in weight_shapes], dtype=int)[:-1]

    parent1_layers = np.split(np.array(parent1), split_indices)
    parent2_layers = np.split(np.array(parent2), split_indices)

    offspring1_layers = parent1_layers[:crossover_point_layer] + parent2_layers[crossover_point_layer:]
    offspring2_layers = parent2_layers[:crossover_point_layer] + parent1_layers[crossover_point_layer:]

    offspring1 = np.concatenate([layer.flatten() for layer in offspring1_layers])
    offspring2 = np.concatenate([layer.flatten() for layer in offspring2_layers])

    return offspring1, offspring2

def mutate_weights(individual, percentage=PERCENTILE):
    for i in range(len(individual)):

        if isinstance(individual[i], (int, float)) and random.random() < MUTPB:
            deviation = individual[i] * percentage * random.uniform(-1, 1)
            individual[i] += deviation
    return individual

def evaluate_individual(individual, model):
    weight_shapes = [w.shape for w in model.get_weights()]
    split_indices = np.cumsum([np.prod(shape) for shape in weight_shapes], dtype=int)[:-1]
    new_weights = np.split(np.array(individual), split_indices)
    reshaped_weights = [w.reshape(shape) for w, shape in zip(new_weights, weight_shapes)]
    model.set_weights(reshaped_weights)

    loss = model.evaluate(test_dataset, verbose=0)

    return loss[0]

def train_individual(individual, model):
    weight_shapes = [w.shape for w in model.get_weights()]
    split_indices = np.cumsum([np.prod(shape) for shape in weight_shapes], dtype=int)[:-1]
    new_weights = np.split(np.array(individual), split_indices)
    reshaped_weights = [w.reshape(shape) for w, shape in zip(new_weights, weight_shapes)]
    model.set_weights(reshaped_weights)

    model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, verbose=0)

    trained_weights = model.get_weights()
    trained_individual = np.concatenate([w.flatten() for w in trained_weights])

    return trained_individual

def good_bad_mutation(individual, original_loss, model):
    original_individual = copy.deepcopy(individual)

    mutated_individual = mutate_weights(individual)
    mutated_loss = evaluate_individual(mutated_individual, model)

    if mutated_loss < original_loss:
        print(f"Good mutation: Loss improved from {original_loss} to {mutated_loss}")
        return mutated_individual, mutated_loss
    else:
        print(f"Bad mutation: Loss worsened from {original_loss} to {mutated_loss}. Reverting mutation.")

        return original_individual, original_loss


def select_population(population, fitnesses, model, num_best=2):
    sorted_indices = np.argsort(np.array(fitnesses, dtype=float)).astype(int)

    print("fitnesses:", fitnesses)

    best_individuals = [population[i] for i in sorted_indices[:num_best]]
    new_population = []

    for ind in best_individuals * (len(population) // num_best):
        trained_individual = train_individual(ind, model)
        new_population.append(trained_individual)

    return new_population


def GRL_optimization(num_generations, pop_size):
    model = load_existing_model()
    population = create_population(pop_size, model)
    best_loss_per_generation = []

    previous_best_loss = float('inf')
    previous_best_individual = None

    for gen in range(num_generations):

        fitnesses = [evaluate_individual(ind, model) for ind in population]
        best_loss = min(fitnesses)
        best_individual = copy.deepcopy(population[np.argmin(fitnesses)])

        print(f"Generation {gen + 1}: Best Loss = {best_loss}")

        if best_loss > previous_best_loss:
            print("Warning: New generation's best loss is worse. Replacing with previous best individual.")
            population = [copy.deepcopy(previous_best_individual) for _ in range(pop_size)]
            best_loss = previous_best_loss
            best_loss_per_generation.append(best_loss)
        else:
            previous_best_loss = best_loss
            previous_best_individual = copy.deepcopy(best_individual)
            best_loss_per_generation.append(best_loss)

            population = select_population(population, fitnesses, model)

        new_population = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < CROPB:
                    child1, child2 = layer_crossover(model, population[i], population[i + 1])
                else:
                    child1, child2 = population[i], population[i + 1]

                original_loss1 = evaluate_individual(child1, model)
                original_loss2 = evaluate_individual(child2, model)

                child1, _ = good_bad_mutation(child1, original_loss1, model)
                child2, _ = good_bad_mutation(child2, original_loss2, model)

                new_population.extend([child1, child2])

        population = new_population

    plt.plot(range(1, len(best_loss_per_generation) + 1), best_loss_per_generation, marker='o')
    plt.title(f"Best Loss per Generation {param_to_train}")
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.grid(True)
    plt.show()

    final_fitnesses = [evaluate_individual(ind, model) for ind in population]
    best_individual = population[np.argmin(final_fitnesses)]
    return best_individual

def main():
    best_individual = GRL_optimization(num_generations=GENS, pop_size=POP_SIZE)
    model = load_existing_model()
    weight_shapes = [w.shape for w in model.get_weights()]
    split_indices = np.cumsum([np.prod(shape) for shape in weight_shapes], dtype=int)[:-1]

    new_weights = np.split(np.array(best_individual), split_indices)
    reshaped_weights = [w.reshape(shape) for w, shape in zip(new_weights, weight_shapes)]
    model.set_weights(reshaped_weights)

    model.save(save_path, save_format='tf')
    print(f"Optimized model saved at: {save_path}")

if __name__ == '__main__':
    main()

