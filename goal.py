import numpy as np
import os
import glob


def get_position(state_layer):
    positions = np.argwhere(state_layer == 1)
    return positions[0] if len(positions) > 0 else None

def calculate_distance_change(prev_position, current_position):
    if prev_position is None or current_position is None:
        return 0
    return np.linalg.norm(current_position - prev_position) - np.linalg.norm(prev_position)

def calculate_current_distance(position1, position2):
    if position1 is None or position2 is None:
        return np.inf
    return np.linalg.norm(position1 - position2)

def calculate_likelihood(distance_change, current_distance):
    base_factor = 1.5
    distance_factor = 1.0 if current_distance == 0 else 1 / current_distance
    if distance_change < 0:  # 猎人靠近目标
        return base_factor + distance_factor * 4.0
    return 1.0  # 距离未缩小或增大时保持原似然值

def update_goal_probabilities(step_state, prev_step_state, prior_probabilities):
    hunter_position = get_position(step_state[0, :, :])
    rabbit_position = get_position(step_state[1, :, :])
    sheep_position = get_position(step_state[2, :, :])

    prev_hunter_position = get_position(prev_step_state[0, :, :]) if prev_step_state is not None else None

    distance_change_rabbit = calculate_distance_change(prev_hunter_position, rabbit_position)
    distance_change_sheep = calculate_distance_change(prev_hunter_position, sheep_position)
    current_distance_rabbit = calculate_current_distance(hunter_position, rabbit_position)
    current_distance_sheep = calculate_current_distance(hunter_position, sheep_position)

    likelihood_rabbit = calculate_likelihood(distance_change_rabbit, current_distance_rabbit)
    likelihood_sheep = calculate_likelihood(distance_change_sheep, current_distance_sheep)

    target_probabilities = prior_probabilities.copy()
    target_probabilities['Rabbit'] *= likelihood_rabbit
    target_probabilities['Sheep'] *= likelihood_sheep

    total_prob = sum(target_probabilities.values())
    for animal in target_probabilities:
        target_probabilities[animal] /= total_prob

    return target_probabilities


def process_file(file_path):
    all_steps = np.load(file_path)
    current_probabilities = {'Rabbit': 0.5, 'Sheep': 0.5}
    prev_step_state = None

    print(f"Processing file: {file_path}")
    for step_number, step_state in enumerate(all_steps):
        current_probabilities = update_goal_probabilities(step_state, prev_step_state, current_probabilities)
        prev_step_state = step_state
        print(f"Step {step_number}: Goal probabilities - Rabbit: {current_probabilities['Rabbit']}, Sheep: {current_probabilities['Sheep']}")

def process_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*.npy')):
        process_file(file_path)


folder_path = 'exp1704597341'
process_folder(folder_path)
