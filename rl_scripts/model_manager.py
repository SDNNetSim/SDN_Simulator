import os

from stable_baselines3 import PPO

from rl_scripts.helpers.setup_helpers import setup_ppo


def get_model(algorithm: str, device: str, env: object):
    """
    Creates or retrieves a new reinforcement learning model based on the specified algorithm.

    :param algorithm: The algorithm type (e.g., 'ppo', 'dqn', or 'a2c').
    :param device: The device on which the model will run (e.g., 'cpu' or 'cuda').
    :param env: The reinforcement learning environment.
    :return: A tuple containing the RL model and a configuration dictionary for the environment.
    """
    model = None
    yaml_dict = None
    env_name = None

    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        model = setup_ppo(env=env, device=device)
    elif algorithm == 'a2c':
        model = None

    return model, yaml_dict[env_name]


def get_trained_model(env: object, sim_dict: dict):
    """
    Loads a pre-trained reinforcement learning model from disk or initializes a new one.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters, including the model type and path.
    :return: The loaded or newly initialized RL model.
    """
    if sim_dict['spectrum_algorithm'] == 'ppo':
        model = PPO.load(os.path.join('logs', sim_dict['spectrum_model'], 'ppo_model.zip'), env=env)
    else:
        model = None

    return model
