import os

from helper_scripts.sim_helpers import parse_yaml_file

from rl_scripts.helpers.general_helpers import determine_model_type
from rl_scripts.args.registry_args import ALGORITHM_REGISTRY


def get_model(sim_dict: dict, device: str, env: object):
    """
    Creates or retrieves a new reinforcement learning model based on the specified algorithm.

    :param sim_dict: Has all simulation parameters.
    :param device: The device on which the model will run (e.g., 'cpu' or 'cuda').
    :param env: The reinforcement learning environment.
    :return: A tuple containing the RL model and a configuration dictionary for the environment.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm = sim_dict.get(model_type)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported.")

    yaml_file = os.path.join('sb3_scripts', 'yml', f'{algorithm}.yml')
    yaml_dict = parse_yaml_file(yaml_file=yaml_file)
    model = ALGORITHM_REGISTRY[algorithm]['setup'](env=env, device=device)

    env_name = list(yaml_dict.keys())[0]
    return model, yaml_dict[env_name]


def get_trained_model(env: object, sim_dict: dict):
    """
    Loads a pre-trained reinforcement learning model from disk or initializes a new one.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters, including the model type and path.
    :return: The loaded or newly initialized RL model.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_info = sim_dict.get(model_type)

    if '_' not in algorithm_info:
        raise ValueError(
            f"Algorithm info '{algorithm_info}' must include both algorithm and agent type (e.g., 'ppo_path').")
    algorithm, agent_type = algorithm_info.split('_', 1)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported for loading.")

    model_key = f"{agent_type}_model"
    model_path = os.path.join('logs', sim_dict[model_key], f"{algorithm_info}_model.zip")
    model = ALGORITHM_REGISTRY[algorithm]['load'](model_path, env=env)

    return model


def save_model(sim_dict: dict, env: object, model):
    """
    Saves the trained model to the appropriate location based on the algorithm and agent type.

    :param sim_dict: Simulation configuration dictionary.
    :param env: The reinforcement learning environment.
    :param model: The trained model to be saved.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    if '_' not in model_type:
        raise ValueError(
            f"Algorithm info '{model_type}' must include both algorithm and agent type (e.g., 'ppo_path').")
    # TODO: (drl_path_agents) If agent type isn't used, remove it
    algorithm = sim_dict.get(model_type)
    save_fp = os.path.join(
        'logs',
        algorithm,
        env.modified_props['network'],
        env.modified_props['date'],
        env.modified_props['sim_start'],
        f"{algorithm}_model.zip"
    )
    model.save(save_fp)
