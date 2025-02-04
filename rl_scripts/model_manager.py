import os

from rl_scripts.args.general_args import ALGORITHM_REGISTRY


def determine_model_type(sim_dict: dict) -> str:
    """
    Determines the type of agent being used based on the provided simulation dictionary.

    :param sim_dict: A dictionary containing simulation configuration.
    :return: A string representing the model type ('path_algorithm', 'core_algorithm', 'spectrum_algorithm').
    """
    if sim_dict.get('path_algorithm') is not None:
        return 'path_algorithm'
    if sim_dict.get('core_algorithm') is not None:
        return 'core_algorithm'
    if sim_dict.get('spectrum_algorithm') is not None:
        return 'spectrum_algorithm'

    raise ValueError("No valid algorithm type found in sim_dict. "
                     "Ensure 'path_algorithm', 'core_algorithm', or 'spectrum_algorithm' is set.")


def get_model(sim_dict: dict, device: str, env: object):
    """
    Creates or retrieves a new reinforcement learning model based on the specified algorithm.

    :param sim_dict: Has all simulation parameters.
    :param device: The device on which the model will run (e.g., 'cpu' or 'cuda').
    :param env: The reinforcement learning environment.
    :return: A tuple containing the RL model and a configuration dictionary for the environment.
    """
    yaml_dict = {}
    env_name = None

    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm = sim_dict.get(model_type)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported.")

    model = ALGORITHM_REGISTRY[algorithm]['setup'](env=env, device=device)
    return model, yaml_dict.get(env_name, {})


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
    algorithm_info = sim_dict.get(model_type)

    if '_' not in algorithm_info:
        raise ValueError(
            f"Algorithm info '{algorithm_info}' must include both algorithm and agent type (e.g., 'ppo_path').")
    # TODO: (drl_path_agents) If agent type isn't used, remove it
    algorithm, _ = algorithm_info.split('_', 1)

    if algorithm_info in sim_dict:
        save_fp = os.path.join(
            'logs',
            algorithm_info,
            env.modified_props['network'],
            env.modified_props['date'],
            env.modified_props['sim_start'],
            f"{algorithm_info}_model.zip"
        )
        model.save(save_fp)
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported for saving.")
