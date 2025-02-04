from rl_scripts.args.general_args import ALGORITHM_REGISTRY


def get_algorithm_instance(sim_dict: dict, model_type: str):
    """
    Retrieve an instance of the algorithm class associated with the model type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param model_type: The type of model to work with.
    """
    algorithm_info = sim_dict.get(model_type)

    if '_' not in algorithm_info:
        raise ValueError("Algorithm info must include both algorithm and agent type (e.g., 'ppo_path').")
    algorithm, _ = algorithm_info.split('_', 1)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not registered.")

    algorithm_class = ALGORITHM_REGISTRY[algorithm]['class']

    rl_props = sim_dict.get('rl_props')
    engine_props = sim_dict.get('engine_props')

    # Initialize and return the algorithm class instance
    return algorithm_class(rl_props=rl_props, engine_props=engine_props)


def get_obs_space(sim_dict: dict, model_type: str):
    """
    Get the observation space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param model_type: The type of model to work with.

    :return: Observation space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, model_type=model_type)
    return algorithm_instance.get_obs_space()


def get_action_space(sim_dict: dict, model_type: str):
    """
    Get the action space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param model_type: The type of model to work with.

    :return: Action space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, model_type=model_type)
    return algorithm_instance.get_action_space()
