from rl_scripts.helpers.general_helpers import determine_model_type
from rl_scripts.args.registry_args import ALGORITHM_REGISTRY


# TODO: (drl_path_agents) Only works for 's1'
# TODO: (drl_path_agents) We need to return the model here to use it...What about model.learn?
def get_algorithm_instance(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Retrieve an instance of the algorithm class associated with the model type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.
    """
    model_type = determine_model_type(sim_dict=sim_dict)

    if '_' not in model_type:
        raise ValueError("Algorithm info must include both algorithm and agent type (e.g., 'ppo_path').")
    algorithm = sim_dict['s1'].get(model_type)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not registered.")

    algorithm_class = ALGORITHM_REGISTRY[algorithm]['class']
    return algorithm_class(rl_props=rl_props, engine_obj=engine_obj)


def get_obs_space(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Get the observation space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.

    :return: Observation space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, rl_props=rl_props, engine_obj=engine_obj)
    return algorithm_instance.get_obs_space()


def get_action_space(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Get the action space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.

    :return: Action space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, rl_props=rl_props, engine_obj=engine_obj)
    return algorithm_instance.get_action_space()
