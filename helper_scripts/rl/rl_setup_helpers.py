import os

from stable_baselines3 import PPO

from helper_scripts.sim_helpers import parse_yaml_file
from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from arg_scripts.rl_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS, VALID_SPECTRUM_ALGORITHMS


def setup_rl_sim():
    """
    Set up a reinforcement learning simulation.

    :return: The simulation dictionary for the RL sim.
    :rtype: dict
    """
    args_dict = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_dict=args_dict, config_path=config_path)

    return sim_dict


def print_info(sim_dict: dict):
    """
    Prints relevant RL simulation information.

    :param sim_dict: Simulation dictionary (engine props).
    """
    if sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
        print(f'Beginning training process for the PATH AGENT using the '
              f'{sim_dict["path_algorithm"].title()} algorithm.')
    elif sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
        print(f'Beginning training process for the CORE AGENT using the '
              f'{sim_dict["core_algorithm"].title()} algorithm.')
    elif sim_dict['spectrum_algorithm'] in VALID_SPECTRUM_ALGORITHMS:
        print(f'Beginning training process for the SPECTRUM AGENT using the '
              f'{sim_dict["spectrum_algorithm"].title()} algorithm.')
    else:
        raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                         f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                         f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')


def setup_ppo(env: object, device: str):
    """
    Setups up the StableBaselines3 PPO model.

    :param env: Custom environment created.
    :param device: Device to use, cpu or gpu.
    :return: A PPO model.
    :rtype: object
    """
    yaml_path = os.path.join('sb3_scripts', 'yml', 'ppo.yml')
    yaml_dict = parse_yaml_file(yaml_path)
    env_name = list(yaml_dict.keys())[0]
    kwargs_dict = eval(yaml_dict[env_name]['policy_kwargs'])  # pylint: disable=eval-used
    model = PPO(env=env, device=device, policy=yaml_dict[env_name]['policy'],
                n_steps=yaml_dict[env_name]['n_steps'],
                batch_size=yaml_dict[env_name]['batch_size'], gae_lambda=yaml_dict[env_name]['gae_lambda'],
                gamma=yaml_dict[env_name]['gamma'], n_epochs=yaml_dict[env_name]['n_epochs'],
                vf_coef=yaml_dict[env_name]['vf_coef'], ent_coef=yaml_dict[env_name]['ent_coef'],
                max_grad_norm=yaml_dict[env_name]['max_grad_norm'],
                learning_rate=yaml_dict[env_name]['learning_rate'], clip_range=yaml_dict[env_name]['clip_range'],
                policy_kwargs=kwargs_dict)

    return model
