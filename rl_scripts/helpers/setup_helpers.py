import os
import copy

from stable_baselines3 import PPO
from torch import nn  # pylint: disable=unused-import

from src.engine import Engine
from src.routing import Routing

from helper_scripts.setup_helpers import create_input, save_input
from helper_scripts.sim_helpers import parse_yaml_file, get_start_time

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config

from rl_scripts.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS


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
    elif sim_dict['spectrum_algorithm']:
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                         f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                         f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')


class SetupHelper:
    """
    A helper class to handle setup-related tasks for the SimEnv environment.
    """

    def __init__(self, sim_env: object):
        """
        Constructor for RLSetupHelper.

        :param sim_env: Reference to the parent SimEnv instance, to update relevant attributes directly.
        """
        self.sim_env = sim_env

    def create_input(self):
        """
        Creates input for RL agents based on the simulation configuration.
        """
        base_fp = os.path.join('data')
        self.sim_env.sim_dict['thread_num'] = 's1'

        get_start_time(sim_dict={'s1': self.sim_env.sim_dict})
        file_name = "sim_input_s1.json"

        self.sim_env.engine_obj = Engine(engine_props=self.sim_env.sim_dict)
        self.sim_env.route_obj = Routing(engine_props=self.sim_env.engine_obj.engine_props,
                                         sdn_props=self.sim_env.rl_props.mock_sdn_dict)

        self.sim_env.sim_props = create_input(base_fp=base_fp, engine_props=self.sim_env.sim_dict)
        self.sim_env.modified_props = copy.deepcopy(self.sim_env.sim_props)

        save_input(base_fp=base_fp, properties=self.sim_env.modified_props, file_name=file_name,
                   data_dict=self.sim_env.modified_props)

    def init_envs(self):
        """
        Sets up environments for RL agents based on the simulation configuration.
        """
        # Environment initialization logic (from the original _init_envs method)
        if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_env.sim_dict['is_training']:
            self.sim_env.path_agent.engine_props = self.sim_env.engine_obj.engine_props
            self.sim_env.path_agent.setup_env(is_path=True)
        elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_env.sim_dict['is_training']:
            self.sim_env.core_agent.engine_props = self.sim_env.engine_obj.engine_props
            self.sim_env.core_agent.setup_env(is_path=False)

    # TODO: Options to have select AI agents (drl_path_agents)
    def load_models(self):
        """
        Loads pretrained models for RL agents and configures agent properties.
        """
        # Model loading logic (from the original _load_models method)
        self.sim_env.path_agent.engine_props = self.sim_env.engine_obj.engine_props
        self.sim_env.path_agent.rl_props = self.sim_env.rl_props
        self.sim_env.path_agent.load_model(
            model_path=self.sim_env.sim_dict['path_model'],
            erlang=self.sim_env.sim_dict['erlang'],
            num_cores=self.sim_env.sim_dict['cores_per_link']
        )

        self.sim_env.core_agent.engine_props = self.sim_env.engine_obj.engine_props
        self.sim_env.core_agent.rl_props = self.sim_env.rl_props
        self.sim_env.core_agent.load_model(
            model_path=self.sim_env.sim_dict['core_model'],
            erlang=self.sim_env.sim_dict['erlang'],
            num_cores=self.sim_env.sim_dict['cores_per_link']
        )
