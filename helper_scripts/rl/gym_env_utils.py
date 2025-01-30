from gym_env_scripts.general_sim_env import SimEnv

from helper_scripts.rl.setup_helpers import setup_rl_sim
from helper_scripts.rl.callback_helpers import GetModelParams


def create_environment():
    """
    Creates the simulation environment and associated callback for RL.

    :return: A tuple consisting of the SimEnv object and its sim_dict.
    """
    callback = GetModelParams()
    env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=setup_rl_sim())
    env.sim_dict['callback'] = callback
    return env, env.sim_dict
