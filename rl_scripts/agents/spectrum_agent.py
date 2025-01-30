import numpy as np
from gymnasium import spaces


# TODO: (drl_path_agents) This class is no longer supported
class SpectrumAgent:
    """
    A class that handles everything related to spectrum assignment in reinforcement learning simulations.
    """

    def __init__(self, spectrum_algorithm: str, rl_props: object):
        self.spectrum_algorithm = spectrum_algorithm
        self.rl_props = rl_props

        self.no_penalty = None
        self.model = None

    def _ppo_obs_space(self):
        """
        Gets the observation space for the DRL agent.

        :return: The observation space.
        :rtype: spaces.Dict
        """
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(15 + 1),
            'source': spaces.MultiBinary(self.rl_props.num_nodes),
            'destination': spaces.MultiBinary(self.rl_props.num_nodes),
            'super_channels': spaces.Box(-0.01, 100.0, shape=(3,), dtype=np.float32)
        })

        return resp_obs

    def get_obs_space(self):
        """
        Gets the observation space for each DRL model.

        :return: The DRL model's observation space.
        """
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_obs_space()

        return None

    def _ppo_action_space(self):
        action_space = spaces.Discrete(self.rl_props.super_channel_space)
        return action_space

    def get_action_space(self):
        """
        Gets the action space for the DRL model.

        :return: The DRL model's action space.
        """
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_action_space()

        return None

    def get_reward(self, was_allocated: bool):
        """
        Gets the reward for the spectrum agent.

        :param was_allocated: If the request was allocated or not.
        :return: The reward.
        :rtype: float
        """
        if self.no_penalty and not was_allocated:
            drl_reward = 0.0
        elif not was_allocated:
            drl_reward = -1.0
        else:
            drl_reward = 1.0

        return drl_reward
