import os
import numpy as np

from rl_scripts.algorithms.q_learning import QLearning
from rl_scripts.algorithms.bandits import EpsilonGreedyBandit, UCBBandit


class BaseAgent:
    """
    A base agent to be used for path, core, and spectrum agents.
    """

    def __init__(self, algorithm: str, rl_props: object, rl_help_obj: object):
        """
        Common initializer for all agents.
        """
        self.algorithm = algorithm
        self.rl_props = rl_props
        self.rl_help_obj = rl_help_obj
        self.agent_obj = None
        self.engine_props = None

    def setup_env(self, is_path: bool):
        """
        Sets up the environment for both core or path agents, depending on the algorithm.
        """
        if self.algorithm == 'q_learning':
            self.agent_obj = QLearning(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props,
                                                 is_path=is_path)
        elif self.algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=is_path)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def calculate_dynamic_penalty(self, core_index: float, req_id: float) -> float:
        """
        Calculate a dynamic penalty after every action.
        """
        return self.engine_props['penalty'] * (1 + self.engine_props['gamma'] * core_index / req_id)

    def calculate_dynamic_reward(self, core_index: float, req_id: float) -> float:
        """
        Calculates a dynamic reward after every action.
        """
        core_decay = self.engine_props['reward'] / (1 + self.engine_props['decay_factor'] * core_index)
        request_weight = ((self.engine_props['num_requests'] - req_id) /
                          self.engine_props['num_requests']) ** self.engine_props['core_beta']
        return core_decay * request_weight

    def get_reward(self, was_allocated: bool, dynamic: bool, core_index: float, req_id: float):
        """
        Generalized reward calculation for both path and core agents.
        """
        if was_allocated:
            if dynamic:
                return self.calculate_dynamic_reward(core_index, req_id)

            return self.engine_props['reward']

        if dynamic:
            return self.calculate_dynamic_penalty(core_index, req_id)

        return self.engine_props['penalty']

    def load_model(self, model_path: str, file_prefix: str, **kwargs):
        """
        Loads a previously-trained model for either a core or path agent.
        """
        self.setup_env(is_path=kwargs.get('is_path', False))
        if self.algorithm == 'q_learning':
            # Assumes similar directory logic
            model_path = os.path.join('logs', model_path,
                                      f"{file_prefix}_e{kwargs['erlang']}_c{kwargs['num_cores']}.npy")
            self.agent_obj.props.cores_matrix = np.load(model_path, allow_pickle=True)
