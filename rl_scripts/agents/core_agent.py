import os

import numpy as np

from rl_scripts.algorithms.q_learning import QLearning
from rl_scripts.algorithms.bandits import EpsilonGreedyBandit, UCBBandit


# TODO: (drl_path_agents) This class is no longer supported
class CoreAgent:
    """
    A class that handles everything related to core assignment in reinforcement learning simulations.
    """

    def __init__(self, core_algorithm: str, rl_props: object, rl_help_obj: object):
        self.core_algorithm = core_algorithm
        self.rl_props = rl_props
        self.engine_props = None
        self.agent_obj = None
        self.rl_help_obj = rl_help_obj

        self.level_index = None
        self.cong_list = list()
        self.no_penalty = False
        self.ramp_up = False

    def end_iter(self):
        """
        Ends an iteration for the core agent.
        """
        raise NotImplementedError

    def setup_env(self):
        """
        Sets up the environment for the core agent.
        """
        if self.core_algorithm == 'q_learning':
            self.agent_obj = QLearning(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False)
        elif self.core_algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def calculate_dynamic_penalty(self, core_index: float, req_id: float):
        """
        Calculate a dynamic penalty after every action.

        :param core_index: Core chosen.
        :param req_id: Current request ID.
        :return: The penalty that's calculated.
        :rtype: float
        """
        return self.engine_props['penalty'] * (1 + self.engine_props['gamma'] * core_index / req_id)

    def calculate_dynamic_reward(self, core_index: float, req_id: float):
        """
        Calculates a dynamic reward after every action.

        :param core_index: Core chosen.
        :param req_id: Current request ID.
        :return: The reward that's calculated.
        :rtype: float
        """
        core_decay = self.engine_props['reward'] / (1 + self.engine_props['decay_factor'] * core_index)
        request_weight = ((self.engine_props['num_requests'] - req_id) /
                          self.engine_props['num_requests']) ** self.engine_props['core_beta']

        return core_decay * request_weight

    def get_reward(self, was_allocated: bool):
        """
        Gets the core agent's reward based on the last action taken.

        :param was_allocated: If the last request was allocated.
        :return: The reward.
        :rtype: float
        """
        req_id = float(self.rl_help_obj.route_obj.sdn_props['req_id'])
        core_index = self.rl_props.core_index

        if was_allocated:
            if self.engine_props['dynamic_reward']:
                reward = self.calculate_dynamic_reward(core_index, req_id)
            else:
                reward = self.engine_props['reward']
            return reward

        if self.engine_props['dynamic_reward']:
            penalty = self.calculate_dynamic_penalty(core_index, req_id)
        else:
            penalty = self.engine_props['penalty']
        return penalty

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int):
        """
        Makes updates to the core agent after each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param iteration: The current iteration
        """
        reward = self.get_reward(was_allocated=was_allocated)

        self.agent_obj.iteration = iteration
        if self.core_algorithm == 'q_learning':
            print('Core Index:', self.rl_props.core_index)
            self.agent_obj.update_cores_matrix(reward=reward, level_index=self.level_index,
                                               net_spec_dict=net_spec_dict, core_index=self.rl_props.core_index)
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.core_index, iteration=iteration)
        elif self.core_algorithm == 'ucb_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.core_index, iteration=iteration)
        else:
            raise NotImplementedError

    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        cores_matrix = self.agent_obj.props.cores_matrix
        cores_matrix = cores_matrix[self.rl_props.source][self.rl_props.destination]
        self.rl_props.cores_list = cores_matrix[self.rl_props.chosen_path_index]
        self.cong_list = self.rl_help_obj.classify_cores(cores_list=self.rl_props.cores_list)

        if random_float < self.agent_obj.props.epsilon:
            self.rl_props.core_index = np.random.randint(0, self.engine_props['cores_per_link'])
            self.level_index = self.cong_list[self.rl_props.core_index][-1]
        else:
            self.rl_props.core_index, self.rl_props.chosen_core = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list,
                matrix_flag='cores_matrix')
            self.level_index = self.cong_list[self.rl_props.core_index][-1]

    def _bandit_core(self, path_index: int, source: str, dest: str):
        self.rl_props.core_index = self.agent_obj.select_core_arm(source=int(source), dest=int(dest),
                                                                  path_index=path_index)

    def get_core(self):
        """
        Assigns a core to the current request.
        """
        if self.core_algorithm == 'q_learning':
            self._ql_core()
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self._bandit_core(path_index=self.rl_props.chosen_path_index, source=self.rl_props.chosen_path_list[0][0],
                              dest=self.rl_props.chosen_path_list[0][-1])
        elif self.core_algorithm == 'ucb_bandit':
            self._bandit_core(path_index=self.rl_props.chosen_path_index, source=self.rl_props.chosen_path_list[0][0],
                              dest=self.rl_props.chosen_path_list[0][-1])
        else:
            raise NotImplementedError

    def load_model(self, model_path: str, erlang: float, num_cores: int):
        """
        Loads a previously trained core agent model.

        :param model_path: The path to the core agent model.
        :param erlang: The Erlang value the model was trained on.
        :param num_cores: The number of cores the model was trained on.
        """
        self.setup_env()
        if self.core_algorithm == 'q_learning':
            model_path = os.path.join('logs', model_path, f'e{erlang}_cores_c{num_cores}.npy')
            self.agent_obj.props.cores_matrix = np.load(model_path, allow_pickle=True)
