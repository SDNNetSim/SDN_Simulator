import os

import numpy as np

from rl_scripts.helpers.hyperparam_helpers import HyperparamConfig
from rl_scripts.algorithms.q_learning import QLearningHelpers
from rl_scripts.algorithms.bandits import EpsilonGreedyBandit, UCBBandit

from rl_scripts.args.general_args import EPISODIC_STRATEGIES

class PathAgent:
    """
    A class that handles everything related to path assignment in reinforcement learning simulations.
    """

    def __init__(self, path_algorithm: str, rl_props: object, rl_help_obj: object):
        self.path_algorithm = path_algorithm
        self.iteration = None
        self.engine_props = dict()
        self.rl_props = rl_props
        self.rl_help_obj = rl_help_obj
        self.agent_obj = None
        self.context_obj = None

        self.level_index = None
        self.cong_list = None

        self.hyperparam_obj = None
        self.state_action_pair = None
        self.action_index = None
        self.reward_penalty_list = None

    def end_iter(self):
        """
        Ends an iteration for the path agent.
        """
        self.hyperparam_obj.iteration += 1
        if self.hyperparam_obj.alpha_strategy in EPISODIC_STRATEGIES:
            if 'bandit' not in self.engine_props['path_algorithm']:
                self.hyperparam_obj.update_alpha()
        if self.hyperparam_obj.epsilon_strategy in EPISODIC_STRATEGIES:
            self.hyperparam_obj.update_eps()

    def setup_env(self):
        """
        Sets up the environment for the path agent.
        """
        self.reward_penalty_list = np.zeros(self.engine_props['max_iters'])
        self.hyperparam_obj = HyperparamConfig(engine_props=self.engine_props, rl_props=self.rl_props, is_path=True)
        if self.path_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)
            self.agent_obj.setup_env()
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True)
        else:
            raise NotImplementedError

    def get_reward(self, was_allocated: bool, path_length: int = None):  # pylint: disable=unused-argument
        """
        Get the current reward for the last agent's action.

        :param was_allocated: If the request was allocated or not.
        :param path_length: The path length.
        :return: The reward.
        :rtype: float
        """
        if was_allocated:
            return self.engine_props['reward']

        return self.engine_props['penalty']

    def _handle_hyperparams(self):
        if not self.hyperparam_obj.fully_episodic:
            self.state_action_pair = (self.rl_props.source, self.rl_props.destination)
            self.action_index = self.rl_props.chosen_path_index
            self.hyperparam_obj.update_timestep_data(state_action_pair=self.state_action_pair,
                                                     action_index=self.action_index)
        if self.hyperparam_obj.alpha_strategy not in EPISODIC_STRATEGIES:
            if 'bandit' not in self.engine_props['path_algorithm']:
                self.hyperparam_obj.update_alpha()
        if self.hyperparam_obj.epsilon_strategy not in EPISODIC_STRATEGIES:
            self.hyperparam_obj.update_eps()

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int, path_length: int):
        """
        Makes updates to the agent for each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param path_length: Length of the path.
        :param iteration: The current iteration.
        """
        if self.hyperparam_obj.iteration >= self.engine_props['max_iters']:
            return

        reward = self.get_reward(was_allocated=was_allocated, path_length=path_length)
        self.reward_penalty_list[self.hyperparam_obj.iteration] += reward
        self.hyperparam_obj.curr_reward = reward
        self.iteration = iteration
        self._handle_hyperparams()

        self.agent_obj.iteration = iteration
        if self.path_algorithm == 'q_learning':
            self.agent_obj.learn_rate = self.hyperparam_obj.curr_alpha
            self.agent_obj.update_routes_matrix(reward=reward, level_index=self.level_index,
                                                net_spec_dict=net_spec_dict)
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration)
        else:
            raise NotImplementedError

    def __ql_route(self, random_float: float):
        if random_float < self.hyperparam_obj.curr_epsilon:
            self.rl_props.chosen_path_index = np.random.choice(self.rl_props.k_paths)
            # The level will always be the last index
            self.level_index = self.cong_list[self.rl_props.chosen_path_index][-1]

            if self.rl_props.chosen_path_index == 1 and self.rl_props.k_paths == 1:
                self.rl_props.chosen_path_index = 0
            self.rl_props.chosen_path_list = self.rl_props.paths_list[self.rl_props.chosen_path_index]
        else:
            self.rl_props.chosen_path_index, self.rl_props.chosen_path_list = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list, matrix_flag='routes_matrix')
            self.level_index = self.cong_list[self.rl_props.chosen_path_index][-1]

    def _ql_route(self):
        random_float = float(np.round(np.random.uniform(0, 1), decimals=1))
        routes_matrix = self.agent_obj.props.routes_matrix
        self.rl_props.paths_list = routes_matrix[self.rl_props.source][self.rl_props.destination]['path']

        self.cong_list = self.rl_help_obj.classify_paths(paths_list=self.rl_props.paths_list)
        if self.rl_props.paths_list.ndim != 1:
            self.rl_props.paths_list = self.rl_props.paths_list[:, 0]

        self.__ql_route(random_float=random_float)

        if len(self.rl_props.chosen_path_list) == 0:
            raise ValueError('The chosen path can not be None')

    # TODO: (drl_path_agents) Change q-learning to be like this (agent_obj.something)
    def _bandit_route(self, route_obj: object):
        paths_list = route_obj.route_props.paths_matrix
        source = paths_list[0][0]
        dest = paths_list[0][-1]

        self.agent_obj.epsilon = self.hyperparam_obj.curr_epsilon
        self.rl_props.chosen_path_index = self.agent_obj.select_path_arm(source=int(source), dest=int(dest))
        self.rl_props.chosen_path_list = route_obj.route_props.paths_matrix[self.rl_props.chosen_path_index]

    def get_route(self, **kwargs):
        """
        Assign a route for the current request.
        """
        if self.path_algorithm == 'q_learning':
            self._ql_route()
        elif self.path_algorithm in ('epsilon_greedy_bandit', 'thompson_sampling_bandit', 'ucb_bandit'):
            self._bandit_route(route_obj=kwargs['route_obj'])
        else:
            raise NotImplementedError

    def load_model(self, model_path: str, erlang: float, num_cores: int):
        """
        Loads a previously trained path agent model.

        :param model_path: The path to the trained model.
        :param erlang: The Erlang value the model was trained with.
        :param num_cores: The number of cores the model was trained with.
        """
        self.setup_env()
        if self.engine_props['path_algorithm'] == 'q_learning':
            model_path = os.path.join('logs', model_path, f'e{erlang}_routes_c{num_cores}.npy')
            self.agent_obj.props.routes_matrix = np.load(model_path, allow_pickle=True)
