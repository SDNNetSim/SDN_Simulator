import numpy as np

from rl_scripts.args.general_args import VALID_DRL_ALGORITHMS
from rl_scripts.agents.base_agent import BaseAgent

from rl_scripts.args.general_args import EPISODIC_STRATEGIES


class PathAgent(BaseAgent):
    """
    A class that handles everything related to path assignment in reinforcement learning simulations.
    """

    def __init__(self, path_algorithm: str, rl_props: object, rl_help_obj: object):
        super().__init__(path_algorithm, rl_props, rl_help_obj)

        self.iteration = None
        self.hyperparam_obj = None
        self.reward_penalty_list = None
        self.level_index = None
        self.cong_list = None

        self.state_action_pair = None
        self.action_index = None

    def end_iter(self):
        """
        Ends an iteration for the path agent.
        """
        self.hyperparam_obj.iteration += 1
        if self.hyperparam_obj.alpha_strategy in EPISODIC_STRATEGIES:
            if 'bandit' not in self.engine_props['path_algorithm']:
                self.hyperparam_obj.update_alpha()
        if self.hyperparam_obj.epsilon_strategy in EPISODIC_STRATEGIES:
            if 'ucb' not in self.engine_props['path_algorithm']:
                self.hyperparam_obj.update_eps()

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

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int,
               path_length: int, trial: int):  # pylint: disable=unused-argument
        """
        Makes updates to the agent for each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param path_length: Length of the path.
        :param trial: The current trial.
        :param iteration: The current iteration.
        """
        # StableBaselines3 will handle all algorithm updates
        if self.algorithm in VALID_DRL_ALGORITHMS:
            return

        if self.hyperparam_obj.iteration >= self.engine_props['max_iters']:
            raise ValueError

        reward = self.get_reward(was_allocated=was_allocated, dynamic=self.engine_props['dynamic_reward'],
                                 core_index=None, req_id=None)
        self.reward_penalty_list[self.hyperparam_obj.iteration] += reward
        self.hyperparam_obj.curr_reward = reward
        self.iteration = iteration
        self.algorithm_obj.learn_rate = self.hyperparam_obj.curr_alpha

        self._handle_hyperparams()

        self.algorithm_obj.iteration = iteration
        if self.algorithm == 'q_learning':
            self.algorithm_obj.learn_rate = self.hyperparam_obj.curr_alpha
            self.algorithm_obj.update_q_matrix(reward=reward, level_index=self.level_index, net_spec_dict=net_spec_dict,
                                               flag='path', trial=trial)
        elif self.algorithm == 'epsilon_greedy_bandit':
            self.algorithm_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration,
                                      trial=trial)
        elif self.algorithm == 'ucb_bandit':
            self.algorithm_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration,
                                      trial=trial)
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
            self.rl_props.chosen_path_index, self.rl_props.chosen_path_list = self.algorithm_obj.get_max_curr_q(
                cong_list=self.cong_list, matrix_flag="routes_matrix")
            self.level_index = self.cong_list[self.rl_props.chosen_path_index][-1]

    def _ql_route(self):
        random_float = float(np.round(np.random.uniform(0, 1), decimals=1))
        routes_matrix = self.algorithm_obj.props.routes_matrix[self.rl_props.source, self.rl_props.destination]['path']
        self.rl_props.paths_list = routes_matrix

        self.cong_list = self.rl_help_obj.classify_paths(paths_list=self.rl_props.paths_list)
        if self.rl_props.paths_list.ndim != 1:
            self.rl_props.paths_list = self.rl_props.paths_list[:, 0]

        self.__ql_route(random_float=random_float)

        if len(self.rl_props.chosen_path_list) == 0:
            raise ValueError('The chosen path can not be None')

    def _bandit_route(self, route_obj: object):
        paths_list = route_obj.route_props.paths_matrix
        source = paths_list[0][0]
        dest = paths_list[0][-1]

        self.algorithm_obj.epsilon = self.hyperparam_obj.curr_epsilon
        self.rl_props.chosen_path_index = self.algorithm_obj.select_path_arm(source=int(source), dest=int(dest))
        self.rl_props.chosen_path_list = route_obj.route_props.paths_matrix[self.rl_props.chosen_path_index]

    def _drl_route(self, route_obj: object, action: int):
        if self.algorithm in ('ppo', 'a2c'):
            self.rl_props.chosen_path_index = action
            self.rl_props.chosen_path_list = route_obj.route_props.paths_matrix[action]
        else:
            raise NotImplementedError

    def get_route(self, **kwargs):
        """
        Assign a route for the current request.
        """
        if self.algorithm == 'q_learning':
            self._ql_route()
        elif self.algorithm in ('epsilon_greedy_bandit', 'thompson_sampling_bandit', 'ucb_bandit'):
            self._bandit_route(route_obj=kwargs['route_obj'])
        elif self.algorithm in ('ppo', 'a2c'):
            self._drl_route(route_obj=kwargs['route_obj'], action=kwargs['action'])
        else:
            raise NotImplementedError
