from .ql_helpers import QLearningHelpers


class PathAgent:
    def __init__(self, path_algorithm: str):
        self.path_algorithm = path_algorithm

        self.agent_obj = None

    def setup_env(self):
        if self.path_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers()
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    # TODO: These will move to multi-agent helpers script
    def _ql_route(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        routes_matrix = self.q_props['routes_matrix']
        # TODO: May need to modify this
        self.rl_props['paths_list'] = routes_matrix[self.rl_props['source']][self.rl_props['destination']]['path']

        self.paths_obj = self.helper_obj.classify_paths(paths_list=self.rl_props['paths_list'])
        if self.rl_props['paths_list'].ndim != 1:
            self.rl_props['paths_list'] = self.rl_props['paths_list'][:, 0]

        if random_float < self.q_props['epsilon']:
            self.rl_props['path_index'] = np.random.choice(self.rl_props['k_paths'])
            # The level will always be the last index
            self.level_index = self.paths_obj[self.rl_props['path_index']][-1]

            if self.rl_props['path_index'] == 1 and self.rl_props['k_paths'] == 1:
                self.rl_props['path_index'] = 0
            self.rl_props['chosen_path'] = self.rl_props['paths_list'][self.rl_props['path_index']]
        else:
            self.rl_props['path_index'], self.rl_props['chosen_path'] = self.helper_obj.get_max_curr_q(
                paths_info=self.paths_obj)
            self.level_index = self.paths_obj[self.rl_props['path_index']][-1]

        if len(self.rl_props['chosen_path']) == 0:
            raise ValueError('The chosen path can not be None')

        try:
            req_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count']]
        except:
            req_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count'] - 1]
        self.helper_obj.update_route_props(bandwidth=req_dict['bandwidth'], chosen_path=self.rl_props['chosen_path'])

    # TODO: To call the above function
    def get_route(self):
        raise NotImplementedError


class CoreAgent:
    def __init__(self):
        raise NotImplementedError

    def setup_env(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    # TODO: Here or in the ql_agent script?
    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        if random_float < self.q_props['epsilon']:
            self.rl_props['core_index'] = np.random.randint(0, self.engine_obj.engine_props['cores_per_link'])
        else:
            cores_matrix = self.q_props['cores_matrix'][self.rl_props['source']][self.rl_props['destination']]
            q_values = cores_matrix[self.rl_props['path_index']]['q_value']
            self.rl_props['core_index'] = np.argmax(q_values)

    # TODO: To call the above function
    def get_core(self):
        raise NotImplementedError


class SpectrumAgent:
    def __init__(self):
        raise NotImplementedError

    def _ppo_obs_space(self):
        """
        Gets the observation space for the DRL agent.

        :return: The observation space.
        :rtype: spaces.Dict
        """
        self.find_maximums()
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(self.drl_props['max_slots_needed'] + 1),
            'source': spaces.MultiBinary(self.ai_props['num_nodes']),
            'destination': spaces.MultiBinary(self.ai_props['num_nodes']),
            # TODO: Change
            'super_channels': spaces.Box(-0.01, 100.0, shape=(3,), dtype=np.float32)
        })

        return resp_obs

    # TODO: Change
    @staticmethod
    def _ppo_action_space(super_channel_space: int = 3):
        """
        Gets the action space for the DRL agent.

        :param super_channel_space: The number of 'J' super-channels that can be selected.
        :return: The action space.
        :rtype: spaces.Discrete
        """
        action_space = spaces.Discrete(super_channel_space)
        return action_space

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    @staticmethod
    def _calc_deep_reward(was_allocated: bool):
        if was_allocated:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def calculate_drl_reward(self, was_allocated: bool):
        """
        Gets the reward for the deep reinforcement learning agent.

        :param was_allocated: Determines if the request was successfully allocated or not.
        :return: The reward.
        :rtype: float
        """
        if self.no_penalty:
            drl_reward = 0.0
        else:
            drl_reward = self._calc_deep_reward(was_allocated=was_allocated)

        return drl_reward

    def get_reward(self):
        raise NotImplementedError

    def _ppo_spectrum(self):
        """
        Returns the spectrum as a binary array along a path.
        A one indicates that channel is taken along one or multiple of the links, a zero indicates that the channel
        is free along every link in the path.

        :return: The binary array of current path occupation.
        :rtype: list
        """
        resp_spec_arr = np.zeros(self.engine_obj.engine_props['spectral_slots'])
        path_list = self.ai_props['paths_list'][self.ai_props['path_index']]
        core_index = self.ai_props['core_index']
        net_spec_dict = self.engine_obj.net_spec_dict
        for source, dest in zip(path_list, path_list[1:]):
            core_arr = net_spec_dict[(source, dest)]['cores_matrix'][core_index]
            resp_spec_arr = combine_and_one_hot(resp_spec_arr, core_arr)

        return resp_spec_arr

    def get_spectrum(self):
        raise NotImplementedError
