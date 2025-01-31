# pylint: disable=unsupported-assignment-operation

import os
import json

import networkx as nx
import numpy as np

from arg_scripts.rl_args import QProps
from helper_scripts.sim_helpers import find_path_cong, classify_cong, calc_matrix_stats, find_core_cong
from helper_scripts.os_helpers import create_dir


class QLearningEnv:
    """
    Initializes the QLearningEnv instance with required properties for the Q-learning environment.
    """

    def __init__(self, rl_props: object, engine_props: object, props: object, path_levels: int):
        self.rl_props = rl_props
        self.engine_props = engine_props
        self.props = props
        self.path_levels = path_levels

    def init_q_tables(self):
        """
        Initializes the q-tables for the agents.
        """
        for source in range(0, self.rl_props.num_nodes):
            for destination in range(0, self.rl_props.num_nodes):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = nx.shortest_simple_paths(G=self.engine_props['topology'],
                                                          source=str(source), target=str(destination), weight='length')
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.rl_props.k_paths:
                        break

                    for level_index in range(self.path_levels):
                        self.props.routes_matrix[source, destination, k, level_index] = (curr_path, 0.0)

                        for core_action in range(self.engine_props['cores_per_link']):
                            core_tuple = (curr_path, core_action, 0.0)
                            self.props.cores_matrix[source, destination, k, core_action, level_index] = core_tuple

    def setup_env(self):
        """
        Sets up the q-learning environments.
        """
        self.props.epsilon = self.engine_props['epsilon_start']
        route_types = [('path', 'O'), ('q_value', 'f8')]
        core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.props.routes_matrix = np.empty((self.rl_props.num_nodes, self.rl_props.num_nodes,
                                             self.rl_props.k_paths, self.path_levels), dtype=route_types)

        self.props.cores_matrix = np.empty((self.rl_props.num_nodes, self.rl_props.num_nodes,
                                            self.rl_props.k_paths, self.engine_props['cores_per_link'],
                                            self.path_levels), dtype=core_types)

        self.init_q_tables()


class QLearningModelSaver:
    """
    Initializes the QLearningModelSaver with necessary data for saving models.
    """

    def __init__(self, engine_props: object, props: object):
        self.engine_props = engine_props
        self.props = props

    def save_params(self, save_dir: str):
        """
        Saves parameters from the given data structures into a JSON file.

        :param save_dir: The directory to save to.
        """
        params_dict = dict()
        for param_type, params_list in self.props.save_params_dict.items():
            for key in params_list:
                if param_type == 'engine_params_list':
                    params_dict[key] = self.engine_props[key]
                else:
                    params_dict[key] = self.props.get_data(key=key)

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']
        param_fp = f"e{erlang}_params_c{cores_per_link}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    def save_model(self, path_algorithm: str, core_algorithm: str):
        """
        Saves the current q-learning model.

        :param path_algorithm: The path algorithm used.
        :param core_algorithm: The core algorithm used.
        """
        date_time = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                 self.engine_props['sim_start'])
        save_dir = os.path.join('logs', 'q_learning', date_time)
        create_dir(file_path=save_dir)

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']

        if path_algorithm == 'q_learning':
            save_fp = f"e{erlang}_routes_c{cores_per_link}.npy"
        elif core_algorithm == 'q_learning':
            save_fp = f"e{erlang}_cores_c{cores_per_link}.npy"
        else:
            raise NotImplementedError

        save_fp = os.path.join(os.getcwd(), save_dir, save_fp)
        if save_fp.split('_')[1] == 'routes':
            np.save(save_fp, self.props.routes_matrix)
        else:
            np.save(save_fp, self.props.cores_matrix)
        self.save_params(save_dir=save_dir)


class QLearningStats:  # pylint: disable=too-few-public-methods
    """
    Initializes the QLearningStats instance with required properties for tracking and updating Q-learning statistics.
    """

    def __init__(self, props: object, engine_props: object, model_saver: object):
        """
        :param props: QProps instance containing rewards and errors dictionaries.
        :param engine_props: Configuration parameters for the engine.
        :param model_saver: Instance of QLearningModelSaver for saving the model.
        """
        self.props = props
        self.engine_props = engine_props
        self.iteration = 0
        self.completed_sim = False
        self.model_saver = model_saver

    def calc_q_averages(self, stats_flag: str, episode: str):
        """
        Calculates averages for Q-values based on the accumulated rewards and errors.

        :param stats_flag: A flag to determine whether to update the path or core agent.
        :param episode: The current episode.
        """
        len_rewards = len(self.props.rewards_dict[stats_flag]['rewards'][episode])

        max_iters = self.engine_props['max_iters']
        num_requests = self.engine_props['num_requests']

        if (self.iteration in (max_iters - 1, (max_iters - 1) % 10)) and len_rewards == num_requests:
            rewards_dict = self.props.rewards_dict[stats_flag]['rewards']
            errors_dict = self.props.errors_dict[stats_flag]['errors']

            if self.iteration == (max_iters - 1):
                self.completed_sim = True
                self.props.rewards_dict[stats_flag] = calc_matrix_stats(input_dict=rewards_dict)
                self.props.errors_dict[stats_flag] = calc_matrix_stats(input_dict=errors_dict)
            else:
                self.props.rewards_dict[stats_flag]['training'] = calc_matrix_stats(input_dict=rewards_dict)
                self.props.errors_dict[stats_flag]['training'] = calc_matrix_stats(input_dict=errors_dict)

            # Delegate saving the model to QLearningModelSaver
            if not self.engine_props['is_training']:
                self.model_saver.save_model(path_algorithm=self.engine_props['path_algorithm'],
                                            core_algorithm='first_fit')
                self.model_saver.save_model(path_algorithm='first_fit',
                                            core_algorithm=self.engine_props['core_algorithm'])
            else:
                self.model_saver.save_model(path_algorithm=self.engine_props['path_algorithm'],
                                            core_algorithm=self.engine_props['core_algorithm'])

    def update_q_stats(self, reward: float, td_error: float, stats_flag: str):
        """
        Update relevant statistics for both q-learning agents.

        :param reward: The current reward.
        :param td_error: The current temporal difference error.
        :param stats_flag: A flag to determine whether to update the path or core agent.
        """
        # To account for a reset even after a sim has completed (how SB3 works)
        if self.completed_sim:
            return

        episode = str(self.iteration)
        if episode not in self.props.rewards_dict[stats_flag]['rewards'].keys():
            self.props.rewards_dict[stats_flag]['rewards'][episode] = [reward]
            self.props.errors_dict[stats_flag]['errors'][episode] = [td_error]
            self.props.sum_rewards_dict[episode] = reward
            self.props.sum_errors_dict[episode] = td_error
        else:
            self.props.rewards_dict[stats_flag]['rewards'][episode].append(reward)
            self.props.errors_dict[stats_flag]['errors'][episode].append(td_error)
            self.props.sum_rewards_dict[episode] += reward
            self.props.sum_errors_dict[episode] += td_error

        self.calc_q_averages(stats_flag=stats_flag, episode=episode)


class QLearning:
    """
    Class dedicated to handling everything related to the Q-learning algorithm.
    """

    def __init__(self, rl_props: object, engine_props: dict):
        """
        Initializes the QLearning instance and its components.

        :param rl_props: Reinforcement Learning properties object.
        :param engine_props: Configuration parameters for the Q-learning environment.
        """
        self.rl_props = rl_props
        self.engine_props = engine_props
        self.props = QProps()

        self.path_levels = engine_props['path_levels']
        self.completed_sim = False
        self.iteration = 0
        self.learn_rate = self.engine_props.get('learn_rate', 0.1)

        self.env = QLearningEnv(rl_props=self.rl_props, engine_props=self.engine_props, props=self.props,
                                path_levels=self.path_levels)

        self.model_saver = QLearningModelSaver(engine_props=self.engine_props, props=self.props)

        self.stats = QLearningStats(props=self.props, engine_props=self.engine_props, model_saver=self.model_saver)

    def get_max_future_q(self, path_list: list, net_spec_dict: dict, matrix: list, flag: str, core_index: int = None):
        """
        Gets the maximum possible future q for the next state s prime.

        :param path_list: The current path.
        :param net_spec_dict: The network spectrum database.
        :param matrix: The matrix to find the maximum current q-value in.
        :param flag: A flag to determine whether the matrix is for the path or core agent.
        :param core_index: The index of the core selected.
        :return: The maximum future q.
        :rtype: float
        """
        if flag == 'path':
            new_cong = find_path_cong(path_list=path_list, net_spec_dict=net_spec_dict)
            new_cong_index = classify_cong(curr_cong=new_cong)
            max_future_q = matrix[self.rl_props.chosen_path_index][new_cong_index]['q_value']
        elif flag == 'core':
            new_cong = find_core_cong(core_index=core_index, net_spec_dict=net_spec_dict, path_list=path_list)
            new_cong_index = classify_cong(curr_cong=new_cong)
            max_future_q = matrix[core_index][new_cong_index]['q_value']
        else:
            raise NotImplementedError

        return max_future_q

    def update_routes_matrix(self, reward: float, level_index: int, net_spec_dict: dict):
        """
        Updates the q-table for the path/routing agent.

        :param reward: The reward received from the last action.
        :param level_index: Index to determine the current state.
        :param net_spec_dict: The network spectrum database.
        """
        routes_matrix = self.props.routes_matrix[self.rl_props.source][self.rl_props.destination]
        path_list = routes_matrix[self.rl_props.chosen_path_index][level_index]
        current_q = path_list['q_value']

        max_future_q = self.get_max_future_q(path_list=routes_matrix[self.rl_props.chosen_path_index][0][0],
                                             net_spec_dict=net_spec_dict, matrix=routes_matrix, flag='path')

        delta = reward + self.engine_props['gamma'] * max_future_q
        td_error = current_q - (reward + self.engine_props['gamma'] * max_future_q)
        self.stats.update_q_stats(reward=reward, stats_flag='routes_dict', td_error=td_error)
        new_q = ((1.0 - self.learn_rate) * current_q) + (self.learn_rate * delta)

        routes_matrix = self.props.routes_matrix[self.rl_props.source][self.rl_props.destination]
        routes_matrix[self.rl_props.chosen_path_index][level_index]['q_value'] = new_q

    def update_cores_matrix(self, reward: float, core_index: int, level_index: int, net_spec_dict: dict):
        """
        Updates the q-table for the core agent.

        :param reward: The reward received from the last action.
        :param core_index: The index of the core selected.
        :param level_index: Index to determine the current state.
        :param net_spec_dict: The network spectrum database.
        """
        cores_matrix = self.props.cores_matrix[self.rl_props.source][self.rl_props.destination]
        cores_matrix = cores_matrix[self.rl_props.chosen_path_index]
        cores_list = cores_matrix[self.rl_props.core_index][level_index]
        current_q = cores_list['q_value']

        max_future_q = self.get_max_future_q(path_list=cores_list['path'], net_spec_dict=net_spec_dict,
                                             matrix=cores_matrix, flag='core', core_index=core_index)

        delta = reward + self.engine_props['gamma'] * max_future_q
        td_error = current_q - (reward + self.engine_props['gamma'] * max_future_q)
        self.stats.update_q_stats(reward=reward, stats_flag='cores_dict', td_error=td_error)
        new_q = ((1.0 - self.learn_rate) * current_q) + (self.learn_rate * delta)

        cores_matrix[core_index][level_index]['q_value'] = new_q

    def get_max_curr_q(self, cong_list: list, matrix_flag: str):
        """
        Gets the maximum current q-value from the current state s.

        :param cong_list: A list determining the congestion levels of cores or paths in the current state.
        :param matrix_flag: A flag to determine whether to update the path or core q-table.
        :return: The maximum q-value index (state) and an object
        :rtype: tuple
        """
        q_values = list()
        for obj_index, _, level_index in cong_list:
            if matrix_flag == 'routes_matrix':
                matrix = self.props.routes_matrix[self.rl_props.source][self.rl_props.destination]
                sub_flag = 'paths_list'
            elif matrix_flag == 'cores_matrix':
                matrix = self.props.cores_matrix[self.rl_props.source][self.rl_props.destination]
                matrix = matrix[self.rl_props.chosen_path_index]
                sub_flag = 'cores_list'
            else:
                raise ValueError

            curr_q = matrix[obj_index][level_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        if sub_flag == 'cores_list':
            max_obj = self.rl_props.cores_list[max_index]
        else:
            max_obj = self.rl_props.paths_list[max_index]
        return max_index, max_obj
