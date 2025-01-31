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
        self.learn_rate = engine_props.get('learn_rate', 0.1)
        self.gamma = engine_props.get('gamma', 0.9)

        self.env = QLearningEnv(rl_props=self.rl_props, engine_props=self.engine_props, props=self.props,
                                path_levels=self.path_levels)
        self.env.net_spec_dict = None

        self.model_saver = QLearningModelSaver(engine_props=self.engine_props, props=self.props)
        self.stats = QLearningStats(props=self.props, engine_props=self.engine_props, model_saver=self.model_saver)

        self.routes_matrix = None
        self.cores_matrix = None
        self.indices = None
        self.current_q = None
        self.max_future_q = None

    def setup_matrices(self, matrix_flag: str, level_index: int, core_index: int = None):
        """
        Sets up shared attributes like routes or cores matrices and indices.

        :param matrix_flag: Defines whether to update 'routes_matrix' or 'cores_matrix'.
        :param level_index: The current level index representing the state.
        :param core_index: Index of the core, required for core updates.
        """
        if matrix_flag == 'routes_matrix':
            self.routes_matrix = self.props.routes_matrix[self.rl_props.source][self.rl_props.destination]
            self.indices = (self.rl_props.chosen_path_index, level_index)

        elif matrix_flag == 'cores_matrix':
            self.cores_matrix = self.props.cores_matrix[self.rl_props.source][self.rl_props.destination]
            self.cores_matrix = self.cores_matrix[self.rl_props.chosen_path_index]
            self.indices = (core_index, level_index)
        else:
            raise ValueError(f"Unsupported matrix_flag '{matrix_flag}'.")

    def fetch_q_value(self, action: str, flag: str):
        """
        Fetches current or maximum future Q-value.

        :param action: Specify whether to fetch "current" or "max_future".
        :param flag: Determines whether working with "path" or "core".
        :return: A structured result with the Q-value and its related metadata.
        """
        # Fetch current Q-value
        if action == 'current':
            if flag == 'path':
                path_index, level_index = self.indices
                q_value = self.routes_matrix[path_index][level_index]['q_value']
                return {"q_value": q_value, "indices": self.indices,
                        "state": {"path_index": path_index, "level_index": level_index}}

            # Return core
            core_index, level_index = self.indices
            q_value = self.cores_matrix[core_index][level_index]['q_value']
            return {"q_value": q_value, "indices": self.indices,
                    "state": {"core_index": core_index, "level_index": level_index}}

        # Fetch maximum future Q-value
        if flag == 'path':
            path_list = self.routes_matrix[self.rl_props.chosen_path_index][0][0]
        elif flag == 'core':
            core_index, level_index = self.indices
            path_list = self.cores_matrix[core_index][level_index]['path']

        new_congestion_index = self.calculate_congestion(path_list, flag)

        if flag == 'path':
            path_index, _ = self.indices
            q_value = self.routes_matrix[path_index][new_congestion_index]['q_value']
            return {"q_value": q_value, "indices": (path_index, new_congestion_index),
                    "state": {"path_index": path_index, "congestion_index": new_congestion_index}}

        core_index, _ = self.indices
        q_value = self.cores_matrix[core_index][new_congestion_index]['q_value']
        return {"q_value": q_value, "indices": (core_index, new_congestion_index),
                "state": {"core_index": core_index, "congestion_index": new_congestion_index}}

    def calculate_congestion(self, path_list, flag):
        """
        Calculates the congestion for a given path or core.

        :param path_list: The path or configuration list.
        :param flag: 'path' or 'core'.
        :return: Congestion index (classified).
        """
        if flag == 'path':
            new_congestion = find_path_cong(path_list=path_list, net_spec_dict=self.env.net_spec_dict)
        elif flag == 'core':
            core_index, _ = self.indices
            new_congestion = find_core_cong(core_index=core_index, net_spec_dict=self.env.net_spec_dict,
                                            path_list=path_list)
        else:
            raise ValueError("Flag must be 'path' or 'core'.")

        return classify_cong(curr_cong=new_congestion)

    def compute_td_error_and_new_q(self, reward: float):
        """
        Compute the temporal difference error and updated Q-value using the Bellman equation.

        :param reward: The received reward.
        :return: td_error and new_q.
        """
        delta = reward + self.gamma * self.max_future_q
        td_error = self.current_q - delta
        new_q = ((1.0 - self.learn_rate) * self.current_q) + (self.learn_rate * delta)
        return td_error, new_q

    def update_matrix(self, new_q: float, flag: str):
        """
        Update the Q-value in the matrix.

        :param new_q: The new Q-value to write into the matrix.
        :param flag: Determines whether to update 'routes_matrix' or 'cores_matrix'.
        """
        if flag == 'path':
            path_index, level_index = self.indices
            self.routes_matrix[path_index][level_index]['q_value'] = new_q
        elif flag == 'core':
            core_index, level_index = self.indices
            self.cores_matrix[core_index][level_index]['q_value'] = new_q
        else:
            raise ValueError(f"Invalid flag '{flag}'.")

    def update_routes_matrix(self, reward: float, level_index: int):
        """
        Updates the Q-table for the path/routing agent.
        """
        self.setup_matrices(matrix_flag='routes_matrix', level_index=level_index)

        current_result = self.fetch_q_value(action='current', flag='path')
        self.current_q = current_result['q_value']
        max_future_result = self.fetch_q_value(action='max_future', flag='path')
        self.max_future_q = max_future_result['q_value']
        td_error, new_q = self.compute_td_error_and_new_q(reward)

        self.update_matrix(new_q=new_q, flag='path')
        self.stats.update_q_stats(reward=reward, stats_flag='routes_dict', td_error=td_error)

    def update_cores_matrix(self, reward: float, core_index: int, level_index: int):
        """
        Updates the Q-table for the core agent.
        """
        self.setup_matrices(matrix_flag='cores_matrix', level_index=level_index, core_index=core_index)

        current_result = self.fetch_q_value(action='current', flag='core')
        self.current_q = current_result['q_value']
        max_future_result = self.fetch_q_value(action='max_future', flag='core')
        self.max_future_q = max_future_result['q_value']
        td_error, new_q = self.compute_td_error_and_new_q(reward)

        self.update_matrix(new_q=new_q, flag='core')
        self.stats.update_q_stats(reward=reward, stats_flag='cores_dict', td_error=td_error)
