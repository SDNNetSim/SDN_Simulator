import os
import json

import numpy as np

from helper_scripts.os_helpers import create_dir
from rl_scripts.algorithms.algorithm_props import BanditProps


def load_model(train_fp: str):
    """
    Loads a pre-trained bandit model.

    :param train_fp: File path the model has been saved on.
    :return: The state-value functions V(s, a)
    :rtype: dict
    """
    train_fp = os.path.join('logs', train_fp)
    with open(train_fp, 'r', encoding='utf-8') as file_obj:
        state_vals_dict = json.load(file_obj)

    return state_vals_dict


def _get_base_fp(is_path: bool, erlang: float, cores_per_link: int):
    if is_path:
        base_fp = f"e{erlang}_routes_c{cores_per_link}"
    else:
        base_fp = f"e{erlang}_cores_c{cores_per_link}"

    return base_fp


def _save_model(state_values_dict: dict, erlang: float, cores_per_link: int, save_dir: str, is_path: bool,
                trial: int):
    if state_values_dict is None:
        return
    # Convert tuples to strings and arrays to lists for JSON format
    state_values_dict = {str(key): value.tolist() for key, value in state_values_dict.items()}

    if is_path:
        state_vals_fp = f"state_vals_e{erlang}_routes_c{cores_per_link}_t{trial + 1}.json"
    else:
        raise NotImplementedError
    save_fp = os.path.join(os.getcwd(), save_dir, state_vals_fp)
    with open(save_fp, 'w', encoding='utf-8') as file_obj:
        json.dump(state_values_dict, file_obj)


def save_model(iteration: int, algorithm: str, self: object, trial: int):
    """
    Saves a trained bandit model.

    :param iteration: Current iteration.
    :param algorithm: The algorithm used.
    :param trial: Current trial number.
    :param self: The object to be saved.
    """
    max_iters = self.engine_props['max_iters']
    rewards_matrix = self.props.rewards_matrix

    if (iteration % self.engine_props['save_step'] == 0 or iteration == max_iters - 1) and \
            len(self.props.rewards_matrix[iteration]) == self.engine_props['num_requests']:
        rewards_matrix = np.array(rewards_matrix)
        rewards_arr = rewards_matrix.mean(axis=0)

        date_time = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                 self.engine_props['sim_start'])
        save_dir = os.path.join('logs', algorithm, date_time)
        create_dir(file_path=save_dir)

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']
        base_fp = _get_base_fp(is_path=self.is_path, erlang=erlang, cores_per_link=cores_per_link)

        rewards_fp = f'rewards_{base_fp}_t{trial + 1}_iter_{iteration}.npy'
        save_fp = os.path.join(os.getcwd(), save_dir, rewards_fp)
        np.save(save_fp, rewards_arr)

        _save_model(state_values_dict=self.values, erlang=erlang, cores_per_link=cores_per_link,
                    save_dir=save_dir, is_path=self.is_path, trial=trial)


def get_q_table(self: object):
    """
    Constructs the q-table.

    :param self: The current bandit object.
    :return: The initial V(s, a) and N(s, a) values.
    :rtype: tuple
    """
    self.counts = {}
    self.values = {}
    for source in range(self.num_nodes):
        for destination in range(self.num_nodes):
            if source == destination:
                continue

            if self.is_path:
                self.counts[(source, destination)] = np.zeros(self.n_arms)
                self.values[(source, destination)] = np.zeros(self.n_arms)
            else:
                for path_index in range(self.engine_props['k_paths']):
                    self.counts[(source, destination, path_index)] = np.zeros(self.n_arms)
                    self.values[(source, destination, path_index)] = np.zeros(self.n_arms)

    return self.counts, self.values


def _update_bandit(self: object, iteration: int, reward: float, arm: int, algorithm: str,
                   trial: int):
    if self.is_path:
        pair = (self.source, self.dest)
    else:
        pair = (self.source, self.dest, self.path_index)

    self.counts[pair][arm] += 1
    n_times = self.counts[pair][arm]
    value = self.values[pair][arm]
    self.values[pair][arm] = value + (reward - value) / n_times

    if self.iteration >= len(self.props.rewards_matrix):
        self.props.rewards_matrix.append([])
    self.props.rewards_matrix[self.iteration].append(reward)

    # Check if we need to save the model
    save_model(iteration=iteration, algorithm=algorithm, self=self, trial=trial)


class EpsilonGreedyBandit:
    """
    Epsilon greedy bandit algorithm, considering N(s, a) using counts to update state-action values Q(s, a).
    """

    def __init__(self, rl_props: object, engine_props: dict, is_path: bool):
        self.props = BanditProps()
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.source = None
        self.dest = None
        self.path_index = None  # Index of the last chosen path

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.epsilon = None
        self.num_nodes = rl_props.num_nodes
        self.counts, self.values = get_q_table(self=self)  # Amount of times an action has been taken and every V(s,a)

    def _get_action(self, state_action_pair: tuple):
        # TODO: (drl_path_agents) Epsilon greedy bandit has not been updated
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)

        return np.argmax(self.values[state_action_pair])

    def select_path_arm(self, source: int, dest: int):
        """
        Selects a bandit's arm for the path agent only.

        :param source: Source node.
        :param dest: Destination node.
        :return: The action selected.
        :rtype: np.float64
        """
        self.source = source
        self.dest = dest
        state_action_pair = (source, dest)
        return self._get_action(state_action_pair=state_action_pair)

    def select_core_arm(self, source: int, dest: int, path_index: int):
        """
        Selects a bandit's arm for the core agent only.

        :param source: Source node.
        :param dest: Destination node.
        :param path_index: Path index selected prior.
        :return: The action selected.
        :rtype: np.float64
        """
        self.source = source
        self.dest = dest
        self.path_index = path_index
        state_action_pair = (source, dest, path_index)
        return self._get_action(state_action_pair=state_action_pair)

    def update(self, arm: int, reward: int, iteration: int, trial: int):
        """
        Make updates to the bandit after each time step or episode.

        :param arm: The arm selected.
        :param reward: Reward received from R(s, a).
        :param iteration: Current episode or iteration.:
        :param trial: Current trial number.
        """
        _update_bandit(self=self, iteration=iteration, reward=reward, arm=arm, algorithm='epsilon_greedy_bandit',
                       trial=trial)


class UCBBandit:
    """
    Upper Confidence Bound bandit algorithm, considering N(s, a) using counts to update state-action values Q(s, a).
    """

    def __init__(self, rl_props: object, engine_props: dict, is_path: bool):
        self.props = BanditProps()
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.path_index = None
        self.source = None
        self.dest = None
        self.num_nodes = rl_props.num_nodes

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.counts, self.values = get_q_table(self=self)

    def _get_action(self, state_action_pair: tuple):
        if 0 in self.counts[state_action_pair]:
            return np.argmin(self.counts[state_action_pair])

        conf_param = self.engine_props['conf_param']
        total_counts = sum(self.counts[state_action_pair])
        ucb_values = self.values[state_action_pair] + \
                     np.sqrt(conf_param * np.log(total_counts) / self.counts[state_action_pair])
        return np.argmax(ucb_values)

    def select_path_arm(self, source: int, dest: int):
        """
        Selects a bandit's arm for the path agent only.

        :param source: Source node.
        :param dest: Destination node.
        :return: The action selected.
        :rtype: np.float64
        """
        self.source = source
        self.dest = dest
        state_action_pair = (source, dest)

        return self._get_action(state_action_pair=state_action_pair)

    def select_core_arm(self, source: int, dest: int, path_index: int):
        """
        Selects a bandit's arm for the core agent only.

        :param source: Source node.
        :param dest: Destination node.
        :param path_index: Path index selected prior.
        :return: The action selected.
        :rtype: np.float64
        """
        self.source = source
        self.dest = dest
        self.path_index = path_index
        state_action_pair = (source, dest, path_index)

        return self._get_action(state_action_pair=state_action_pair)

    def update(self, arm: int, reward: int, iteration: int, trial: int):
        """
        Make updates to the bandit after each time step or episode.

        :param arm: The arm selected.
        :param reward: Reward received from R(s, a).
        :param iteration: Current episode or iteration.
        :param trial: Current trial number.
        """
        _update_bandit(iteration=iteration, arm=arm, reward=reward, self=self, algorithm='ucb_bandit',
                       trial=trial)
