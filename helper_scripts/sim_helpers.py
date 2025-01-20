import copy
import os
import json
import pickle
from datetime import datetime

import networkx as nx
import numpy as np
import yaml


def get_path_mod(mods_dict: dict, path_len: int):
    """
    Choose a modulation format that will allocate a network request.

    :param mods_dict: Information for maximum reach for each modulation format.
    :param path_len: The length of the path to be taken.
    :return: The chosen modulation format.
    :rtype: str
    """
    # Pycharm auto-formats it like this for comparisons...I'd rather this look weird than look at PyCharm warnings
    if mods_dict['QPSK']['max_length'] >= path_len > mods_dict['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mods_dict['16-QAM']['max_length'] >= path_len > mods_dict['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mods_dict['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def find_max_path_len(source: int, destination: int, topology: nx.Graph):
    """
    Find the maximum path length possible of a path in the network.

    :param source: The source node.
    :param destination: The destination node.
    :param topology: The network topology.
    :return: The length of the longest path possible.
    :rtype: float
    """
    all_paths_list = list(nx.shortest_simple_paths(topology, source, destination))
    path_list = all_paths_list[-1]
    resp = find_path_len(path_list=path_list, topology=topology)

    return resp


def sort_nested_dict_vals(original_dict: dict, nested_key: str):
    """
    Sort a dictionary by a value which belongs to a nested key.

    :param original_dict: The original dictionary.
    :param nested_key: The nested key to sort by.
    :return: The sorted dictionary, ascending.
    :rtype: dict
    """
    sorted_items = sorted(original_dict.items(), key=lambda x: x[1][nested_key])
    sorted_dict = dict(sorted_items)
    return sorted_dict


def sort_dict_keys(dictionary: dict):
    """
    Sort a dictionary by keys in descending order.

    :param dictionary: The dictionary to sort.
    :return: The newly sorted dictionary.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path_list: list, topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path_list: A list of integers representing the nodes in the path.
    :param topology: The network topology.
    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path_list) - 1):
        path_len += topology[path_list[i]][path_list[i + 1]]['length']

    return path_len


# TODO: Defaulting to 'c' band
def find_path_cong(path_list: list, net_spec_dict: dict, band: str = 'c'):
    """
    Finds the average percentage of congestion for a given path.

    :param path_list: The path to be analyzed.
    :param net_spec_dict: The current up-to-date network spectrum database.
    :param band: Band to find congestion on.
    :return: The average congestion as a decimal.
    :rtype: float
    """
    # Divide by the total length of that array
    links_cong_list = list()
    for src, dest in zip(path_list, path_list[1:]):
        src_dest = (src, dest)
        cores_matrix = net_spec_dict[src_dest]['cores_matrix']
        cores_per_link = float(len(cores_matrix[band]))

        for curr_band in net_spec_dict[src_dest]['cores_matrix']:
            # Every core will have the same number of spectral slots
            total_slots = len(cores_matrix[curr_band][0])
            slots_taken = 0
            for curr_core in cores_matrix[curr_band]:
                core_slots_taken = float(len(np.where(curr_core != 0.0)[0]))
                slots_taken += core_slots_taken

            links_cong_list.append(slots_taken / (total_slots * cores_per_link))

    average_path_cong = np.mean(links_cong_list)
    return average_path_cong


def find_core_cong(core_index: int, net_spec_dict: dict, path_list: list):
    """
    Finds the current percentage of congestion on a core along a path.

    :param core_index: Index of the core.
    :param net_spec_dict: Network spectrum database.
    :param path_list: Current path.
    :return: The average congestion percentage on the core.
    :rtype: float
    """
    links_cong_list = list()
    for src, dest in zip(path_list, path_list[1:]):
        src_dest = (src, dest)
        cores_matrix = net_spec_dict[src_dest]['cores_matrix']
        total_slots = 0
        slots_taken = 0
        for band in cores_matrix:
            # Every core will have the same number of spectral slots
            total_slots += len(cores_matrix[band][0])
            core_slots_taken = float(len(np.where(cores_matrix[band][core_index] != 0.0)[0]))
            slots_taken += core_slots_taken

        links_cong_list.append(slots_taken / total_slots)

    average_core_cong = np.mean(links_cong_list)
    return average_core_cong


def find_core_frag_cong(net_spec_db: dict, path: list, core: int, band: str):
    """
    Finds the congestion and fragmentation scores for a specific request.

    :param net_spec_db: Current network spectrum database.
    :param path: Current path.
    :param core: Current core.
    :param band: Current allocated band.
    :return: The congestion and fragmentation scores.
    :rtype: float
    """
    frag_resp = 0.0
    cong_resp = 0.0
    for src, dest in zip(path, path[1:]):
        src_dest = (src, dest)
        core_arr = net_spec_db[src_dest]['cores_matrix'][band][core]

        if len(core_arr) != 256:
            raise NotImplementedError('Only works for 256 spectral slots.')

        cong_resp += len(np.where(core_arr != 0)[0])

        count = 0
        in_zero_group = False

        for number in core_arr:
            if number == 0:
                if not in_zero_group:
                    in_zero_group = True
            else:
                if in_zero_group:
                    count += 1
                    in_zero_group = False

        frag_resp += count

    num_links = len(path) - 1
    # The lowest number of slots a request can take is 2, the max number of times [1, 1, 0, 2, 2, 0, ..., 5, 5, 0]
    # fragmentation can happen is 43 for 128 spectral slots and 86 for 256 spectral slots (Rounded)
    frag_resp = frag_resp / 86.0 / num_links
    cong_resp = cong_resp / 256.0 / num_links
    return frag_resp, cong_resp


def get_channel_overlaps(free_channels_dict: dict, free_slots_dict: dict):
    """
    Find the number of overlapping and non-overlapping channels between adjacent cores.

    :param free_channels_dict: The free super-channels found on a path.
    :param free_slots_dict: The free slots found on the given path.
    :return: The overlapping and non-overlapping channels for every core.
    :rtype: dict
    """
    resp_dict = dict()
    for link in free_channels_dict.keys():  # pylint: disable=too-many-nested-blocks
        resp_dict.update({link: {'overlapped_dict': {}, 'non_over_dict': {}}})
        for band, free_channels in free_channels_dict[link].items():
            num_cores = int(len(free_channels.keys()))
            resp_dict[link]['overlapped_dict'][band] = dict()
            resp_dict[link]['non_over_dict'][band] = dict()
            for core_num, channels_list in free_channels.items():
                resp_dict[link]['overlapped_dict'][band][core_num] = list()
                resp_dict[link]['non_over_dict'][band][core_num] = list()

                for curr_channel in channels_list:
                    for sub_core, slots_dict in free_slots_dict[link][band].items():
                        if sub_core == core_num:
                            continue
                        # The final core overlaps with all other cores
                        if core_num == num_cores - 1:
                            result_arr = np.isin(curr_channel, slots_dict[sub_core])
                        else:
                            # Only certain cores neighbor each other on a fiber
                            first_neighbor = 5 if core_num == 0 else core_num - 1
                            second_neighbor = 0 if core_num == 5 else core_num + 1

                            result_arr = np.isin(curr_channel, free_slots_dict[link][band][first_neighbor])
                            result_arr = np.append(result_arr,
                                                   np.isin(curr_channel, free_slots_dict[link][band][second_neighbor]))
                            result_arr = np.append(result_arr,
                                                   np.isin(curr_channel, free_slots_dict[link][band][num_cores - 1]))

                        if result_arr is False:
                            resp_dict[link]['overlapped_dict'][band][core_num].append(curr_channel)
                            break

                    resp_dict[link]['non_over_dict'][band][core_num].append(curr_channel)

    return resp_dict


def find_free_slots(net_spec_dict: dict, link_tuple: tuple):
    """
    Find every unallocated spectral slot for a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to find the free slots on.
    :return: The indexes of the free spectral slots on the link for each core.
    :rtype: dict
    """
    resp_dict = {}
    for band in net_spec_dict[link_tuple]['cores_matrix'].keys():
        resp_dict[band] = dict()

        num_cores = len(net_spec_dict[link_tuple]['cores_matrix'][band])
        for core_num in range(num_cores):  # pylint: disable=consider-using-enumerate
            free_slots_list = np.where(net_spec_dict[link_tuple]['cores_matrix'][band][core_num] == 0)[0]
            resp_dict[band].update({core_num: free_slots_list})

    return resp_dict


def find_free_channels(net_spec_dict: dict, slots_needed: int, link_tuple: tuple):
    """
    Finds the free super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param slots_needed: The number of slots needed for the request.
    :param link_tuple: The link to search on.
    :return: Available super-channels for every core.
    :rtype: dict
    """
    resp_dict = {}
    for band in net_spec_dict[link_tuple]['cores_matrix'].keys():
        cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'][band])
        resp_dict.update({band: {}})
        for core_num, link_list in enumerate(cores_matrix):
            indexes = np.where(link_list == 0)[0]
            channels_list = []
            curr_channel_list = []

            for i, free_index in enumerate(indexes):
                if i == 0:
                    curr_channel_list.append(free_index)
                    if len(curr_channel_list) == slots_needed:
                        channels_list.append(curr_channel_list.copy())
                        curr_channel_list.pop(0)
                elif free_index == indexes[i - 1] + 1:
                    curr_channel_list.append(free_index)
                    if len(curr_channel_list) == slots_needed:
                        channels_list.append(curr_channel_list.copy())
                        curr_channel_list.pop(0)
                else:
                    curr_channel_list = [free_index]

            resp_dict[band].update({core_num: channels_list})

    return resp_dict


def find_taken_channels(net_spec_dict: dict, link_tuple: tuple):
    """
    Finds the taken super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to search on.
    :return: Unavailable super-channels for every core.
    :rtype: dict
    """
    resp_dict = {}
    for band in net_spec_dict[link_tuple]['cores_matrix'].keys():
        resp_dict.update({band: {}})
        cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'][band])
        for core_num, link_list in enumerate(cores_matrix):
            channels_list = []
            curr_channel_list = []

            for value in link_list:
                if value > 0:
                    curr_channel_list.append(value)
                elif value < 0 and curr_channel_list:
                    channels_list.append(curr_channel_list)
                    curr_channel_list = []

            if curr_channel_list:
                channels_list.append(curr_channel_list)

            resp_dict[band][core_num] = channels_list

    return resp_dict


def snake_to_title(snake_str: str):
    """
    Converts a snake string to a title string.

    :param snake_str: The string to convert in snake case.
    :return: Another string in title case.
    :rtype: str
    """
    words_list = snake_str.split('_')
    title_str = ' '.join(word.capitalize() for word in words_list)
    return title_str


def int_to_string(number: int):
    """
    Converts an integer to a string.

    :param number: The number to convert.
    :return: The original number as a string.
    :rtype: str
    """
    return '{:,}'.format(number)  # pylint: disable=consider-using-f-string


def dict_to_list(data_dict: dict, nested_key: str, path_list: list = None, find_mean: bool = False):
    """
    Creates a list from a dictionary taken values from a specified key.

    :param data_dict: The dictionary to search.
    :param nested_key: Where to take values from.
    :param path_list: If the key is nested, the path is to that nested key.
    :param find_mean: Flag to return mean or not.
    :return: A list or single value.
    :rtype: list or float
    """
    if path_list is None:
        path_list = []

    extracted_list = []
    for value_dict in data_dict.values():
        for key in path_list:
            value_dict = value_dict.get(key, {})
        nested_value = value_dict.get(nested_key)
        if nested_value is not None:
            extracted_list.append(nested_value)

    if find_mean:
        return np.mean(extracted_list)

    return np.array(extracted_list)


def list_to_title(input_list: list):
    """
    Converts a list to a title case.

    :param input_list: The input list to convert, each element is a word.
    :return: A title string.
    :rtype: str
    """
    if not input_list:
        return ""

    unique_list = list()
    for item in input_list:
        if item[0] not in unique_list:
            unique_list.append(item[0])

    if len(unique_list) > 1:
        return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

    return unique_list[0]


def calc_matrix_stats(input_dict: dict):
    """
    Creates a matrix based on dict values and takes the min, max, and average of columns.
    :param input_dict: The input dict with values as lists.
    :return: The min, max, and average of columns.
    :rtype: dict
    """
    resp_dict = dict()
    tmp_matrix = np.array([])
    for episode, curr_list in input_dict.items():
        if episode == '0':
            tmp_matrix = np.array([curr_list])
        else:
            tmp_matrix = np.vstack((tmp_matrix, curr_list))

    resp_dict['min'] = tmp_matrix.min(axis=0, initial=np.inf).tolist()
    resp_dict['max'] = tmp_matrix.max(axis=0, initial=np.inf * -1.0).tolist()
    resp_dict['average'] = tmp_matrix.mean(axis=0).tolist()

    return resp_dict


def combine_and_one_hot(arr1: np.array, arr2: np.array):
    """
    Or operation of two arrays to find overlaps.
    :param arr1: The first input array.
    :param arr2: The second input array.
    :return: The output of the or operation.
    :rtype: np.array
    """
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length.")

    one_hot_arr1 = (arr1 != 0).astype(int)
    one_hot_arr2 = (arr2 != 0).astype(int)

    result = one_hot_arr1 | one_hot_arr2
    return result


def get_start_time(sim_dict: dict):
    """
    Gets the start time of a simulation.
    :param sim_dict: Holds the simulation parameters.
    """
    sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
    sim_dict['s1']['date'] = sim_start.split('_')[0]
    tmp_list = sim_start.split('_')

    time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
    sim_dict['s1']['sim_start'] = time_string

    return sim_dict


def min_max_scale(value: float, min_value: float, max_value: float):
    """
    Scales a value with respect to a min and a max value.

    :param value: The value to be scaled.
    :param min_value: The minimum value to scale by.
    :param max_value: The maximum value to scale by.
    :return: The final scaled value.
    :rtype: float
    """
    return (value - min_value) / (max_value - min_value)


def get_super_channels(input_arr: np.array, slots_needed: int):
    """
    Gets available super-channels w.r.t. the current request's needs.

    :param input_arr: The current spectrum (a single core).
    :param slots_needed: Slots needed by the request.
    :return: A matrix of positions of available super-channels.
    :rtype: np.array
    """
    potential_super_channels = []
    consecutive_zeros = 0

    for i in range(len(input_arr)):  # pylint: disable=consider-using-enumerate
        if input_arr[i] == 0:
            consecutive_zeros += 1
            # Plus one to account for the guard band
            if consecutive_zeros >= (slots_needed + 1):
                start_position = i - slots_needed
                end_position = i

                if start_position == end_position:
                    potential_super_channels.append([start_position])
                else:
                    potential_super_channels.append([start_position, end_position])
        else:
            consecutive_zeros = 0

    return np.array(potential_super_channels)


# TODO: (drl_path_agents) Add reference
# Please refer to this paper for the formulation:
def _get_hfrag_score(sc_index_mat: np.array, spectral_slots: int):
    big_n = len(sc_index_mat) * -1.0
    if big_n == 0.0:
        return np.inf

    channel_len = len(sc_index_mat[0])
    resp_score = big_n * (channel_len / spectral_slots) * np.log((channel_len / spectral_slots))
    return resp_score


def get_hfrag(path_list: list, core_num: int, band: str, slots_needed: int, spectral_slots: int, net_spec_dict: dict):
    """
    Gets the shannon entropy fragmentation scores for allocating a request.

    :param path_list: The current path.
    :param core_num: The core number.
    :param band: Current allocated band.
    :param slots_needed: The slots needed by the request.
    :param spectral_slots: The number of spectral slots on a single core.
    :param net_spec_dict: The up-to-date network spectrum database.
    :return: An array with all shannon entropy fragmentation scores.
    """
    path_alloc_arr = np.zeros(spectral_slots)
    resp_frag_arr = np.ones(spectral_slots)
    # TODO: (drl_path_agents) First fit for core, only use in testing
    if core_num is None:
        core_num = 0

    for source, dest in zip(path_list, path_list[1:]):
        core_arr = net_spec_dict[(source, dest)]['cores_matrix'][band][core_num]
        path_alloc_arr = combine_and_one_hot(path_alloc_arr, core_arr)

    sc_index_mat = get_super_channels(input_arr=path_alloc_arr, slots_needed=slots_needed)
    hfrag_before = _get_hfrag_score(sc_index_mat=sc_index_mat, spectral_slots=spectral_slots)
    for super_channel in sc_index_mat:
        mock_alloc_arr = copy.deepcopy(path_alloc_arr)
        for index in super_channel:
            mock_alloc_arr[index] = 1

        tmp_sc_mat = get_super_channels(input_arr=mock_alloc_arr, slots_needed=slots_needed)
        hfrag_after = _get_hfrag_score(sc_index_mat=tmp_sc_mat, spectral_slots=spectral_slots)
        delta_hfrag = hfrag_before - hfrag_after
        start_index = super_channel[0]
        resp_frag_arr[start_index] = np.round(delta_hfrag, 3)

    resp_frag_arr = np.where(resp_frag_arr == 1, np.inf, resp_frag_arr)

    return sc_index_mat, resp_frag_arr


def classify_cong(curr_cong: float):
    """
    Classifies congestion percentages to 'levels'.

    :param curr_cong: Current congestion percentage.
    :return: The congestion indexes or level.
    :rtype: int
    """
    if curr_cong <= 0.3:
        cong_index = 0
    else:
        cong_index = 1

    return cong_index


def parse_yaml_file(yaml_file: str):
    """
    Parses a YAML file.

    :param yaml_file: YAML file name.
    :return: The YAML data as an object.
    :rtype: object
    """
    with open(yaml_file, "r", encoding='utf-8') as file_obj:
        try:
            yaml_data = yaml.safe_load(file_obj)
            return yaml_data
        except yaml.YAMLError as exc:
            return exc


def get_arrival_rates(arrival_dict: dict):
    """
    Generate a list of arrival rates based on the configuration dictionary.

    :param arrival_dict: The configuration dictionary containing values for generating the arrival rates.
    :return: A list of arrival rates generated from the configuration.
    :rtype: list
    """
    start = int(arrival_dict['start'])
    stop = int(arrival_dict['stop'])
    step = int(arrival_dict['step'])

    return list(range(start, stop + 1, step))


def run_simulation_for_arrival_rates(env, arrival_list: list, run_func):
    """
    Run the simulation for each arrival rate in the given list.

    :param env: The simulation environment instance.
    :param arrival_list: A list of arrival rates to simulate.
    :param run_func: The function to run a simulation.
    :return: The mean of total rewards from all simulations.
    :rtype: float
    """
    total_rewards = []
    for arrival_rate in arrival_list:
        env.engine_obj.engine_props['erlang'] = arrival_rate / env.sim_dict['holding_time']
        env.engine_obj.engine_props['arrival_rate'] = arrival_rate * env.sim_dict['cores_per_link']
        run_func(env=env, sim_dict=env.sim_dict)
        sum_returns = np.sum(env.path_agent.reward_penalty_list)
        total_rewards.append(sum_returns)

    return np.mean(total_rewards)


def save_study_results(study, env, study_name: str, best_params: dict, best_reward: float, start_time: str):
    """
    Save the results of the study, including the best hyperparameters and the best reward value.

    :param study: The Optuna study object containing the results.
    :param env: The simulation environment instance.
    :param study_name: The name of the study file to save.
    :param best_params: The best hyperparameters found by Optuna.
    :param best_reward: The best reward value from the study.
    :param start_time: Start time of simulation run.
    """
    date_time = os.path.join(env.engine_obj.engine_props['network'], env.engine_obj.engine_props['date'],
                             env.engine_obj.engine_props['sim_start'])
    save_dir = os.path.join('logs', env.engine_obj.engine_props['path_algorithm'], date_time)
    os.makedirs(save_dir, exist_ok=True)

    save_fp = os.path.join(save_dir, study_name)
    with open(save_fp, 'wb') as file_path:
        pickle.dump(study, file_path)

    save_fp = os.path.join(save_dir, 'best_hyperparams.txt')
    with open(save_fp, 'w', encoding='utf-8') as file_path:
        file_path.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            file_path.write(f"{key}: {value}\n")
        file_path.write(f"\nBest Trial Reward: {best_reward}\n")
        file_path.write(f"\nBest Trial Start Time: {start_time}\n")


# TODO: Only support for one process
def modify_multiple_json_values(file_path: str, update_list: list):
    """
    Opens a JSON file, modifies multiple key-value pairs in a dictionary, and saves it back to the file.

    :param file_path: The path to the JSON file.
    :param update_list: A list of tuples containing keys and their new values to be updated.
                        Example: [('key1', 'new_value1'), ('key2', 'new_value2')]
    """
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for key, new_value in update_list:
        if key in data:
            data[key] = new_value
        else:
            raise KeyError(f"Key '{key}' not found in the JSON file.")

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
