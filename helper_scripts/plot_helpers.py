import os
import json
from statistics import mean

import numpy as np

from helper_scripts.sim_helpers import dict_to_list, list_to_title
from arg_scripts.plot_args import PlotArgs, PlotProps


class PlotHelpers:  # pylint: disable=too-few-public-methods
    """
    A class to assist with various tasks related when plotting statistics.
    """

    def __init__(self, plot_props: PlotProps, net_names_list: list):
        self.plot_props = plot_props

        self.plot_props.title_names = list_to_title(input_list=net_names_list)
        self.file_info = None
        self.erlang_dict = None
        self.erlang = None
        self.time = None
        self.sim_num = None
        self.data_dict = None

    # TODO: (drl_path_agents) This function is only partially functional for q-learning
    def _find_ai_stats(self, cores_per_link: int):
        # TODO: Generalize
        # TODO: Save the date of a simulation
        ai_fp = os.path.join('..', 'logs', 'ql', self.data_dict['network'], self.data_dict['date'], self.time)
        ai_fp = os.path.join(ai_fp, f"e{self.erlang}_params_c{cores_per_link}.json")

        ai_dict = self._read_json_file(file_path=ai_fp)
        for ai_key in ('learn_rate', 'discount_factor', 'epsilon_start', 'sum_rewards_dict', 'sum_errors_dict'):
            if ai_key in ('sum_rewards_dict', 'sum_errors_dict'):
                label_list = ai_key.split('_')
                label = f"{label_list[0]}_{label_list[1]}_list"
                self.plot_props.plot_dict[self.time][self.sim_num][label].append(list(ai_dict[ai_key].values()))
            else:
                self.plot_props.plot_dict[self.time][self.sim_num][ai_key] = ai_dict[ai_key]

    def _find_misc_stats(self):
        lengths_list = dict_to_list(self.erlang_dict['iter_stats'], 'lengths_mean')
        hops_list = dict_to_list(self.erlang_dict['iter_stats'], 'hops_mean')
        times_list = dict_to_list(self.erlang_dict['iter_stats'], 'route_times_mean') * 10 ** 3

        cong_list = dict_to_list(self.erlang_dict['iter_stats'], 'congestion', ['block_reasons_dict'])
        dist_list = dict_to_list(self.erlang_dict['iter_stats'], 'distance', ['block_reasons_dict'])

        average_length = np.nanmean(lengths_list) if len(lengths_list) > 0 else 0
        average_hop = np.nanmean(hops_list) if len(hops_list) > 0 else 0
        average_time = np.nanmean(times_list) if len(times_list) > 0 else 0
        average_cong = np.nanmean(cong_list) if len(cong_list) > 0 else 0
        average_dist = np.nanmean(dist_list) if len(dist_list) > 0 else 0

        self.plot_props.plot_dict[self.time][self.sim_num].lengths_list.append(average_length)
        self.plot_props.plot_dict[self.time][self.sim_num].hops_list.append(average_hop)
        self.plot_props.plot_dict[self.time][self.sim_num].times_list.append(average_time)
        self.plot_props.plot_dict[self.time][self.sim_num].cong_block_list.append(average_cong)
        self.plot_props.plot_dict[self.time][self.sim_num].dist_block_list.append(average_dist)

    @staticmethod
    def _dict_to_np_array(snap_val_list: list, key: str):
        return np.nan_to_num([np.nan if d.get(key) is None else d.get(key) for d in snap_val_list])

    def _process_snapshots(self, snap_val_list: list):
        active_req_list = self._dict_to_np_array(snap_val_list=snap_val_list, key='active_requests')
        block_req_list = self._dict_to_np_array(snap_val_list=snap_val_list, key='blocking_prob')
        occ_slot_list = self._dict_to_np_array(snap_val_list=snap_val_list, key='occ_slots')

        return active_req_list, block_req_list, occ_slot_list

    def _find_snapshot_usage(self):
        req_num_list, active_req_matrix, block_req_matrix, occ_slot_matrix = [], [], [], []
        for _, stats_dict in self.erlang_dict['iter_stats'].items():
            snapshots_dict = stats_dict['snapshots_dict']
            req_num_list = [int(req_num) for req_num in snapshots_dict.keys()]

            snap_val_list = list(snapshots_dict.values())
            active_req_list, block_req_list, occ_slot_list = self._process_snapshots(snap_val_list=snap_val_list)
            active_req_matrix.append(active_req_list)
            block_req_matrix.append(block_req_list)
            occ_slot_matrix.append(occ_slot_list)

        self.plot_props.plot_dict[self.time][self.sim_num].req_num_list = req_num_list
        self.plot_props.plot_dict[self.time][self.sim_num].active_req_matrix = np.mean(active_req_matrix, axis=0)
        self.plot_props.plot_dict[self.time][self.sim_num].block_req_matrix = np.mean(block_req_matrix, axis=0)
        self.plot_props.plot_dict[self.time][self.sim_num].occ_slot_matrix = np.mean(occ_slot_matrix, axis=0)

    def _find_mod_info(self):
        mods_used_dict = self.erlang_dict['iter_stats']['0']['mods_used_dict']
        for bandwidth, mod_dict in mods_used_dict.items():
            for modulation in mod_dict:
                filters_list = ['mods_used_dict', bandwidth]
                mod_usages = dict_to_list(data_dict=self.erlang_dict['iter_stats'], nested_key=modulation,
                                          path_list=filters_list)

                modulations_dict = self.plot_props.plot_dict[self.time][self.sim_num].modulations_dict
                modulations_dict.setdefault(bandwidth, {})
                modulations_dict[bandwidth].setdefault(modulation, []).append(mean(mod_usages))

    def _find_sim_info(self, input_dict: dict):
        # TODO: (drl_path_agents) Does not support all bands/slots
        info_item_list = ['holding_time', 'cores_per_link', 'c_band', 'network', 'num_requests',
                          'cores_per_link', 'max_segments']
        self.plot_props = self.plot_props.plot_dict[self.time][self.sim_num].update_info_dict(
            plot_props=self.plot_props,
            input_dict=input_dict,
            info_item_list=info_item_list,
            time=self.time,
            sim_num=self.sim_num)

    def _update_plot_dict(self):
        if self.plot_props.plot_dict is None:
            self.plot_props.plot_dict = {self.time: {}}
        elif self.time not in self.plot_props.plot_dict:
            self.plot_props.plot_dict[self.time] = {}

        self.plot_props.plot_dict[self.time][self.sim_num] = PlotArgs()

    @staticmethod
    def _read_json_file(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            try:
                return json.load(file_obj)
            except json.JSONDecodeError:
                print('JSON file did not finishing writing, skipping!')
                return None

    def _read_input_output(self):
        base_fp = os.path.join(self.data_dict['network'], self.data_dict['date'], self.time)
        file_name = f'{self.erlang}_erlang.json'
        output_fp = os.path.join(self.plot_props.output_dir, base_fp, self.sim_num, file_name)
        erlang_dict = self._read_json_file(file_path=output_fp)

        file_name = f'sim_input_{self.sim_num}.json'
        input_fp = os.path.join(self.plot_props.input_dir, base_fp, file_name)
        input_dict = self._read_json_file(file_path=input_fp)

        return input_dict, erlang_dict

    def _get_data(self):
        for time, data_dict in self.file_info.items():
            self.time = time
            self.data_dict = data_dict
            for sim_num, erlang_list in self.data_dict['sim_dict'].items():
                self.sim_num = sim_num
                self._update_plot_dict()
                for erlang in erlang_list:
                    self.erlang = erlang
                    input_dict, self.erlang_dict = self._read_input_output()

                    if input_dict is None or self.erlang_dict is None:
                        continue

                    self.plot_props.plot_dict[time][sim_num].erlang_list.append(float(erlang))

                    self.plot_props.erlang_dict = self.erlang_dict
                    blocking_mean = self.plot_props.erlang_dict['blocking_mean']
                    last_iter = list(self.erlang_dict['iter_stats'].keys())[-1]
                    if blocking_mean is None:
                        blocking_mean = mean(self.erlang_dict['iter_stats'][last_iter]['sim_block_list'])

                    block_per_iter = self.erlang_dict['iter_stats'][last_iter]['sim_block_list']
                    self.plot_props.plot_dict[time][sim_num].block_per_iter.append(block_per_iter)
                    self.plot_props.plot_dict[time][sim_num].blocking_list.append(blocking_mean)

                    self._find_sim_info(input_dict=input_dict)
                    self._find_mod_info()
                    self._find_snapshot_usage()
                    self._find_misc_stats()

    def get_file_info(self, sims_info_dict: dict):
        """
        Retrieves all necessary file information to plot.

        :param sims_info_dict: A dictionary of specified configurations to find.
        """
        self.file_info = dict()
        matrix_count = 0
        networks_matrix = sims_info_dict['networks_matrix']
        dates_matrix = sims_info_dict['dates_matrix']
        times_matrix = sims_info_dict['times_matrix']

        for network_list, dates_list, times_list in zip(networks_matrix, dates_matrix, times_matrix):
            for network, date, time, in zip(network_list, dates_list, times_list):
                self.file_info[time] = {'network': network, 'date': date, 'sim_dict': dict()}
                curr_dir = os.path.join(self.plot_props.output_dir, network, date, time)
                # Sort by sim number
                try:
                    sim_dirs_list = os.listdir(curr_dir)
                except FileNotFoundError:
                    print(f'File: {curr_dir} was not found. Skipping!')
                    continue

                sim_dirs_list = [sim_dir for sim_dir in os.listdir(curr_dir) if 'train_data' not in sim_dir]
                sim_dirs_list = sorted(sim_dirs_list, key=lambda x: int(x[1:]))

                for sim in sim_dirs_list:
                    # User selected to not run this simulation
                    if sim not in sims_info_dict['sims_matrix'][matrix_count]:
                        continue

                    curr_fp = os.path.join(curr_dir, sim)
                    self.file_info[time]['sim_dict'][sim] = list()
                    files_list = os.listdir(curr_fp)
                    sorted_files_list = sorted(files_list, key=lambda x: float(x.split('_')[0]))

                    for erlang_file in sorted_files_list:
                        self.file_info[time]['sim_dict'][sim].append(erlang_file.split('_')[0])

            matrix_count += 1

        self._get_data()


def _not_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    for flags_list in filter_dict['not_filter_list']:
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        file_value = None
        for curr_key in keys_list:
            file_value = file_dict.get(curr_key)

        if file_value == check_value:
            keep_config = False
            break

        keep_config = True

    return keep_config


def _or_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    for flags_list in filter_dict['or_filter_list']:
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        file_value = None
        for curr_key in keys_list:
            file_value = file_dict.get(curr_key)

        if file_value == check_value:
            keep_config = True
            break

        keep_config = False

    return keep_config


def _and_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    for flags_list in filter_dict['and_filter_list']:
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        file_value = None
        for curr_key in keys_list:
            file_value = file_dict.get(curr_key)

            if isinstance(file_value, dict):
                file_dict = file_value
            else:
                break

        if file_value != check_value:
            keep_config = False
            break

    return keep_config


def _check_filters(file_dict: dict, filter_dict: dict):
    keep_config = _and_filters(filter_dict=filter_dict, file_dict=file_dict)

    if keep_config:
        keep_config = _or_filters(filter_dict=filter_dict, file_dict=file_dict)

        if keep_config:
            keep_config = _not_filters(filter_dict=filter_dict, file_dict=file_dict)

    return keep_config


def find_times(dates_dict: dict, filter_dict: dict):
    """
    Searches output directories based on filters and retrieves simulation directory information.

    :param dates_dict: The date directory to search.
    :param filter_dict: A dictionary containing all search filters.
    :return: A dictionary with all times, sim numbers, networks, and dates that matched the filter dict.
    :rtype: dict
    """
    resp = {
        'times_matrix': list(),
        'sims_matrix': list(),
        'networks_matrix': list(),
        'dates_matrix': list(),
    }
    info_dict = dict()
    for date, network in dates_dict.items():
        times_path = os.path.join('..', 'data', 'input', network, date)
        times_list = [curr_dir for curr_dir in os.listdir(times_path)
                      if os.path.isdir(os.path.join(times_path, curr_dir))]

        for curr_time in times_list:
            sims_path = os.path.join(times_path, curr_time)
            input_file_list = [input_file for input_file in os.listdir(sims_path) if
                               'sim' in input_file]

            for input_file in input_file_list:
                file_path = os.path.join(sims_path, input_file)
                with open(file_path, 'r', encoding='utf-8') as file_obj:
                    try:
                        file_dict = json.load(file_obj)
                    except json.JSONDecodeError:
                        print('Skipping file, it did not complete writing.')
                        continue

                keep_config = _check_filters(file_dict=file_dict, filter_dict=filter_dict)
                if keep_config:
                    if curr_time not in info_dict:
                        info_dict[curr_time] = {'sim_list': list(), 'network_list': list(), 'dates_list': list()}

                    sim = input_file.split('_')[2]
                    sim = sim.split('.')[0]
                    info_dict[curr_time]['sim_list'].append(sim)
                    info_dict[curr_time]['network_list'].append(network)
                    info_dict[curr_time]['dates_list'].append(date)

    # Convert info dict to lists
    for time, obj in info_dict.items():
        resp['times_matrix'].append([time])
        resp['sims_matrix'].append(obj['sim_list'])
        resp['networks_matrix'].append(obj['network_list'])
        resp['dates_matrix'].append(obj['dates_list'])

    return resp
