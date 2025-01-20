import json
import os
import math
import copy
from statistics import mean, variance, stdev

import numpy as np
import pandas as pd

from arg_scripts.stats_args import StatsProps
from arg_scripts.stats_args import SNAP_KEYS_LIST
from helper_scripts.sim_helpers import find_path_len, find_core_cong
from helper_scripts.os_helpers import create_dir


# TODO: (drl_path_agents) Note that many of these dictionaries were converted to objects, this will affect saving/calculating
class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, engine_props: dict, sim_info: str, stats_props: dict = None):
        # TODO: Implement ability to pick up from previously run simulations
        if stats_props is not None:
            self.stats_props = stats_props
        else:
            self.stats_props = StatsProps()

        self.engine_props = engine_props
        self.sim_info = sim_info

        self.save_dict = {'iter_stats': {}}

        # Used to track transponders for a single allocated request
        self.curr_trans = 0
        # Used to track transponders for an entire simulation on average
        self.total_trans = 0
        self.blocked_reqs = 0
        self.block_mean = None
        self.block_variance = None
        self.block_ci = None
        self.block_ci_percent = None
        self.bit_rate_request = None
        self.bit_rate_blocked = 0
        self.bit_rate_block_mean = None
        self.bit_rate_block_variance = None
        self.bit_rate_block_ci = None
        self.bit_rate_block_ci_percent = None
        self.topology = None
        self.iteration = None

        self.train_data_list = list()

    @staticmethod
    def _get_snapshot_info(net_spec_dict: dict, path_list: list):
        """
        Retrieves relative information for simulation snapshots.

        :param net_spec_dict: The current network spectrum database.
        :param path_list: A path to find snapshot info, if empty, does this for the entire network.
        :return: The occupied slots, number of guard bands, and active requests.
        :rtype: tuple
        """
        active_reqs_set = set()
        occupied_slots = 0
        guard_slots = 0
        # Skip by two because the link is bidirectional, no need to check both arrays e.g., (0, 1) and (1, 0)
        for link in list(net_spec_dict.keys())[::2]:
            if path_list is not None and link not in path_list:
                continue
            link_data = net_spec_dict[link]
            for core in link_data['cores_matrix']:
                requests_set = set(core[core > 0])
                for curr_req in requests_set:
                    active_reqs_set.add(curr_req)
                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        return occupied_slots, guard_slots, len(active_reqs_set)

    def update_train_data(self, old_req_info_dict: dict, req_info_dict: dict, net_spec_dict: dict):
        """
        Updates the training data list with the current request information.

        :param old_req_info_dict: Request dictionary before any potential slicing.
        :param req_info_dict: Request dictionary after potential slicing.
        :param net_spec_dict: Network spectrum database.
        """
        path_list = req_info_dict['path']
        cong_arr = np.array([])

        for core_num in range(self.engine_props['cores_per_link']):
            curr_cong = find_core_cong(core_index=core_num, net_spec_dict=net_spec_dict, path_list=path_list)
            cong_arr = np.append(cong_arr, curr_cong)

        path_length = find_path_len(path_list=path_list, topology=self.engine_props['topology'])
        tmp_info_dict = {
            'old_bandwidth': old_req_info_dict['bandwidth'],
            'path_length': path_length,
            'longest_reach': np.max(old_req_info_dict['mod_formats']['QPSK']['max_length']),
            'ave_cong': float(np.mean(cong_arr)),
            'num_segments': self.curr_trans,
        }
        self.train_data_list.append(tmp_info_dict)

    def update_snapshot(self, net_spec_dict: dict, req_num: int, path_list: list = None):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network or a specific path.

        :param net_spec_dict: The current network spectrum database.
        :param req_num: The current request number.
        :param path_list: The desired path to find the occupied slots on.
        :return: None
        """
        occupied_slots, guard_slots, active_reqs = self._get_snapshot_info(net_spec_dict=net_spec_dict,
                                                                           path_list=path_list)
        blocking_prob = self.blocked_reqs / req_num
        bit_rate_block_prob = self.bit_rate_blocked / self.bit_rate_request

        self.stats_props.snapshots_dict[req_num]['occupied_slots'].append(occupied_slots)
        self.stats_props.snapshots_dict[req_num]['guard_slots'].append(guard_slots)
        self.stats_props.snapshots_dict[req_num]['active_requests'].append(active_reqs)
        self.stats_props.snapshots_dict[req_num]["blocking_prob"].append(blocking_prob)
        self.stats_props.snapshots_dict[req_num]['num_segments'].append(self.curr_trans)
        self.stats_props.snapshots_dict[req_num]["bit_rate_blocking_prob"].append(bit_rate_block_prob)

    def _init_snapshots(self):
        for req_num in range(0, self.engine_props['num_requests'] + 1, self.engine_props['snapshot_step']):
            self.stats_props.snapshots_dict[req_num] = dict()
            for key in SNAP_KEYS_LIST:
                self.stats_props.snapshots_dict[req_num][key] = list()

    def _init_mods_weights_bws(self):
        for bandwidth, obj in self.engine_props['mod_per_bw'].items():
            self.stats_props.mods_used_dict[bandwidth] = dict()
            self.stats_props.weights_dict[bandwidth] = dict()
            for modulation in obj.keys():
                self.stats_props.weights_dict[bandwidth][modulation] = list()
                self.stats_props.mods_used_dict[bandwidth][modulation] = 0
                if modulation not in self.stats_props.mods_used_dict or isinstance(
                        self.stats_props.mods_used_dict[modulation]['length']['overall'], dict):
                    self.stats_props.mods_used_dict[modulation] = dict()
                    self.stats_props.mods_used_dict[modulation]['length'] = dict()
                    self.stats_props.mods_used_dict[modulation]['length']['overall'] = list()
                    for band in self.engine_props['band_list']:
                        self.stats_props.mods_used_dict[modulation][band] = 0
                        self.stats_props.mods_used_dict[modulation]['length'][band] = list()

            self.stats_props.block_bw_dict[bandwidth] = 0

    def _init_stat_dicts(self):
        for stat_key, data_type in vars(self.stats_props).items():
            if not isinstance(data_type, dict):
                continue
            if stat_key in ('mods_used_dict', 'weights_dict', 'block_bw_dict'):
                self._init_mods_weights_bws()
            elif stat_key == 'snapshots_dict':
                if self.engine_props['save_snapshots']:
                    self._init_snapshots()
            elif stat_key == 'cores_dict':
                self.stats_props.cores_dict = {core: 0 for core in range(self.engine_props['cores_per_link'])}
            elif stat_key == 'block_reasons_dict':
                self.stats_props.block_reasons_dict = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
            elif stat_key != 'iter_stats':
                raise ValueError('Dictionary statistic was not reset in props.')

    def _init_stat_lists(self):
        for stat_key in vars(self.stats_props).keys():
            data_type = getattr(self.stats_props, stat_key)
            if isinstance(data_type, list):
                # Only reset sim_block_list when we encounter a new traffic volume
                if self.iteration != 0 and stat_key in ['sim_block_list', 'sim_br_block_list']:
                    continue
                setattr(self.stats_props, stat_key, list())

    def init_iter_stats(self):
        """
        Initializes data structures used in other methods of this class.

        :return: None
        """
        self._init_stat_dicts()
        self._init_stat_lists()

        self.blocked_reqs = 0
        self.bit_rate_blocked = 0
        self.bit_rate_request = 0
        self.total_trans = 0

    def get_blocking(self):
        """
        Gets the current blocking probability.

        :return: None
        """
        if self.engine_props['num_requests'] == 0:
            blocking_prob = 0
            bit_rate_blocking_prob = 0
        else:
            blocking_prob = self.blocked_reqs / self.engine_props['num_requests']
            bit_rate_blocking_prob = self.bit_rate_blocked / self.bit_rate_request

        self.stats_props.sim_block_list.append(blocking_prob)
        self.stats_props.sim_br_block_list.append(bit_rate_blocking_prob)

    def _handle_iter_lists(self, sdn_data: object):
        for stat_key in sdn_data.stat_key_list:
            # TODO: (drl_path_agents) This name should be changed to 'sdn_data'
            curr_sdn_data = sdn_data.get_data(key=stat_key)
            if stat_key == 'xt_list':
                # (drl_path_agents) fixme
                if curr_sdn_data == [None]:
                    break
            for i, data in enumerate(curr_sdn_data):
                if stat_key == 'core_list':
                    self.stats_props.cores_dict[data] += 1
                elif stat_key == 'modulation_list':
                    bandwidth = sdn_data.bandwidth_list[i]
                    band = sdn_data.band_list[i]
                    self.stats_props.mods_used_dict[bandwidth][data] += 1
                    self.stats_props.mods_used_dict[data][band] += 1
                    self.stats_props.mods_used_dict[data]['length'][band].append(sdn_data.path_weight)
                    self.stats_props.mods_used_dict[data]['length']['overall'].append(sdn_data.path_weight)
                elif stat_key == 'start_slot_list':
                    self.stats_props.start_slot_list.append(int(data))
                elif stat_key == 'end_slot_list':
                    self.stats_props.end_slot_list.append(int(data))
                elif stat_key == 'modulation_list':
                    self.stats_props.modulation_list.append(int(data))
                elif stat_key == 'bandwidth_list':
                    self.stats_props.bandwidth_list.append(int(data))

    def iter_update(self, req_data: dict, sdn_data: object):
        """
        Continuously updates the statistical data for each request allocated/blocked in the current iteration.

        :param req_data: Holds data relevant to the current request.
        :param sdn_data: Hold the response data from the sdn controller.
        :return: None
        """
        # Request was blocked
        if not sdn_data.was_routed:
            self.blocked_reqs += 1
            self.bit_rate_blocked += int(sdn_data.bandwidth)
            self.bit_rate_request += int(sdn_data.bandwidth)
            self.stats_props.block_reasons_dict[sdn_data.block_reason] += 1
            self.stats_props.block_bw_dict[req_data['bandwidth']] += 1
        else:
            num_hops = len(sdn_data.path_list) - 1
            self.stats_props.hops_list.append(num_hops)

            path_len = find_path_len(path_list=sdn_data.path_list, topology=self.topology)
            self.stats_props.lengths_list.append(round(float(path_len), 2))

            self._handle_iter_lists(sdn_data=sdn_data)
            self.stats_props.route_times_list.append(sdn_data.route_time)
            self.total_trans += sdn_data.num_trans
            bandwidth = sdn_data.bandwidth
            mod_format = sdn_data.modulation_list[0]

            self.bit_rate_request += int(sdn_data.bandwidth)
            self.stats_props.weights_dict[bandwidth][mod_format].append(round(float(sdn_data.path_weight), 2))

    def _get_iter_means(self):
        for _, curr_snapshot in self.stats_props.snapshots_dict.items():
            for snap_key, data_list in curr_snapshot.items():
                if data_list:
                    curr_snapshot[snap_key] = mean(data_list)
                else:
                    curr_snapshot[snap_key] = None

        for _, mod_obj in self.stats_props.weights_dict.items():
            for modulation, data_list in mod_obj.items():
                # Modulation was never used
                if len(data_list) == 0:
                    mod_obj[modulation] = {'mean': None, 'std': None, 'min': None, 'max': None}
                else:
                    if len(data_list) == 1:
                        deviation = 0.0
                    else:
                        deviation = stdev(data_list)
                    mod_obj[modulation] = {'mean': mean(data_list), 'std': deviation,
                                           'min': min(data_list), 'max': max(data_list)}
                for key, value in self.stats_props.mods_used_dict[modulation]['length'].items():
                    if not isinstance(value, list):
                        continue
                    if len(value) == 0:
                        self.stats_props.mods_used_dict[modulation]['length'][key] = {'mean': None, 'std': None,
                                                                                      'min': None, 'max': None}
                    else:
                        # TODO: Is this ever equal to one?
                        if len(value) == 1:
                            deviation = 0.0
                        else:
                            deviation = stdev(value)
                        self.stats_props.mods_used_dict[modulation]['length'][key] = {
                            'mean': round(float(mean(value)), 2), 'std': round(float(deviation), 2),
                            'min': round(float(min(value)), 2), 'max': round(float(max(value)), 2)}

    def end_iter_update(self):
        """
        Updates relevant stats after an iteration has finished.

        :return: None
        """
        if self.engine_props['num_requests'] == self.blocked_reqs:
            self.stats_props.trans_list.append(0)
        else:
            trans_mean = self.total_trans / float(self.engine_props['num_requests'] - self.blocked_reqs)
            self.stats_props.trans_list.append(trans_mean)

        if self.blocked_reqs > 0:
            for block_type, num_times in self.stats_props.block_reasons_dict.items():
                self.stats_props.block_reasons_dict[block_type] = num_times / float(self.blocked_reqs)

        self._get_iter_means()

    def get_conf_inter(self):
        """
        Get the confidence interval for every iteration so far.

        :return: Whether the simulations should end for this erlang.
        :rtype: bool
        """
        self.block_mean = mean(self.stats_props.sim_block_list)
        self.bit_rate_block_mean = mean(self.stats_props.sim_br_block_list)
        if len(self.stats_props.sim_block_list) <= 1:
            return False

        self.block_variance = variance(self.stats_props.sim_block_list)
        self.bit_rate_block_variance = variance(self.stats_props.sim_br_block_list)

        if self.block_mean == 0.0:
            return False

        try:
            # 1.645 for 90% confidence level and 1.96 for 95% confidence level
            block_ci_rate = 1.96 * (math.sqrt(self.block_variance) / math.sqrt(len(self.stats_props.sim_block_list)))
            self.block_ci = block_ci_rate
            block_ci_percent = ((2 * block_ci_rate) / self.block_mean) * 100
            self.block_ci_percent = block_ci_percent
            # bit rate blcoking
            bit_rate_block_ci = 1.96 * (
                        math.sqrt(self.bit_rate_block_variance) / math.sqrt(len(self.stats_props.sim_br_block_list)))
            self.bit_rate_block_ci = bit_rate_block_ci
            bit_rate_block_ci_percent = ((2 * bit_rate_block_ci) / self.bit_rate_block_mean) * 100
            self.bit_rate_block_ci_percent = bit_rate_block_ci_percent
        except ZeroDivisionError:
            return False

        # TODO: (drl_path_agents) Add CI percent to configuration file
        if block_ci_percent <= 5:
            print(f"Confidence interval of {round(block_ci_percent, 2)}% reached. "
                  f"{self.iteration + 1}, ending and saving results for Erlang: {self.engine_props['erlang']}")
            self.save_stats(base_fp='data')
            return True

        return False

    def save_train_data(self, base_fp: str):
        """
        Saves training data file.

        :param base_fp: Base file path.
        """
        if self.iteration == (self.engine_props['max_iters'] - 1):
            save_df = pd.DataFrame(self.train_data_list)
            save_df.to_csv(f"{base_fp}/output/{self.sim_info}/{self.engine_props['erlang']}_train_data.csv",
                           index=False)

    def save_stats(self, base_fp: str):
        """
        Saves simulations stats as either a json or csv file.

        :return: None
        """
        if self.engine_props['file_type'] not in ('json', 'csv'):
            raise NotImplementedError(f"Invalid file type: {self.engine_props['file_type']}, expected csv or json.")

        self.save_dict['blocking_mean'] = self.block_mean
        self.save_dict['blocking_variance'] = self.block_variance
        self.save_dict['ci_rate_block'] = self.block_ci
        self.save_dict['ci_percent_block'] = self.block_ci_percent

        self.save_dict['bit_rate_blocking_mean'] = self.bit_rate_block_mean
        self.save_dict['bit_rate_blocking_variance'] = self.bit_rate_block_variance
        self.save_dict['ci_rate_bit_rate_block'] = self.bit_rate_block_ci
        self.save_dict['ci_percent_bit_rate_block'] = self.bit_rate_block_ci_percent

        self.save_dict['iter_stats'][self.iteration] = dict()
        for stat_key in vars(self.stats_props).keys():
            if stat_key in ('trans_list', 'hops_list', 'lengths_list', 'route_times_list', 'xt_list'):
                save_key = f"{stat_key.split('list')[0]}"
                if stat_key == 'xt_list':
                    stat_array = [0 if stat is None else stat for stat in getattr(self.stats_props, stat_key)]
                else:
                    stat_array = getattr(self.stats_props, stat_key)

                # Every request was blocked
                if len(stat_array) == 0:
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}mean'] = None
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}min'] = None
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}max'] = None
                else:
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}mean'] = round(float(mean(stat_array)), 2)
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}min'] = round(float(min(stat_array)), 2)
                    self.save_dict['iter_stats'][self.iteration][f'{save_key}max'] = round(float(max(stat_array)), 2)
            else:
                if stat_key in ['start_slot_list', 'end_slot_list'] and not self.engine_props['save_start_end_slots']:
                    self.save_dict['iter_stats'][self.iteration][stat_key] = []
                    continue
                self.save_dict['iter_stats'][self.iteration][stat_key] = copy.deepcopy(getattr(self.stats_props,
                                                                                               stat_key))

        if base_fp is None:
            base_fp = 'data'
        save_fp = os.path.join(base_fp, 'output', self.sim_info, self.engine_props['thread_num'])
        create_dir(save_fp)
        if self.engine_props['file_type'] == 'json':
            with open(f"{save_fp}/{self.engine_props['erlang']}_erlang.json", 'w', encoding='utf-8') as file_path:
                json.dump(self.save_dict, file_path, indent=4)
        else:
            raise NotImplementedError

        if self.engine_props['output_train_data']:
            self.save_train_data(base_fp=base_fp)

    def print_iter_stats(self, max_iters: int, print_flag: bool):
        """
        Prints iteration stats, mostly used to ensure simulations are running fine.

        :param max_iters: The maximum number of iterations.
        :param print_flag: Determine if we want to print or not.
        :return: None
        """
        if print_flag:
            print(f"Iteration {self.iteration + 1} out of {max_iters} completed for "
                  f"Erlang: {self.engine_props['erlang']}")
            print(f"Mean of blocking: {round(mean(self.stats_props.sim_block_list), 4)}")
