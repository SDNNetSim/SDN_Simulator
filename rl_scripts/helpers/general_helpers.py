import os

import numpy as np

from src.spectrum_assignment import SpectrumAssignment

from helper_scripts.sim_helpers import find_path_len, get_path_mod, get_hfrag
from helper_scripts.sim_helpers import find_path_cong, classify_cong, find_core_cong

from rl_scripts.args.general_args import VALID_PATH_ALGORITHMS
from arg_scripts.sdn_args import SDNProps


class CoreUtilHelpers:
    """
    Contains methods to assist with reinforcement learning simulations.
    """

    def __init__(self, rl_props: object, engine_obj: object, route_obj: object):
        self.rl_props = rl_props

        self.engine_obj = engine_obj
        self.route_obj = route_obj

        self.topology = None

        self.core_num = None
        self.super_channel = None
        self.super_channel_indexes = list()
        self.mod_format = None
        self._last_processed_index = 0

    def update_snapshots(self):
        """
        Updates snapshot saves for the simulation.
        """
        arrival_count = self.rl_props.arrival_count
        snapshot_step = self.engine_obj.engine_props['snapshot_step']

        if self.engine_obj.engine_props['save_snapshots'] and (arrival_count + 1) % snapshot_step == 0:
            self.engine_obj.stats_obj.update_snapshot(net_spec_dict=self.engine_obj.net_spec_dict,
                                                      req_num=arrival_count + 1)

    def get_super_channels(self, slots_needed: int, num_channels: int):
        """
        Gets the available 'J' super-channels for the agent to choose from along with a fragmentation score.

        :param slots_needed: Slots needed by the current request.
        :param num_channels: Number of channels needed by the current request.
        :return: A matrix of super-channels with their fragmentation score.
        :rtype: list
        """
        # TODO: (drl_path_agents) The 'c' band used by default
        path_list = self.rl_props.chosen_path_list[0]
        sc_index_mat, hfrag_arr = get_hfrag(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict,
                                            spectral_slots=self.rl_props.spectral_slots, core_num=self.core_num,
                                            slots_needed=slots_needed, band='c')

        self.super_channel_indexes = sc_index_mat[:num_channels]
        # There were not enough super-channels, do not penalize the agent
        no_penalty = len(self.super_channel_indexes) == 0

        resp_frag_mat = list()
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            resp_frag_mat.append(hfrag_arr[start_index])

        resp_frag_mat = np.where(np.isinf(resp_frag_mat), 100.0, resp_frag_mat)
        difference = self.rl_props.super_channel_space - len(resp_frag_mat)

        if len(resp_frag_mat) < self.rl_props.super_channel_space or np.any(np.isinf(resp_frag_mat)):
            for _ in range(difference):
                resp_frag_mat = np.append(resp_frag_mat, 100.0)

        return resp_frag_mat, no_penalty

    def classify_paths(self, paths_list: list):
        """
        Classify paths by their current congestion level.

        :param paths_list: A list of paths from source to destination.
        :return: The index of the path, the path itself, and its congestion index for every path.
        :rtype: list
        """
        info_list = list()
        paths_list = paths_list[:, 0]
        for path_index, curr_path in enumerate(paths_list):
            curr_cong = find_path_cong(path_list=curr_path, net_spec_dict=self.engine_obj.net_spec_dict)
            cong_index = classify_cong(curr_cong=curr_cong)

            info_list.append((path_index, curr_path, cong_index))

        return info_list

    def classify_cores(self, cores_list: list):
        """
        Classify cores by their congestion level.

        :param cores_list: A list of cores.
        :return: The core index, the core itself, and the congestion level of that core for every core.
        :rtype: list
        """
        info_list = list()

        for core_index, curr_core in enumerate(cores_list):
            path_list = curr_core['path'][0]
            curr_cong = find_core_cong(core_index=core_index, net_spec_dict=self.engine_obj.net_spec_dict,
                                       path_list=path_list)
            cong_index = classify_cong(curr_cong=curr_cong)

            info_list.append((core_index, curr_core[cong_index], cong_index))

        return info_list

    def update_route_props(self, bandwidth: str, chosen_path: list):
        """
        Updates the route properties.

        :param bandwidth: Bandwidth of the current request.
        :param chosen_path: Path of the current request.
        """
        self.route_obj.route_props.paths_matrix = chosen_path
        path_len = find_path_len(path_list=chosen_path[0], topology=self.engine_obj.engine_props['topology'])
        mod_format = get_path_mod(mods_dict=self.engine_obj.engine_props['mod_per_bw'][bandwidth], path_len=path_len)
        self.route_obj.route_props.mod_formats_matrix = [[mod_format]]
        self.route_obj.route_props.weights_list.append(path_len)

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.rl_props.arrival_list[min(self.rl_props.arrival_count,
                                                   len(self.rl_props.arrival_list) - 1)]['arrive']

        depart_list = self.rl_props.depart_list
        while self._last_processed_index < len(depart_list):
            req_obj = depart_list[self._last_processed_index]
            if req_obj['depart'] > curr_time:
                break

            self.engine_obj.handle_release(curr_time=req_obj['depart'])
            self._last_processed_index += 1

    def allocate(self):
        """
        Attempts to allocate a request.
        """
        curr_time = self.rl_props.arrival_list[self.rl_props.arrival_count]['arrive']
        if self.rl_props.forced_index is not None:
            try:
                forced_index = self.super_channel_indexes[self.rl_props.forced_index][0]
            # DRL agent picked a super-channel that is not available, block
            except IndexError:
                self.engine_obj.stats_obj.blocked_reqs += 1
                self.engine_obj.stats_obj.stats_props['block_reasons_dict']['congestion'] += 1
                bandwidth = self.rl_props.arrival_list[self.rl_props.arrival_count]['bandwidth']
                self.engine_obj.stats_obj.stats_props['block_bw_dict'][bandwidth] += 1
                return
        else:
            forced_index = None

        force_mod_format = self.route_obj.route_props.mod_formats_matrix[0]
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=self.rl_props.chosen_path_list,
                                       force_core=self.rl_props.core_index,
                                       forced_index=forced_index, force_mod_format=force_mod_format)

    @staticmethod
    def mock_handle_arrival(engine_props: dict, sdn_props: dict, path_list: list, mod_format_list: list):
        """
        Function to mock an arrival process or allocation in the network.

        :param engine_props: Properties of engine.
        :param sdn_props: Properties of the SDN controller.
        :param path_list: List of nodes, the current path.
        :param mod_format_list: Valid modulation formats.
        :return: If there are available spectral slots.
        :rtype: bool
        """
        route_props = None
        spectrum_obj = SpectrumAssignment(engine_props=engine_props, sdn_props=sdn_props, route_props=route_props)

        spectrum_obj.spectrum_props.forced_index = None
        spectrum_obj.spectrum_props.forced_core = None
        spectrum_obj.spectrum_props.path_list = path_list
        spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
        # Request was blocked for this path
        if spectrum_obj.spectrum_props.is_free is not True:
            return False

        return True

    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary to find select routes.

        :param curr_req: The current request.
        :return: The mock return of the SDN controller.
        :rtype: dict
        """
        mock_sdn = SDNProps()
        params = {
            'req_id': curr_req['req_id'],
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.engine_obj.net_spec_dict,
            'topology': self.topology,
            'mod_formats_dict': curr_req['mod_formats'],
            'num_trans': 1.0,
            'block_reason': None,
            'modulation_list': list(),
            'xt_list': list(),
            'is_sliced': False,
            'core_list': list(),
            'bandwidth_list': list(),
            'path_weight': list(),
        }

        for key, value in params.items():
            setattr(mock_sdn, key, value)

        return mock_sdn

    def reset_reqs_dict(self, seed: int):
        """
        Resets the request dictionary.

        :param seed: The random seed.
        """
        self._last_processed_index = 0
        self.engine_obj.generate_requests(seed=seed)

        for req_time in self.engine_obj.reqs_dict:
            if self.engine_obj.reqs_dict[req_time]['request_type'] == 'arrival':
                self.rl_props.arrival_list.append(self.engine_obj.reqs_dict[req_time])
            else:
                self.rl_props.depart_list.append(self.engine_obj.reqs_dict[req_time])


class SimEnvHelpers:
    """
    Encapsulates high-level helper methods tailored for managing and enhancing the behavior of the `SimEnv` class during
    reinforcement learning simulations.
    """

    def __init__(self, sim_env: object):
        """
        Initializes the helper methods class with shared context.

        :param sim_env: The main simulation environment object.
        """
        self.sim_env = sim_env

    def update_helper_obj(self, action: int, bandwidth: str):
        """
        Updates the helper object with new actions and configurations.
        """
        if self.sim_env.engine_obj.engine_props['is_drl_agent']:
            self.sim_env.rl_help_obj.path_index = action
        else:
            self.sim_env.rl_help_obj.path_index = self.sim_env.rl_props.path_index

        self.sim_env.rl_help_obj.core_num = self.sim_env.rl_props.core_index

        if self.sim_env.sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            self.sim_env.rl_help_obj.rl_props.forced_index = action
        else:
            self.sim_env.rl_help_obj.rl_props.forced_index = None

        self.sim_env.rl_help_obj.rl_props = self.sim_env.rl_props
        self.sim_env.rl_help_obj.engine_obj = self.sim_env.engine_obj
        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_help_obj.update_route_props(chosen_path=self.sim_env.rl_props.chosen_path_list,
                                                    bandwidth=bandwidth)

    def determine_core_penalty(self):
        """
        Determines penalty for the core algorithm based on path availability.
        """
        # Default to first fit if all paths fail
        self.sim_env.rl_props.chosen_path = [self.sim_env.route_obj.route_props.paths_matrix[0]]
        self.sim_env.rl_props.chosen_path_index = 0
        for path_index, path_list in enumerate(self.sim_env.route_obj.route_props.paths_matrix):
            mod_format_list = self.sim_env.route_obj.route_props.mod_formats_matrix[path_index]

            was_allocated = self.sim_env.rl_help_obj.mock_handle_arrival(
                engine_props=self.sim_env.engine_obj.engine_props,
                sdn_props=self.sim_env.rl_props.mock_sdn_dict,
                mod_format_list=mod_format_list,
                path_list=path_list)

            if was_allocated:
                self.sim_env.rl_props.chosen_path_list = [path_list]
                self.sim_env.rl_props.chosen_path_index = path_index
                break

    def handle_test_train_obs(self, curr_req: dict):  # pylint: disable=unused-argument
        """
        Handles path and core selection during training/testing phases based on the current request.

        Returns:
            Path modulation format, if available.
        """
        if self.sim_env.sim_dict['is_training']:
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.sim_env.step_helper.handle_path_train_test()
            else:
                raise NotImplementedError
        else:
            self.sim_env.step_helpers.handle_path_train_test()

    def get_drl_obs(self):
        """
        Creates observation data for Deep Reinforcement Learning (DRL) in a graph-based
        environment.
        """
        source_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        source_obs[self.sim_env.rl_props.source] = 1.0
        dest_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        dest_obs[self.sim_env.rl_props.destination] = 1.0

        return source_obs, dest_obs


# TODO: (drl_path_agents) Only works for s1
def determine_model_type(sim_dict: dict) -> str:
    """
    Determines the type of agent being used based on the provided simulation dictionary.

    :param sim_dict: A dictionary containing simulation configuration.
    :return: A string representing the model type ('path_algorithm', 'core_algorithm', 'spectrum_algorithm').
    """
    if 's1' in sim_dict:
        sim_dict = sim_dict['s1']
    if sim_dict.get('path_algorithm') is not None:
        return 'path_algorithm'
    if sim_dict.get('core_algorithm') is not None:
        return 'core_algorithm'
    if sim_dict.get('spectrum_algorithm') is not None:
        return 'spectrum_algorithm'

    raise ValueError("No valid algorithm type found in sim_dict. "
                     "Ensure 'path_algorithm', 'core_algorithm', or 'spectrum_algorithm' is set.")


def save_arr(arr: np.array, sim_dict: dict, file_name: str):
    """
    Save a numpy array to a specific file path constructed from simulation details.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_type = sim_dict[model_type]

    network, date, time = sim_dict['network'], sim_dict['date'], sim_dict['sim_start']
    file_path = os.path.join('logs', algorithm_type, network, date, time, file_name)
    np.save(file_path, arr)
