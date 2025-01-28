import time

import numpy as np

from helper_scripts.sim_helpers import sort_dict_keys, get_path_mod, find_path_len
from helper_scripts.ml_helpers import get_ml_obs
from arg_scripts.sdn_args import SDNProps
from src.routing import Routing
from src.spectrum_assignment import SpectrumAssignment


class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.sdn_props = SDNProps()

        self.ai_obj = None
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.spectrum_obj = SpectrumAssignment(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                               route_props=self.route_obj.route_props)

    def release(self):
        """
        Removes a previously allocated request from the network.
        """
        for source, dest in zip(self.sdn_props.path_list, self.sdn_props.path_list[1:]):
            for band in self.engine_props['band_list']:
                for core_num in range(self.engine_props['cores_per_link']):
                    core_arr = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num]
                    req_id_arr = np.where(core_arr == self.sdn_props.req_id)
                    gb_arr = np.where(core_arr == (self.sdn_props.req_id * -1))

                    for req_index in req_id_arr:
                        self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num][req_index] = 0
                        self.sdn_props.net_spec_dict[(dest, source)]['cores_matrix'][band][core_num][req_index] = 0
                    for gb_index in gb_arr:
                        self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num][gb_index] = 0
                        self.sdn_props.net_spec_dict[(dest, source)]['cores_matrix'][band][core_num][gb_index] = 0

    def _allocate_gb(self, band: str, core_matrix: list, rev_core_matrix: list, core_num: int, end_slot: int):
        if core_matrix[band][core_num][end_slot] != 0.0 or rev_core_matrix[band][core_num][end_slot] != 0.0:
            raise BufferError("Attempted to allocate a taken spectrum.")

        core_matrix[band][core_num][end_slot] = self.sdn_props.req_id * -1
        rev_core_matrix[band][core_num][end_slot] = self.sdn_props.req_id * -1

    def allocate(self):
        """
        Allocates a network request.
        """
        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_num
        band = self.spectrum_obj.spectrum_props.curr_band

        if self.engine_props['guard_slots'] != 0:
            end_slot = end_slot - 1
        else:
            end_slot += 1

        for link_tuple in zip(self.sdn_props.path_list, self.sdn_props.path_list[1:]):
            # Remember, Python list indexing is up to and NOT including!
            link_dict = self.sdn_props.net_spec_dict[(link_tuple[0], link_tuple[1])]
            rev_link_dict = self.sdn_props.net_spec_dict[(link_tuple[1], link_tuple[0])]

            tmp_set = set(link_dict['cores_matrix'][band][core_num][start_slot:end_slot])
            rev_tmp_set = set(rev_link_dict['cores_matrix'][band][core_num][start_slot:end_slot])

            if tmp_set == {} or rev_tmp_set == {}:
                raise ValueError('Nothing detected on the spectrum when allocating.')

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            core_matrix = link_dict['cores_matrix']
            rev_core_matrix = rev_link_dict['cores_matrix']
            core_matrix[band][core_num][start_slot:end_slot] = self.sdn_props.req_id
            rev_core_matrix[band][core_num][start_slot:end_slot] = self.sdn_props.req_id

            if self.engine_props['guard_slots']:
                self._allocate_gb(core_matrix=core_matrix, rev_core_matrix=rev_core_matrix, end_slot=end_slot,
                                  core_num=core_num, band=band)

    # TODO: No support for multi-band
    def _update_req_stats(self, bandwidth: str):
        self.sdn_props.bandwidth_list.append(bandwidth)
        for stat_key in self.sdn_props.stat_key_list:
            spectrum_key = stat_key.split('_')[0]  # pylint: disable=use-maxsplit-arg
            if spectrum_key == 'xt':
                spectrum_key = 'xt_cost'
            elif spectrum_key == 'core':
                spectrum_key = 'core_num'
            elif spectrum_key == 'band':
                spectrum_key = 'curr_band'
            elif spectrum_key == 'start':
                spectrum_key = 'start_slot'
            elif spectrum_key == 'end':
                spectrum_key = 'end_slot'

            self.sdn_props.update_params(key=stat_key, spectrum_key=spectrum_key, spectrum_obj=self.spectrum_obj)

    def _allocate_slicing(self, num_segments: int, mod_format: str, path_list: list, bandwidth: str):
        self.sdn_props.num_trans = num_segments
        self.spectrum_obj.spectrum_props.path_list = path_list
        mod_format_list = [mod_format]
        for _ in range(num_segments):
            self.spectrum_obj.get_spectrum(mod_format_list=mod_format_list, slice_bandwidth=bandwidth)
            if self.spectrum_obj.spectrum_props.is_free:
                self.allocate()
                self._update_req_stats(bandwidth=bandwidth)
            else:
                self.sdn_props.was_routed = False
                self.sdn_props.block_reason = 'congestion'
                self.release()
                break

    def _handle_slicing(self, path_list: list, forced_segments: int):
        bw_mod_dict = sort_dict_keys(dictionary=self.engine_props['mod_per_bw'])
        for bandwidth, mods_dict in bw_mod_dict.items():
            # We can't slice to a larger or equal bandwidth
            if int(bandwidth) >= int(self.sdn_props.bandwidth):
                continue

            path_len = find_path_len(path_list=path_list, topology=self.engine_props['topology'])
            mod_format = get_path_mod(mods_dict=mods_dict, path_len=path_len)
            if not mod_format:
                continue

            self.sdn_props.was_routed = True
            num_segments = int(int(self.sdn_props.bandwidth) / int(bandwidth))
            if num_segments > self.engine_props['max_segments']:
                self.sdn_props.was_routed = False
                self.sdn_props.block_reason = 'max_segments'
                break

            if forced_segments not in (-1, num_segments):
                self.sdn_props.was_routed = False
                continue

            self._allocate_slicing(num_segments=num_segments, mod_format=mod_format, path_list=path_list,
                                   bandwidth=bandwidth)

            if self.sdn_props.was_routed:
                self.sdn_props.is_sliced = True
                return

            self.sdn_props.is_sliced = False

    def _handle_congestion(self, remaining_bw):
        """
        Handles the case where allocation fails due to congestion.
        """
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = 'congestion'
        self.sdn_props.num_trans = 1

        if remaining_bw != int(self.sdn_props.bandwidth):
            self.release()

        self.sdn_props.is_sliced = False

    def _handle_dynamic_slicing(self, path_list: list, path_index: int,
                                forced_segments: int):  # pylint: disable=unused-argument
        """
        Handles dynamic slicing for a network request.
        """
        remaining_bw = int(self.sdn_props.bandwidth)
        # TODO: Ignored?
        path_len = find_path_len(path_list=path_list, topology=self.engine_props['topology'])  # pylint: disable=unused-variable
        bw_mod_dict = sort_dict_keys(self.engine_props['mod_per_bw'])  # pylint: disable=unused-variable

        self.spectrum_obj.spectrum_props.path_list = path_list
        self.sdn_props.num_trans = 0

        while remaining_bw > 0:
            if not self.engine_props['fixed_grid']:
                raise NotImplementedError("Dynamic slicing for non-fixed grid is not implemented.")

            # TODO: Mod format list ignored?
            self.sdn_props.was_routed = True
            _, bandwidth = self.spectrum_obj.get_spectrum_dynamic_slicing(
                mod_format_list=[], path_index=path_index, path_len=path_len
            )

            if self.spectrum_obj.spectrum_props.is_free:
                self.allocate()
                dedicated_bw = min(bandwidth, remaining_bw)
                self._update_req_stats(bandwidth=str(dedicated_bw))
                remaining_bw -= bandwidth
                self.sdn_props.num_trans += 1
                self.sdn_props.is_sliced = True
            else:
                self._handle_congestion(remaining_bw=remaining_bw)
                break

    def _init_req_stats(self):
        self.sdn_props.bandwidth_list = list()
        self.sdn_props.reset_params()

    def handle_event(self, req_dict: dict, request_type: str, force_slicing: bool = False,
                     # pylint: disable=too-many-statements
                     force_route_matrix: list = None, forced_index: int = None,
                     force_core: int = None, ml_model=None, force_mod_format: str = None, forced_band: str = None):
        """
        Handles any event that occurs in the simulation, controls this class.

        :param req_dict: Request dictionary.
        :param request_type: Whether the request is an arrival or departure.
        :param force_slicing: Whether to force light segment slicing or not.
        :param force_route_matrix: Whether to force a path or not.
        :param forced_index: Whether to force a start index for a request.
        :param force_mod_format: Forces a modulation format.
        :param force_core: Force a specific core.
        :param ml_model: An optional machine learning model.
        :param forced_band: Whether to force a band or not.
        """
        self._init_req_stats()
        # Even if the request is blocked, we still consider one transponder
        self.sdn_props.num_trans = 1

        if request_type == "release":
            self.release()
            return

        start_time = time.time()
        if force_route_matrix is None:
            self.route_obj.get_route()
            route_matrix = self.route_obj.route_props.paths_matrix
        else:
            route_matrix = force_route_matrix
            # TODO: (drl_path_agents) This is an inconsistency when forcing a modulation format
            self.route_obj.route_props.mod_formats_matrix = [force_mod_format]
            # fixme (drl_path_agents)
            self.route_obj.route_props.weights_list = [0]
        route_time = time.time() - start_time

        segment_slicing = False
        # TODO: Improve SDN controller's structure
        while True:  # pylint: disable=too-many-nested-blocks
            for path_index, path_list in enumerate(route_matrix):
                if path_list is not False:
                    self.sdn_props.path_list = path_list
                    self.sdn_props.path_index = path_index
                    mod_format_list = self.route_obj.route_props.mod_formats_matrix[path_index]

                    if ml_model is not None:
                        input_df = get_ml_obs(req_dict=req_dict, engine_props=self.engine_props,
                                              sdn_props=self.sdn_props)
                        forced_segments = ml_model.predict(input_df)[0]
                    else:
                        forced_segments = -1.0

                    # fixme: Looping twice (Due to segment slicing flag)
                    if segment_slicing or force_slicing or forced_segments > 1:
                        force_slicing = True
                        if self.engine_props['dynamic_lps']:
                            self._handle_dynamic_slicing(path_list=path_list, path_index=path_index,
                                                         forced_segments=force_slicing)
                            if not self.sdn_props.was_routed:
                                continue
                        else:
                            self._handle_slicing(path_list=path_list, forced_segments=forced_segments)
                            if not self.sdn_props.was_routed:
                                self.sdn_props.num_trans = 1
                                continue
                    else:
                        self.spectrum_obj.spectrum_props.forced_index = forced_index
                        self.spectrum_obj.spectrum_props.forced_core = force_core
                        self.spectrum_obj.spectrum_props.path_list = path_list
                        self.spectrum_obj.spectrum_props.forced_band = forced_band
                        self.spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
                        # Request was blocked for this path
                        if self.spectrum_obj.spectrum_props.is_free is not True:
                            self.sdn_props.block_reason = 'congestion'
                            continue
                        self._update_req_stats(bandwidth=self.sdn_props.bandwidth)

                    self.sdn_props.was_routed = True
                    self.sdn_props.route_time = route_time
                    self.sdn_props.path_weight = self.route_obj.route_props.weights_list[path_index]
                    self.sdn_props.spectrum_object = self.spectrum_obj.spectrum_props

                    if not segment_slicing and not force_slicing:
                        self.sdn_props.is_sliced = False
                        self.allocate()
                    return

            if self.engine_props['max_segments'] > 1 and self.sdn_props.bandwidth != '25' and not segment_slicing:
                segment_slicing = True
                continue

            self.sdn_props.block_reason = 'distance'
            self.sdn_props.was_routed = False
            return
