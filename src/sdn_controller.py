import time

import numpy as np

from helper_scripts.sim_helpers import sort_dict_keys, get_path_mod, find_path_len, sort_nested_dict_vals
from helper_scripts.ml_helpers import get_ml_obs
from arg_scripts.sdn_args import SDNProps
from src.routing import Routing
from src.spectrum_assignment import SpectrumAssignment
from src.grooming import Grooming


class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.sdn_props = SDNProps()

        self.ai_obj = None
        self.grooming_obj = Grooming(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.spectrum_obj = SpectrumAssignment(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                               route_props=self.route_obj.route_props)

    def release(self, lightpath_id: int, slicing_flag: bool = False):
        """
        Removes a previously allocated request from the network.
        :param lightpath_id: Lightpath id for release resources
        """
        for source, dest in zip(self.sdn_props.path_list, self.sdn_props.path_list[1:]):
            for band in self.engine_props['band_list']:
                for core_num in range(self.engine_props['cores_per_link']):
                    core_arr = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num]
                    req_id_arr = np.where(core_arr == lightpath_id)
                    gb_arr = np.where(core_arr == (lightpath_id * -1))

                    for req_index in req_id_arr:
                        self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num][req_index] = 0
                        self.sdn_props.net_spec_dict[(dest, source)]['cores_matrix'][band][core_num][req_index] = 0
                    for gb_index in gb_arr:
                        self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][core_num][gb_index] = 0
                        self.sdn_props.net_spec_dict[(dest, source)]['cores_matrix'][band][core_num][gb_index] = 0

        # Remove lightpath from lightpath_status_dict
        if not  slicing_flag:

            # update the transponder dict
            for node in [self.sdn_props.source, self.sdn_props.destination]:

                if node not in self.sdn_props.transponder_usage_dict:
                    raise KeyError(f"Node '{node}' not found in transponder usage dictionary.")
                    
                self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1
        

            light_id = tuple(sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]]))
            if self.sdn_props.lightpath_status_dict[light_id][lightpath_id]['requests_dict'] and self.engine_props['is_grooming_enabled']:
                raise ValueError('The releasing lightpath are currently using')
            self.sdn_props.lightpath_status_dict[light_id].pop(lightpath_id)

    def _allocate_gb(self, band: str, core_matrix: list, rev_core_matrix: list, core_num: int, end_slot: int, lightpath_id: int):
        if core_matrix[band][core_num][end_slot] != 0.0 or rev_core_matrix[band][core_num][end_slot] != 0.0:
            raise BufferError("Attempted to allocate a taken spectrum.")

        core_matrix[band][core_num][end_slot] = lightpath_id * -1
        rev_core_matrix[band][core_num][end_slot] = lightpath_id * -1

    def allocate(self):
        """
        Allocates a network request.
        """
        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_num
        band = self.spectrum_obj.spectrum_props.curr_band
        lightpath_id = self.spectrum_obj.spectrum_props.lightpath_id

        if self.engine_props['guard_slots'] != 0:
            if self.spectrum_obj.spectrum_props.slots_needed != 1:
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
            core_matrix[band][core_num][start_slot:end_slot] = lightpath_id
            rev_core_matrix[band][core_num][start_slot:end_slot] = lightpath_id

            if self.engine_props['guard_slots']:
                self._allocate_gb(core_matrix=core_matrix, rev_core_matrix=rev_core_matrix, end_slot=end_slot,
                                  core_num=core_num, band=band, lightpath_id=lightpath_id)

    # TODO: No support for multi-band
    def _update_req_stats(self, bandwidth: str, remaining: str):
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
            elif spectrum_key == 'lightpath' and stat_key.split('_')[1] == 'id':
                spectrum_key = 'lightpath_id'
            elif spectrum_key == 'lightpath' and stat_key.split('_')[1] == 'bandwidth':
                spectrum_key = 'lightpath_bandwidth'
            elif spectrum_key == 'remaining':
                self.sdn_props.remaining_bw = remaining
                continue

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

    def _handle_dynamic_slicing(self, path_list: list, path_index: int, forced_segments: int ):
        remaining_bw = self.sdn_props.remaining_bw if self.sdn_props.was_partially_groomed else int(self.sdn_props.bandwidth)
        path_len = find_path_len(path_list=path_list, topology=self.engine_props['topology'])
        bw_mod_dict = sort_dict_keys(dictionary=self.engine_props['mod_per_bw'])
        self.spectrum_obj.spectrum_props.path_list = path_list
        self.sdn_props.num_trans = 0
        while remaining_bw > 0:
            if self.engine_props['fixed_grid']:
                self.sdn_props.was_routed = True
                mod_format, bw = self.spectrum_obj.get_spectrum_dynamic_slicing(mod_format_dict = {}, path_index = path_index)
                if self.spectrum_obj.spectrum_props.is_free:
                    lp_id = self.sdn_props.get_lightpath_id()
                    self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                    self.spectrum_obj.spectrum_props.lightpath_bandwidth = bw
                    # self.spectrum_obj
                    self.allocate()
                    dedicated_bw = bw if remaining_bw > bw else remaining_bw
                    self._update_req_stats(bandwidth=str(dedicated_bw), remaining= str(remaining_bw-dedicated_bw if remaining_bw > dedicated_bw else 0))
                    remaining_bw -= bw
                    self.sdn_props.num_trans += 1
                    self.sdn_props.is_sliced = True
                    # self.sdn_props.was_new_lp_established = True
                    self.sdn_props.was_new_lp_established.append(lp_id)
                    self.sdn_props.was_partially_routed = False
                else:
                    self.sdn_props.was_routed = False
                    self.sdn_props.block_reason = 'congestion'
                    if self.engine_props['can_partially_serve'] and remaining_bw != int(self.sdn_props.bandwidth):
                        if self.sdn_props.was_partially_groomed or path_index == self.engine_props['k_paths'] - 1:
                            self.sdn_props.is_sliced = True
                            self.sdn_props.was_partially_routed = True
                            return
                    if remaining_bw != int(self.sdn_props.bandwidth):
                        for lpid in list(self.sdn_props.was_new_lp_established):
                            self.release(lightpath_id = lpid, slicing_flag = True)
                            self.sdn_props.was_new_lp_established.remove(lpid)
                            lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                            self.sdn_props.lightpath_id_list.pop(lp_idx)
                            self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                            self.sdn_props.start_slot_list.pop(lp_idx)
                            self.sdn_props.band_list.pop(lp_idx)
                            self.sdn_props.core_list.pop(lp_idx)
                            self.sdn_props.end_slot_list.pop(lp_idx)
                            self.sdn_props.xt_list.pop(lp_idx)
                            self.sdn_props.bandwidth_list.pop(lp_idx)
                            self.sdn_props.modulation_list.pop(lp_idx)
                    self.sdn_props.num_trans = 1
                    self.sdn_props.is_sliced = False
                    self.sdn_props.was_partially_routed = False
                    self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)
                    self.sdn_props.was_new_lp_established = list()

                    break
            else:
                for bandwidth, mods_dict in bw_mod_dict.items():
                    # We can't slice to a larger or equal bandwidth
                    if int(bandwidth) >= int(self.sdn_props.bandwidth):
                        continue

                    while remaining_bw > 0:
                        if remaining_bw < int(bandwidth):
                            break 
                        self.sdn_props.was_routed = True
                        mod_format, bw = self.spectrum_obj.get_spectrum_dynamic_slicing(mod_format_dict = mods_dict, path_index = path_index)
                        # in flexigrid slicing, bandwidth in pre-calculated
                        bw = int(bandwidth)
                        if self.spectrum_obj.spectrum_props.is_free:
                            lp_id = self.sdn_props.get_lightpath_id()
                            self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                            self.spectrum_obj.spectrum_props.lightpath_bandwidth = bw
                            self.allocate()
                            dedicated_bw = bw if remaining_bw > bw else remaining_bw
                            self._update_req_stats(bandwidth=str(dedicated_bw), remaining= str(remaining_bw-dedicated_bw if remaining_bw > dedicated_bw else 0))
                            remaining_bw -= bw
                            self.sdn_props.num_trans += 1
                            self.sdn_props.is_sliced = True
                            self.sdn_props.was_new_lp_established.append(lp_id)
                            self.sdn_props.was_partially_routed = False
                        else:
                            break


                    if remaining_bw <= 0:
                        break

                if remaining_bw <= 0:
                    if self.sdn_props.was_routed:
                        self.sdn_props.is_sliced = True
                        return
                else:
                    self.sdn_props.was_routed = False
                    self.sdn_props.block_reason = 'congestion'
                    if self.engine_props['can_partially_serve'] and remaining_bw != int(self.sdn_props.bandwidth):
                        if self.sdn_props.was_partially_groomed or path_index == self.engine_props['k_paths'] - 1:
                            self.sdn_props.is_sliced = True
                            self.sdn_props.was_partially_routed = True
                            return
                    if remaining_bw != int(self.sdn_props.bandwidth):
                        for lpid in list(self.sdn_props.was_new_lp_established):
                            self.release(lightpath_id = lpid, slicing_flag = True)
                            self.sdn_props.was_new_lp_established.remove(lpid)
                            lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                            self.sdn_props.lightpath_id_list.pop(lp_idx)
                            self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                            self.sdn_props.start_slot_list.pop(lp_idx)
                            self.sdn_props.band_list.pop(lp_idx)
                            self.sdn_props.core_list.pop(lp_idx)
                            self.sdn_props.end_slot_list.pop(lp_idx)
                            self.sdn_props.xt_list.pop(lp_idx)
                            self.sdn_props.bandwidth_list.pop(lp_idx)
                            self.sdn_props.modulation_list.pop(lp_idx)
                    self.sdn_props.num_trans = 1
                    self.sdn_props.is_sliced = False
                    self.sdn_props.was_partially_routed = False
                    self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)
                    self.sdn_props.was_new_lp_established = list()
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
        :param lightpath_id: Lightpath id for release
        """


        if request_type == "release":
            if self.engine_props['is_grooming_enabled']:
                lightpath_id_list = self.grooming_obj.handle_grooming(request_type)
            else:
                lightpath_id_list = self.sdn_props.lightpath_id_list
            for lightpath_id in lightpath_id_list:
                self.release(lightpath_id=lightpath_id)
            return
        
        self._init_req_stats()
        # Even if the request is blocked, we still consider one transponder
        self.sdn_props.num_trans = 1
        if self.engine_props['is_grooming_enabled']:
            self.grooming_obj.lightpath_status_dict = self.sdn_props.lightpath_status_dict
            groom_res = self.grooming_obj.handle_grooming(request_type)
            if groom_res:
                return
            else:
                self.sdn_props.was_new_lp_established = list()
                if self.sdn_props.was_partially_groomed: 
                    force_route_matrix = [self.sdn_props.path_list]
                    mod_formats_dict = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                                        nested_key='max_length')
                    force_mod_format = [list(mod_formats_dict.keys())[::-1]]
        start_time = time.time()
        if force_route_matrix is None:
            self.route_obj.get_route()
            route_matrix = self.route_obj.route_props.paths_matrix
        else:
            route_matrix = force_route_matrix
            # TODO: This is an inconsistency
            self.route_obj.route_props.mod_formats_matrix = force_mod_format
            if self.sdn_props.was_partially_groomed:
                self.route_obj.route_props.weights_list = [self.sdn_props.path_weight]
            else:
                # TODO: This must be fixed
                self.route_obj.route_props.weights_list = [0]
        route_time = time.time() - start_time

        segment_slicing = False
        while True:
            for path_index, path_list in enumerate(route_matrix):
                if path_list is not False:
                    self.sdn_props.path_list = path_list
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
                        self.spectrum_obj.spectrum_props.slicing_flag = True
                        if self.engine_props['dynamic_lps']:
                            self._handle_dynamic_slicing(path_list= path_list, path_index= path_index, forced_segments= force_slicing )
                            if not self.sdn_props.was_routed:
                                if self.engine_props['can_partially_serve']:
                                    if not self.sdn_props.was_partially_groomed and not self.sdn_props.was_partially_routed:
                                        continue
                                else:
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
                        self.spectrum_obj.spectrum_props.slicing_flag = False
                        self.spectrum_obj.get_spectrum(mod_format_list=mod_format_list, path_index = path_index)
                        # Request was blocked for this path
                        if self.spectrum_obj.spectrum_props.is_free is not True:
                            self.sdn_props.block_reason = 'congestion'
                            continue
                        lp_id = self.sdn_props.get_lightpath_id()
                        self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                        if self.sdn_props.was_partially_groomed:
                            allocate_br_lp = str(self.sdn_props.remaining_bw)
                        else:
                            allocate_br_lp = self.sdn_props.bandwidth
                        self._update_req_stats(bandwidth = allocate_br_lp, remaining = '0')

                    self.sdn_props.was_routed = True
                    self.sdn_props.route_time = route_time
                    self.sdn_props.path_weight = self.route_obj.route_props.weights_list[path_index]
                    self.sdn_props.spectrum_object = self.spectrum_obj.spectrum_props

                    if not segment_slicing and not force_slicing:
                        if self.sdn_props.was_partially_groomed:
                            self.sdn_props.is_sliced = True
                        else:
                            self.sdn_props.is_sliced = False
                        self.sdn_props.was_new_lp_established.append(lp_id)
                        self.allocate()
                    return

            if self.engine_props['max_segments'] > 1 and self.sdn_props.bandwidth != '25' and not segment_slicing:
                segment_slicing = True
                continue

            if self.engine_props['can_partially_serve']: 
                if self.sdn_props.remaining_bw < int(self.sdn_props.bandwidth):
                    self.sdn_props.was_partially_routed = True
            else:
                if self.sdn_props.was_partially_groomed:
                    self.grooming_obj.handle_grooming("release")
                    self.sdn_props.was_partially_routed = False
                    self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)


            self.sdn_props.block_reason = 'distance'
            self.sdn_props.was_routed = False
            return
