import os

import networkx as nx
import numpy as np

from arg_scripts.routing_args import RoutingProps
from helper_scripts.routing_helpers import RoutingHelpers
from helper_scripts.sim_helpers import find_path_len, get_path_mod, find_free_slots, sort_nested_dict_vals, get_path_mod_fixedgrid


class Routing:
    """
    This class contains methods related to routing network requests.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.route_props = RoutingProps()

        self.route_help_obj = RoutingHelpers(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                             route_props=self.route_props)

    def _find_most_cong_link(self, path_list: list):
        most_cong_link = None
        most_cong_slots = -1

        for i in range(len(path_list) - 1):
            link_dict = self.sdn_props.net_spec_dict[(path_list[i], path_list[i + 1])]
            free_slots = 0
            for band in link_dict['cores_matrix']:
                cores_matrix = link_dict['cores_matrix'][band]
                for core_arr in cores_matrix:
                    free_slots += np.sum(core_arr == 0)

            if free_slots < most_cong_slots or most_cong_link is None:
                most_cong_slots = free_slots
                most_cong_link = link_dict

        self.route_props.paths_matrix.append({'path_list': path_list,
                                              'link_dict': {'link': most_cong_link,
                                                            'free_slots': most_cong_slots}})

    def _find_least_cong(self):
        # Sort dictionary by number of free slots, descending
        sorted_paths_list = sorted(self.route_props.paths_matrix, key=lambda d: d['link_dict']['free_slots'],
                                   reverse=True)

        self.route_props.paths_matrix = [sorted_paths_list[0]['path_list']]
        self.route_props.weights_list = [int(sorted_paths_list[0]['link_dict']['free_slots'])]
        # TODO: Constant QPSK format (Ask Arash)
        self.route_props.mod_formats_matrix.append(['QPSK'])

    def find_least_cong(self):
        """
        Find the least congested path in the network.
        """
        all_paths_obj = nx.shortest_simple_paths(self.engine_props['topology'], self.sdn_props.source,
                                                 self.sdn_props.destination)
        min_hops = None
        for i, path_list in enumerate(all_paths_obj):
            num_hops = len(path_list)
            if i == 0:
                min_hops = num_hops
                self._find_most_cong_link(path_list=path_list)
            else:
                if num_hops <= min_hops + 1:
                    self._find_most_cong_link(path_list=path_list)
                # We exceeded minimum hops plus one, return the best path
                else:
                    self._find_least_cong()
                    return

    def find_least_weight(self, weight: str):
        """
        Find the path with respect to a given weight.

        :param weight: Determines the weight to consider for finding the path.
        """
        paths_obj = nx.shortest_simple_paths(G=self.sdn_props.topology, source=self.sdn_props.source,
                                             target=self.sdn_props.destination, weight=weight)

        for path_list in paths_obj:
            # If cross-talk, our path weight is the summation across the path
            if weight == 'xt_cost':
                resp_weight = sum(self.sdn_props.topology[path_list[i]][path_list[i + 1]][weight]
                                  for i in range(len(path_list) - 1))

                mod_formats = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                                    nested_key='max_length')
                path_len = find_path_len(path_list=path_list, topology=self.sdn_props.topology)
                mod_format_list = list()
                for mod_format in mod_formats:
                    if self.sdn_props.mod_formats_dict[mod_format]['max_length'] >= path_len:
                        mod_format_list.append(mod_format)
                    else:
                        mod_format_list.append(False)

                self.route_props.mod_formats_matrix.append(mod_format_list)
            else:
                resp_weight = find_path_len(path_list=path_list, topology=self.sdn_props.topology)
                mod_format = get_path_mod(self.sdn_props.mod_formats, resp_weight)
                self.route_props.mod_formats_matrix.append([mod_format])

            self.route_props.weights_list.append(resp_weight)
            self.route_props.paths_matrix.append(path_list)
            break

    def find_k_shortest(self):
        """
        Finds the k-shortest paths with respect to length from source to destination.
        """
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.engine_props['topology'], source=self.sdn_props.source,
                                             target=self.sdn_props.destination, weight='length')

        for k, path_list in enumerate(paths_obj):
            if k > self.engine_props['k_paths'] - 1:
                break
            path_len = find_path_len(path_list=path_list, topology=self.engine_props['topology'])
            chosen_bw = self.sdn_props.bandwidth
            if not self.engine_props['pre_calc_mod_selection']:
                mod_formats_list = [get_path_mod_fixedgrid(mods_dict=self.engine_props['mod_per_bw'][chosen_bw], path_len=path_len) if self.engine_props['fixed_grid'] else
                    get_path_mod(mods_dict=self.engine_props['mod_per_bw'][chosen_bw], path_len=path_len)]
            else:
                mod_formats_dict = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                                         nested_key='max_length')
                mod_formats_list = list(mod_formats_dict.keys())
            self.route_props.paths_matrix.append(path_list)
            self.route_props.mod_formats_matrix.append(mod_formats_list)
            self.route_props.weights_list.append(path_len)

    def find_least_nli(self):
        """
        Finds and selects the path with the least amount of non-linear impairment.
        """
        # Bidirectional links are identical, therefore, we don't have to check each one
        for link_tuple in list(self.sdn_props.net_spec_dict.keys())[::2]:
            source, destination = link_tuple[0], link_tuple[1]
            num_spans = self.sdn_props.topology[source][destination]['length'] / self.route_props.span_len
            bandwidth = self.sdn_props.bandwidth
            # TODO: Constant QPSK for slots needed (Ask Arash)
            slots_needed = self.engine_props['mod_per_bw'][bandwidth]['QPSK']['slots_needed']
            self.sdn_props.slots_needed = slots_needed

            link_cost = self.route_help_obj.get_nli_cost(link_tuple=link_tuple, num_span=num_spans)
            self.sdn_props.topology[source][destination]['nli_cost'] = link_cost

        self.find_least_weight(weight='nli_cost')

    def find_least_xt(self):
        """
        Finds the path with the least amount of intra-core crosstalk interference.

        :return: The selected path with the least amount of interference.
        :rtype: list
        """
        # At the moment, we have identical bidirectional links (no need to loop over all links)
        for link_list in list(self.sdn_props.net_spec_dict.keys())[::2]:
            source, destination = link_list[0], link_list[1]
            num_spans = self.sdn_props.topology[source][destination]['length'] / self.route_props.span_len

            free_slots_dict = find_free_slots(net_spec_dict=self.sdn_props.net_spec_dict, link_tuple=link_list)
            xt_cost = self.route_help_obj.find_xt_link_cost(free_slots_dict=free_slots_dict, link_list=link_list)

            if self.engine_props['xt_type'] == 'with_length':
                if self.route_props.max_link_length is None:
                    self.route_help_obj.get_max_link_length()

                link_cost = self.sdn_props.topology[source][destination]['length'] / \
                            self.route_props.max_link_length
                link_cost *= self.engine_props['beta']
                link_cost += (1 - self.engine_props['beta']) * xt_cost
            elif self.engine_props['xt_type'] == 'without_length':
                link_cost = num_spans * xt_cost
            else:
                raise ValueError(f"XT type not recognized, expected with or without_length, "
                                 f"got: {self.engine_props['xt_type']}")

            self.sdn_props.topology[source][destination]['xt_cost'] = link_cost
            self.sdn_props.topology[destination][source]['xt_cost'] = link_cost

        self.find_least_weight(weight='xt_cost')

    def _init_route_info(self):
        self.route_props.paths_matrix = list()
        self.route_props.mod_formats_matrix = list()
        self.route_props.weights_list = list()
        self.route_props.path_index_list = list()
        self.route_props.connection_index = None

    # TODO: Put all pre-loaded stuff in a JSON or similar file structure
    def load_k_shortest(self):
        """
        Load the k-shortest paths from an external file.
        """
        base_path = os.path.join('data', 'pre_calc', self.engine_props['network'], 'paths')

        if self.engine_props['network'] == 'USbackbone60':
            file_path = os.path.join(base_path, 'USB6014-10SP.npy')
            loaded_data_dict = np.load(file_path, allow_pickle=True)
        elif self.engine_props['network'] == 'Spainbackbone30':
            file_path = os.path.join(base_path, 'SPNB3014-10SP.npy')
            loaded_data_dict = np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Missing precalculated path dataset for '{self.engine_props['network']}' topology")
        src_des_list = [int(self.sdn_props.source), int(self.sdn_props.destination)]

        path_cnt = 0
        for pre_comp_matrix in loaded_data_dict:
            paths_calculated = 0
            src_dest_index = 5
            first_node = pre_comp_matrix[src_dest_index][0][0][0][0]
            last_node = pre_comp_matrix[src_dest_index][0][0][0][-1]
            if first_node in src_des_list and last_node in src_des_list:
                self.route_props.connection_index = path_cnt
                for path in pre_comp_matrix[5][0]:
                    if paths_calculated == self.engine_props['k_paths']:
                        break
                    if first_node == int(self.sdn_props.source):
                        temp_path = list(path[0])
                    else:
                        temp_path = list(path[0][::-1])

                    temp_path = list(map(str, temp_path))
                    path_len = pre_comp_matrix[3][0][paths_calculated]
                    if path_len.dtype != np.float64:
                        path_len = path_len.astype(np.float64)
                    mod_formats_dict = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                                             nested_key='max_length')
                    mod_formats_list = list(mod_formats_dict.keys())
                    self.route_props.paths_matrix.append(temp_path)
                    self.route_props.mod_formats_matrix.append(mod_formats_list[::-1])
                    self.route_props.weights_list.append(path_len)
                    self.route_props.path_index_list.append(paths_calculated)

                    paths_calculated += 1
                break
            path_cnt += 1

    def get_route(self):
        """
        Controls the class by finding the appropriate routing function.

        :return: None
        """
        self._init_route_info()

        if self.engine_props['route_method'] == 'nli_aware':
            self.find_least_nli()
        elif self.engine_props['route_method'] == 'xt_aware':
            self.find_least_xt()
        elif self.engine_props['route_method'] == 'least_congested':
            self.find_least_cong()
        elif self.engine_props['route_method'] == 'shortest_path':
            self.find_least_weight(weight='length')
        elif self.engine_props['route_method'] == 'k_shortest_path':
            self.find_k_shortest()
        elif self.engine_props['route_method'] == 'external_ksp':
            self.load_k_shortest()
        else:
            raise NotImplementedError(f"Routing method not recognized, got: {self.engine_props['route_method']}.")
