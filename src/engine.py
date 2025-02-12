# Standard library imports
import copy
import os
import signal

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from src.request_generator import get_requests
from src.sdn_controller import SDNController
from helper_scripts.stats_helpers import SimStats
from helper_scripts.ml_helpers import load_model


class Engine:
    """
    Controls a single simulation.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.net_spec_dict = dict()
        self.reqs_dict = None
        self.reqs_status_dict = dict()

        self.iteration = 0
        self.topology = nx.Graph()
        self.sim_info = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                     self.engine_props['sim_start'])

        self.sdn_obj = SDNController(engine_props=self.engine_props)
        self.stats_obj = SimStats(engine_props=self.engine_props, sim_info=self.sim_info)

        self.ml_model = None

    def update_arrival_params(self, curr_time: float):
        """
        Updates parameters for a request after attempted allocation.

        :param curr_time: The current simulated time.
        """
        sdn_props = self.sdn_obj.sdn_props
        self.stats_obj.iter_update(req_data=self.reqs_dict[curr_time], sdn_data=sdn_props)
        if sdn_props.was_routed:
            self.stats_obj.curr_trans = sdn_props.num_trans

            self.reqs_status_dict.update({self.reqs_dict[curr_time]['req_id']: {
                "mod_format": sdn_props.modulation_list,
                "path": sdn_props.path_list,
                "is_sliced": sdn_props.is_sliced,
                "was_routed": sdn_props.was_routed,
                "core_list": sdn_props.core_list,
                "band": sdn_props.band_list,
                "start_slot_list": sdn_props.start_slot_list,
                "end_slot_list": sdn_props.end_slot_list,
                "bandwidth_list": sdn_props.bandwidth_list,
                "snr_cost": sdn_props.xt_list,
            }})

    def handle_arrival(self, curr_time: float, force_route_matrix: list = None, force_core: int = None,
                       force_slicing: bool = False, forced_index: int = None, force_mod_format: str = None):
        """
        Updates the SDN controller to handle an arrival request and retrieves relevant request statistics.

        :param curr_time: The arrival time of the request.
        :param force_route_matrix: Passes forced routes to the SDN controller.
        :param force_slicing: Forces slicing in the SDN controller.
        :param forced_index: Forces an index in the SDN controller.
        :param force_mod_format: Forces a modulation format.
        :param force_core: Force a certain core for allocation.
        """
        for req_key, req_value in self.reqs_dict[curr_time].items():
            # TODO: This should be changed in reqs_dict directly
            if req_key == 'mod_formats':
                req_key = 'mod_formats_dict'
            self.sdn_obj.sdn_props.update_params(key=req_key, spectrum_key=None, spectrum_obj=None, value=req_value)

        self.sdn_obj.handle_event(request_type='arrival', force_route_matrix=force_route_matrix,
                                  force_slicing=force_slicing, forced_index=forced_index, force_core=force_core,
                                  ml_model=self.ml_model, req_dict=self.reqs_dict[curr_time],
                                  force_mod_format=force_mod_format)
        self.net_spec_dict = self.sdn_obj.sdn_props.net_spec_dict
        self.update_arrival_params(curr_time=curr_time)

    def handle_release(self, curr_time: float):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request.
        """
        for req_key, req_value in self.reqs_dict[curr_time].items():
            # TODO: This should be changed in reqs_dict directly
            if req_key == 'mod_formats':
                req_key = 'mod_formats_dict'
            self.sdn_obj.sdn_props.update_params(key=req_key, spectrum_key=None, spectrum_obj=None, value=req_value)

        if self.reqs_dict[curr_time]['req_id'] in self.reqs_status_dict:
            self.sdn_obj.sdn_props.path_list = self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']]['path']
            self.sdn_obj.handle_event(req_dict=self.reqs_dict[curr_time], request_type='release')
            self.net_spec_dict = self.sdn_obj.sdn_props.net_spec_dict
        # Request was blocked, nothing to release
        else:
            pass

    def create_topology(self):
        """
        Create the physical topology of the simulation.
        """
        self.net_spec_dict = {}
        self.topology.add_nodes_from(self.engine_props['topology_info']['nodes'])

        # TODO: (drl_path_agents) This list should be stored somewhere else, like an arguments script
        self.engine_props['band_list'] = list()
        for band in ['c', 'l', 's', 'o', 'e']:
            try:
                if self.engine_props[f'{band}_band']:
                    self.engine_props['band_list'].append(band)
            except KeyError:
                continue

        for link_num, link_data in self.engine_props['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']

            cores_matrix = dict()
            for band in self.engine_props['band_list']:
                # TODO: This variable name for bands changes and is not consistent
                band_slots = self.engine_props[f'{band}_band']
                cores_matrix[band] = np.zeros((link_data['fiber']['num_cores'], band_slots))

            self.net_spec_dict[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_dict[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)

        self.engine_props['topology'] = self.topology
        self.stats_obj.topology = self.topology
        self.sdn_obj.sdn_props.net_spec_dict = self.net_spec_dict
        self.sdn_obj.sdn_props.topology = self.topology

    def generate_requests(self, seed: int):
        """
        Calls the request generator to generate requests.

        :param seed: The seed to use for the random generation.
        """
        self.reqs_dict = get_requests(seed=seed, engine_props=self.engine_props)
        self.reqs_dict = dict(sorted(self.reqs_dict.items()))

    def handle_request(self, curr_time: float, req_num: int):
        """
        Carries out arrival or departure functions for a given request.

        :param curr_time: The current simulated time.
        :param req_num: The request number.
        """
        req_type = self.reqs_dict[curr_time]["request_type"]
        if req_type == "arrival":
            old_net_spec_dict = copy.deepcopy(self.net_spec_dict)
            old_req_info_dict = copy.deepcopy(self.reqs_dict[curr_time])
            self.handle_arrival(curr_time=curr_time)

            if self.engine_props['save_snapshots'] and req_num % self.engine_props['snapshot_step'] == 0:
                self.stats_obj.update_snapshot(net_spec_dict=self.net_spec_dict, req_num=req_num)

            if self.engine_props['output_train_data']:
                was_routed = self.sdn_obj.sdn_props.was_routed
                if was_routed:
                    req_info_dict = self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']]
                    self.stats_obj.update_train_data(old_req_info_dict=old_req_info_dict, req_info_dict=req_info_dict,
                                                     net_spec_dict=old_net_spec_dict)

        elif req_type == "release":
            self.handle_release(curr_time=curr_time)
        else:
            raise NotImplementedError(f'Request type unrecognized. Expected arrival or release, '
                                      f'got: {req_type}')

    def end_iter(self, iteration: int, print_flag: bool = True, base_fp: str = None):
        """
        Updates iteration statistics.

        :param iteration: The current iteration.
        :param print_flag: Whether to print or not.
        :param base_fp: The base file path to save output statistics.
        """
        self.stats_obj.get_blocking()
        self.stats_obj.end_iter_update()
        # Some form of ML/RL is being used, ignore confidence intervals for training and testing
        if not self.engine_props['is_training']:
            resp = bool(self.stats_obj.get_conf_inter())
        else:
            resp = False
        if (iteration + 1) % self.engine_props['print_step'] == 0 or iteration == 0:
            self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'], print_flag=print_flag)

        if (iteration + 1) % self.engine_props['save_step'] == 0 or iteration == 0 or (iteration + 1) == \
                self.engine_props['max_iters']:
            self.stats_obj.save_stats(base_fp=base_fp)

        return resp

    def init_iter(self, iteration: int, seed: int = None, print_flag: bool = True, trial: int = None):
        """
        Initializes an iteration.

        :param iteration: The current iteration number.
        :param seed: The seed to use for the random generation.
        :param trial: The trial number.
        :param print_flag: Flag to determine printing iter info.
        """
        if trial is not None:
            self.engine_props['thread_num'] = f's{trial + 1}'

        self.iteration = iteration

        self.stats_obj.iteration = iteration
        self.stats_obj.init_iter_stats()
        # To prevent incomplete saves
        try:
            signal.signal(signal.SIGINT, self.stats_obj.save_stats)
            signal.signal(signal.SIGTERM, self.stats_obj.save_stats)
        # Signal only works in the main thread...
        except ValueError:
            pass

        if iteration == 0 and print_flag:
            print(f"Simulation started for Erlang: {self.engine_props['erlang']} "
                  f"simulation number: {self.engine_props['thread_num']}.")

            if self.engine_props['deploy_model']:
                self.ml_model = load_model(engine_props=self.engine_props)

        # You can pass a list of seeds, a constant seed, or default to iteration number
        if seed is None:
            seed = self.engine_props["seeds"][iteration] if self.engine_props["seeds"] else iteration + 1

        self.generate_requests(seed)

    def run(self, seed: int = None):
        """
        Controls the Engine class methods.
        """
        self.create_topology()
        for iteration in range(self.engine_props["max_iters"]):
            self.init_iter(iteration=iteration, seed=seed)
            req_num = 1
            for curr_time in self.reqs_dict:
                self.handle_request(curr_time=curr_time, req_num=req_num)

                if self.reqs_dict[curr_time]['request_type'] == 'arrival':
                    req_num += 1

            end_iter = self.end_iter(iteration=iteration)
            if end_iter:
                break

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
