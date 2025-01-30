import os

import gymnasium as gym
import numpy as np

from helper_scripts.rl.rl_setup_helpers import setup_rl_sim, RLSetupHelper
from helper_scripts.rl.rl_helpers import RLHelpers
from helper_scripts.sim_helpers import find_path_len, get_path_mod
from helper_scripts.rl.multi_agent_helpers import PathAgent, CoreAgent, SpectrumAgent

from arg_scripts.rl_args import RLProps, VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS


class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Controls all reinforcement learning assisted simulations.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, custom_callback: object = None, sim_dict: dict = None,
                 **kwargs):  # pylint: disable=unused-argument
        super().__init__()

        self.rl_props = RLProps()

        if sim_dict is None:
            self.sim_dict = setup_rl_sim()['s1']
        else:
            self.sim_dict = sim_dict['s1']
        self.rl_props.super_channel_space = self.sim_dict['super_channel_space']

        self.iteration = 0
        self.options = None
        self.optimize = None
        self.callback = custom_callback
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None

        # TODO: (drl_path_agents) Change all inputs to account for the new object
        self.rl_help_obj = RLHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj)
        # TODO: (drl_path_agents) Check these parameters and when setup helper is defined
        self._setup_agents()

        self.modified_props = None
        self.sim_props = None
        self.setup_helper = RLSetupHelper(sim_env=self)

        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        self.observation_space = self.spectrum_agent.get_obs_space()
        self.action_space = self.spectrum_agent.get_action_space()

    def _setup_agents(self):
        self.path_agent = PathAgent(path_algorithm=self.sim_dict['path_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.core_agent = CoreAgent(core_algorithm=self.sim_dict['core_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.spectrum_agent = SpectrumAgent(spectrum_algorithm=self.sim_dict['spectrum_algorithm'],
                                            rl_props=self.rl_props)

    def _check_terminated(self):
        if self.rl_props.arrival_count == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            # The spectrum agent is handled by SB3 automatically
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_dict['is_training']:
                self.path_agent.end_iter()
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_dict['is_training']:
                self.core_agent.end_iter()
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    def _update_helper_obj(self, action: list, bandwidth: str):
        self.rl_help_obj.path_index = self.rl_props.path_index
        self.rl_help_obj.core_num = self.rl_props.core_index

        if self.sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            self.rl_help_obj.rl_props.forced_index = action
        else:
            self.rl_help_obj.rl_props.forced_index = None

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.handle_releases()
        self.rl_help_obj.update_route_props(chosen_path=self.rl_props.chosen_path_list, bandwidth=bandwidth)

    def _handle_test_train_step(self, was_allocated: bool, path_length: int):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration, path_length=path_length)
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration)
            else:
                raise NotImplementedError
        else:
            self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration, path_length=path_length)
            self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration)

    def step(self, action: list):
        """
        Handles a single time step in the simulation.

        :param action: A list of actions from the DRL agent.
        :return: The new observation, reward, if terminated, if truncated, and misc. info.
        :rtype: tuple
        """
        req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count]
        req_id = req_info_dict['req_id']
        bandwidth = req_info_dict['bandwidth']

        self._update_helper_obj(action=action, bandwidth=bandwidth)
        self.rl_help_obj.allocate()
        reqs_status_dict = self.engine_obj.reqs_status_dict

        was_allocated = req_id in reqs_status_dict
        path_length = self.route_obj.route_props.weights_list[0]
        self._handle_test_train_step(was_allocated=was_allocated, path_length=path_length)
        self.rl_help_obj.update_snapshots()
        drl_reward = self.spectrum_agent.get_reward(was_allocated=was_allocated)

        self.rl_props.arrival_count += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    def _handle_path_train_test(self):
        if 'bandit' in self.sim_dict['path_algorithm']:
            self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
            self.route_obj.engine_props['route_method'] = 'k_shortest_path'
            self.route_obj.get_route()

        self.path_agent.get_route(route_obj=self.route_obj)
        self.rl_help_obj.rl_props.chosen_path_list = [self.rl_props.chosen_path_list]
        self.route_obj.route_props.paths_matrix = self.rl_help_obj.rl_props.chosen_path_list
        self.rl_props.core_index = None
        self.rl_props.forced_index = None

    def _determine_core_penalty(self):
        # Default to first fit if all paths fail
        self.rl_props.chosen_path = [self.route_obj.route_props.paths_matrix[0]]
        self.rl_props.chosen_path_index = 0
        for path_index, path_list in enumerate(self.route_obj.route_props.paths_matrix):
            mod_format_list = self.route_obj.route_props.mod_formats_matrix[path_index]

            was_allocated = self.rl_help_obj.mock_handle_arrival(engine_props=self.engine_obj.engine_props,
                                                                 sdn_props=self.rl_props.mock_sdn_dict,
                                                                 mod_format_list=mod_format_list, path_list=path_list)

            if was_allocated:
                self.rl_props.chosen_path_list = [path_list]
                self.rl_props.chosen_path_index = path_index
                self.core_agent.no_penalty = False
                break

            self.core_agent.no_penalty = True

    def _handle_core_train(self):
        self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
        self.route_obj.engine_props['route_method'] = 'k_shortest_path'
        self.route_obj.get_route()
        self._determine_core_penalty()

        self.rl_props.forced_index = None
        # TODO: (drl_path_agents) Check to make sure this doesn't affect anything
        # try:
        #     req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count]
        # except IndexError:
        #     req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count - 1]

        self.core_agent.get_core()

    def _handle_spectrum_train(self):
        self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
        self.route_obj.engine_props['route_method'] = 'shortest_path'
        self.route_obj.get_route()
        # TODO: (drl_path_agents) Change name in rl props
        self.rl_props.paths_list = self.route_obj.route_props.paths_matrix
        self.rl_props.chosen_path = self.route_obj.route_props.paths_matrix
        self.rl_props.path_index = 0
        self.rl_props.core_index = None

    def _handle_test_train_obs(self, curr_req: dict):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self._handle_path_train_test()
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                self._handle_core_train()
            elif self.sim_dict['spectrum_algorithm'] not in ('first_fit', 'best_fit', ' last_fit'):
                self._handle_spectrum_train()
            else:
                raise NotImplementedError
        else:
            self._handle_path_train_test()
            self.core_agent.get_core()

        path_len = find_path_len(path_list=self.rl_props.chosen_path_list[0],
                                 topology=self.engine_obj.topology)
        path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)

        return path_mod

    # fixme (drl_path_agents)
    def _get_spectrum_obs(self, curr_req: dict):  # pylint: disable=unused-argument
        # path_mod = self._handle_test_train_obs(curr_req=curr_req)
        # if path_mod is not False:
        #     slots_needed = curr_req['mod_formats'][path_mod]['slots_needed']
        # super_channels, no_penalty = self.rl_help_obj.get_super_channels(slots_needed=slots_needed,
        #                                                                  num_channels=self.rl_props[
        #                                                                      'super_channel_space'])
        # No penalty for DRL agent, mistake not made by it
        # else:
        slots_needed = -1
        no_penalty = True
        super_channels = np.array([100.0, 100.0, 100.0])

        self.spectrum_agent.no_penalty = no_penalty
        source_obs = np.zeros(self.rl_props.num_nodes)
        source_obs[self.rl_props.source] = 1.0
        dest_obs = np.zeros(self.rl_props.num_nodes)
        dest_obs[self.rl_props.destination] = 1.0

        return slots_needed, source_obs, dest_obs, super_channels

    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.rl_props.arrival_count == self.engine_obj.engine_props['num_requests']:
            curr_req = self.rl_props.arrival_list[self.rl_props.arrival_count - 1]
        else:
            curr_req = self.rl_props.arrival_list[self.rl_props.arrival_count]

        self.rl_help_obj.handle_releases()
        self.rl_props.source = int(curr_req['source'])
        self.rl_props.destination = int(curr_req['destination'])
        self.rl_props.mock_sdn_dict = self.rl_help_obj.update_mock_sdn(curr_req=curr_req)

        # TODO: (drl_path_agents) This is for spectrum assignment, ignored return for now
        _ = self._handle_test_train_obs(curr_req=curr_req)
        slots_needed, source_obs, dest_obs, super_channels = self._get_spectrum_obs(curr_req=curr_req)
        obs_dict = {
            'slots_needed': slots_needed,
            'source': source_obs,
            'destination': dest_obs,
            'super_channels': super_channels,
        }
        return obs_dict

    def _init_envs(self):
        self.setup_helper.init_envs()

    def _create_input(self):
        self.setup_helper.create_input()

    def _load_models(self):
        self.setup_helper.load_models()

    def setup(self):
        """
        Sets up this class.
        """
        self.optimize = self.sim_dict['optimize']
        self.rl_props.k_paths = self.sim_dict['k_paths']
        self.rl_props.cores_per_link = self.sim_dict['cores_per_link']
        # TODO: Only support for 'c' band...Maybe add multi-band (drl_path_agents)
        self.rl_props.spectral_slots = self.sim_dict['c_band']

        self._create_input()

        self.sim_dict['arrival_dict'] = {
            'start': self.sim_dict['erlang_start'],
            'stop': self.sim_dict['erlang_stop'],
            'step': self.sim_dict['erlang_step'],
        }

        self.engine_obj.engine_props['erlang'] = float(self.sim_dict['erlang_start'])
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        erlang = self.engine_obj.engine_props['erlang']
        holding_time = self.engine_obj.engine_props['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = (cores_per_link * erlang) / holding_time

        self.engine_obj.engine_props['band_list'] = ['c']

    def _init_props_envs(self):
        self.rl_props.arrival_count = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.rl_help_obj.topology = self.engine_obj.topology
        self.rl_props.num_nodes = len(self.engine_obj.topology.nodes)

        if self.iteration == 0:
            self._init_envs()

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.route_obj = self.route_obj

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        """
        Resets necessary variables after each iteration of the simulation.

        :param seed: Seed for random generation.
        :param options: Custom option input.
        :return: The first observation and misc. information.
        :rtype: tuple
        """
        super().reset(seed=seed)
        self.rl_props.arrival_list = list()
        self.rl_props.depart_list = list()

        # TODO: fixme statement breaks for DRL (drl_path_agents)
        if self.optimize is None:
            self.iteration = 0
            self.setup()

        self._init_props_envs()
        if not self.sim_dict['is_training'] and self.iteration == 0:
            self._load_models()
        if seed is None:
            seed = self.iteration + 1

        self.rl_help_obj.reset_reqs_dict(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
