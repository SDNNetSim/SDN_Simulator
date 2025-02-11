import os

from rl_scripts.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS


class SimEnvUtils:
    """
    Provides helper methods for managing steps, training/testing logic, and observations
    in the SimEnv reinforcement learning environment.
    """

    def __init__(self, sim_env):
        """
        Initializes the RL step helper with access to the SimEnv instance.

        :param sim_env: The main simulation environment object.
        """
        self.sim_env = sim_env

    def check_terminated(self):
        """
        Checks whether the simulation has reached termination conditions.

        :return: A boolean indicating if the simulation is terminated.
        """
        if self.sim_env.rl_props.arrival_count == (self.sim_env.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_env.sim_dict[
                'is_training']:
                self.sim_env.path_agent.end_iter()
            elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_env.sim_dict[
                'is_training']:
                self.sim_env.core_agent.end_iter()
            self.sim_env.engine_obj.end_iter(iteration=self.sim_env.iteration, print_flag=False, base_fp=base_fp)
            self.sim_env.iteration += 1
        else:
            terminated = False

        return terminated

    def handle_test_train_step(self, was_allocated: bool, path_length: int, trial: int):
        """
        Handles updates specific to training or testing during the current simulation step.

        :param was_allocated: Whether the resource allocation was successful.
        :param trial: The current trial number.
        :param path_length: The length of the chosen path.
        """
        if self.sim_env.sim_dict['is_training']:
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.sim_env.path_agent.update(was_allocated=was_allocated,
                                               net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                               iteration=self.sim_env.iteration, path_length=path_length,
                                               trial=trial)
            elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.sim_env.path_agent.update(was_allocated=was_allocated,
                                           net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                           iteration=self.sim_env.iteration, path_length=path_length)
            self.sim_env.core_agent.update(was_allocated=was_allocated,
                                           net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                           iteration=self.sim_env.iteration)

    def handle_step(self, action: int, is_drl_agent: bool):
        """
        Handles path-related decisions during training and testing phases.
        """
        # Q-learning has access to its own paths, everything else needs the route object
        if 'bandit' in self.sim_env.sim_dict['path_algorithm'] or is_drl_agent:
            self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
            self.sim_env.route_obj.engine_props['route_method'] = 'k_shortest_path'
            self.sim_env.route_obj.get_route()

        self.sim_env.path_agent.get_route(route_obj=self.sim_env.route_obj, action=action)
        self.sim_env.rl_help_obj.rl_props.chosen_path_list = [self.sim_env.rl_props.chosen_path_list]
        self.sim_env.route_obj.route_props.paths_matrix = self.sim_env.rl_help_obj.rl_props.chosen_path_list
        self.sim_env.rl_props.core_index = None
        self.sim_env.rl_props.forced_index = None

    def handle_core_train(self):
        """
        Handles core-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props['route_method'] = 'k_shortest_path'
        self.sim_env.route_obj.get_route()
        self.sim_env.sim_env_helper.determine_core_penalty()

        self.sim_env.rl_props.forced_index = None

        self.sim_env.core_agent.get_core()

    def handle_spectrum_train(self):
        """
        Handles spectrum-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props['route_method'] = 'shortest_path'
        self.sim_env.route_obj.get_route()
        self.sim_env.rl_props.paths_list = self.sim_env.route_obj.route_props.paths_matrix
        self.sim_env.rl_props.chosen_path = self.sim_env.route_obj.route_props.paths_matrix
        self.sim_env.rl_props.path_index = 0
        self.sim_env.rl_props.core_index = None

    def get_obs(self):
        """
        Generates the current observation for the agent based on the environment state.

        :return: A dictionary containing observation components.
        """
        if self.sim_env.rl_props.arrival_count == self.sim_env.engine_obj.engine_props['num_requests']:
            curr_req = self.sim_env.rl_props.arrival_list[self.sim_env.rl_props.arrival_count - 1]
        else:
            curr_req = self.sim_env.rl_props.arrival_list[self.sim_env.rl_props.arrival_count]

        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_props.source = int(curr_req['source'])
        self.sim_env.rl_props.destination = int(curr_req['destination'])
        self.sim_env.rl_props.mock_sdn_dict = self.sim_env.rl_help_obj.update_mock_sdn(curr_req=curr_req)

        source_obs, dest_obs = self.sim_env.sim_env_helper.get_drl_obs()
        obs_dict = {
            'source': source_obs,
            'destination': dest_obs,
        }
        return obs_dict
