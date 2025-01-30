import os

import numpy as np
import optuna

from stable_baselines3 import PPO

from src.spectrum_assignment import SpectrumAssignment

from helper_scripts.rl.rl_setup_helpers import setup_ppo
from helper_scripts.sim_helpers import find_path_len, get_path_mod, get_hfrag
from helper_scripts.sim_helpers import find_path_cong, classify_cong, find_core_cong
from helper_scripts.sim_helpers import modify_multiple_json_values
from helper_scripts.sim_helpers import get_arrival_rates, run_simulation_for_arrival_rates, save_study_results
from helper_scripts.rl.rl_setup_helpers import print_info
from helper_scripts.rl.rl_zoo_helpers import run_rl_zoo

from arg_scripts.rl_args import get_optuna_hyperparams
from arg_scripts.rl_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS
from arg_scripts.rl_args import VALID_SPECTRUM_ALGORITHMS
from arg_scripts.sdn_args import SDNProps


# TODO: (drl_path_agents) Name changed
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

        # TODO: (drl_path_agents) Fix inconsistency e.g., if route object isn't the same in sdn controller
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
        # TODO: (drl_path_agents) Fix this in RL merge
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

    def update_helper_obj(self, action: list, bandwidth: str):
        """
        Updates the helper object with new actions and configurations.
        """
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

    def handle_test_train_obs(self, curr_req: dict):
        """
        Handles path and core selection during training/testing phases based on the current request.

        Returns:
            Path modulation format, if available.
        """
        if self.sim_env.sim_dict['is_training']:
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.sim_env.step_helper.handle_path_train_test()
            elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                self.sim_env.step_helper.handle_core_train()
            elif self.sim_env.sim_dict['spectrum_algorithm'] not in ('first_fit', 'best_fit', 'last_fit'):
                self.sim_env.step_helper.handle_spectrum_train()
            else:
                raise NotImplementedError
        else:
            self.sim_env.step_helpers.handle_path_train_test()
            self.sim_env.core_agent.get_core()

        path_len = find_path_len(path_list=self.sim_env.rl_props.chosen_path_list[0],
                                 topology=self.sim_env.engine_obj.topology)
        path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)

        return path_mod

    # fixme: (drl_path_agents)
    def get_spectrum_obs(self, curr_req: dict):  # pylint: disable=unused-argument
        """
        Generates the spectrum observation for the given request.

        Returns:
            Spectrum-related observation components.
        """
        # TODO: (drl_path_agents) Add logic for full spectrum assignment, skipping penalty for now
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
        super_channels = np.array([100.0, 100.0, 100.0])  # Placeholder values

        self.sim_env.spectrum_agent.no_penalty = no_penalty
        source_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        source_obs[self.sim_env.rl_props.source] = 1.0
        dest_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        dest_obs[self.sim_env.rl_props.destination] = 1.0

        return slots_needed, source_obs, dest_obs, super_channels


def get_model(algorithm: str, device: str, env: object):
    """
    Creates or retrieves a new reinforcement learning model based on the specified algorithm.

    :param algorithm: The algorithm type (e.g., 'ppo', 'dqn', or 'a2c').
    :param device: The device on which the model will run (e.g., 'cpu' or 'cuda').
    :param env: The reinforcement learning environment.
    :return: A tuple containing the RL model and a configuration dictionary for the environment.
    """
    model = None
    yaml_dict = None
    env_name = None

    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        model = setup_ppo(env=env, device=device)
    elif algorithm == 'a2c':
        model = None

    return model, yaml_dict[env_name]


def get_trained_model(env: object, sim_dict: dict):
    """
    Loads a pre-trained reinforcement learning model from disk or initializes a new one.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters, including the model type and path.
    :return: The loaded or newly initialized RL model.
    """
    if sim_dict['spectrum_algorithm'] == 'ppo':
        model = PPO.load(os.path.join('logs', sim_dict['spectrum_model'], 'ppo_model.zip'), env=env)
    else:
        model = None

    return model


def run_iters(env: object, sim_dict: dict, is_training: bool, model=None):
    """
    Runs the specified number of episodes in the reinforcement learning environment.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation settings, such as maximum iterations.
    :param is_training: A boolean flag indicating whether the model should train or evaluate.
    :param model: The RL model to be used; required only if not in training mode.
    """
    completed_episodes = 0
    obs, _ = env.reset()
    while True:
        if is_training:
            obs, _, is_terminated, is_truncated, _ = env.step([0])
        else:
            # TODO: Implement (drl_path_agents)
            action, _states = model.predict(obs)
            # action = [0]
            obs, _, is_terminated, is_truncated, _ = env.step(action)

        if completed_episodes >= sim_dict['max_iters']:
            break
        if is_terminated or is_truncated:
            obs, _ = env.reset()
            completed_episodes += 1
            print(f'{completed_episodes} episodes completed out of {sim_dict["max_iters"]}.')


def run_testing(env: object, sim_dict: dict):
    """
    Runs pre-trained RL model evaluation in the environment for the number of episodes specified in `sim_dict`.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters (e.g., model type, paths).
    """
    model = get_trained_model(env=env, sim_dict=sim_dict)
    run_iters(env=env, sim_dict=sim_dict, is_training=False, model=model)
    # fixme: Hard coded (drl_path_agents)
    save_fp = os.path.join('logs', 'ppo', env.modified_props['network'], env.modified_props['date'],
                           env.modified_props['sim_start'], 'ppo_model.zip')
    model.save(save_fp)


def run_spectrum(sim_dict: dict, env: object):
    """
    Handles training RL models for spectral resource allocation or hyperparameter optimization.

    :param sim_dict: A dictionary containing simulation configurations, including RL-related parameters.
    :param env: The reinforcement learning environment.
    """
    if sim_dict['optimize_hyperparameters']:
        run_rl_zoo(sim_dict=sim_dict)
    else:
        model, yaml_dict = get_model(algorithm=sim_dict['spectrum_algorithm'], device=sim_dict['device'],
                                     env=env)
        model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                    callback=sim_dict['callback'])

        save_fp = os.path.join('logs', 'ppo', env.modified_props['network'], env.modified_props['date'],
                               env.modified_props['sim_start'], 'ppo_model.zip')
        model.save(save_fp)


def run(env: object, sim_dict: dict):
    """
    Manages the execution of simulations for training or testing RL models.

    Delegates to either training or testing based on flags within the simulation configuration.

    :param env: The reinforcement learning environment.
    :param sim_dict: The simulation configuration dictionary containing paths, algorithms, and statistical parameters.
    """
    print_info(sim_dict=sim_dict)

    if sim_dict['is_training']:
        # Print info function should already error check valid input, no need to raise an error here
        if sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS or sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
            run_iters(env=env, sim_dict=sim_dict, is_training=True)
        elif sim_dict['spectrum_algorithm'] in VALID_SPECTRUM_ALGORITHMS:
            run_spectrum(sim_dict=sim_dict, env=env)
    else:
        run_testing(sim_dict=sim_dict, env=env)


def run_optuna_study(env, sim_dict):
    """
    Runs Optuna study for hyperparameter optimization.

    :param env: Initialized simulation environment.
    :param sim_dict: The simulation configuration dictionary.
    """

    # Define the Optuna objective function
    def objective(trial: optuna.Trial):
        hyperparam_dict = get_optuna_hyperparams(sim_dict=sim_dict, trial=trial)
        update_list = [(param, value) for param, value in hyperparam_dict.items() if param in sim_dict]
        file_path = os.path.join('data', 'input', sim_dict['network'], sim_dict['date'],
                                 sim_dict['sim_start'], 'sim_input_s1.json')
        modify_multiple_json_values(file_path=file_path, update_list=update_list)
        arrival_list = get_arrival_rates(arrival_dict=sim_dict['arrival_dict'])
        mean_reward = run_simulation_for_arrival_rates(env=env, arrival_list=arrival_list, run_func=run)
        trial.set_user_attr("sim_start_time", sim_dict['sim_start'])
        return mean_reward

    # Run the optimization
    study_name = "hyperparam_study.pkl"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    n_trials = sim_dict['n_trials']
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    best_trial = study.best_trial
    save_study_results(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        study=study,
        env=env,
        study_name=study_name,
        best_params=best_trial.params,
        best_reward=best_trial.value,
        best_start_time=best_trial.user_attrs.get("sim_start_time")
    )
