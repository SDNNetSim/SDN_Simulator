import os

import optuna
import numpy as np

from helper_scripts.sim_helpers import modify_multiple_json_values, update_dict_from_list
from helper_scripts.sim_helpers import get_erlang_vals, run_simulation_for_erlangs, save_study_results
from rl_scripts.helpers.rl_zoo_helpers import run_rl_zoo
from rl_scripts.gymnasium_envs.general_sim_env import SimEnv
from rl_scripts.helpers.setup_helpers import print_info, setup_rl_sim
from rl_scripts.helpers.callback_helpers import GetModelParams
from rl_scripts.model_manager import get_trained_model, get_model, save_model

from rl_scripts.helpers.hyperparam_helpers import get_optuna_hyperparams
from rl_scripts.helpers.general_helpers import save_arr

from rl_scripts.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS, VALID_DRL_ALGORITHMS


def _run_drl_training(env: object, sim_dict: dict):
    """
    Trains a deep reinforcement learning model with StableBaselines3.
    """
    if sim_dict['optimize_hyperparameters'] or sim_dict['optimize']:
        run_rl_zoo(sim_dict=sim_dict)
    else:
        model, yaml_dict = get_model(sim_dict=sim_dict, device=sim_dict['device'], env=env)
        model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                    callback=sim_dict['callback'])

        save_model(sim_dict=sim_dict, env=env, model=model)


# TODO: (drl_path_agents) Break this function up for organizational purposes
def run_iters(env: object, sim_dict: dict, is_training: bool, drl_agent: bool, model=None):
    """
    Runs the specified number of episodes in the reinforcement learning environment.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation settings, such as maximum iterations.
    :param is_training: A boolean flag indicating whether the model should train or evaluate.
    :param drl_agent: A boolean flag indicating whether the model is a DRL agent.
    :param model: The RL model to be used; required only if not in training mode.
    """
    completed_episodes = 0
    completed_trials = 0
    episodic_reward = 0
    rewards_matrix = np.zeros((sim_dict['n_trials'], sim_dict['max_iters']))
    episodic_rew_arr = np.array([])

    obs, _ = env.reset(seed=completed_trials)

    while completed_trials < sim_dict['n_trials']:
        if is_training:
            if drl_agent:
                _run_drl_training(env=env, sim_dict=sim_dict)
                print("Training of DRL agent completed. No trial/reward tracking required.")
                break

            obs, reward, is_terminated, is_truncated, _ = env.step(0)
        else:
            action, _states = model.predict(obs)
            obs, reward, is_terminated, is_truncated, _ = env.step(action)

        episodic_reward += reward
        if is_terminated or is_truncated:
            episodic_rew_arr = np.append(episodic_rew_arr, episodic_reward)
            episodic_reward = 0
            completed_episodes += 1

            print(f"{completed_episodes} episodes completed out of {sim_dict['max_iters']}.")

            if completed_episodes == sim_dict['max_iters']:
                env.iteration = 0
                env.trial += 1
                rewards_matrix[completed_trials] = episodic_rew_arr
                episodic_rew_arr = np.array([])

                completed_trials += 1
                completed_episodes = 0
                print(f"{completed_trials} trials completed out of {sim_dict['n_trials']}.")

            obs, _ = env.reset(seed=completed_trials)

    if not (is_training and drl_agent):
        means_arr = np.mean(rewards_matrix, axis=0)
        cum_reward = np.sum(means_arr)
        save_arr(arr=means_arr, sim_dict=sim_dict, file_name="average_rewards.npy")

        return cum_reward

    return None


def run_testing(env: object, sim_dict: dict):
    """
    Runs pre-trained RL model evaluation in the environment for the number of episodes specified in `sim_dict`.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters (e.g., model type, paths).
    """
    model = get_trained_model(env=env, sim_dict=sim_dict)
    run_iters(env=env, sim_dict=sim_dict, is_training=False, model=model)


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
            sum_returns = run_iters(env=env, sim_dict=sim_dict, is_training=True,
                                    drl_agent=sim_dict['path_algorithm'] in VALID_DRL_ALGORITHMS)
        else:
            raise NotImplementedError
    else:
        sum_returns = run_testing(sim_dict=sim_dict, env=env)

    return sum_returns


def run_optuna_study(sim_dict):
    """
    Runs Optuna study for hyperparameter optimization.
    
    :param env: The reinforcement learning environment.
    :param sim_dict: The simulation configuration dictionary containing paths, algorithms, and statistical parameters.
    """

    # Define the Optuna objective function
    def objective(trial: optuna.Trial):
        callback = GetModelParams()
        env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=setup_rl_sim())
        env.sim_dict['callback'] = callback

        hyperparam_dict = get_optuna_hyperparams(sim_dict=sim_dict, trial=trial)
        update_list = [(param, value) for param, value in hyperparam_dict.items() if param in sim_dict]

        modify_multiple_json_values(trial_num=trial.number, file_path=file_path, update_list=update_list)
        env.sim_dict = update_dict_from_list(input_dict=env.sim_dict, updates_list=update_list)

        erlang_list = get_erlang_vals(sim_dict=sim_dict)
        mean_reward = run_simulation_for_erlangs(env=env, erlang_list=erlang_list, sim_dict=sim_dict, run_func=run)
        mean_reward = mean_reward / sim_dict['n_trials'] / sim_dict['max_iters']

        trial.set_user_attr("sim_start_time", sim_dict['sim_start'])
        trial.set_user_attr("env", env)
        return mean_reward

    # Run the optimization
    study_name = "hyperparam_study.pkl"
    file_path = os.path.join('data', 'input', sim_dict['network'], sim_dict['date'],
                             sim_dict['sim_start'])
    study = optuna.create_study(direction='maximize', study_name=study_name)
    n_trials = sim_dict['optuna_trials']
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    best_trial = study.best_trial
    save_study_results(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        study=study,
        env=best_trial.user_attrs.get("env"),
        study_name=study_name,
        best_params=best_trial.params,
        best_reward=best_trial.value,
        best_sim_start=best_trial.user_attrs.get("sim_start_time"),
    )
