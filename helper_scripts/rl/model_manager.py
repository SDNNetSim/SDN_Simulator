import os

import optuna

from stable_baselines3 import PPO

from helper_scripts.rl.setup_helpers import setup_ppo
from helper_scripts.sim_helpers import modify_multiple_json_values
from helper_scripts.sim_helpers import get_arrival_rates, run_simulation_for_arrival_rates, save_study_results
from helper_scripts.rl.rl_zoo_helpers import run_rl_zoo
from helper_scripts.rl.workflow_runner import run

from arg_scripts.rl_args import get_optuna_hyperparams


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
