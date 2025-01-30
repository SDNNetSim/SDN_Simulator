import os

from helper_scripts.rl.rl_setup_helpers import print_info
from helper_scripts.rl.model_manager import get_trained_model, run_spectrum

from arg_scripts.rl_args import VALID_SPECTRUM_ALGORITHMS, VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS


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
