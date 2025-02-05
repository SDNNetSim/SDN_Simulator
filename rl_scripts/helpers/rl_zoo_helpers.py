import subprocess

from rl_scripts.args.general_args import LOCAL_RL_COMMANDS_LIST


def run_rl_zoo(sim_dict: dict):
    """
    Executes RL Zoo commands for training or running simulations using the specified algorithm.

    :param sim_dict: A dictionary containing simulation configuration, including the spectrum algorithm to use.
    :raises NotImplementedError: If the specified algorithm is not implemented.
    """
    # TODO: Detect if working locally or on the cluster (drl_path_agents)
    for command in LOCAL_RL_COMMANDS_LIST:
        subprocess.run(command, shell=True, check=True)

    if sim_dict['spectrum_algorithm'] == 'ppo':
        subprocess.run('python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file '
                       './sb3_scripts/yml/ppo.yml -optimize --n-trials 5 --n-timesteps 20000', shell=True, check=True)
    else:
        raise NotImplementedError(f"Spectrum Algorithm has not been implemented: {sim_dict['spectrum_algorithm']}")
