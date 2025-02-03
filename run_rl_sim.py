from rl_scripts.workflow_runner import run_optuna_study, run
from rl_scripts.utils.gym_env_util import create_environment


# TODO: (drl_path_agents) Put 'utils' file ending (imports) in the standards and guidelines
# TODO: (drl_path_agents) No support for core or spectrum assignment
# TODO: (drl_path_agents) Does not support multi-band
# fixme: Saves extra input directory (drl_path_agents)
# fixme: Saves to second traffic volume file (400 to 500) (drl_path_agents)
# TODO: (drl_path_agents) The goal is to get the path agent working for DRL and then update spectrum and core

def run_rl_sim():
    """
    The main function that controls reinforcement learning simulations, including hyperparameter optimization.
    """
    # TODO: (drl_path_agents) Bug is in setup, functions not being called properly
    env, sim_dict = create_environment()

    if not sim_dict['optimize']:
        run(env=env, sim_dict=sim_dict)
    else:
        run_optuna_study(env=env, sim_dict=sim_dict)


if __name__ == '__main__':
    run_rl_sim()
