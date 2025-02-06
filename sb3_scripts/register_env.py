import argparse
import shutil

import gymnasium
from gymnasium.envs.registration import register


def copy_yml_file(algorithm: str):
    """
    Copies a yaml file to the RLZoo3 library.

    :param algorithm: The algorithm being used for RLZoo3.
    """
    source_file = f'sb3_scripts/yml/{algorithm}.yml'
    destination_file = f'venvs/unity_venv/venv/lib/python3.11/site-packages/rl_zoo3/hyperparams/{algorithm}.yml'
    shutil.copy(source_file, destination_file)


def main():
    """
    Controls this script.
    """
    parser = argparse.ArgumentParser(description='Register your custom Gymnasium environment.')
    parser.add_argument('--algo', help='The custom algorithm yml file.')
    parser.add_argument('--env-name', help='The environment to register')
    args = parser.parse_args()

    register(
        id=args.env_name,
        entry_point=f'rl_scripts.gymnasium_envs.general_sim_env:{args.env_name}',
    )

    print('\n=== Registered Environments with Gymnasium ===\n')
    gymnasium.pprint_registry()
    print('\n')

    copy_yml_file(algorithm=args.algo)


if __name__ == '__main__':
    main()
