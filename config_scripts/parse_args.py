import argparse

from arg_scripts.config_args import COMMAND_LINE_PARAMS


def parse_args():
    """
    Parse and error check command line arguments passed.

    :return: An object containing the values for each valid parameter.
    :rtype: obj
    """
    parser = argparse.ArgumentParser(description='Software-Defined Networking Simulator.')

    for args_lst in COMMAND_LINE_PARAMS:
        argument, arg_type, arg_help = args_lst[0], args_lst[1], args_lst[2]
        parser.add_argument(f'--{argument}', type=arg_type, help=arg_help)
        parser.add_argument(f'-{argument}', type=arg_type, help=arg_help)

    # fixme: (drl_path_agents) Two 'optimize' variables for SB3 and our variable interferes, needs to be
    #  fixed for DRL
    # parser.add_argument('-optimize', action='store_true', help='Enable optimization')
    args = parser.parse_args()

    return vars(args)
