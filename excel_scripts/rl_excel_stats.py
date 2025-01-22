# pylint: disable=cell-var-from-loop

# TODO: (drl_path_agents) Generalize for many scenarios

import os
import json

import numpy as np
import pandas as pd

from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import find_times, PlotHelpers
from arg_scripts.plot_args import PlotProps

NETWORK_LIST = ['NSFNet', 'USNet']
ARRIVAL_RATE_LIST = [10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 190]

inf_baselines = {
    # With 2,000 requests and k=3
    ('NSFNet', 40): 0.00065,
    ('NSFNet', 60): 0.0176,
    ('NSFNet', 80): 0.04785,
    ('NSFNet', 100): 0.0765,
    ('NSFNet', 120): 0.0986,
    ('NSFNet', 140): 0.119,
    ('NSFNet', 160): 0.136,
    ('NSFNet', 180): 0.148,
    ('NSFNet', 200): 0.16,

    # With 2,000 requests and k=4
    ('USNet', 40): 0.031,
    ('USNet', 60): 0.041,
    ('USNet', 80): 0.0528,
    ('USNet', 100): 0.063,
    ('USNet', 120): 0.0733,
    ('USNet', 140): 0.0823,
    ('USNet', 160): 0.09315,
    ('USNet', 180): 0.1009,
    ('USNet', 200): 0.109,

    # With 4,000 requests and k=4
    ('Pan-European', 40): 0.0,
    ('Pan-European', 60): 0.0,
    ('Pan-European', 80): 0.0045,
    ('Pan-European', 100): 0.019,
    ('Pan-European', 120): 0.0399,
    ('Pan-European', 140): 0.0586,
    ('Pan-European', 160): 0.0757,
    ('Pan-European', 180): 0.0934,
    ('Pan-European', 200): 0.1081,
}

ksp_baselines = {
    ('NSFNet', 10): 0.0,
    ('NSFNet', 20): 0.0,
    ('NSFNet', 30): 0.00165,
    ('NSFNet', 40): 0.01155,
    ('NSFNet', 50): 0.0274,
    ('NSFNet', 60): 0.0402,
    ('NSFNet', 70): 0.05345,
    ('NSFNet', 80): 0.0685,
    ('NSFNet', 90): 0.084,
    ('NSFNet', 100): 0.0948,
    ('NSFNet', 110): 0.1078,
    ('NSFNet', 120): 0.11935,
    ('NSFNet', 130): 0.127,
    ('NSFNet', 140): 0.13985,
    ('NSFNet', 150): 0.1456,
    ('NSFNet', 160): 0.1545,
    ('NSFNet', 170): 0.161,
    ('NSFNet', 180): 0.1665,
    ('NSFNet', 190): 0.173,
    ('NSFNet', 200): 0.17875,

    ('USNet', 10): 0.0247,
    ('USNet', 20): 0.0247,
    ('USNet', 30): 0.0274,
    ('USNet', 40): 0.03485,
    ('USNet', 50): 0.041,
    ('USNet', 60): 0.04965,
    ('USNet', 70): 0.0581,
    ('USNet', 80): 0.0677,
    ('USNet', 90): 0.076,
    ('USNet', 100): 0.08125,
    ('USNet', 110): 0.0878,
    ('USNet', 120): 0.0929,
    ('USNet', 130): 0.09715,
    ('USNet', 140): 0.1024,
    ('USNet', 150): 0.1068,
    ('USNet', 160): 0.1102,
    ('USNet', 170): 0.114,
    ('USNet', 180): 0.118,
    ('USNet', 190): 0.121,
    ('USNet', 200): 0.125,

    ('Pan-European', 40): 0.0,
    ('Pan-European', 60): 0.0008,
    ('Pan-European', 80): 0.012,
    ('Pan-European', 100): 0.0295,
    ('Pan-European', 120): 0.0534,
    ('Pan-European', 140): 0.076,
    ('Pan-European', 160): 0.095,
    ('Pan-European', 180): 0.1133,
    ('Pan-European', 200): 0.1294,
}

spf_baselines = {
    ('NSFNet', 10): 0.0,
    ('NSFNet', 20): 0.0,
    ('NSFNet', 30): 0.0068,
    ('NSFNet', 40): 0.0341,
    ('NSFNet', 50): 0.0667,
    ('NSFNet', 60): 0.093,
    ('NSFNet', 70): 0.115,
    ('NSFNet', 80): 0.1338,
    ('NSFNet', 90): 0.15075,
    ('NSFNet', 100): 0.16245,
    ('NSFNet', 110): 0.172,
    ('NSFNet', 120): 0.1844,
    ('NSFNet', 130): 0.1942,
    ('NSFNet', 140): 0.202,
    ('NSFNet', 150): 0.212,
    ('NSFNet', 160): 0.2165,
    ('NSFNet', 170): 0.225,
    ('NSFNet', 180): 0.2304,
    ('NSFNet', 190): 0.2367,
    ('NSFNet', 200): 0.24105,

    ('USNet', 10): 0.0247,
    ('USNet', 20): 0.0247,
    ('USNet', 30): 0.0326,
    ('USNet', 40): 0.055,
    ('USNet', 50): 0.0705,
    ('USNet', 60): 0.08375,
    ('USNet', 70): 0.0951,
    ('USNet', 80): 0.10845,
    ('USNet', 90): 0.1184,
    ('USNet', 100): 0.1282,
    ('USNet', 110): 0.1357,
    ('USNet', 120): 0.144,
    ('USNet', 130): 0.1502,
    ('USNet', 140): 0.1566,
    ('USNet', 150): 0.1629,
    ('USNet', 160): 0.16815,
    ('USNet', 170): 0.1729,
    ('USNet', 180): 0.1778,
    ('USNet', 190): 0.18305,
    ('USNet', 200): 0.1855,

    ('Pan-European', 40): 0.0,
    ('Pan-European', 60): 0.007,
    ('Pan-European', 80): 0.029,
    ('Pan-European', 100): 0.0565,
    ('Pan-European', 120): 0.081425,
    ('Pan-European', 140): 0.107,
    ('Pan-European', 160): 0.1262,
    ('Pan-European', 180): 0.145,
    ('Pan-European', 200): 0.161,
}

for network in NETWORK_LIST:
    for arrival_rate in ARRIVAL_RATE_LIST:
        filter_dict = {
            'and_filter_list': [
                ['arrival_start', arrival_rate],
                # TODO: Filter function has limited functionality
                # ['max_iters', 200],
            ],
            'or_filter_list': [
            ],
            'not_filter_list': [
                # ['max_segments', 4],
                # ['max_segments', 8],
            ]
        }

        sims_info_dict = find_times(dates_dict={'1014': network, '1015': network},
                                    filter_dict=filter_dict)

        is_empty = True  # pylint: disable=invalid-name
        for _, data_list in sims_info_dict.items():
            if len(data_list) > 0:
                is_empty = False  # pylint: disable=invalid-name
                break

        if is_empty:
            print(f'No file matches for: {filter_dict}')
            continue

        helpers_obj = PlotHelpers(plot_props=PlotProps(), net_names_list=sims_info_dict['networks_matrix'])
        helpers_obj.get_file_info(sims_info_dict=sims_info_dict)

        counter = 0  # pylint: disable=invalid-name
        dict_list = []
        BATCH_SIZE = 200

        save_fp = os.path.join('..', 'data', 'excel')
        create_dir(file_path=save_fp)
        csv_file = os.path.join(save_fp, f'{network}_analysis_{arrival_rate / 0.2}.csv')


        def read_files():
            """
            Reads a file from a single reinforcement learning simulation run.

            :return: The input and output JSON files.
            :rtype: tuple
            """
            output_fp = os.path.join('..', 'data', 'output', network, date, run_time, 's1',
                                     f'{erlang}_erlang.json')
            input_fp = os.path.join('..', 'data', 'input', network, date, run_time, 'sim_input_s1.json')

            if not os.path.exists(output_fp):
                print(f'Output file not found! Skipping: {output_fp}')
                return False, False
            if not os.path.exists(input_fp):
                print(f'Input file not found! Skipping: {input_fp}')
                return False, False

            with open(output_fp, 'r', encoding='utf-8') as file_path:
                try:
                    output_dict = json.load(file_path)
                except json.JSONDecodeError:
                    print('Output dict not found.')
                    return False, False
            with open(input_fp, 'r', encoding='utf-8') as file_path:
                try:
                    input_dict = json.load(file_path)
                except json.JSONDecodeError:
                    print('Input dict not found.')
                    return False, False

            return input_dict, output_dict


        def calculate_baseline_reductions(tmp_dict, network, arrival_rate):
            """
            Calculates percentage reductions compared to SPF, KSP, and KSP-Inf baselines.
            Updates tmp_dict in-place.

            :param tmp_dict: Dictionary containing 'Blocking' key.
            :param network: The network name.
            :param arrival_rate: The arrival rate.
            """
            # Calculate percentage reductions compared to SPF baseline
            spf_baseline = spf_baselines.get((network, arrival_rate))
            if spf_baseline is not None and spf_baseline != 0:
                spf_reduction = ((spf_baseline - tmp_dict['Blocking']) / spf_baseline) * 100
                tmp_dict['SPF Reduction (%)'] = spf_reduction
            else:
                tmp_dict['SPF Reduction (%)'] = np.inf

            # Calculate percentage reductions compared to KSP baseline
            ksp_baseline = ksp_baselines.get((network, arrival_rate))
            if ksp_baseline is not None and ksp_baseline != 0:
                ksp_reduction = ((ksp_baseline - tmp_dict['Blocking']) / ksp_baseline) * 100
                tmp_dict['KSP Reduction (%)'] = ksp_reduction
            else:
                tmp_dict['KSP Reduction (%)'] = np.inf

            # Calculate percentage reductions compared to KSP-Inf baseline
            inf_baseline = inf_baselines.get((network, arrival_rate))
            if inf_baseline is not None and inf_baseline != 0:
                inf_reduction = ((inf_baseline - tmp_dict['Blocking']) / inf_baseline) * 100
                tmp_dict['KSP-Inf Reduction (%)'] = inf_reduction
            else:
                tmp_dict['KSP-Inf Reduction (%)'] = np.inf


        def get_dict(input_dict: dict, output_dict: dict, network: str, arrival_rate: int):
            """
            Gets desired information from input and output for a single RL simulation.
            Calculates percentage reductions compared to SPF and KSP baselines.

            :param input_dict: Input dictionary.
            :param output_dict: Output dictionary.
            :param network: The network name.
            :param arrival_rate: The arrival rate.
            :return: Relevant data from input/output with percentage reductions.
            :rtype: dict
            """
            tmp_dict = dict()
            last_key = list(output_dict['iter_stats'].keys())[-1]
            tmp_dict['Blocking'] = np.mean(output_dict['iter_stats'][last_key]['sim_block_list'][-10:])
            tmp_dict['Completed Iters'] = len(output_dict['iter_stats'][last_key]['sim_block_list'])
            tmp_dict['Sim Start'] = input_dict['sim_start'].split('_')[-1]

            tmp_dict['Alpha Start'] = input_dict['alpha_start']
            tmp_dict['Alpha End'] = input_dict['alpha_end']
            tmp_dict['Alpha Update'] = input_dict['alpha_update']

            tmp_dict['Epsilon Start'] = input_dict['epsilon_start']
            tmp_dict['Epsilon End'] = input_dict['epsilon_end']
            tmp_dict['Epsilon Update'] = input_dict['epsilon_update']

            tmp_dict['Reward'] = input_dict['reward']
            tmp_dict['Penalty'] = input_dict['penalty']
            tmp_dict['Path Algorithm'] = input_dict['path_algorithm']

            calculate_baseline_reductions(tmp_dict, network, arrival_rate)

            return tmp_dict


        if not os.path.exists(csv_file):  # pylint: disable=simplifiable-if-statement
            HEADER = True
        else:
            HEADER = False

        # TODO: Only supports 's1'
        for run_time, run_obj in helpers_obj.file_info.items():
            net_key, date_key, sim_key = list(run_obj.keys())
            sim_network = run_obj[net_key]
            date = run_obj[date_key]
            sim_dict = run_obj[sim_key]

            try:
                erlang = sim_dict['s1'][0]
            except KeyError:
                print(f'No data found in dictionary. Skipping: {run_time}')
                continue

            input_dict, output_dict = read_files()

            if not input_dict or not output_dict:
                continue
            if '19' in input_dict['sim_start'][0:2]:
                continue

            tmp_dict = get_dict(
                input_dict=input_dict,
                output_dict=output_dict,
                network=network,
                arrival_rate=arrival_rate
            )
            dict_list.append(tmp_dict)
            counter += 1

            print(f'{counter} dictionaries created in this batch.')
            if counter == BATCH_SIZE:
                print(f'Completed one batch of {BATCH_SIZE}, appending to a CSV!')
                df_to_write = pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list])
                df_to_write.to_csv(
                    csv_file, mode='a', index=False, header=HEADER, encoding='utf-8')
                counter = 0  # pylint: disable=invalid-name
                dict_list = []
                HEADER = False

        if dict_list:
            pd.concat([pd.DataFrame(d, index=[0]) for d in dict_list]).to_csv(
                csv_file, mode='a', index=False, header=HEADER, encoding='utf-8')

        df = pd.read_csv(csv_file, encoding='utf-8')
        df = df.sort_values(by='Blocking')
        df.to_csv(csv_file, index=False, encoding='utf-8')
