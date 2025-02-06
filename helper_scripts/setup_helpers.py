import json
import os
import copy

from data_scripts.structure_data import create_network
from data_scripts.generate_data import create_bw_info, create_pt
from helper_scripts.os_helpers import create_dir


def create_input(base_fp: str, engine_props: dict):
    """
    Creates input data to run simulations.

    :param base_fp: The base file path to save input data.
    :param engine_props: Input of properties to engine.
    :return: Engine props modified with network, physical topology, and bandwidth information.
    """
    bw_info_dict = create_bw_info(
        mod_assumption=engine_props['mod_assumption'],
        mod_assumptions_path=engine_props['mod_assumption_path']
    )
    bw_file = f"bw_info_{engine_props['thread_num']}.json"
    save_input(base_fp=base_fp, properties=engine_props, file_name=bw_file, data_dict=bw_info_dict)

    save_path = os.path.join(base_fp, 'input', engine_props['network'], engine_props['date'],
                             engine_props['sim_start'], bw_file)
    with open(save_path, 'r', encoding='utf-8') as file_object:
        engine_props['mod_per_bw'] = json.load(file_object)

    network_dict, core_nodes_list = create_network(base_fp=base_fp, const_weight=engine_props['const_link_weight'],
                                                   net_name=engine_props['network'],
                                                   is_only_core_node=engine_props['is_only_core_node'])
    engine_props['topology_info'] = create_pt(cores_per_link=engine_props['cores_per_link'],
                                              net_spec_dict=network_dict)
    engine_props['core_nodes'] = core_nodes_list

    return engine_props


def save_input(base_fp: str, properties: dict, file_name: str, data_dict: dict):
    """
    Saves simulation input.

    :param base_fp: The base file path to save input.
    :param properties: Properties of the simulation, used for name when saving.
    :param file_name: The desired file name.
    :param data_dict: A dictionary containing the data to save.
    """
    path = os.path.join(base_fp, 'input', properties['network'], properties['date'],
                        properties['sim_start'])
    create_dir(path)
    create_dir(os.path.join('data', 'output'))

    save_path = os.path.join(path, file_name)

    save_dict = copy.deepcopy(data_dict)

    if 'topology' in save_dict:
        del save_dict['topology']

    if 'callback' in save_dict:
        del save_dict['callback']

    with open(save_path, 'w', encoding='utf-8') as file_path:
        json.dump(save_dict, file_path, indent=4)
