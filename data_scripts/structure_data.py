import os


def assign_link_lengths(network_fp: str, node_pairs_dict: dict, constant_weight: bool = False):
    """
    Assign a length to every link that exists in the topology.

    :param node_pairs_dict: Maps node numbers to names.
    :param constant_weight: Sets all link weights to one.
    :param network_fp: File path that contains topology information.
    :return: A mapping of nodes to links and links to weights.
    :rtype: dict
    """
    response_dict = {}
    with open(network_fp, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            src, dest, link_len_str = line.strip().split('\t')
            link_len = float(link_len_str) if not constant_weight else 1.0

            if node_pairs_dict != {}:
                src_dest_tuple = (node_pairs_dict[src], node_pairs_dict[dest])
            else:
                src_dest_tuple = (src, dest)

            if src_dest_tuple not in response_dict:
                response_dict[src_dest_tuple] = link_len

    return response_dict


def assign_core_nodes(core_nodes_fp: str):
    """
    List a core nodes of network topology. 

    :param core_nodes_fp: File path that contains core nodes  information.
    :return: A list of core nodes.
    :rtype: list
    """
    response_list = []
    with open(core_nodes_fp, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            response_list.append(line.strip().split('\t')[0])
    return response_list


# TODO: Return type is no longer a dictionary
def create_network(net_name: str, base_fp: str = None, const_weight: bool = False, is_only_core_node: bool = False):
    """
    The main structure data function.

    :param net_name: The desired network name, used to read the data file.
    :param const_weight: Set all links' weights to one if true.
    :param base_fp: The base of the file path to read the network from.
    :param is_only_core_node: Network includes only core nodes or not.
    :return: The network spectrum database.
    :rtype: dict
    """
    core_nodes_list = []
    if base_fp is None:
        base_fp = 'data/raw'
    else:
        base_fp = os.path.join(base_fp, 'raw')

    if net_name == 'USNet':
        network_fp = os.path.join(base_fp, 'us_network.txt')
    elif net_name == 'NSFNet':
        network_fp = os.path.join(base_fp, 'nsf_network.txt')
    elif net_name == 'Pan-European':
        network_fp = os.path.join(base_fp, 'europe_network.txt')
    elif net_name == 'USbackbone60':
        network_fp = os.path.join(base_fp, 'USB6014.txt')
        if not is_only_core_node:
            core_nodes_fp = os.path.join(base_fp, 'USB6014_core_nodes.txt')
            core_nodes_list = assign_core_nodes(core_nodes_fp=core_nodes_fp)
    elif net_name == 'Spainbackbone30':
        network_fp = os.path.join(base_fp, 'SPNB3014.txt')
        if not is_only_core_node:
            core_nodes_fp = os.path.join(base_fp, 'SPNB3014_core_nodes.txt')
            core_nodes_list = assign_core_nodes(core_nodes_fp=core_nodes_fp)
    else:
        raise NotImplementedError(f"Unknown network name. Expected USNet, NSFNet, or Pan-European. Got: {net_name}")

    resp = assign_link_lengths(constant_weight=const_weight, network_fp=network_fp, node_pairs_dict={}), core_nodes_list

    return resp
