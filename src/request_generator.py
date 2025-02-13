from random import random
import random


from helper_scripts.random_helpers import set_seed, get_uniform_rv, get_exponential_rv


def get_requests(seed: int, engine_props: dict):
    """
    Generates requests for a single simulation.

    :param seed: Seed for random generation.
    :param engine_props: Properties from the engine class.
    :return: The generated requests and request information.
    :rtype: dict
    """
    # sources, destinations == MSMD
    # source, dest == single source single destination, may need to name them better
    global sources, destinations, source, dest
    num_sources = max(2, min(15, engine_props.get('num_sources', 2)))            # ensures num_sources is within 2 and 15
    num_destinations = max(2, min(15, engine_props.get('num_destinations', 2)))  # ensures num_destinations is within 2 and 15
    requests_dict = {}
    current_time = 0
    request_id = 1

    # Means ALL nodes are core nodes
    if engine_props['is_only_core_node']:
        nodes_list = list(engine_props['topology_info']['nodes'].keys())
    # Means some nodes are nodes
    else:
        nodes_list = engine_props['core_nodes']
    set_seed(seed=seed)

    #Check is MSMD is enabled
    if engine_props['multi_source_multi_destination']:

        #Check to see that we don't select more sources or destinations than available
        if num_sources > len(nodes_list) or num_destinations > len(nodes_list):
            raise ValueError("Number of sources or destinations specified exceeds the number of available nodes.")

    bw_counts_dict = {bandwidth: int(engine_props['request_distribution'][bandwidth] * engine_props['num_requests'])
                      for bandwidth in engine_props['mod_per_bw']}
    bandwidth_list = list(engine_props['mod_per_bw'].keys())

    # Check to see if the number of requests can be distributed
    difference = engine_props['num_requests'] - sum(bw_counts_dict.values())
    if difference != 0:
        raise ValueError('The number of requests could not be distributed in the percentage distributed input. Please'
                         'either change the number of requests, or change the percentages for the bandwidth values'
                         'selected.')

    # Generate requests, multiply the number of requests by two since we have arrival and departure types
    while len(requests_dict) < (engine_props['num_requests'] * 2): # while we haven't generated the total amount of requests
        current_time += get_exponential_rv(scale_param=engine_props['arrival_rate'])
        depart_time = current_time + get_exponential_rv(scale_param=1 / engine_props['holding_time'])

        if engine_props['multi_source_multi_destination']:
            sources = random.sample(nodes_list, num_sources) # random.sample randomly selects a unique item from this list without replacement
            remaining_nodes = [node for node in nodes_list if node not in sources] # this line only allows destinations to pick from the remaining nodes that [sources] did not choose above, ensuring that no nodes are both sources and destinations
            destinations = random.sample(remaining_nodes, num_destinations)
            #destinations = random.sample(nodes_list, num_destinations)
        else:
            source = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]
            dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]
            while dest == source:
                dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]

        while True:
            chosen_bandwidth = bandwidth_list[get_uniform_rv(scale_param=len(bandwidth_list))]
            if bw_counts_dict[chosen_bandwidth] > 0:
                bw_counts_dict[chosen_bandwidth] -= 1
                break

        if current_time not in requests_dict and depart_time not in requests_dict:
            if engine_props['multi_source_multi_destination']: #MSMD enabled
                requests_dict.update({current_time: { #add variables in body to accommodate multi source multi destination
                    "req_id": request_id,
                    "source": sources, #adds list of sources if MSMD enabled
                    "destination": destinations, #adds list of destinations if MSMD enabled
                    "arrive": current_time,
                    "depart": depart_time,
                    "request_type": "arrival",
                    "bandwidth": chosen_bandwidth,
                    "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
                }})
                requests_dict.update({depart_time: {
                    "req_id": request_id,
                    "source": sources,
                    "destination": destinations,
                    "arrive": current_time,
                    "depart": depart_time,
                    "request_type": "release",
                    "bandwidth": chosen_bandwidth,
                    "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
                }})
            else: #MSMD disabled
                requests_dict.update({current_time: {
                    "req_id": request_id,
                    "source": source,    #stays as singular source if MSMD not enabled
                    "destination": dest, #stays as singular destination if MSMD not enabled
                    "arrive": current_time,
                    "depart": depart_time,
                    "request_type": "arrival",
                    "bandwidth": chosen_bandwidth,
                    "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
                }})
                requests_dict.update({depart_time: {
                    "req_id": request_id,
                    "source": source,
                    "destination": dest,
                    "arrive": current_time,
                    "depart": depart_time,
                    "request_type": "release",
                    "bandwidth": chosen_bandwidth,
                    "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
                }})
            request_id += 1
        # Bandwidth was not chosen due to either arrival or depart time already existing, add back to distribution
        else:
            bw_counts_dict[chosen_bandwidth] += 1

    return requests_dict # returns dict containing all the generated requests once the loop is done
