# pylint: disable=too-few-public-methods
import copy
import os


class PlotProps:
    """
    Properties used in the main plot_stats.py script.
    """

    def __init__(self):
        self.sims_info_dict = None  # Contains all necessary information for each simulation run to be plotted
        self.plot_dict = None  # Contains only information related to plotting for each simulation run
        self.output_dir = os.path.join('..', 'data', 'output')  # The base output directory when saving graphs
        self.input_dir = os.path.join('..', 'data', 'input')  # The base input directory when reading simulation input
        self.erlang_dict = None  # Has the information for one simulation run for each every Erlang value under it
        self.num_requests = None  # The number of requests used for each iteration for the simulation run
        self.num_cores = None  # Number of cores used for each iteration for the simulation run

        self.color_list = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']  # Colors used for lines
        self.style_list = ['solid', 'dashed', 'dotted', 'dashdot']  # Styles used for lines
        self.marker_list = ['o', '^', 's', 'x']  # Marker styles used for lines
        self.x_tick_list = [50, 100, 200, 300, 400, 500, 600, 700]  # X-tick labels
        self.title_names = None  # Important names used for titles in plots (one string)

    def __repr__(self):
        return f"PlotProps({self.__dict__})"


class PlotArgs:
    """
    Arguments used in the plot_helpers.py script.
    """

    def __init__(self):
        self.erlang_list = []  # Numerical Erlang values we are plotting
        self.blocking_list = []  # Blocking values to be plotted
        self.lengths_list = []  # Average path length values
        self.hops_list = []  # Average path hop
        self.occ_slot_matrix = []  # Occupied slots in the entire network at different snapshots
        self.active_req_matrix = []  # Number of requests allocated in the network (snapshots)
        self.block_req_matrix = []  # Running average blocking probabilities (snapshots)
        self.req_num_list = []  # Active request identification numbers (snapshots)
        self.times_list = []  # Simulation start times
        self.modulations_dict = dict()  # Modulation formats used
        self.dist_block_list = []  # Percentage of blocking due to a reach constraint
        self.cong_block_list = []  # Percentage of blocking due to a congestion constraint
        self.holding_time = None  # Holding time for the simulation run
        self.cores_per_link = None  # Number of cores per link
        # TODO: (drl_path_agents) Does not support all bands, check on this
        self.c_band = None  # Spectral slots per core for the c-band
        self.learn_rate = None  # For artificial intelligence (AI), learning rate used if any
        self.discount_factor = None  # For AI, discount factor used if any

        self.block_per_iter = []  # Blocking probability per iteration of one simulation configuration
        self.sum_rewards_list = []  # For reinforcement learning (RL), sum of rewards per episode
        self.sum_errors_list = []  # For RL, sum of errors per episode
        self.epsilon_list = []  # For RL, decay of epsilon w.r.t. each episode

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(f'{key} not found') from exc

    def __delitem__(self, key):
        try:
            delattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"'{key}' not found") from exc

    def __contains__(self, key):
        return hasattr(self, key)

    @staticmethod
    def update_info_dict(plot_props: PlotProps, input_dict: dict, info_item_list: list, time: str, sim_num: str):
        """
        Updates various items in the plot dictionary.

        :param plot_props: Main plot script properties object.
        :param input_dict: Input dictionary containing information for each item (blocking, length, etc.)
        :param info_item_list: Keys of the dictionary for their to be updated.
        :param time: Simulation start time.
        :param sim_num: Simulation number.
        :return: The updated plot properties with the simulation information.
        :rtype: object
        """
        resp_plot_props = copy.deepcopy(plot_props)
        for info_item in info_item_list:
            resp_plot_props.plot_dict[time][sim_num][info_item] = input_dict[info_item]

        return resp_plot_props

    def __repr__(self):
        return f"PlotArgs({self.__dict__})"
