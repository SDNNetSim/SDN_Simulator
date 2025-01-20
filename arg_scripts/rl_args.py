# pylint: disable=too-few-public-methods
import optuna


# TODO: (drl_path_agents) Support only for path selection for all functions
# TODO: (drl_path_agents) UCB Bandit 'c' variable is a constant
def get_optuna_hyperparams(sim_dict: dict, trial: optuna.trial):
    """
    Suggests hyperparameters for the Optuna trial.
    """
    resp_dict = dict()

    # There is no alpha in bandit algorithms
    if 'bandit' not in sim_dict['path_algorithm']:
        resp_dict['alpha_start'] = trial.suggest_float('alpha_start', low=0.01, high=0.5, log=False, step=0.01)
        resp_dict['alpha_end'] = trial.suggest_float('alpha_end', low=0.01, high=0.1, log=False, step=0.01)
    else:
        resp_dict['alpha_start'], resp_dict['alpha_end'] = None, None

    resp_dict['epsilon_start'] = trial.suggest_float('epsilon_start', low=0.01, high=0.5, log=False, step=0.01)
    resp_dict['epsilon_end'] = trial.suggest_float('epsilon_end', low=0.01, high=0.1, log=False, step=0.01)

    if 'q_learning' in (sim_dict['path_algorithm']):
        resp_dict['discount_factor'] = trial.suggest_float('discount_factor', low=0.8, high=1.0, step=0.01)
    else:
        resp_dict['discount_factor'] = None

    if 'exp_decay' in (sim_dict['epsilon_update'], sim_dict['alpha_update']):
        resp_dict['decay_rate'] = trial.suggest_float('decay_rate', low=0.1, high=0.5, step=0.01)
    else:
        resp_dict['decay_rate'] = None

    return resp_dict


class RLProps:
    """
    Main reinforcement learning properties used in run_rl_sim.py script.
    """

    def __init__(self):
        self.k_paths = None  # Number of paths the agent has to choose from
        self.cores_per_link = None  # Number of cores on every link
        self.spectral_slots = None  # Numerical value of spectral slots on every core
        self.num_nodes = None  # Total nodes in the network topology

        self.arrival_list = []  # Inter-arrival times for every request
        self.depart_list = []  # Departure times for every request

        self.mock_sdn_dict = {}  # A virtual SDN dictionary
        self.source = None  # Source node for a single request
        self.destination = None  # Destination node for a single request

        self.paths_list = []  # Potential paths from source to destination for a single request
        self.path_index = None  # Index of the last path chosen in a reinforcement learning (RL) simulation
        self.chosen_path_list = []  # The actual chosen path (including the nodes) for a single request
        self.core_index = None  # Index of the last core chosen for a request

    def __repr__(self):
        return f"RLProps({self.__dict__})"


class QProps:
    """
    Properties object used in the ql_helpers.py script.
    """

    def __init__(self):
        self.epsilon = None  # Current epsilon used at a certain point in time
        self.epsilon_start = None  # Starting value of epsilon
        self.epsilon_end = None  # Ending value of epsilon to be linearly decayed
        self.epsilon_list = []  # A list of every value at each time step

        self.is_training = None  # Flag to determine whether to load an already trained agent

        # Rewards for the core and path q-learning agents
        self.rewards_dict = {
            'routes_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}},
            'cores_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}}
        }
        # Temporal difference (TD) errors for the core and path agents
        self.errors_dict = {
            'routes_dict': {'average': [], 'min': [], 'max': [], 'errors': {}},
            'cores_dict': {'average': [], 'min': [], 'max': [], 'errors': {}}
        }
        # Total sum of rewards for each episode (episode as a key, sum of rewards as a value)
        self.sum_rewards_dict = {}
        # Total sum of TD errors each episode
        self.sum_errors_dict = {}

        self.routes_matrix = None  # Main routing q-table used by the path agent
        self.cores_matrix = None  # Main core q-table used by the core agent
        self.num_nodes = None  # Total number of nodes in the topology

        # All important parameters to be saved in a QL simulation run
        self.save_params_dict = {
            'q_params_list': ['rewards_dict', 'errors_dict', 'epsilon_list', 'sum_rewards_dict', 'sum_errors_dict'],
            'engine_params_list': ['epsilon_start', 'epsilon_end', 'max_iters', 'alpha_start', 'alpha_end',
                                   'discount_factor', 'epsilon_update', 'alpha_update']
        }

    def get_data(self, key: str):
        """
        Retrieve a property of the object.

        :param key: The property name.
        :return: The value of the property.
        """
        if hasattr(self, key):
            return getattr(self, key)

        raise AttributeError(f"'SDNProps' object has no attribute '{key}'")

    def __repr__(self):
        return f"QProps({self.__dict__})"


class BanditProps:
    """
    Properties object used in the bandit_helpers.py script.
    """

    def __init__(self):
        self.rewards_matrix = []  # Total sum of rewards for each episode
        self.counts_list = []  # Total number of counts for each action taken for every episode
        self.state_values_list = []  # Every possible V(s)

    def __repr__(self):
        return f"BanditProps({self.__dict__})"


# TODO: (drl_path_agents) Add support for DRL agents
class PPOProps:
    """
    Not implemented at this time.
    """
    pass  # pylint: disable=unnecessary-pass


VALID_PATH_ALGORITHMS = [
    'q_learning',
    'epsilon_greedy_bandit',
    'ucb_bandit',
]

VALID_CORE_ALGORITHMS = [
    'q_learning',
    'epsilon_greedy_bandit',
    'ucb_bandit',
]

VALID_SPECTRUM_ALGORITHMS = [
    'dqn',
    'ppo',
    'a2c',
]

# TODO: (drl_path_agents) Detect if running on Unity cluster or locally
LOCAL_RL_COMMANDS_LIST = [
    # 'rm -rf venvs/unity_venv/venv',
    # 'module load python/3.11.0',
    # './bash_scripts/make_venv.sh venvs/unity_venv python3.11',
    # 'source venvs/unity_venv/venv/bin/activate',
    # 'pip install -r requirements.txt',

    # './bash_scripts/register_rl_env.sh ppo SimEnv'
]
