import numpy as np
import optuna

from rl_scripts.algorithms.bandits import get_q_table

from rl_scripts.args.general_args import EPISODIC_STRATEGIES


class HyperparamConfig:  # pylint: disable=too-few-public-methods
    """
    Controls all hyperparameter starts, ends, and episodic and or time step modifications.
    """

    def __init__(self, engine_props: dict, rl_props: object, is_path: bool):
        self.engine_props = engine_props
        self.total_iters = engine_props['max_iters']
        self.num_nodes = rl_props.num_nodes
        self.is_path = is_path
        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.iteration = 0
        self.curr_reward = None
        self.state_action_pair = None
        self.action_index = None
        self.alpha_strategy = engine_props['alpha_update']
        self.epsilon_strategy = engine_props['epsilon_update']

        if self.alpha_strategy not in EPISODIC_STRATEGIES or self.epsilon_strategy not in EPISODIC_STRATEGIES:
            self.fully_episodic = False
        else:
            self.fully_episodic = True

        self.alpha_start = engine_props['alpha_start']
        self.alpha_end = engine_props['alpha_end']
        self.curr_alpha = self.alpha_start

        self.epsilon_start = engine_props['epsilon_start']
        self.epsilon_end = engine_props['epsilon_end']
        self.curr_epsilon = self.epsilon_start

        self.temperature = None
        self.counts = None
        self.values = None
        self.reward_list = None
        self.decay_rate = engine_props['decay_rate']

        self.alpha_strategies = {
            'softmax': self._softmax_alpha,
            'reward_based': self._reward_based_alpha,
            'state_based': self._state_based_alpha,
            'exp_decay': self._exp_alpha,
            'linear_decay': self._linear_alpha,
        }
        self.epsilon_strategies = {
            'softmax': self._softmax_eps,
            'reward_based': self._reward_based_eps,
            'state_based': self._state_based_eps,
            'exp_decay': self._exp_eps,
            'linear_decay': self._linear_eps,
        }

        if self.iteration == 0:
            self.reset()

    def _softmax(self, q_vals_list: list):
        """
        Compute the softmax probabilities for a given set of Q-values
        """
        exp_values = np.exp(np.array(q_vals_list) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def _softmax_eps(self):
        """
        Softmax epsilon update rule.
        """
        raise NotImplementedError

    def _softmax_alpha(self):
        """
        Softmax alpha update rule.
        """
        raise NotImplementedError

    def _reward_based_eps(self):
        """
        Reward-based epsilon update.
        """
        if len(self.reward_list) != 2:
            print('Did not update epsilon due to the length of the reward list.')
            return

        curr_reward, last_reward = self.reward_list
        reward_diff = abs(curr_reward - last_reward)
        self.curr_epsilon = self.epsilon_start * (1 / (1 + reward_diff))

    def _reward_based_alpha(self):
        """
        Reward-based alpha update.
        """
        if len(self.reward_list) != 2:
            print('Did not update alpha due to the length of the reward list.')
            return

        curr_reward, last_reward = self.reward_list
        reward_diff = abs(curr_reward - last_reward)
        self.curr_alpha = self.alpha_start * (1 / (1 + reward_diff))

    def _state_based_eps(self):
        """
        State visitation epsilon update.
        """
        self.counts[self.state_action_pair][self.action_index] += 1
        total_visits = self.counts[self.state_action_pair][self.action_index]
        self.curr_epsilon = self.epsilon_start / (1 + total_visits)

    def _state_based_alpha(self):
        """
        State visitation alpha update.
        """
        self.counts[self.state_action_pair][self.action_index] += 1
        total_visits = self.counts[self.state_action_pair][self.action_index]
        self.curr_alpha = 1 / (1 + total_visits)

    def _exp_eps(self):
        """
        Exponential distribution epsilon update.
        """
        self.curr_epsilon = self.epsilon_start * (self.decay_rate ** self.iteration)

    def _exp_alpha(self):
        """
        Exponential distribution alpha update.
        """
        self.curr_alpha = self.alpha_start * (self.decay_rate ** self.iteration)

    def _linear_eps(self):
        """
        Linear decay epsilon update.
        """
        self.curr_epsilon = self.epsilon_end + (
                (self.epsilon_start - self.epsilon_end) * (self.total_iters - self.iteration) / self.total_iters
        )

    def _linear_alpha(self):
        """
        Linear decay alpha update.
        """
        self.curr_alpha = self.alpha_end + (
                (self.alpha_start - self.alpha_end) * (self.total_iters - self.iteration) / self.total_iters
        )

    def update_timestep_data(self, state_action_pair: tuple, action_index: int):
        """
        Updates data structures used for updating alpha and epsilon.
        """
        self.state_action_pair = state_action_pair
        self.action_index = action_index

        if len(self.reward_list) == 2:
            # Moves old current reward to now last reward, current reward always first index
            self.reward_list = [self.curr_reward, self.reward_list[0]]
        elif len(self.reward_list) == 1:
            last_reward = self.reward_list[0]
            self.reward_list = [self.curr_reward, last_reward]
        else:
            self.reward_list.append(self.curr_reward)

    def update_eps(self):
        """
        Update epsilon.
        """
        if self.epsilon_strategy in self.epsilon_strategies:
            self.epsilon_strategies[self.epsilon_strategy]()
        else:
            raise NotImplementedError(f'{self.epsilon_strategy} not in any known strategies: {self.epsilon_strategies}')

    def update_alpha(self):
        """
        Updates alpha.
        """
        if self.alpha_strategy in self.alpha_strategies:
            self.alpha_strategies[self.alpha_strategy]()
        else:
            raise NotImplementedError(f'{self.alpha_strategy} not in any known strategies: {self.alpha_strategies}')

    def reset(self):
        """
        Resets certain class variables.
        """
        self.reward_list = list()
        self.counts, self.values = get_q_table(self=self)


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

    if 'ucb' in sim_dict['path_algorithm']:
        resp_dict['conf_param'] = trial.suggest_float('conf_param (c)', low=1.0, high=5.0, log=False, step=0.01)
        resp_dict['epsilon_start'] = None
        resp_dict['epsilon_end'] = None
    else:
        resp_dict['epsilon_start'] = trial.suggest_float('epsilon_start', low=0.01, high=0.5, log=False, step=0.01)
        resp_dict['epsilon_end'] = trial.suggest_float('epsilon_end', low=0.01, high=0.1, log=False, step=0.01)

    if 'q_learning' in (sim_dict['path_algorithm']):
        resp_dict['discount_factor'] = trial.suggest_float('discount_factor', low=0.8, high=1.0, step=0.01)
    else:
        resp_dict['discount_factor'] = None

    if ('exp_decay' in (sim_dict['epsilon_update'], sim_dict['alpha_update']) and
            'ucb' not in sim_dict['path_algorithm']):
        resp_dict['decay_rate'] = trial.suggest_float('decay_rate', low=0.1, high=0.5, step=0.01)
    else:
        resp_dict['decay_rate'] = None

    return resp_dict
