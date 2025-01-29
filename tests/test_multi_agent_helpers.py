# pylint: disable=protected-access

import unittest

from unittest.mock import MagicMock, patch
from helper_scripts.rl.multi_agent_helpers import PathAgent, CoreAgent, HyperparamConfig


class TestPathAgent(unittest.TestCase):
    """
    Test the path agent in multi_agent_helpers.py
    """

    def setUp(self):
        """
        Set up common test variables and mock objects.
        """
        self.rl_props = MagicMock()
        self.rl_help_obj = MagicMock()
        self.engine_props = {
            'reward': 10,
            'penalty': -5,
            'path_algorithm': 'q_learning',
            'k_paths': 3,
            'path_levels': 2,
            'cores_per_link': 4,
            'max_iters': 100,
            'alpha_update': 'linear_decay',
            'alpha_start': 1.0,
            'alpha_end': 0.1,
            'epsilon_update': 'linear_decay',
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'decay_rate': 1.0,
        }
        self.path_agent = PathAgent(path_algorithm='q_learning', rl_props=self.rl_props, rl_help_obj=self.rl_help_obj)
        self.path_agent.engine_props = self.engine_props

    def test_get_reward_allocated(self):
        """
        Test get_reward method when the request is allocated.
        """
        reward = self.path_agent.get_reward(was_allocated=True, path_length=5)
        self.assertEqual(reward, 10)

    def test_get_reward_not_allocated(self):
        """
        Test get_reward method when the request is not allocated.
        """
        reward = self.path_agent.get_reward(was_allocated=False, path_length=5)
        self.assertEqual(reward, -5)

    def test_update_q_learning(self):
        """
        Test update method for 'q_learning' algorithm.
        """
        self.path_agent.path_algorithm = 'q_learning'
        self.path_agent.agent_obj = MagicMock()
        self.path_agent.hyperparam_obj = MagicMock()
        self.path_agent.hyperparam_obj.curr_reward = 1.0
        self.path_agent.hyperparam_obj.iteration = 5
        self.path_agent.reward_penalty_list = [0] * self.engine_props['max_iters']
        self.path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
        self.path_agent.agent_obj.update_routes_matrix.assert_called_once()

    def test_update_epsilon_greedy_bandit(self):
        """
        Test update method for 'epsilon_greedy_bandit' algorithm.
        """
        self.path_agent.path_algorithm = 'epsilon_greedy_bandit'
        self.path_agent.agent_obj = MagicMock()
        self.path_agent.hyperparam_obj = MagicMock()
        self.path_agent.hyperparam_obj.curr_reward = 1.0
        self.path_agent.hyperparam_obj.iteration = 0
        self.path_agent.reward_penalty_list = [0] * self.engine_props['max_iters']
        self.path_agent.update(was_allocated=True, net_spec_dict={}, iteration=1, path_length=5)
        self.path_agent.agent_obj.update.assert_called_once()

    def test_get_route_q_learning(self):
        """
        Test get_route method for 'q_learning' algorithm.
        """
        self.path_agent.path_algorithm = 'q_learning'
        with patch.object(self.path_agent, '_ql_route') as mock_ql_route:
            self.path_agent.get_route()
            mock_ql_route.assert_called_once()

    def test_get_route_bandit(self):
        """
        Test get_route method for bandit algorithms.
        """
        self.path_agent.path_algorithm = 'epsilon_greedy_bandit'
        with patch.object(self.path_agent, '_bandit_route') as mock_bandit_route:
            self.path_agent.get_route(route_obj=MagicMock())
            mock_bandit_route.assert_called_once()

    def test_load_model_q_learning(self):
        """
        Test load_model method when path_algorithm is 'q_learning'.
        """
        self.engine_props['path_algorithm'] = 'q_learning'
        with patch('numpy.load') as mock_load:
            self.path_agent.load_model(model_path='model', erlang=10.0, num_cores=4)
            mock_load.assert_called_once()

    def test_raise_not_implemented(self):
        """
        Test to ensure NotImplementedError is raised for unknown path_algorithm.
        """
        self.path_agent.path_algorithm = 'unknown_algorithm'
        with self.assertRaises(NotImplementedError):
            self.path_agent.setup_env()


class TestCoreAgent(unittest.TestCase):
    """
    Test the core agent in multi_agent_helpers.py
    """

    def setUp(self):
        """
        Set up common test variables and mock objects.
        """
        self.rl_props = MagicMock()
        self.rl_help_obj = MagicMock()
        self.engine_props = {
            'reward': 10,
            'penalty': -5,
            'core_algorithm': 'q_learning',
            'gamma': 0.1,
            'decay_factor': 0.2,
            'num_requests': 100,
            'core_beta': 0.5,
            'dynamic_reward': True,
            'cores_per_link': 4,
            'path_levels': 2,
            'epsilon_start': 1.0,
        }
        self.core_agent = CoreAgent(core_algorithm='q_learning', rl_props=self.rl_props, rl_help_obj=self.rl_help_obj)
        self.core_agent.engine_props = self.engine_props

    def test_end_iter_q_learning(self):
        """
        Test end_iter method when core_algorithm is 'q_learning'.
        Expect NotImplementedError since the method is not implemented yet.
        """
        self.core_agent.agent_obj = MagicMock()

        with self.assertRaises(NotImplementedError):
            self.core_agent.end_iter()

    def test_setup_env_q_learning(self):
        """
        Test setup_env method when core_algorithm is 'q_learning'.
        """
        with patch('helper_scripts.multi_agent_helpers.QLearningHelpers') as mock_q_helpers:
            self.core_agent.setup_env()
            mock_q_helpers.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props)
            self.core_agent.agent_obj.setup_env.assert_called_once()

    def test_setup_env_epsilon_greedy_bandit(self):
        """
        Test setup_env method when core_algorithm is 'epsilon_greedy_bandit'.
        """
        self.core_agent.core_algorithm = 'epsilon_greedy_bandit'
        with patch('helper_scripts.multi_agent_helpers.EpsilonGreedyBandit') as mock_eps_bandit:
            self.core_agent.setup_env()
            mock_eps_bandit.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props,
                                                    is_path=False)

    def test_setup_env_ucb_bandit(self):
        """
        Test setup_env method when core_algorithm is 'ucb_bandit'.
        """
        self.core_agent.core_algorithm = 'ucb_bandit'
        with patch('helper_scripts.multi_agent_helpers.UCBBandit') as mock_ucb:
            self.core_agent.setup_env()
            mock_ucb.assert_called_once_with(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False)

    def test_calculate_dynamic_penalty(self):
        """
        Test calculate_dynamic_penalty method.
        """
        penalty = self.core_agent.calculate_dynamic_penalty(core_index=1, req_id=50)
        expected_penalty = -5 * (1 + 0.1 * 1 / 50)
        self.assertAlmostEqual(penalty, expected_penalty)

    def test_calculate_dynamic_reward(self):
        """
        Test calculate_dynamic_reward method.
        """
        reward = self.core_agent.calculate_dynamic_reward(core_index=1, req_id=50)
        core_decay = 10 / (1 + 0.2 * 1)
        request_weight = ((100 - 50) / 100) ** 0.5
        expected_reward = core_decay * request_weight
        self.assertAlmostEqual(reward, expected_reward)

    def test_get_reward_allocated(self):
        """
        Test get_reward method when the request is allocated.
        """
        self.rl_props.core_index = 1
        self.rl_help_obj.route_obj.sdn_props = {'req_id': 50}
        reward = self.core_agent.get_reward(was_allocated=True)
        self.assertAlmostEqual(reward, self.core_agent.calculate_dynamic_reward(1, 50))

    def test_get_reward_not_allocated(self):
        """
        Test get_reward method when the request is not allocated.
        """
        self.rl_props.core_index = 1
        self.rl_help_obj.route_obj.sdn_props = {'req_id': 50}
        reward = self.core_agent.get_reward(was_allocated=False)
        self.assertAlmostEqual(reward, self.core_agent.calculate_dynamic_penalty(1, 50))

    def test_update_q_learning(self):
        """
        Test update method for 'q_learning' algorithm.
        """
        self.core_agent.core_algorithm = 'q_learning'
        self.core_agent.agent_obj = MagicMock()
        self.rl_props.core_index = 1
        self.core_agent.update(was_allocated=True, net_spec_dict={}, iteration=1)
        self.core_agent.agent_obj.update_cores_matrix.assert_called_once()

    def test_update_epsilon_greedy_bandit(self):
        """
        Test update method for 'epsilon_greedy_bandit' algorithm.
        """
        self.core_agent.core_algorithm = 'epsilon_greedy_bandit'
        self.core_agent.agent_obj = MagicMock()
        self.rl_props.core_index = 1
        self.core_agent.update(was_allocated=True, net_spec_dict={}, iteration=1)
        self.core_agent.agent_obj.update.assert_called_once()

    def test_get_core_q_learning(self):
        """
        Test get_core method for 'q_learning' algorithm.
        """
        self.core_agent.core_algorithm = 'q_learning'
        with patch.object(self.core_agent, '_ql_core') as mock_ql_core:
            self.core_agent.get_core()
            mock_ql_core.assert_called_once()

    def test_get_core_bandit(self):
        """
        Test get_core method for bandit algorithms.
        """
        self.core_agent.core_algorithm = 'epsilon_greedy_bandit'
        with patch.object(self.core_agent, '_bandit_core') as mock_bandit_core:
            self.core_agent.get_core()
            mock_bandit_core.assert_called_once()

    def test_load_model_q_learning(self):
        """
        Test load_model method when core_algorithm is 'q_learning'.
        """
        self.core_agent.core_algorithm = 'q_learning'
        with patch('numpy.load') as mock_load:
            self.core_agent.load_model(model_path='model', erlang=10.0, num_cores=4)
            mock_load.assert_called_once()

    def test_raise_not_implemented(self):
        """
        Test to ensure NotImplementedError is raised for unknown core_algorithm.
        """
        self.core_agent.core_algorithm = 'unknown_algorithm'
        with self.assertRaises(NotImplementedError):
            self.core_agent.setup_env()


class TestHyperParamConfig(unittest.TestCase):
    """
    Test the hyperparameter updates in multi_agent_helpers.py
    """

    def setUp(self):
        """
        Sets up this class.
        """
        engine_props = {
            'num_requests': 100,
            'alpha_start': 0.5,
            'alpha_end': 0.1,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'temperature': 1.0,
            'decay_rate': 0.99,
            'alpha_update': 'linear_decay',
            'epsilon_update': 'linear_decay',
            'k_paths': 5,
            'cores_per_link': 3,
            'max_iters': 100,
            'path_algorithm': 'epsilon_greedy_bandit'
        }
        rl_props = type('RLProps', (object,), {'num_nodes': 4})()  # Mock object for rl_props
        self.config_obj = HyperparamConfig(engine_props, rl_props, is_path=True)

        self.config_obj.time_step = 0
        self.config_obj.state_action_pair = ('state1', 'action1')
        self.config_obj.counts = {('state1', 'action1'): [10, 0, 1]}
        self.config_obj.reward_list = [15.0, 10.0]
        self.config_obj.values = {'state1': 9, 'state2': 3}

        self.config_obj.iteration = 0
        self.config_obj.curr_reward = 20.0

    def test_update_eps(self):
        """
        Tests the update epsilon strategy.
        """
        self.config_obj.action_index = 2
        for strategy in self.config_obj.epsilon_strategies:
            self.config_obj.epsilon_strategy = strategy
            if strategy == 'softmax':
                with self.assertRaises(NotImplementedError):
                    self.config_obj.update_eps()
            else:
                self.config_obj.update_eps()
                self.assertIsNotNone(self.config_obj.curr_epsilon)

    def test_update_alpha(self):
        """
        Tests the update alpha strategy.
        """
        self.config_obj.action_index = 0
        for strategy in self.config_obj.alpha_strategies:
            self.config_obj.alpha_strategy = strategy
            if strategy == 'softmax':
                with self.assertRaises(NotImplementedError):
                    self.config_obj.update_alpha()
            else:
                self.config_obj.update_alpha()
                self.assertIsNotNone(self.config_obj.curr_alpha)

    def test_reset(self):
        """
        Tests the reset method.
        """
        self.config_obj.reset()
        self.assertEqual(len(self.config_obj.reward_list), 0)
        if 'bandit' in self.config_obj.engine_props['path_algorithm']:
            self.assertIsNotNone(self.config_obj.counts)
            self.assertIsNotNone(self.config_obj.values)

    def test_softmax_eps(self):
        """
        Test the softmax epsilon update rule.
        """
        # This method should raise NotImplementedError as per current class definition
        with self.assertRaises(NotImplementedError):
            self.config_obj._softmax_eps()

    def test_softmax_alpha(self):
        """
        Test the softmax alpha update rule.
        """
        # This method should raise NotImplementedError as per current class definition
        with self.assertRaises(NotImplementedError):
            self.config_obj._softmax_alpha()

    def test_reward_based_eps(self):
        """
        Test the reward-based epsilon update rule.
        """
        curr_reward = self.config_obj.reward_list[0]
        last_reward = self.config_obj.reward_list[1]
        reward_diff = abs(curr_reward - last_reward)
        expected_epsilon = self.config_obj.epsilon_start * (1 / (1 + reward_diff))
        self.config_obj._reward_based_eps()
        self.assertAlmostEqual(self.config_obj.curr_epsilon, expected_epsilon, places=5)

    def test_reward_based_alpha(self):
        """
        Test the reward-based alpha update rule.
        """
        curr_reward = self.config_obj.reward_list[0]
        last_reward = self.config_obj.reward_list[1]
        reward_diff = abs(curr_reward - last_reward)
        expected_alpha = self.config_obj.alpha_start * (1 / (1 + reward_diff))
        self.config_obj._reward_based_alpha()
        self.assertAlmostEqual(self.config_obj.curr_alpha, expected_alpha, places=5)

    def test_state_based_eps(self):
        """
        Test the state-based epsilon update rule.
        """
        # Plus one since this is incremented in this function
        self.config_obj.action_index = 1
        num_visits = self.config_obj.counts[self.config_obj.state_action_pair][self.config_obj.action_index] + 1
        expected_epsilon = self.config_obj.epsilon_start / (1 + num_visits)
        self.config_obj._state_based_eps()
        self.assertAlmostEqual(self.config_obj.curr_epsilon, expected_epsilon, places=5)

    def test_state_based_alpha(self):
        """
        Test the state-based alpha update rule.
        """
        self.config_obj.action_index = 0
        # Plus one since this is incremented in this function
        num_visits = self.config_obj.counts[self.config_obj.state_action_pair][self.config_obj.action_index] + 1
        expected_alpha = 1 / (1 + num_visits)
        self.config_obj._state_based_alpha()
        self.assertAlmostEqual(self.config_obj.curr_alpha, expected_alpha, places=5)

    def test_exp_eps(self):
        """
        Test the exponential decay epsilon update rule.
        """
        self.config_obj.time_step = 10
        self.config_obj.decay_rate = 0.99
        self.config_obj._exp_eps()
        expected_epsilon = self.config_obj.epsilon_start * (self.config_obj.decay_rate ** self.config_obj.iteration)
        self.assertAlmostEqual(self.config_obj.curr_epsilon, expected_epsilon, places=5)

    def test_exp_alpha(self):
        """
        Test the exponential decay alpha update rule.
        """
        self.config_obj.iteration = 50
        self.config_obj.decay_rate = 0.1
        self.config_obj._exp_alpha()
        expected_alpha = self.config_obj.alpha_start * (self.config_obj.decay_rate ** self.config_obj.iteration)
        self.assertAlmostEqual(self.config_obj.curr_alpha, expected_alpha, places=5)

    def test_linear_eps(self):
        """
        Test the linear decay epsilon rule.
        """
        self.config_obj.iteration = 50
        self.config_obj.total_iters = 100
        self.config_obj._linear_eps()
        expected_epsilon = (self.config_obj.epsilon_end +
                            (self.config_obj.epsilon_start - self.config_obj.epsilon_end) *
                            (self.config_obj.total_iters - self.config_obj.iteration) / self.config_obj.total_iters)
        self.assertAlmostEqual(self.config_obj.curr_epsilon, expected_epsilon, places=5)

    def test_linear_alpha(self):
        """
        Test the linear decay alpha rule.
        """
        self.config_obj.iteration = 1
        self.config_obj.total_iters = 100
        self.config_obj._linear_alpha()
        expected_alpha = (self.config_obj.alpha_end +
                          (self.config_obj.alpha_start - self.config_obj.alpha_end) *
                          (self.config_obj.total_iters - self.config_obj.iteration) / self.config_obj.total_iters)
        self.assertAlmostEqual(self.config_obj.curr_alpha, expected_alpha, places=5)


if __name__ == '__main__':
    unittest.main()
