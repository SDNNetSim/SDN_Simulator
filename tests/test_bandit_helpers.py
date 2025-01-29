# pylint: disable=protected-access

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import shutil

import numpy as np

from helper_scripts.rl.bandit_helpers import load_model, _get_base_fp, _save_model, save_model, EpsilonGreedyBandit
from helper_scripts.rl.bandit_helpers import UCBBandit


class TestBanditHelpers(unittest.TestCase):
    """
    Unit tests for the functions in the bandit_helpers script.
    """

    @patch('builtins.open', new_callable=mock_open, read_data='{"state1": [1, 2, 3]}')
    def test_load_model(self, mock_file):
        """
        Test loading a model from a JSON file.
        """
        expected_output = {'state1': [1, 2, 3]}
        result = load_model('model.json')
        self.assertEqual(result, expected_output)
        mock_file.assert_called_once_with(os.path.join('logs', 'model.json'), 'r', encoding='utf-8')

    def test_get_base_fp(self):
        """
        Test the generation of the base file path based on input parameters.
        """
        self.assertEqual(_get_base_fp(True, 10.0, 2), "e10.0_routes_c2.npy")
        self.assertEqual(_get_base_fp(False, 10.0, 2), "e10.0_cores_c2.npy")

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model(self, mock_file, mock_json_dump):
        """
        Test saving a model to a JSON file.
        """
        state_values_dict = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}
        expected_output = {"('a', 'b')": [1, 2, 3], "('c', 'd')": [4, 5, 6]}
        _save_model(state_values_dict, 10.0, 2, 'save_dir', True)
        mock_file.assert_called_once_with(os.path.join(os.getcwd(), 'save_dir', 'state_vals_e10.0_routes_c2.json'), 'w',
                                          encoding='utf-8')
        mock_json_dump.assert_called_once_with(expected_output, mock_file())

    @patch('helper_scripts.bandit_helpers.create_dir')
    @patch('numpy.save')
    @patch('helper_scripts.bandit_helpers._save_model')
    def test_save_model_function(self, mock_save_model, mock_np_save, mock_create_dir):
        """
        Test the overall save_model function that saves both the rewards array and the model.
        """
        max_iters = 100
        num_requests = 10
        iteration = max_iters - 1

        engine_props = {
            'max_iters': max_iters,
            'num_requests': num_requests,
            'network': 'network',
            'date': '2024-01-01',
            'sim_start': 'start',
            'erlang': 10.0,
            'cores_per_link': 2
        }
        props = MagicMock()

        rewards_matrix = [[1] * num_requests for _ in range(iteration + 1)]
        props.rewards_matrix = rewards_matrix

        self_obj = MagicMock()
        self_obj.engine_props = engine_props
        self_obj.props = props
        self_obj.is_path = True
        self_obj.values = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}

        save_model(iteration, 'alg', self_obj)

        save_dir = os.path.join('logs', 'alg', 'network', '2024-01-01', 'start')
        mock_create_dir.assert_called_once_with(file_path=save_dir)

        rewards_fp = os.path.join(save_dir, 'rewards_e10.0_routes_c2.npy')
        abs_rewards_fp = os.path.join(os.getcwd(), rewards_fp)
        args, _ = mock_np_save.call_args
        np.testing.assert_array_equal(args[1], np.mean(props.rewards_matrix, axis=0))
        self.assertEqual(args[0], abs_rewards_fp)

        # Print actual call arguments for debugging
        print("Actual call arguments for _save_model:", mock_save_model.call_args)

        actual_call = mock_save_model.call_args
        expected_state_values_dict = {('a', 'b'): np.array([1, 2, 3]), ('c', 'd'): np.array([4, 5, 6])}
        actual_state_values_dict = actual_call[1]['state_values_dict']

        for key in expected_state_values_dict:  # pylint: disable=consider-using-dict-items
            np.testing.assert_array_equal(expected_state_values_dict[key], actual_state_values_dict[key])

        expected_call_kwargs = {
            'state_values_dict': expected_state_values_dict,
            'erlang': 10.0,
            'cores_per_link': 2,
            'save_dir': save_dir,
            'is_path': True
        }

        self.assertEqual(expected_call_kwargs['erlang'], actual_call[1]['erlang'])
        self.assertEqual(expected_call_kwargs['cores_per_link'], actual_call[1]['cores_per_link'])
        self.assertEqual(expected_call_kwargs['save_dir'], actual_call[1]['save_dir'])
        self.assertEqual(expected_call_kwargs['is_path'], actual_call[1]['is_path'])

    def tearDown(self):
        """
        Clean up any created directories or files.
        """
        test_dir = os.path.join('logs', 'alg')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


class TestEpsilonGreedyBandit(unittest.TestCase):
    """
    Unit tests for the EpsilonGreedyBandit class.
    """

    def setUp(self):
        """
        Common setup for all tests.
        """
        self.rl_props = MagicMock()
        self.rl_props.num_nodes = 3
        self.engine_props = {'k_paths': 3, 'cores_per_link': 2, 'epsilon_start': 0.0}  # No randomness for testing
        self.q_table_mock = (np.zeros((3, 3)), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def create_bandit(self, is_path=True):
        """
        Helper method to create and return an EpsilonGreedyBandit instance.
        """
        with patch('helper_scripts.bandit_helpers.get_q_table', return_value=self.q_table_mock):
            return EpsilonGreedyBandit(self.rl_props, self.engine_props, is_path=is_path)

    def test_get_action(self):
        """
        Test the _get_action method.
        """
        bandit = self.create_bandit(is_path=True)
        state_action_pair = (0, 0)

        bandit.epsilon = 0.5
        bandit.n_arms = 5
        bandit.values = {state_action_pair: [0, 1, 2, 3, 4]}

        with patch('numpy.random.rand', return_value=0.6):
            action = bandit._get_action(state_action_pair=state_action_pair)
            self.assertEqual(action, 4)

        with patch('numpy.random.rand', return_value=0.4):
            with patch('numpy.random.randint', return_value=2):
                action = bandit._get_action(state_action_pair=state_action_pair)
                self.assertEqual(action, 2)

    def test_select_path_arm(self):
        """
        Test the select_path_arm method.
        """
        bandit = self.create_bandit(is_path=True)
        bandit.epsilon = 0.0
        action = bandit.select_path_arm(source=0, dest=0)

        self.assertEqual(action, 0)


class TestUCBBandit(unittest.TestCase):
    """
    Unit tests for the UCBBandit class.
    """

    def setUp(self):
        """
        Common setup for all tests.
        """
        self.rl_props = MagicMock()
        self.rl_props.num_nodes = 3
        self.engine_props = {'k_paths': 3, 'cores_per_link': 2}

        self.counts_mock = np.array([[[1, 2, 3]], [[1, 1, 1]], [[1, 1, 1]]])
        self.q_table_mock = np.array(
            [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])

    def create_bandit(self, is_path=True):
        """
        Helper method to create and return a UCBBandit instance.
        """
        with patch('helper_scripts.bandit_helpers.get_q_table', return_value=(self.counts_mock, self.q_table_mock)):
            return UCBBandit(self.rl_props, self.engine_props, is_path=is_path)

    def test_get_action_with_zero_count(self):
        """
        Test the _get_action method when some actions have not been taken yet.
        """
        self.counts_mock = np.array([[[0, 1, 2]], [[1, 1, 1]], [[1, 1, 1]]])
        bandit = self.create_bandit(is_path=True)
        state_action_pair = (0, 0)
        action = bandit._get_action(state_action_pair=state_action_pair)  # pylint: disable=protected-access

        self.assertEqual(action, 0)

    def test_get_action_with_all_counts(self):
        """
        Test the _get_action method when all actions have non-zero counts.
        """
        # All actions have non-zero counts
        self.counts_mock = np.array([[[1, 1, 1]], [[1, 1, 1]], [[1, 1, 1]]])
        bandit = self.create_bandit(is_path=True)
        state_action_pair = (0, 0)
        action = bandit._get_action(state_action_pair=state_action_pair)  # pylint: disable=protected-access

        self.assertEqual(action, 2)

    def test_select_path_arm(self):
        """
        Test the select_path_arm method.
        """
        bandit = self.create_bandit(is_path=True)
        action = bandit.select_path_arm(source=0, dest=0)

        self.assertEqual(action, 2)

    @patch('helper_scripts.bandit_helpers._update_bandit')
    def test_update(self, mock_update_bandit):
        """
        Test the update method.
        """
        bandit = self.create_bandit(is_path=True)

        arm = 1
        reward = 10
        iteration = 5

        bandit.update(arm=arm, reward=reward, iteration=iteration)
        mock_update_bandit.assert_called_once_with(iteration=iteration, arm=arm, reward=reward, self=bandit,
                                                   algorithm='ucb_bandit')


if __name__ == '__main__':
    unittest.main()
