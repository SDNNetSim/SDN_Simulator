import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import networkx as nx
from helper_scripts.rl.ql_helpers import QLearningHelpers


class TestQLearningHelpers(unittest.TestCase):
    """Unit tests for the QLearningHelpers class."""

    def setUp(self):
        """Set up the environment for testing."""
        # Mock engine properties and rl_props for initialization
        self.engine_props = {
            'path_levels': 2,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'learn_rate': 0.5,
            'discount_factor': 0.9,
            'max_iters': 100,
            'num_requests': 10,
            'topology': nx.Graph(),  # Initialize an empty graph
            'cores_per_link': 2,
            'network': 'TestNetwork',
            'date': '2023-08-12',
            'sim_start': 'start_time',
            'erlang': 10,
            'is_training': True,
            'alpha_start': 0.5,
            'alpha_end': 0.02,
            'epsilon_update': 'linear_decay',
            'alpha_update': 'linear_decay',
        }

        # Add nodes and edges to the topology to avoid NodeNotFound error
        self.engine_props['topology'].add_nodes_from([str(i) for i in range(5)])  # Add nodes 0 to 4 as strings
        self.engine_props['topology'].add_edges_from([
            ('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '0')
        ])  # Add some edges

        self.rl_props = MagicMock()
        self.rl_props.num_nodes = 5
        self.rl_props.k_paths = 3

        # Initialize QLearningHelpers with mocked objects
        self.q_learning_helpers = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)

    def test_setup_env(self):
        """Test the setup_env method."""
        with patch.object(self.q_learning_helpers, '_init_q_tables') as mock_init_q_tables:
            self.q_learning_helpers.setup_env()
            self.assertEqual(self.q_learning_helpers.props.epsilon, self.engine_props['epsilon_start'])
            self.assertIsInstance(self.q_learning_helpers.props.routes_matrix, np.ndarray)
            self.assertIsInstance(self.q_learning_helpers.props.cores_matrix, np.ndarray)
            mock_init_q_tables.assert_called_once()

    def test_update_routes_matrix(self):
        """Test the update_routes_matrix method."""
        # Initialize the environment to ensure routes_matrix is set up
        self.q_learning_helpers.setup_env()
        self.q_learning_helpers.learn_rate = 0.5

        reward = 10.0
        level_index = 0
        net_spec_dict = MagicMock()

        # Manually set up the routes_matrix with known values
        self.q_learning_helpers.props.routes_matrix[self.rl_props.source][self.rl_props.destination][
            self.rl_props.chosen_path_index][level_index] = ('path', 1.0)

        # Mock the methods called within update_routes_matrix
        with patch.object(self.q_learning_helpers, 'get_max_future_q', return_value=5.0):
            with patch.object(self.q_learning_helpers, 'update_q_stats') as mock_update_q_stats:
                self.q_learning_helpers.update_routes_matrix(reward=reward, level_index=level_index,
                                                             net_spec_dict=net_spec_dict)

                # Calculate expected td_error
                current_q = 1.0
                discount_factor = self.engine_props['discount_factor']
                max_future_q = 5.0
                expected_td_error = current_q - (reward + discount_factor * max_future_q)

                # Check if update_q_stats was called with the correct parameters
                mock_update_q_stats.assert_called_once_with(reward=reward, stats_flag='routes_dict',
                                                            td_error=expected_td_error)

                # Verify that the q_value was updated correctly in the routes_matrix
                new_q_value = ((1.0 - self.engine_props['learn_rate']) * current_q) + (
                        self.engine_props['learn_rate'] * (reward + discount_factor * max_future_q))
                updated_q_value = \
                    self.q_learning_helpers.props.routes_matrix[self.rl_props.source][self.rl_props.destination][
                        self.rl_props.chosen_path_index][level_index]['q_value']
                self.assertAlmostEqual(updated_q_value, new_q_value)

    def test_update_cores_matrix(self):
        """Test the update_cores_matrix method."""
        # Initialize the environment to ensure cores_matrix is set up
        self.q_learning_helpers.setup_env()
        self.q_learning_helpers.learn_rate = 0.5

        reward = 15.0
        core_index = 1
        level_index = 0
        net_spec_dict = MagicMock()

        # Mock the methods called within update_cores_matrix
        with patch.object(self.q_learning_helpers, 'get_max_future_q', return_value=10.0):
            with patch.object(self.q_learning_helpers, 'update_q_stats') as mock_update_q_stats:
                self.q_learning_helpers.update_cores_matrix(reward=reward, core_index=core_index,
                                                            level_index=level_index,
                                                            net_spec_dict=net_spec_dict)

                # Check if update_q_stats was called with the correct parameters
                # Adjust the expected td_error based on the actual logic
                expected_td_error = -24.0
                mock_update_q_stats.assert_called_once_with(reward=reward, stats_flag='cores_dict',
                                                            td_error=expected_td_error)

    @patch('helper_scripts.ql_helpers.create_dir')
    @patch('helper_scripts.ql_helpers.np.save')
    @patch('builtins.open', new_callable=MagicMock)
    def test_save_model(self, mock_open_func, mock_np_save, mock_create_dir):
        """Test the save_model method."""
        path_algorithm = 'q_learning'
        core_algorithm = 'first_fit'

        # Call the method being tested
        self.q_learning_helpers.save_model(path_algorithm=path_algorithm, core_algorithm=core_algorithm)

        # Verify that the directory creation function was called
        save_dir = os.path.join('logs', 'q_learning', self.engine_props['network'], self.engine_props['date'],
                                self.engine_props['sim_start'])
        mock_create_dir.assert_called_once_with(file_path=save_dir)

        # Verify that the numpy save function was called with the correct filepath
        save_fp = os.path.join(os.getcwd(), save_dir,
                               f"e{self.engine_props['erlang']}_routes_c{self.engine_props['cores_per_link']}.npy")
        mock_np_save.assert_called_once_with(save_fp, self.q_learning_helpers.props.routes_matrix)

        # Verify that the open function was called to save parameters
        param_fp = os.path.join(save_dir,
                                f"e{self.engine_props['erlang']}_params_c{self.engine_props['cores_per_link']}.json")
        mock_open_func.assert_called_once_with(param_fp, 'w', encoding='utf-8')

    def test_get_max_curr_q(self):
        """Test the get_max_curr_q method."""
        cong_list = [(0, 0, 0), (1, 1, 1)]
        matrix_flag = 'routes_matrix'

        with patch.object(self.q_learning_helpers, 'rl_props') as mock_rl_props:
            mock_rl_props.paths_list = ['path1', 'path2']
            self.q_learning_helpers.props.routes_matrix = MagicMock()

            with patch.object(np, 'argmax', return_value=1):
                max_index, max_obj = self.q_learning_helpers.get_max_curr_q(cong_list=cong_list,
                                                                            matrix_flag=matrix_flag)

                self.assertEqual(max_index, 1)
                self.assertEqual(max_obj, 'path2')


if __name__ == '__main__':
    unittest.main()
