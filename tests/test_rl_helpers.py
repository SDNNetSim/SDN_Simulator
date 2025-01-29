# pylint: disable=protected-access

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from helper_scripts.rl.rl_helpers import RLHelpers


class TestRLHelpers(unittest.TestCase):
    """Unit tests for the RLHelpers class."""

    def setUp(self):
        """Set up the environment for testing."""
        # Mock the required objects
        self.rl_props = MagicMock()
        self.engine_obj = MagicMock()
        self.route_obj = MagicMock()

        # Initialize RLHelpers with mocked objects
        self.rl_helpers = RLHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj)

    def test_update_snapshots(self):
        """Test the update_snapshots method."""
        self.rl_props.arrival_count = 9
        self.engine_obj.engine_props = {'snapshot_step': 10, 'save_snapshots': True}
        self.rl_helpers.update_snapshots()
        self.engine_obj.stats_obj.update_snapshot.assert_called_once_with(net_spec_dict=self.engine_obj.net_spec_dict,
                                                                          req_num=10)

    def test_get_super_channels(self):
        """Test the get_super_channels method."""
        self.rl_props.chosen_path_list = [MagicMock()]
        self.rl_props.spectral_slots = 10
        self.rl_props.super_channel_space = 2  # Set a valid integer value for super_channel_space
        self.rl_helpers.core_num = 2

        # Mock get_hfrag to return valid indexes and a matching hfrag_arr size
        with patch('helper_scripts.rl_helpers.get_hfrag',
                   return_value=(np.array([[0, 1], [1, 2]]), np.array([0.1, 0.2, 0.3]))):
            frag_matrix, no_penalty = self.rl_helpers.get_super_channels(slots_needed=5, num_channels=2)
            self.assertEqual(frag_matrix.tolist(), [0.1, 0.2])
            self.assertFalse(no_penalty)

    def test_classify_paths(self):
        """Test the classify_paths method."""
        paths_list = np.array([['path1'], ['path2']])
        with patch('helper_scripts.rl_helpers.find_path_cong', return_value=0.2):
            with patch('helper_scripts.rl_helpers.classify_cong', return_value=1):
                result = self.rl_helpers.classify_paths(paths_list)
                self.assertEqual(result, [(0, 'path1', 1), (1, 'path2', 1)])

    def test_classify_cores(self):
        """Test the classify_cores method."""
        cores_list = [{'path': ['path1'], 2: 'congestion_level'}]  # Ensure the dictionary has the expected keys
        with patch('helper_scripts.rl_helpers.find_core_cong', return_value=0.3):
            with patch('helper_scripts.rl_helpers.classify_cong', return_value=2):
                result = self.rl_helpers.classify_cores(cores_list)
                self.assertEqual(result, [(0, 'congestion_level', 2)])

    def test_update_route_props(self):
        """Test the update_route_props method."""
        self.engine_obj.engine_props = {
            'topology': MagicMock(),
            'mod_per_bw': {'100G': [1, 2, 3]}
        }
        self.route_obj.route_props = MagicMock()
        chosen_path = [['A', 'B', 'C']]
        with patch('helper_scripts.rl_helpers.find_path_len', return_value=100):
            with patch('helper_scripts.rl_helpers.get_path_mod', return_value=2):
                self.rl_helpers.update_route_props(bandwidth='100G', chosen_path=chosen_path)
                self.route_obj.route_props.paths_matrix = chosen_path
                self.route_obj.route_props.mod_formats_matrix = [[2]]
                self.route_obj.route_props.weights_list.append(100)

    def test_handle_releases(self):
        """Test the handle_releases method."""
        self.rl_props.arrival_list = [{'arrive': 5}]
        self.rl_props.arrival_count = 0
        self.rl_props.depart_list = [{'depart': 4}, {'depart': 6}]
        self.rl_helpers._last_processed_index = 0

        self.rl_helpers.handle_releases()
        self.assertEqual(self.rl_helpers._last_processed_index, 1)
        self.engine_obj.handle_release.assert_called_once_with(curr_time=4)

    def test_allocate(self):
        """Test the allocate method."""
        self.rl_props.arrival_list = [{'arrive': 5}]
        self.rl_props.arrival_count = 0
        self.rl_props.forced_index = 0
        self.rl_helpers.super_channel_indexes = [[1, 2], [3, 4]]

        with patch('helper_scripts.rl_helpers.SpectrumAssignment') as mock_spectrum_assignment:
            mock_spectrum_assignment_instance = mock_spectrum_assignment.return_value
            mock_spectrum_assignment_instance.get_spectrum.return_value = True

            self.rl_helpers.allocate()
            self.engine_obj.handle_arrival.assert_called_once()

    @patch('helper_scripts.rl_helpers.SpectrumAssignment')
    def test_mock_handle_arrival(self, mock_spectrum_assignment):
        """Test the mock_handle_arrival method."""
        mock_spectrum_assignment_instance = mock_spectrum_assignment.return_value
        mock_spectrum_assignment_instance.spectrum_props.is_free = False

        engine_props = {'key': 'value'}
        sdn_props = {'key': 'value'}
        path_list = ['A', 'B', 'C']
        mod_format_list = [1, 2, 3]

        result = RLHelpers.mock_handle_arrival(engine_props, sdn_props, path_list, mod_format_list)
        self.assertFalse(result)
        mock_spectrum_assignment_instance.get_spectrum.assert_called_once()

    @patch('helper_scripts.rl_helpers.SDNProps')
    def test_update_mock_sdn(self, mock_sdn_props):
        """Test the update_mock_sdn method."""
        curr_req = {
            'req_id': 1,
            'source': 'A',
            'destination': 'B',
            'bandwidth': '100G',
            'mod_formats': [1, 2, 3]
        }

        # Mock SDNProps
        mock_sdn_instance = mock_sdn_props.return_value

        # Call the method
        result = self.rl_helpers.update_mock_sdn(curr_req)

        # Verify that the returned object is the mocked instance
        self.assertEqual(result, mock_sdn_instance)

        # Verify that the attributes were set correctly on the mock
        self.assertEqual(mock_sdn_instance.req_id, curr_req['req_id'])
        self.assertEqual(mock_sdn_instance.source, curr_req['source'])
        self.assertEqual(mock_sdn_instance.destination, curr_req['destination'])
        self.assertEqual(mock_sdn_instance.bandwidth, curr_req['bandwidth'])
        self.assertEqual(mock_sdn_instance.mod_formats_dict, curr_req['mod_formats'])

    def test_reset_reqs_dict(self):
        """Test the reset_reqs_dict method."""
        # Ensure that arrival_list and depart_list are initialized as empty lists
        self.rl_props.arrival_list = []
        self.rl_props.depart_list = []

        # Mock the reqs_dict that generate_requests would populate
        self.engine_obj.reqs_dict = {
            1: {'request_type': 'arrival', 'req_id': 1, 'arrive': 10, 'depart': 20},
            2: {'request_type': 'departure', 'req_id': 1, 'arrive': 10, 'depart': 20}
        }

        # Mock generate_requests to populate reqs_dict based on the mock above
        with patch.object(self.engine_obj, 'generate_requests') as mock_generate_requests:
            self.rl_helpers.reset_reqs_dict(seed=42)

            # Ensure that generate_requests was called
            mock_generate_requests.assert_called_once_with(seed=42)

            # Check that the arrival and depart lists have been populated
            self.assertEqual(len(self.rl_props.arrival_list), 1)
            self.assertEqual(len(self.rl_props.depart_list), 1)
            self.assertEqual(self.rl_helpers._last_processed_index, 0)


if __name__ == '__main__':
    unittest.main()
