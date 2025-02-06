from enum import Enum

SETTINGS_CONFIG_DICT = [
    {
        "category": "General",
        "settings": [
            {"type": "combo", "label": "Sim Type:", "default": "yue", "options": ["yue", "arash"]},
            {"type": "double_spin", "label": "Holding Time:", "default": 0.2, "min": 0.0, "step": 0.1},
            {"type": "spin", "label": "Arrival Rate Start:", "default": 150},
            {"type": "spin", "label": "Arrival Rate Stop:", "default": 152},
            {"type": "spin", "label": "Arrival Rate Step:", "default": 50},
            {"type": "check", "label": "Thread Erlangs:", "default": False},
            {"type": "spin", "label": "Guard Slots:", "default": 1, "min": 1},
            {"type": "spin", "label": "Number of Requests:", "default": 2000, "max": 100000},
            {"type": "line", "label": "Request Distribution:",
             "default": "{\"25\": 0.3, \"50\": 0.5, \"100\": 0.2, \"200\": 0.0, \"400\": 0.0}"},
            {"type": "spin", "label": "Max Iters:", "default": 10, "min": 1},
            {"type": "spin", "label": "Max Segments:", "default": 1, "min": 1},
            {"type": "check", "label": "Dynamic LPS:", "default": False},
            {"type": "combo", "label": "Allocation Method:", "default": "first_fit",
             "options": ["best_fit", "first_fit", "last_fit", "priority_first", "priority_last", "xt_aware"]},
            {"type": "spin", "label": "K Paths:", "default": 3, "min": 1},
            {"type": "combo", "label": "Route Method:", "default": "k_shortest_path",
             "options": ["nli_aware", "xt_aware", "least_congested", "shortest_path", "k_shortest_path"]},
            {"type": "check", "label": "Save Snapshots:", "default": False},
            {"type": "spin", "label": "Snapshot Step:", "default": 10, "min": 1},
            {"type": "spin", "label": "Print Step:", "default": 1, "min": 1},
            {"type": "header", "label": "Simulation Output"},
            {"type": "check", "label": "Show iteration step:", "default": True},
            {"type": "check", "label": "Show blocking probability:", "default": True}
        ]
    },
    {
        "category": "Topology",
        "settings": [
            {"type": "line", "label": "Network:", "default": "NSFNet"},
            {"type": "spin", "label": "Spectral Slots:", "default": 128},
            {"type": "double_spin", "label": "BW per Slot:", "default": 12.5},
            {"type": "spin", "label": "Cores per Link:", "default": 4},
            {"type": "check", "label": "Const Link Weight:", "default": False}
        ]
    },
    {
        "category": "SNR",
        "settings": [
            {"type": "line", "label": "SNR Type:", "default": "None"},
            {"type": "line", "label": "XT Type:", "default": "without_length"},
            {"type": "double_spin", "label": "Beta:", "default": 0.5},
            {"type": "double_spin", "label": "Theta:", "default": 0.0},
            {"type": "double_spin", "label": "Input Power:", "default": 0.001},
            {"type": "check", "label": "EGN Model:", "default": False},
            {"type": "line", "label": "Phi:",
             "default": "{\"QPSK\": 1, \"16-QAM\": 0.68, \"64-QAM\": 0.6190476190476191}"},
            {"type": "check", "label": "Bi-Directional:", "default": True},
            {"type": "check", "label": "XT Noise:", "default": False},
            {"type": "line", "label": "Requested XT:",
             "default": "{\"QPSK\": -26.19, \"16-QAM\": -36.69, \"64-QAM\": -41.69}"}
        ]
    },
    {
        "category": "RL",
        "settings": [
            {"type": "line", "label": "Device:", "default": "cpu"},
            {"type": "check", "label": "Optimize:", "default": False},
            {"type": "check", "label": "Is Training:", "default": True},
            {"type": "line", "label": "Path Algorithm:", "default": "ucb_bandit"},
            {"type": "line", "label": "Path Model:",
             "default": "greedy_bandit/NSFNet/0617/16_47_22_694727/state_vals_e750.0_routes_c4.json"},
            {"type": "combo", "label": "Core Algorithm:", "default": "first_fit", "options": ["first_fit"]},
            {"type": "line", "label": "Core Model:",
             "default": "greedy_bandit/NSFNet/0617/16_57_13_315030/state_vals_e750.0_cores_c4.json"},
            {"type": "combo", "label": "Spectrum Algorithm:", "default": "first_fit", "options": ["first_fit"]},
            {"type": "line", "label": "Spectrum Model:", "default": "ppo/NSFNet/0512/12_57_55_484293"},
            {"type": "line", "label": "Render Mode:", "default": "None"},
            {"type": "spin", "label": "Super Channel Space:", "default": 3, "min": 1},
            {"type": "double_spin", "label": "Learn Rate:", "default": 0.01},
            {"type": "double_spin", "label": "Discount Factor:", "default": 0.95},
            {"type": "double_spin", "label": "Epsilon Start:", "default": 0.2},
            {"type": "double_spin", "label": "Epsilon End:", "default": 0.05},
            {"type": "spin", "label": "Reward:", "default": 1},
            {"type": "spin", "label": "Penalty:", "default": -100},
            {"type": "check", "label": "Dynamic Reward:", "default": False},
            {"type": "spin", "label": "Path Levels:", "default": 2, "min": 1},
            {"type": "double_spin", "label": "Decay Factor:", "default": 0.01},
            {"type": "double_spin", "label": "Core Beta:", "default": 0.1},
            {"type": "double_spin", "label": "Gamma:", "default": 0.1}
        ]
    },
    {
        "category": "ML",
        "settings": [
            {"type": "check", "label": "Deploy Model:", "default": False},
            {"type": "check", "label": "Output Train Data:", "default": False},
            {"type": "check", "label": "ML Training:", "default": True},
            {"type": "line", "label": "ML Model:", "default": "decision_tree"},
            {"type": "line", "label": "Train File Path:", "default": "Pan-European/0531/22_00_16_630834"},
            {"type": "double_spin", "label": "Test Size:", "default": 0.3, "min": 0.0, "max": 1.0}
        ]
    },
    {
        "category": "File",
        "settings": [
            {"type": "line", "label": "File Type:", "default": "json"}
        ]
    }
]

# TODO: update documentation to reflect this.
GUI_DEFAULTS = {
    'supported_networks': ['USNet', 'NSFNet', 'Pan-European', 'USbackbone60'],
    'plots': ['Blocking', 'Block Reasons', 'Errors', 'Hops', 'Path Length', 'Rewards'],
    'license_file_path': 'LICENSE'
}


class AlertCode(Enum):
    INFORMATION = 0
    CRITICAL = 1
    QUESTION = 2
    WARNING = 3
