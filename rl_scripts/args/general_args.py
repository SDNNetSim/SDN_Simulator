

EPISODIC_STRATEGIES = ['exp_decay', 'linear_decay']


VALID_PATH_ALGORITHMS = [
    'q_learning',
    'epsilon_greedy_bandit',
    'ucb_bandit',
    'ppo',
]

VALID_CORE_ALGORITHMS = [
    'q_learning',
    'epsilon_greedy_bandit',
    'ucb_bandit',
]

VALID_DRL_ALGORITHMS = [
    'ppo',
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
