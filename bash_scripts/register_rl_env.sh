#!/bin/bash

# This script registers a custom reinforcement learning environment in the Gymnasium RL library
# and Stable Baselines 3. It is intended to be used with a specific algorithm and environment
# name provided as arguments.

# Usage: register_rl_env.sh <algo> <env_name>
# Example: ./register_rl_env.sh PPO SimEnv

# Check if the correct number of arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: register_rl_env.sh <algo> <env_name>"
  exit 1
fi

# Assign arguments to variables for better readability
algo="$1"
env_name="$2"

# Run the Python script to register the environment with the specified algorithm and environment name
python sb3_scripts/register_env.py --algo "$algo" --env-name "$env_name"

# Append the custom environment registration to the import_envs.py file in the Stable Baselines 3 installation
cat >>venvs/unity_venv/venv/lib/python3.11/site-packages/rl_zoo3/import_envs.py <<EOL

from rl_scripts.gymnasium_envs.general_sim_env import SimEnv

register(id='SimEnv', entry_point='rl_scripts.gymnasium_envs.general_sim_env:SimEnv')
EOL

echo "Custom environment 'SimEnv' registered for algorithm '$algo' in Stable Baselines 3 and Gymnasium RL library."
