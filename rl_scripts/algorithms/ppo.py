from gymnasium import spaces


class PPO:
    """
    Facilitates Proximal Policy Optimization (PPO) for reinforcement learning.

    This class provides functionalities for handling observation space, action space,
    and rewards specific to the PPO framework for reinforcement learning. It's driven
    by the properties passed during initialization to define the behavior and attributes
    of the reinforcement learning environment and its engine.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the ppo reinforcement learning framework.
        """
        resp_obs = spaces.Dict({
            'source': spaces.MultiBinary(self.rl_props.num_nodes),
            'destination': spaces.MultiBinary(self.rl_props.num_nodes),
        })

        return resp_obs

    def get_action_space(self):
        """
        Get the action space for the environment.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
