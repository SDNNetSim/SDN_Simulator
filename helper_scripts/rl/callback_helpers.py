from stable_baselines3.common.callbacks import BaseCallback


class GetModelParams(BaseCallback):
    """
    Handles all methods related to custom callbacks in the StableBaselines3 library.
    """

    def __init__(self, verbose: int = 0):
        super(GetModelParams, self).__init__(verbose)  # pylint: disable=super-with-arguments

        self.model_params = None
        self.value_estimate = 0.0

    def _on_step(self) -> bool:
        """
        Every step of the model this method is called. Retrieves the estimated value function for the PPO algorithm.
        """
        self.model_params = self.model.get_parameters()

        obs = self.locals['obs_tensor']
        self.value_estimate = self.model.policy.predict_values(obs=obs)[0][0].item()
        return True
