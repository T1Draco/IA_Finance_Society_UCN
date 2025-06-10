from stable_baselines3.common.callbacks import BaseCallback

class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("custom/capital", self.training_env.get_attr("total_asset")[0])
        self.logger.record("custom/balance", self.training_env.get_attr("balance")[0])
        self.logger.record("custom/shares", self.training_env.get_attr("shares_held")[0])
        return True
