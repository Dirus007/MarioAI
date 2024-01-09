from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, env, learning_rate=0.000001, n_steps=512):
        self.model = PPO('CnnPolicy', env, verbose=1, learning_rate=learning_rate, n_steps=n_steps)

    def train(self, total_timesteps, callback):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        return PPO.load(path)
