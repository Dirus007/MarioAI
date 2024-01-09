import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env.mario_env import create_mario_env
from training.TrainAndLoggingCallback import TrainAndLoggingCallback


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LearningRateScheduler(BaseCallback):
    def __init__(self, schedule_function, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.schedule_function = schedule_function

    def _on_step(self):
        # Update the learning rate
        total_steps = self.model.num_timesteps
        new_learning_rate = self.schedule_function(total_steps)
        self.model.lr_schedule = lambda _: new_learning_rate
        return True


def create_schedule_function(initial_learning_rate, step_size, decay_rate):
    def schedule_function(total_steps):
        return initial_learning_rate * (decay_rate ** (total_steps // step_size))
    return schedule_function


def train_agent(args):
    env = create_mario_env("train")
    model = PPO('CnnPolicy', env, verbose=1, learning_rate=args.learning_rate, tensorboard_log="./ppo_tensorboard/")

    project_root = get_project_root()
    model_save_path = os.path.join(project_root, 'models', 'final_mario_model')
    checkpoint_dir = os.path.join(project_root, 'models')

    callback = TrainAndLoggingCallback(check_freq=args.check_freq, save_path=checkpoint_dir)
    schedule_fn = create_schedule_function(args.learning_rate, args.step_size, args.decay_rate)
    lr_scheduler = LearningRateScheduler(schedule_fn)

    model.learn(total_timesteps=args.total_timesteps, callback=[callback, lr_scheduler])
    model.save(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Super Mario Bros")
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps for training the model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate for the model')
    parser.add_argument('--step_size', type=int, default=10000, help='Number of steps after which learning rate decays')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate of the learning rate')
    parser.add_argument('--check_freq', type=int, default=10000, help='Frequency of saving the model')
    parser.add_argument('--n_steps', type=int, default=512, help='Number of steps to run for each environment per update')

    args = parser.parse_args()
    train_agent(args)
