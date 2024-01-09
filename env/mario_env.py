import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gym

def create_mario_env(mode):
    if mode == "train":
        env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
    else:
        env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    return env
