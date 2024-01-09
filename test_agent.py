import os
import argparse
from ppo_agent import PPOAgent
from mario_env import create_mario_env


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def test_agent(model_path):
    env = create_mario_env("test")
    model = PPOAgent.load(model_path)

    state = env.reset()
    while True:
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    project_root = get_project_root()
    default_path = os.path.join(project_root, 'models', 'final_mario_model.zip')
    parser = argparse.ArgumentParser(description="Test a trained PPO agent on Super Mario Bros")
    parser.add_argument('--model_path', type=str, default=default_path, required=False, help='Path to the trained model')

    args = parser.parse_args()

    model_full_path = args.model_path

    test_agent(model_full_path)