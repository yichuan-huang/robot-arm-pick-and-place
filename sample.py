import gymnasium as gym
import env

env = gym.make("RobotArmPickAndPlace-v0", render_mode="human")

try:
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

finally:
    env.close()  # Ensure the environment is closed correctly
