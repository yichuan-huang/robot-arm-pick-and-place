import gymnasium as gym
from stable_baselines3 import SAC
import env  # noqa: F401


# Load the trained model
model = SAC.load("model/SAC_pick_and_place_mocap.zip")

# Create environment with human rendering
env = gym.make("RobotArmPickAndPlaceMocap-v0", render_mode="human")

# Test for multiple episodes
num_episodes = 10
success_count = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    step_count = 0

    print(f"\n--- Episode {episode + 1} ---")

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        if done:
            success_count += 1
            print(f"Success! Episode reward: {episode_reward:.2f}, Steps: {step_count}")
        elif truncated:
            print(
                f"Truncated. Episode reward: {episode_reward:.2f}, Steps: {step_count}"
            )

    print(f"Distance to goal: {info['distance_to_goal']:.4f}")
    print(f"Is gripping: {info['is_gripping']}")

env.close()

print(f"\n=== Summary ===")
print(
    f"Success rate: {success_count}/{num_episodes} = {success_count/num_episodes*100:.1f}%"
)
