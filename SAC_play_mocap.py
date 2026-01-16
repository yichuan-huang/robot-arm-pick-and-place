import os
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import SAC
import numpy as np
import env as custom_env  # noqa: F401 Ensure custom envs are registered

# ---- Config ----
env_id = "RobotArmPickAndPlaceMocap-v0"
model_path = "model/SAC_pick_and_place_mocap.zip"
num_episodes = 10
video_fps = 30

# Output
video_folder = "videos"
video_filename = "SAC_pick_and_place_mocap_all_episodes.mp4"

os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, video_filename)

# ---- Create environment ----
# Use rgb_array for stable frame capture
env = gym.make(env_id, render_mode="rgb_array")

# ---- Load model ----
model = SAC.load(model_path)


# ---- Helper to grab a frame ----
def get_frame_from_env(env, info=None):
    frame = env.render()  # should return an RGB array (H, W, 3), dtype=uint8
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Ensure render_mode='rgb_array' or adapt frame extraction."
        )
    # Ensure uint8
    if frame.dtype != np.uint8:
        frame = (
            (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if frame.max() <= 1.0
            else frame.astype(np.uint8)
        )
    return frame


# ---- Prepare video writer ----
writer = None
total_frames_written = 0
success_count = 0

try:
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        print(f"\n--- Episode {episode + 1} ---")

        # Capture an initial frame right after reset
        first_frame = get_frame_from_env(env, info)
        if writer is None:
            # Initialize writer with the frame size
            h, w = first_frame.shape[:2]
            writer = imageio.get_writer(
                video_path,
                fps=video_fps,
                codec="libx264",
                quality=8,  # 0(worst)-10(best)
                macro_block_size=None,  # avoid resizing issues
                pixelformat="yuv420p",  # broad compatibility
            )
        # Write the initial frame
        writer.append_data(first_frame)
        total_frames_written += 1

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            step_count += 1

            frame = get_frame_from_env(env, info)
            writer.append_data(frame)
            total_frames_written += 1

        if done:
            success_count += 1
            print(f"Success! Episode reward: {episode_reward:.2f}, Steps: {step_count}")
        elif truncated:
            print(
                f"Truncated. Episode reward: {episode_reward:.2f}, Steps: {step_count}"
            )

        print(f"Distance to goal: {info['distance_to_goal']:.4f}")
        print(f"Is gripping: {info['is_gripping']}")

finally:
    if writer is not None:
        writer.close()
    env.close()

print(f"\n=== Summary ===")
print(
    f"Success rate: {success_count}/{num_episodes} = {success_count/num_episodes*100:.1f}%"
)
print(f"Video saved to: {os.path.abspath(video_path)}")
print(f"Frames written: {total_frames_written}, playback FPS: {video_fps}")
