import os
import time
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import SAC
import numpy as np
import env # Ensure custom envs are registered

# ---- Config ----
env_id = "RobotArmPickAndPlace-v0"
model_path = "model/SAC_pick_and_place.zip"
episodes = 50
video_fps = 30
# Output
video_folder = "videos"
video_filename = "SAC_pick_and_place_all_episodes.mp4"

os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, video_filename)

# ---- Create environment ----
# Prefer rgb_array for stable frame capture
env = gym.make(env_id, render_mode="rgb_array")

# ---- Load model ----
model = SAC.load(model_path)


# ---- Helper to grab a frame ----
def get_frame_from_env(env, info=None):
    # Most Gymnasium Mujoco envs expose frames via env.render(). With render_mode="rgb_array", step() already updates internal buffer.
    frame = env.render()  # should return an RGB array (H, W, 3), dtype=uint8
    if frame is None:
        # Fallback: some envs provide 'rgb' inside info or via env.sim.render. Adjust if needed.
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
# We will open the writer lazily after getting the first frame to know frame size
writer = None

total_frames_written = 0

try:
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0

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
        # Write the single initial frame
        writer.append_data(first_frame)
        total_frames_written += 1

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            frame = get_frame_from_env(env, info)
            # Write the single frame
            writer.append_data(frame)
            total_frames_written += 1

        print(
            f"Episode {ep + 1} finished. reward={ep_reward:.3f}, terminated={terminated}, truncated={truncated}"
        )

finally:
    if writer is not None:
        writer.close()
    env.close()

print(f"Video saved to: {os.path.abspath(video_path)}")
print(f"Frames written: {total_frames_written}, playback FPS: {video_fps}")
