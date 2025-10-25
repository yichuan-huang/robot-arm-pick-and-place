import time

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import env  # noqa: F401


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True  # Continue training

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


env = gym.make("RobotArmPickAndPlace-v0", render_mode="rgb_array")
model = SAC(
    "MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/SAC_pick_and_place/"
)
start_time = time.time()
progress_callback = ProgressBarCallback(total_timesteps=1000000)
model.learn(total_timesteps=1000000, callback=progress_callback)
model.save("model/SAC_pick_and_place.zip")
end_time = time.time()
with open("./logs/SAC_pick_and_place/SAC_pick_and_place_train_time.txt", "w") as opener:
    opener.write("spend_tine:{}".format(end_time - start_time))

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
