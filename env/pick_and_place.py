import os
import numpy as np
from env.robot_arm_env import RobotArmEnv

MODEL_XML_PATH = os.path.join(
    os.path.dirname(__file__), "../assets/", "pick_and_place.xml"
)


class RobotArmPickAndPlaceEnv(RobotArmEnv):
    def __init__(self, **kwargs):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            distance_threshold=0.05,
            obj_xy_range=0.25,
            obj_x_offset=0.6,
            obj_y_offset=0.3,
            max_episode_steps=150,
            **kwargs,
        )
