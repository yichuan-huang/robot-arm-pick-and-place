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
            max_episode_steps=50,
            # ---- Trajectory and teacher assistance are enabled by default, no need to specify for the left arm environment ----
            use_trajectory_teacher=True,
            traj_hover_z=0.18,
            traj_pick_z=0.035,
            traj_place_z=0.035,
            traj_pos_tol=0.0125,
            teacher_lambda=0.01,
            teacher_gain=0.08,  # Smaller step size for more stability
            assist_init=1.0,  # High teacher proportion in the early stage
            assist_decay=3.0,  # Gradually decay to 0 (within the whole scene)
            track_reward_scale=2.0,  # Trajectory tracking shaping
            **kwargs,
        )
