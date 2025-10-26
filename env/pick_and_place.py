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
            # ---- Trajectory and teacher assistance are enabled by default ----
            use_trajectory_teacher=True,
            traj_hover_z=0.18,
            traj_pick_z=0.04,  # 稍微提高pick高度，避免过早接触
            traj_place_z=0.04,
            traj_pos_tol=0.015,  # 稍微放松容差
            teacher_lambda=0.01,
            teacher_gain=0.1,  # 增加引导力度
            assist_init=1.0,
            assist_decay=2.3,  # 减慢衰减速度，让teacher辅助更持久
            track_reward_scale=2.0,  # 增加轨迹跟踪奖励
            suction_distance_threshold=0.05,  # 调整吸盘阈值
            **kwargs,
        )
