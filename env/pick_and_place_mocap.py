import os
from env.robot_arm_mocap_env import RobotArmMocapEnv

MODEL_XML_PATH = os.path.join(
    os.path.dirname(__file__), "../assets/", "pick_and_place_mocap.xml"
)


class RobotArmPickAndPlaceMocapEnv(RobotArmMocapEnv):
    def __init__(
        self,
        reward_type="dense",
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=50,
            distance_threshold=0.05,
            obj_xy_range=0.25,
            obj_x_offset=0.6,
            obj_y_offset=0.3,
            max_episode_steps=100,
            pos_ctrl_scale=0.05,
            # Reward weights
            w_progress=2.0,
            w_distance=-1.0,
            w_action=-0.001,
            w_action_change=-0.005,
            w_smooth=-0.002,
            w_height=0.5,
            w_gripper=0.3,
            success_reward=10.0,
            terminal_bonus=20.0,
            **kwargs,
        )
