import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.6, 0.5, 0.3]),
}


class RobotArmMocapEnv(MujocoRobotEnv):
    """
    Robot arm environment with mocap control (similar to Franka Panda).

    This environment controls the end-effector position directly through mocap body,
    which is welded to the suction cup. The underlying joints are automatically
    driven by MuJoCo's constraint solver to match the mocap pose.

    Action space:
        - action[0:3]: End-effector position delta (dx, dy, dz)
        - action[3]: Gripper control (suction on/off)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        distance_threshold: float = 0.05,
        goal_position: Optional[np.ndarray] = None,
        obj_xy_range: float = 0.25,
        obj_x_offset: float = 0.6,
        obj_y_offset: float = 0.3,
        max_episode_steps: int = 200,
        # Action control parameters (for end-effector position control)
        pos_ctrl_scale: float = 0.05,  # meters per action unit
        # Reward weights (similar to Panda env)
        w_progress: float = 2.0,
        w_distance: float = -1.0,
        w_action: float = -0.001,
        w_action_change: float = -0.005,
        w_smooth: float = -0.002,
        w_height: float = 0.5,
        w_gripper: float = 0.3,
        success_reward: float = 10.0,
        terminal_bonus: float = 20.0,
        progress_clip: float = 0.1,
        vel_ema_tau: float = 0.95,
        # Suction gripper parameters
        suction_distance_threshold: float = 0.06,
        gripper_target_width: float = 0.04,
        **kwargs,
    ):
        self.model_path = model_path
        action_size = 4  # 3 for position control + 1 for gripper

        # Initialize goal
        if goal_position is None:
            self.goal = np.array([0.6032, 0.5114, 0.025], dtype=np.float64)
        else:
            self.goal = np.array(goal_position, dtype=np.float64)

        # Position control scaling
        self.pos_ctrl_scale = float(pos_ctrl_scale)

        # Episode tracking
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.trajectory_points = []
        self.previous_ee_position = None
        self.total_path_length = 0.0
        self.initial_object_height = 0.025

        # Gripper state
        self.is_gripping = False
        self.suction_distance_threshold = suction_distance_threshold
        self.gripper_target_width = gripper_target_width

        # Reward weights
        self.w_progress = float(w_progress)
        self.w_distance = float(w_distance)
        self.w_action = float(w_action)
        self.w_action_change = float(w_action_change)
        self.w_smooth = float(w_smooth)
        self.w_height = float(w_height)
        self.w_gripper = float(w_gripper)
        self.success_reward = float(success_reward)
        self.terminal_bonus = float(terminal_bonus)
        self.progress_clip = float(progress_clip)
        self.vel_ema_tau = float(vel_ema_tau)

        # Caching for reward calculation
        self.vel_ema = np.zeros(3, dtype=np.float64)
        self.prev_action: Optional[np.ndarray] = None
        self.prev_achieved_goal: Optional[np.ndarray] = None
        self.previous_distance: Optional[float] = None

        self.distance_threshold = distance_threshold

        # Object sampling range
        self.obj_xy_range = obj_xy_range
        self.obj_x_offset = obj_x_offset
        self.obj_y_offset = obj_y_offset
        self.obj_range_low = np.array(
            [
                self.obj_x_offset - self.obj_xy_range / 2,
                self.obj_y_offset - self.obj_xy_range / 2,
                0.025,
            ],
            dtype=np.float64,
        )
        self.obj_range_high = np.array(
            [
                self.obj_x_offset + self.obj_xy_range / 2,
                self.obj_y_offset + self.obj_xy_range / 2,
                0.025,
            ],
            dtype=np.float64,
        )

        # Neutral/initial joint positions
        self.neutral_joint_values = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Control range
        self.ctrl_range = self.model.actuator_ctrlrange

    # ---------- Mujoco / base lifecycle ----------

    def _initialize_simulation(self) -> None:
        # Ensure goal is initialized before any other operations
        if not hasattr(self, "goal") or self.goal is None or self.goal.shape[0] == 0:
            self.goal = np.array([0.6032, 0.5114, 0.025], dtype=np.float64)

        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # Joint names
        self.joint_names = ["joint1", "joint2", "joint3_slide"]

        # Get suction weld constraint ID
        self.suction_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_weld"
        )

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        """Setup environment, initialize mocap body position"""
        self.set_joint_neutral()

        # Set initial joint positions
        for i, name in enumerate(self.joint_names):
            self._utils.set_joint_qpos(
                self.model, self.data, name, neutral_joint_values[i]
            )

        # Reset mocap welds
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)

        # Get initial mocap position from end-effector
        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()

        # Get initial orientation
        self.grasp_site_pose = self.get_ee_orientation().copy()

        # Set mocap pose
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        # Step simulation to apply constraints
        self._mujoco_step()

    def reset_mocap_welds(self, model, data) -> None:
        """Reset mocap weld constraints to relative pose"""
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # Set relative pose to identity
                    model.eq_data[i, 3:10] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        self._mujoco.mj_forward(model, data)

    # ---------- Core step ----------

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get current object position
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        # Initialize distances on first step
        if self.current_step == 0:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()
        else:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()

        # Apply action and step simulation
        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        # Track trajectory
        ee_pos = self.get_ee_position()
        self.trajectory_points.append(ee_pos.copy())

        # Update total path length
        step_distance = np.linalg.norm(ee_pos - self.previous_ee_position)
        self.total_path_length += step_distance

        self.current_step += 1

        # Get new observation
        obs = self._get_obs().copy()

        # Calculate distances
        current_distance = self.goal_distance(obs["achieved_goal"], self.goal)
        ee_to_obj_distance = np.linalg.norm(ee_pos - obs["achieved_goal"])

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "steps": self.current_step,
            "total_path_length": self.total_path_length,
            "distance_to_goal": current_distance,
            "ee_to_object_distance": ee_to_obj_distance,
            "is_gripping": self.is_gripping,
        }

        terminated = bool(info["is_success"])
        truncated = self.current_step >= self.max_episode_steps

        reward = self.compute_reward(
            obs["achieved_goal"], self.goal, info, action=action, obs_dict=obs
        )

        if terminated and self.terminal_bonus != 0.0:
            reward = float(reward) + float(self.terminal_bonus)

        self.prev_action = action.copy()
        self.prev_achieved_goal = obs["achieved_goal"].copy()

        return obs, reward, terminated, truncated, info

    def _set_action(self, action) -> None:
        """
        Set action for mocap control.
        action[0:3]: End-effector position delta (dx, dy, dz)
        action[3]: Gripper control (suction on/off)
        """
        action = action.copy()

        # Position control
        pos_ctrl = action[:3]
        gripper_ctrl = action[3]

        # Scale position control and add to current position
        pos_ctrl *= self.pos_ctrl_scale
        pos_ctrl += self.get_ee_position().copy()

        # Limit z to be above ground
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        # Set mocap pose
        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

        # Handle gripper (suction) control
        ee_pos = self.get_ee_position()
        object_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        ee_to_obj_dist = np.linalg.norm(ee_pos - object_pos)

        # Activate suction if close enough and gripper command is positive
        if gripper_ctrl > 0.0 and ee_to_obj_dist < self.suction_distance_threshold:
            if not self.is_gripping:
                self._activate_suction()
                self.is_gripping = True
        elif gripper_ctrl <= 0.0:
            if self.is_gripping:
                self._deactivate_suction()
                self.is_gripping = False

    def _activate_suction(self) -> None:
        """Activate suction weld constraint"""
        self.model.eq_active0[self.suction_weld_id] = 1

    def _deactivate_suction(self) -> None:
        """Deactivate suction weld constraint"""
        self.model.eq_active0[self.suction_weld_id] = 0

    def _get_obs(self) -> dict:
        """Get observation"""
        # End-effector state
        ee_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()

        ee_velocity = (
            self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy()
            * self.dt
        )

        # Object state
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        object_rotation = rotations.mat2euler(
            self._utils.get_site_xmat(self.model, self.data, "obj_site")
        ).copy()

        object_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy()
            * self.dt
        )

        object_velr = (
            self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy()
            * self.dt
        )

        # Relative positions
        if hasattr(self, "goal") and self.goal is not None and self.goal.shape[0] > 0:
            object_goal_distance = np.linalg.norm(object_position - self.goal)
            ee_object_distance = np.linalg.norm(ee_position - object_position)
            goal_rel_pos = self.goal - object_position
            object_rel_ee = object_position - ee_position
        else:
            object_goal_distance = 0.0
            ee_object_distance = 0.0
            goal_rel_pos = np.zeros(3)
            object_rel_ee = np.zeros(3)

        # Normalized time
        normalized_time = self.current_step / self.max_episode_steps

        # Gripper state
        gripper_state = np.array([1.0 if self.is_gripping else 0.0])

        obs = {
            "observation": np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    gripper_state,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                    goal_rel_pos,
                    object_rel_ee,
                    [ee_object_distance],
                    [object_goal_distance],
                    [normalized_time],
                ]
            ).copy(),
            "achieved_goal": object_position.copy(),
            "desired_goal": (
                self.goal.copy()
                if hasattr(self, "goal") and self.goal is not None
                else object_position.copy()
            ),
        }

        return obs

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
        action: Optional[np.ndarray] = None,
        obs_dict: Optional[dict] = None,
    ) -> SupportsFloat:
        """Compute dense reward (similar to Panda env)"""
        d = float(self.goal_distance(achieved_goal, desired_goal))

        reward_components: dict[str, float] = {}

        # 1) Distance penalty
        reward_components["distance"] = self.w_distance * d

        # 2) Progress reward
        prev_distance = None
        if self.prev_achieved_goal is not None:
            prev_distance = float(
                self.goal_distance(self.prev_achieved_goal, desired_goal)
            )
        elif self.previous_distance is not None:
            prev_distance = float(self.previous_distance)

        progress_raw = 0.0
        if prev_distance is not None:
            progress_raw = float(prev_distance - d)
            # Clip progress
            if progress_raw > 0:
                progress_raw = min(progress_raw, self.progress_clip * 2)
            else:
                progress_raw = max(progress_raw, -self.progress_clip)
        reward_components["progress"] = self.w_progress * progress_raw

        # 3) Gripper reward (encourage gripping when close)
        gripper_component = 0.0
        ee_pos = self.get_ee_position()
        ee_obj_distance = float(np.linalg.norm(ee_pos - achieved_goal))
        if ee_obj_distance < 0.05:
            # Reward for activating gripper when close to object
            if self.is_gripping:
                gripper_component = self.w_gripper * 1.0
        reward_components["gripper"] = gripper_component

        # 4) Height reward (encourage lifting)
        height_bonus = max(0.0, float(achieved_goal[2] - self.initial_object_height))
        reward_components["height"] = self.w_height * height_bonus

        # 5) Smoothness penalty
        smooth_penalty = 0.0
        if obs_dict is not None:
            observation = obs_dict["observation"]
            ee_vel_dt = observation[3:6]
            ee_velocity = ee_vel_dt / max(self.dt, 1e-8)
            self.vel_ema = (
                self.vel_ema_tau * self.vel_ema + (1.0 - self.vel_ema_tau) * ee_velocity
            )
            smooth_penalty = float(np.linalg.norm(self.vel_ema))
        reward_components["smooth"] = self.w_smooth * smooth_penalty

        # 6) Action penalties
        action_component = 0.0
        action_change_component = 0.0
        if action is not None:
            action_norm = float(np.linalg.norm(action))
            action_component = self.w_action * action_norm
            if self.prev_action is not None:
                action_diff = float(np.linalg.norm(action - self.prev_action))
                action_change_component = self.w_action_change * action_diff
        reward_components["action"] = action_component
        reward_components["action_change"] = action_change_component

        # 7) Success reward
        success_component = self.success_reward if d < self.distance_threshold else 0.0
        reward_components["success"] = success_component

        total_reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        return total_reward

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        """Check if goal is achieved"""
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        """Visualize goal site"""
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        """Reset simulation"""
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Reset joints
        self.set_joint_neutral()

        # Reset mocap
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        # Sample object position
        self._sample_object()

        # Deactivate suction
        self.is_gripping = False
        self._deactivate_suction()

        # Reset episode tracking
        self.current_step = 0
        self.trajectory_points = []
        self.previous_ee_position = None
        self.total_path_length = 0.0
        self.previous_distance = None

        # Reset reward caching
        self.prev_action = None
        self.prev_achieved_goal = None
        self.vel_ema[:] = 0.0

        # Sample goal
        self.goal = self._sample_goal()

        # Initialize previous distance
        initial_object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()
        self.previous_distance = float(
            self.goal_distance(initial_object_position, self.goal)
        )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        """Step MuJoCo simulation"""
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # ---------- Helper methods ----------

    def set_joint_neutral(self) -> None:
        """Set joints to neutral position"""
        for i, name in enumerate(self.joint_names):
            self._utils.set_joint_qpos(
                self.model, self.data, name, self.neutral_joint_values[i]
            )

    def set_mocap_pose(self, position, orientation) -> None:
        """Set mocap body pose"""
        self._utils.set_mocap_pos(self.model, self.data, "arm_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "arm_mocap", orientation)

    def get_ee_position(self) -> np.ndarray:
        """Get end-effector position"""
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_ee_orientation(self) -> np.ndarray:
        """Get end-effector orientation as quaternion"""
        site_mat = self._utils.get_site_xmat(
            self.model, self.data, "ee_center_site"
        ).reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        """Calculate distance between two goals"""
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal position"""
        goal = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)

        # Sometimes place goal at table height, sometimes elevated
        if self.np_random.random() < 0.3:
            noise[2] = 0.0

        goal += noise
        return goal

    def _sample_object(self) -> None:
        """Sample object position"""
        object_position = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)
