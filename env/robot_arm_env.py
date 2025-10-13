import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from typing import Optional, Any, SupportsFloat

# Default camera configuration for rendering the environment
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.6, 0.5, 0.3]),
}


class RobotArmEnv(MujocoRobotEnv):
    """
    A custom robot arm environment for a pick-and-place task using a suction gripper.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 25,
        distance_threshold: float = 0.05,
        goal_position: Optional[np.ndarray] = None,
        obj_xy_range: float = 0.25,
        obj_x_offset: float = 0.6,
        obj_y_offset: float = 0.3,
        max_episode_steps: int = 200,
        # Action control parameters
        action_scale: Optional[
            np.ndarray
        ] = None,  # Scaling factor for each joint's action
        # Reward weights
        w_progress: float = 2.0,
        w_distance: float = -1.0,
        w_action: float = -0.001,
        w_action_change: float = -0.005,
        w_smooth: float = -0.002,
        w_height: float = 0.5,
        success_reward: float = 10.0,
        terminal_bonus: float = 20.0,
        progress_clip: float = 0.1,
        vel_ema_tau: float = 0.95,
        # Suction gripper parameters
        suction_distance_threshold: float = 0.06,  # Activation distance threshold
        **kwargs,
    ):
        self.model_path = model_path
        action_size = 4  # 3 joints + 1 gripper

        # *** Initialize the goal first, as _get_obs() uses it ***
        if goal_position is None:
            self.goal = np.array([0.6032, 0.5114, 0.025], dtype=np.float64)
        else:
            self.goal = np.array(goal_position, dtype=np.float64)

        # Action scaling factors
        if action_scale is None:
            # Default values: [joint1_rot, joint2_rot, joint3_slide]
            # Rotational joints: ±0.1 rad/step ≈ ±5.7 deg/step
            # Prismatic joint: ±0.02 m/step = ±2 cm/step
            self.action_scale = np.array([0.1, 0.1, 0.02], dtype=np.float64)
        else:
            self.action_scale = np.array(action_scale, dtype=np.float64)

        # Episode tracking
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.previous_ee_position = None
        self.initial_object_height = 0.025

        # Gripper state
        self.is_gripping = False
        self.suction_distance_threshold = suction_distance_threshold

        # Reward weights
        self.w_progress = float(w_progress)
        self.w_distance = float(w_distance)
        self.w_action = float(w_action)
        self.w_action_change = float(w_action_change)
        self.w_smooth = float(w_smooth)
        self.w_height = float(w_height)
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

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=None,  # Use initial position defined in the XML file
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Control range
        self.ctrl_range = self.model.actuator_ctrlrange

        # Get suction weld constraint ID
        self.suction_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_weld"
        )

    def _initialize_simulation(self) -> None:
        """Initializes the MuJoCo simulation."""
        # Ensure goal is initialized
        if not hasattr(self, "goal"):
            self.goal = np.array([0.6032, 0.5114, 0.025], dtype=np.float64)

        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.joint_names = ["joint1", "joint2", "joint3_slide"]

        self._env_setup()
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_qpos = np.copy(self.data.qpos)

    def _env_setup(self) -> None:
        """Sets up the environment using initial positions from the XML."""
        self._mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Executes one timestep in the environment."""
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get current object position for progress calculation
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        # Store state from the previous step
        if self.current_step == 0:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()
        else:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()

        # Apply action and step simulation
        self._set_action(action)
        self._mujoco_step()
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        # 🔥 Update step count *before* getting the new observation
        self.current_step += 1

        # Get new observation
        obs = self._get_obs().copy()

        # Calculate distances
        current_distance = self.goal_distance(obs["achieved_goal"], self.goal)
        ee_to_obj_distance = np.linalg.norm(
            self.get_ee_position() - obs["achieved_goal"]
        )

        # Build info dictionary
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "steps": self.current_step,
            "distance_to_goal": current_distance,
            "ee_to_object_distance": ee_to_obj_distance,
            "is_gripping": self.is_gripping,
            "object_height": obs["achieved_goal"][2],
        }

        # 🔥 Determine termination conditions
        terminated = bool(info["is_success"])
        truncated = self.current_step >= self.max_episode_steps

        # Calculate reward
        reward = self.compute_reward(
            obs["achieved_goal"], self.goal, info, action=action, obs_dict=obs
        )

        # Add bonus for successful termination
        if terminated and self.terminal_bonus != 0.0:
            reward = float(reward) + float(self.terminal_bonus)

        # Cache action and goal for the next step
        self.prev_action = action.copy()
        self.prev_achieved_goal = obs["achieved_goal"].copy()

        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
        action: Optional[np.ndarray] = None,
        obs_dict: Optional[dict] = None,
    ) -> SupportsFloat:
        """Computes the reward for the current step."""
        d = float(self.goal_distance(achieved_goal, desired_goal))
        reward_components: dict[str, float] = {}

        # 1) Distance penalty: penalize distance to goal
        reward_components["distance"] = self.w_distance * d

        # 2) Progress reward: reward for moving closer to the goal
        progress_raw = 0.0
        if self.prev_achieved_goal is not None:
            prev_distance = float(
                self.goal_distance(self.prev_achieved_goal, desired_goal)
            )
            progress_raw = float(prev_distance - d)
            # Clip progress to prevent large reward fluctuations
            if progress_raw > 0:
                progress_raw = min(progress_raw, self.progress_clip * 2)
            else:
                progress_raw = max(progress_raw, -self.progress_clip)
        reward_components["progress"] = self.w_progress * progress_raw

        # 3) Height reward: encourage lifting the object
        height_diff = float(achieved_goal[2] - self.initial_object_height)
        if height_diff > 0:
            # Give a larger reward for lifting while gripping
            height_bonus = height_diff * (2.0 if self.is_gripping else 1.0)
            reward_components["height"] = self.w_height * height_bonus
        else:
            reward_components["height"] = 0.0

        # 4) Smoothness penalty: penalize jerky movements
        smooth_penalty = 0.0
        if obs_dict is not None:
            observation = obs_dict["observation"]
            ee_vel_dt = observation[3:6]
            ee_velocity = ee_vel_dt / max(self.dt, 1e-8)
            # Use an exponential moving average to smooth the velocity
            self.vel_ema = (
                self.vel_ema_tau * self.vel_ema + (1.0 - self.vel_ema_tau) * ee_velocity
            )
            smooth_penalty = float(np.linalg.norm(self.vel_ema))
        reward_components["smooth"] = self.w_smooth * smooth_penalty

        # 5) Action penalties: encourage smaller and more consistent actions
        if action is not None:
            action_norm = float(np.linalg.norm(action))
            reward_components["action"] = self.w_action * action_norm
            # Penalty for change in action to encourage continuity
            if self.prev_action is not None:
                action_diff = float(np.linalg.norm(action - self.prev_action))
                reward_components["action_change"] = self.w_action_change * action_diff
            else:
                reward_components["action_change"] = 0.0
        else:
            reward_components["action"] = 0.0
            reward_components["action_change"] = 0.0

        # 6) Success reward: large bonus for reaching the goal
        reward_components["success"] = (
            self.success_reward if d < self.distance_threshold else 0.0
        )

        # 7) Gripping bonus: small reward for maintaining a grip
        reward_components["gripping_bonus"] = 1.0 if self.is_gripping else 0.0

        # Calculate total reward
        total_reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        return total_reward

    def _set_action(self, action: np.ndarray) -> None:
        """Applies the given action to the simulation (position control)."""
        action = action.copy()

        # 1. Control arm joints
        arm_action = action[:3]
        current_joints_list = []
        for name in self.joint_names:
            qpos = self._utils.get_joint_qpos(self.model, self.data, name)
            # Ensure qpos is a scalar float
            current_joints_list.append(
                float(qpos.flat[0]) if isinstance(qpos, np.ndarray) else float(qpos)
            )
        current_joints = np.array(current_joints_list, dtype=np.float64)

        # Calculate target position: current + (action * scale)
        target_joints = current_joints + arm_action * self.action_scale

        # Clip target to joint limits
        target_joints = np.clip(
            target_joints, self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        )

        # Send position command to actuators
        self.data.ctrl[:3] = target_joints

        # 2. Control suction gripper
        gripper_action = action[3]
        self._set_gripper_action(gripper_action)

    def _set_gripper_action(self, gripper_action: float) -> None:
        """Controls the suction gripper based on the action value."""
        if gripper_action > 0:  # Command: Activate
            if not self.is_gripping:
                # If not gripping, try to activate suction
                if self.try_activate_suction():
                    self.is_gripping = True
            else:
                # If already gripping, ensure the weld is active
                self.data.eq_active[self.suction_weld_id] = 1
        else:  # Command: Deactivate
            if self.is_gripping:
                self.deactivate_suction()
                self.is_gripping = False

    def try_activate_suction(self) -> bool:
        """Attempts to activate suction and returns True if successful."""
        ee_pos = self.get_ee_position()
        obj_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site")
        distance = np.linalg.norm(ee_pos - obj_pos)

        if distance < self.suction_distance_threshold:
            self.data.eq_active[self.suction_weld_id] = 1
            return True
        return False

    def deactivate_suction(self) -> None:
        """Deactivates the suction gripper."""
        self.data.eq_active[self.suction_weld_id] = 0

    def _get_obs(self) -> dict:
        """Returns the current observation from the environment."""
        # Defensive check to ensure goal is initialized
        if not hasattr(self, "goal") or self.goal.size == 0:
            self.goal = np.array([0.6032, 0.5114, 0.025], dtype=np.float64)

        # End-effector state
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_site").copy()
        ee_velocity = (
            self._utils.get_site_xvelp(self.model, self.data, "ee_site").copy()
            * self.dt
        )

        # Object state
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()
        object_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy()
            * self.dt
        )

        # Joint positions
        joint_positions_list = []
        for name in self.joint_names:
            qpos = self._utils.get_joint_qpos(self.model, self.data, name)
            # Ensure value is a scalar float
            if isinstance(qpos, np.ndarray):
                val = float(qpos) if qpos.ndim == 0 else float(qpos.flat[0])
                joint_positions_list.append(val)
            else:
                joint_positions_list.append(float(qpos))
        joint_positions = np.array(joint_positions_list, dtype=np.float64)

        # Geometric relationships
        object_goal_distance = np.linalg.norm(object_position - self.goal)
        ee_object_distance = np.linalg.norm(ee_position - object_position)
        goal_rel_pos = self.goal - object_position
        object_rel_ee = object_position - ee_position

        # Normalized time
        normalized_time = self.current_step / self.max_episode_steps

        # Gripper state
        gripper_state = float(self.is_gripping)

        observation = np.concatenate(
            [
                ee_position,  # 3 - End-effector position
                ee_velocity,  # 3 - End-effector velocity
                joint_positions,  # 3 - Joint positions
                object_position,  # 3 - Object position
                object_velp,  # 3 - Object velocity
                goal_rel_pos,  # 3 - Goal position relative to object
                object_rel_ee,  # 3 - Object position relative to EE
                [ee_object_distance],  # 1 - EE-to-object distance
                [object_goal_distance],  # 1 - Object-to-goal distance
                [normalized_time],  # 1 - Normalized time
                [gripper_state],  # 1 - Gripper state (0 or 1)
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": object_position.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        """Determines if the task has been successfully completed."""
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        """Optional callback for rendering."""
        pass

    def _reset_sim(self) -> bool:
        """Resets the simulation to an initial state."""
        # Reset time and velocities
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self.data.qpos[:] = np.copy(self.initial_qpos)

        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize object position
        self._sample_object()

        # Reset episode-specific states
        self.current_step = 0
        self.previous_ee_position = None
        self.previous_distance = None
        self.prev_action = None
        self.prev_achieved_goal = None
        self.vel_ema[:] = 0.0

        # Reset gripper
        self.is_gripping = False
        self.data.eq_active[self.suction_weld_id] = 0

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self) -> None:
        """Advances the MuJoCo simulation."""
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        """Calculates the Euclidean distance between two goals."""
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_object(self) -> None:
        """Randomly samples a new position for the object within a safe range."""
        # Custom range: in front of the arm, on the same side
        # x: 0.28-0.55 (not too close, not too far)
        # y: 0.25-0.55 (same side as the arm base)
        # z: 0.025 (table height)
        custom_range_low = np.array([0.28, 0.25, 0.025])
        custom_range_high = np.array([0.55, 0.55, 0.025])

        object_position = self.np_random.uniform(custom_range_low, custom_range_high)

        # Set object position (quaternion [w, x, y, z] = [1, 0, 0, 0] for no rotation)
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def _sample_goal(self) -> np.ndarray:
        """Returns the fixed goal position."""
        return self.goal.copy()

    def get_ee_position(self) -> np.ndarray:
        """Gets the current position of the end-effector."""
        return self._utils.get_site_xpos(self.model, self.data, "ee_site")

    # Kept for backward compatibility
    def activate_suction(self) -> None:
        """Simplified method to activate suction."""
        self.try_activate_suction()
