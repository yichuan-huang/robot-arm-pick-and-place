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


class _TrajectoryPlanner:
    """
    Simple kinematic waypoint planner in task-space (EE position).
    Phases (implicitly left/right agnostic):
      0: move above object  (x=obj.x, y=obj.y, z=hover_z)
      1: descend to pick    (x=obj.x, y=obj.y, z=pick_z)
      2: lift               (x=obj.x, y=obj.y, z=hover_z)  [after suction]
      3: move above goal    (x=goal.x, y=goal.y, z=hover_z)
      4: descend to place   (x=goal.x, y=goal.y, z=place_z)
      5: release & retreat  (x=goal.x, y=goal.y, z=hover_z)
    """

    def __init__(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        hover_z: float = 0.18,
        pick_z: float = 0.035,
        place_z: float = 0.035,
        pos_tol: float = 0.01,
    ):
        self.hover_z = float(hover_z)
        self.pick_z = float(pick_z)
        self.place_z = float(place_z)
        self.pos_tol = float(pos_tol)

        self.update_anchors(obj_pos, goal_pos)
        self.phase = 0
        self.done = False

    def update_anchors(self, obj_pos: np.ndarray, goal_pos: np.ndarray):
        self.obj = np.asarray(obj_pos, dtype=np.float64)
        self.goal = np.asarray(goal_pos, dtype=np.float64)

        self.waypoints = [
            np.array([self.obj[0], self.obj[1], self.hover_z], dtype=np.float64),  # 0
            np.array([self.obj[0], self.obj[1], self.pick_z], dtype=np.float64),  # 1
            np.array([self.obj[0], self.obj[1], self.hover_z], dtype=np.float64),  # 2
            np.array([self.goal[0], self.goal[1], self.hover_z], dtype=np.float64),  # 3
            np.array([self.goal[0], self.goal[1], self.place_z], dtype=np.float64),  # 4
            np.array([self.goal[0], self.goal[1], self.hover_z], dtype=np.float64),  # 5
        ]

    def current_target(self) -> np.ndarray:
        return self.waypoints[min(self.phase, len(self.waypoints) - 1)]

    def phase_one_hot(self) -> np.ndarray:
        oh = np.zeros(6, dtype=np.float64)
        oh[min(self.phase, 5)] = 1.0
        return oh

    def advance_if_reached(self, ee_pos: np.ndarray, is_gripping: bool) -> None:
        if self.done:
            return

        tgt = self.current_target()
        if np.linalg.norm(ee_pos - tgt) < self.pos_tol:
            # Special gating around suction events:
            if self.phase == 1 and not is_gripping:
                # Wait at pick position until suction is successful
                return

            # ---- 修改部分：允许在 phase=4 等待有限步后继续 ----
            if self.phase == 4:
                if not hasattr(self, "release_wait_counter"):
                    self.release_wait_counter = 0
                if is_gripping:
                    self.release_wait_counter += 1
                    # 等待几步给 agent 松手机会（约0.1秒左右）
                    if self.release_wait_counter < 10:
                        return
                # 重置计数器
                self.release_wait_counter = 0

            self.phase += 1
            if self.phase >= 6:
                self.done = True


class RobotArmEnv(MujocoRobotEnv):
    """
    A custom robot arm environment for a pick-and-place task using a suction gripper.
    Adds a task-space trajectory planner and a residual "teacher" controller
    to ease learning when the policy only outputs delta joint commands.
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
        use_polar_object_sampling: bool = True,
        obj_radius_range: tuple[float, float] = (0.28, 0.52),
        obj_angle_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
        obj_table_bounds_low: Optional[np.ndarray] = None,
        obj_table_bounds_high: Optional[np.ndarray] = None,
        max_episode_steps: int = 200,
        # Action control parameters
        action_scale: Optional[np.ndarray] = None,  # scaling for 3 joints
        # Reward weights
        w_progress: float = 3.0,
        w_distance: float = -0.5,
        w_action: float = -0.0005,
        w_action_change: float = -0.0015,
        w_smooth: float = -0.001,
        w_height: float = 0.5,
        success_reward: float = 30.0,
        terminal_bonus: float = 40.0,
        progress_clip: float = 0.1,
        vel_ema_tau: float = 0.95,
        # Suction gripper parameters
        suction_distance_threshold: float = 0.06,
        # ---- Trajectory & residual-teacher params (NEW) ----
        use_trajectory_teacher: bool = True,
        traj_hover_z: float = 0.18,
        traj_pick_z: float = 0.035,
        traj_place_z: float = 0.035,
        traj_pos_tol: float = 0.0125,
        # residual IK step (per env step) using damped least squares
        teacher_lambda: float = 0.01,  # damping in IK
        teacher_gain: float = 0.8,  # target EE step gain (meters per step upper bound ~ action_scale)
        # assist scheduling: alpha = assist_init * exp(-decay * t/T)
        assist_init: float = 1.0,
        assist_decay: float = 3.0,
        track_reward_scale: float = 1.5,  # shaping reward weight for tracking target EE
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
            # [joint1_rot, joint2_rot, joint3_slide]
            # 增大 joint1 的动作幅度，鼓励更多使用
            self.action_scale = np.array([0.15, 0.12, 0.025], dtype=np.float64)
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
        self.grip_stabilization_steps = 0  # 吸附稳定计数器

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

        # Object sampling range (Keep same-side sampling, do not distinguish between left and right arms)
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

        # Object sampling configuration for encouraging joint1 motion
        self.use_polar_object_sampling = bool(use_polar_object_sampling)
        self.obj_radius_range = np.array(obj_radius_range, dtype=np.float64)
        self.obj_angle_range = np.array(obj_angle_range, dtype=np.float64)
        self.base_xy = np.array([0.245, 0.0], dtype=np.float64)
        if obj_table_bounds_low is None:
            obj_table_bounds_low = np.array([0.2, -0.25], dtype=np.float64)
        if obj_table_bounds_high is None:
            obj_table_bounds_high = np.array([0.75, 0.6], dtype=np.float64)
        self.obj_bounds_low = np.array(obj_table_bounds_low, dtype=np.float64)
        self.obj_bounds_high = np.array(obj_table_bounds_high, dtype=np.float64)

        # ---- Trajectory / Teacher ----
        self.use_trajectory_teacher = bool(use_trajectory_teacher)
        self.traj_hover_z = float(traj_hover_z)
        self.traj_pick_z = float(traj_pick_z)
        self.traj_place_z = float(traj_place_z)
        self.traj_pos_tol = float(traj_pos_tol)
        self.teacher_lambda = float(teacher_lambda)
        self.teacher_gain = float(teacher_gain)
        self.assist_init = float(assist_init)
        self.assist_decay = float(assist_decay)
        self.track_reward_scale = float(track_reward_scale)
        self._planner: Optional[_TrajectoryPlanner] = None

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

        # Cache site id for jacobian
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

    # ---------- Mujoco / base lifecycle ----------

    def _initialize_simulation(self) -> None:
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
        self._mujoco.mj_forward(self.model, self.data)

    # ---------- Core step ----------

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get current object position for progress calculation
        object_position = self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

        if self.current_step == 0:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()
        else:
            self.previous_distance = self.goal_distance(object_position, self.goal)
            self.previous_ee_position = self.get_ee_position().copy()

        # ---- Residual teacher blending (joint part only) ----
        if self.use_trajectory_teacher and self._planner is not None:
            # Update target based on current phase and states
            self._planner.update_anchors(object_position, self.goal)
            target_ee = self._planner.current_target()
            ee_now = self.get_ee_position().copy()

            # smooth progress of planner (phase auto-advance)
            self._planner.advance_if_reached(ee_now, self.is_gripping)

            # compute teacher joint-delta (same unit as qpos)
            teacher_joint_delta = self._ik_step_to_target(target_ee, ee_now)

            # convert to action units (divide by scale) and clip
            teacher_action = np.zeros_like(action)
            if teacher_joint_delta is not None:
                teacher_action[:3] = np.clip(
                    teacher_joint_delta / self.action_scale, -1.0, 1.0
                )

            # schedule alpha
            t = self.current_step / max(1, self.max_episode_steps)
            alpha = self.assist_init * np.exp(-self.assist_decay * t)
            # residual blend
            action = action.copy()
            action[:3] = (1.0 - alpha) * action[:3] + alpha * teacher_action[:3]

        # Apply action and step simulation
        self._set_action(action)
        self._mujoco_step()
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        self.current_step += 1

        # Get new observation
        obs = self._get_obs().copy()

        # Calculate distances
        current_distance = self.goal_distance(obs["achieved_goal"], self.goal)
        ee_to_obj_distance = np.linalg.norm(
            self.get_ee_position() - obs["achieved_goal"]
        )

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "steps": self.current_step,
            "distance_to_goal": current_distance,
            "ee_to_object_distance": ee_to_obj_distance,
            "is_gripping": self.is_gripping,
            "object_height": obs["achieved_goal"][2],
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

    # ---------- Reward with trajectory shaping ----------

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
        action: Optional[np.ndarray] = None,
        obs_dict: Optional[dict] = None,
    ) -> SupportsFloat:
        d = float(self.goal_distance(achieved_goal, desired_goal))
        reward_components: dict[str, float] = {}

        # 1) Distance penalty to final goal (object to goal)
        reward_components["distance"] = self.w_distance * d

        # 2) Progress shaping (Incremental progress towards the goal)
        progress_raw = 0.0
        if self.prev_achieved_goal is not None:
            prev_distance = float(
                self.goal_distance(self.prev_achieved_goal, desired_goal)
            )
            progress_raw = float(prev_distance - d)
            if progress_raw > 0:
                progress_raw = min(progress_raw, self.progress_clip * 2)
            else:
                progress_raw = max(progress_raw, -self.progress_clip)
        reward_components["progress"] = self.w_progress * progress_raw

        # 3) Height reward: Lifting reward (greater after gripping)
        height_diff = float(achieved_goal[2] - self.initial_object_height)
        if height_diff > 0:
            height_bonus = height_diff * (2.0 if self.is_gripping else 1.0)
            reward_components["height"] = self.w_height * height_bonus
        else:
            reward_components["height"] = 0.0

        # 4) Smoothness penalty
        smooth_penalty = 0.0
        ee_position = None
        ee_object_distance = None
        if obs_dict is not None:
            observation = obs_dict["observation"]
            ee_position = observation[:3]
            ee_vel_dt = observation[3:6]
            ee_velocity = ee_vel_dt / max(self.dt, 1e-8)
            self.vel_ema = (
                self.vel_ema_tau * self.vel_ema + (1.0 - self.vel_ema_tau) * ee_velocity
            )
            smooth_penalty = float(np.linalg.norm(self.vel_ema))
            ee_object_distance = float(observation[21])
        reward_components["smooth"] = self.w_smooth * smooth_penalty

        # 5) Action penalties - 减少对不同关节的惩罚差异
        if action is not None:
            action_norm = float(np.linalg.norm(action))
            reward_components["action"] = self.w_action * action_norm
            if self.prev_action is not None:
                action_diff = float(np.linalg.norm(action - self.prev_action))
                reward_components["action_change"] = self.w_action_change * action_diff
            else:
                reward_components["action_change"] = 0.0
        else:
            reward_components["action"] = 0.0
            reward_components["action_change"] = 0.0

        # 6) Success reward - 只有完整完成任务才给予
        is_success = d < self.distance_threshold and achieved_goal[2] < 0.04
        reward_components["success"] = self.success_reward if is_success else 0.0

        # 7) Gripping bonus - 分阶段奖励
        if self.is_gripping:
            if height_diff > 0.05:
                reward_components["gripping_bonus"] = 2.0
            else:
                reward_components["gripping_bonus"] = 1.0
        else:
            reward_components["gripping_bonus"] = 0.0

        # 7b) Encourage initiating and finishing suction actions
        if ee_object_distance is None:
            ee_position = self.get_ee_position() if ee_position is None else ee_position
            ee_object_distance = float(np.linalg.norm(ee_position - achieved_goal))
        if not self.is_gripping and ee_object_distance < 0.03:
            reward_components["pre_grip_bonus"] = 1.5
        else:
            reward_components["pre_grip_bonus"] = 0.0
        if self.is_gripping and d < self.distance_threshold * 3 and achieved_goal[2] > 0.04:
            reward_components["carry_bonus"] = 1.0
        else:
            reward_components["carry_bonus"] = 0.0
        if (
            not self.is_gripping
            and d < self.distance_threshold * 1.5
            and achieved_goal[2] < 0.045
        ):
            reward_components["release_bonus"] = 5.0
        else:
            reward_components["release_bonus"] = 0.0

        # 8) Placement reward - 如果物体在目标区域上方，鼓励放下
        if d < self.distance_threshold * 2 and achieved_goal[2] > 0.04:
            reward_components["placement_incentive"] = 1.5
        else:
            reward_components["placement_incentive"] = 0.0

        # 9) Trajectory tracking shaping（NEW）
        if self.use_trajectory_teacher and self._planner is not None:
            if ee_position is None:
                ee_position = self.get_ee_position()
            tgt = self._planner.current_target()
            track_err = float(np.linalg.norm(ee_position - tgt))
            target_window = 0.15
            track_term = max(target_window - track_err, 0.0)
            phase_bonus = 0.2 * float(self._planner.phase)
            reward_components["track"] = (
                self.track_reward_scale * track_term + phase_bonus
            )
        else:
            reward_components["track"] = 0.0

        total_reward = float(sum(reward_components.values()))
        info["reward_components"] = reward_components
        return total_reward

    # ---------- Actions & gripper ----------

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()

        # 1. Control arm joints
        arm_action = action[:3]
        current_joints_list = []
        for name in self.joint_names:
            qpos = self._utils.get_joint_qpos(self.model, self.data, name)
            current_joints_list.append(
                float(qpos.flat[0]) if isinstance(qpos, np.ndarray) else float(qpos)
            )
        current_joints = np.array(current_joints_list, dtype=np.float64)

        # target joints = current + delta (scaled)
        target_joints = current_joints + arm_action * self.action_scale
        target_joints = np.clip(
            target_joints, self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        )
        self.data.ctrl[:3] = target_joints

        # 2. Control suction gripper
        gripper_action = action[3]
        self._set_gripper_action(gripper_action)

    def _set_gripper_action(self, gripper_action: float) -> None:
        if gripper_action > 0:  # Activate
            if not self.is_gripping:
                if self.try_activate_suction():
                    self.is_gripping = True
                    self.grip_stabilization_steps = 0
            else:
                # 保持吸附状态
                self.data.eq_active[self.suction_weld_id] = 1
                self.grip_stabilization_steps += 1
        else:  # Deactivate
            if self.is_gripping:
                # 只有在稳定吸附一段时间后才允许释放
                if self.grip_stabilization_steps > 3:
                    self.deactivate_suction()
                    self.is_gripping = False
                    self.grip_stabilization_steps = 0

    def try_activate_suction(self) -> bool:
        ee_pos = self.get_ee_position()
        obj_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site")
        d_vec = ee_pos - obj_pos
        planar = np.linalg.norm(d_vec[:2])
        vertical = abs(d_vec[2])

        planar_threshold = 0.015
        vertical_threshold = min(self.suction_distance_threshold, 0.03)

        if planar < planar_threshold and vertical < vertical_threshold:
            obj_vel = self._utils.get_site_xvelp(self.model, self.data, "obj_site")
            if np.linalg.norm(obj_vel) < 0.5:
                self.data.eq_active[self.suction_weld_id] = 1
                return True
        return False

    def deactivate_suction(self) -> None:
        self.data.eq_active[self.suction_weld_id] = 0

    # ---------- Observations ----------

    def _get_obs(self) -> dict:
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

        # Trajectory target & phase (NEW)
        if self.use_trajectory_teacher and self._planner is not None:
            target = self._planner.current_target()
            ee_to_target = target - ee_position
            phase_oh = self._planner.phase_one_hot()
        else:
            ee_to_target = np.zeros(3, dtype=np.float64)
            phase_oh = np.zeros(6, dtype=np.float64)

        observation = np.concatenate(
            [
                ee_position,  # 3
                ee_velocity,  # 3
                joint_positions,  # 3
                object_position,  # 3
                object_velp,  # 3
                goal_rel_pos,  # 3
                object_rel_ee,  # 3
                [ee_object_distance],  # 1
                [object_goal_distance],  # 1
                [normalized_time],  # 1
                [gripper_state],  # 1
                ee_to_target,  # 3  (NEW)
                phase_oh,  # 6  (NEW)
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": object_position.copy(),
            "desired_goal": self.goal.copy(),
        }

    # ---------- Success ----------

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        # 只有在物体在目标位置且已经释放吸盘的情况下才算成功
        object_at_goal = d < self.distance_threshold
        object_on_ground = achieved_goal[2] < 0.04  # 物体接近地面
        released = not self.is_gripping
        return np.float32(object_at_goal and object_on_ground and released)

    def _render_callback(self) -> None:
        pass

    # ---------- Reset ----------

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        self.data.qpos[:] = np.copy(self.initial_qpos)

        if self.model.na != 0:
            self.data.act[:] = None

        self._sample_object()

        self.current_step = 0
        self.previous_ee_position = None
        self.previous_distance = None
        self.prev_action = None
        self.prev_achieved_goal = None
        self.vel_ema[:] = 0.0

        # Reset gripper
        self.is_gripping = False
        self.grip_stabilization_steps = 0
        self.data.eq_active[self.suction_weld_id] = 0

        self._mujoco.mj_forward(self.model, self.data)

        if self.use_trajectory_teacher:
            obj = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
            self._planner = _TrajectoryPlanner(
                obj_pos=obj,
                goal_pos=self.goal,
                hover_z=self.traj_hover_z,
                pick_z=self.traj_pick_z,
                place_z=self.traj_place_z,
                pos_tol=self.traj_pos_tol,
            )
        else:
            self._planner = None

        return True

    # ---------- Physics ----------

    def _mujoco_step(self) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # ---------- Utilities ----------

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_object(self) -> None:
        # Encourage wide joint1 usage by sampling around the base in polar coordinates
        min_dist = 0.12
        object_position = None

        if self.use_polar_object_sampling:
            for _ in range(96):
                radius = self.np_random.uniform(
                    float(self.obj_radius_range[0]), float(self.obj_radius_range[1])
                )
                angle = self.np_random.uniform(
                    float(self.obj_angle_range[0]), float(self.obj_angle_range[1])
                )
                offset = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
                candidate_xy = self.base_xy + radius * offset

                if not np.all(candidate_xy >= self.obj_bounds_low) or not np.all(
                    candidate_xy <= self.obj_bounds_high
                ):
                    continue

                candidate = np.array(
                    [candidate_xy[0], candidate_xy[1], self.obj_range_low[2]],
                    dtype=np.float64,
                )
                if np.linalg.norm(candidate - self.goal) >= min_dist:
                    object_position = candidate
                    break

        if object_position is None:
            custom_range_low = np.array([0.25, 0.1, self.obj_range_low[2]])
            custom_range_high = np.array([0.6, 0.6, self.obj_range_high[2]])
            for _ in range(64):
                candidate = self.np_random.uniform(custom_range_low, custom_range_high)
                if np.linalg.norm(candidate - self.goal) >= min_dist:
                    object_position = candidate
                    break
            if object_position is None:
                object_position = candidate

        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def _sample_goal(self) -> np.ndarray:
        return self.goal.copy()

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_site")

    def activate_suction(self) -> None:
        self.try_activate_suction()

    # ---------- IK helper for residual teacher ----------

    def _ik_step_to_target(
        self, target_ee: np.ndarray, ee_now: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        One damped least-squares IK step to move EE towards target_ee.
        Returns delta_q (in joint units) for [joint1, joint2, joint3_slide].
        """
        try:
            # pose error (position only)
            dx = np.asarray(target_ee - ee_now, dtype=np.float64)
            # limit the instantaneous target step to avoid instability
            if np.linalg.norm(dx) > self.teacher_gain:
                dx = dx * (self.teacher_gain / (np.linalg.norm(dx) + 1e-8))

            # get site jacobian: 3x nq for translation
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

            # Map jacobian columns to the configured joints to avoid hard-coded dof order assumptions.
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ]
            dof_cols = [self.model.jnt_dofadr[jid] for jid in joint_ids]
            J = jacp[:, dof_cols]

            # damped least squares: dq = J^T (J J^T + λ^2 I)^-1 dx
            JJt = J @ J.T
            lam2I = (self.teacher_lambda**2) * np.eye(3)
            inv = np.linalg.inv(JJt + lam2I)
            dq = J.T @ (inv @ dx)

            # clip to joint control range per step
            # (not strictly necessary here; _set_action will clip final qpos)
            return np.asarray(dq, dtype=np.float64)
        except Exception:
            # fall back: no teacher step
            return None
