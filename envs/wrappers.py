import warnings
from abc import ABC, abstractmethod
from typing import (Tuple, SupportsFloat)

import numpy as np
import mujoco

from envs.humanoid_light_v2.reference_motion_loader import (
    HumanoidLightReferenceMotion,
    ReferencePhaseClock,
)

try:
    from prettytable import PrettyTable
except Exception:
    PrettyTable = None


class BaseEnv(ABC):
    """
    Abstract base class for an environment wrapper for the Sim2Sim framework.
    Defines the standard interface that all environment implementations must follow.
    """

    def __init__(self):
        """
        Initializes the environment.
        This constructor does not perform any specific initialization and should be
        overridden by subclasses if necessary.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        :return:
            - observation (np.ndarray): The initial observation after resetting the environment.
            - info (dict): Additional information about the environment's state, which may
              include metadata or debugging information.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Executes a step in the environment given an action.

        :param action:
            - np.ndarray: The action to be taken by the agent.

        :return:
            - observation (np.ndarray): The new observation after executing the action.
            - terminated (bool): Whether the episode has ended due to reaching a terminal state.
            - truncated (bool): Whether the episode was truncated due to external constraints
              such as time limits.
            - info (dict): Additional information about the environment’s state, which may
              include diagnostic data or auxiliary variables.
        """
        pass

    @abstractmethod
    def event(self, event: str, value):
        """
        Triggers an event.

        :param event: Name of the event (e.g., "push").
        :param value: Associated value to be passed with the event (e.g., velocity vector).
        """
        pass

    @abstractmethod
    def get_data(self):
        """Retrieve low-level environment data from the wrapped environment."""
        return self.env.get_data()

    @abstractmethod
    def render(self):
        """
        Renders the environment.

        Provides a visual representation of the environment, such as a GUI window
        or textual output. This is useful for debugging and analysis.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the environment and releases resources.

        Ensures proper cleanup, such as closing windows or stopping background processes.
        Should be called when the environment is no longer needed.
        """
        pass


class StateBuildWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.settings_cfg = self.config.get("settings", self.config.get("observation", {}))
        self.quiet = bool(self.config.get("env", {}).get("quiet", False))
        self.last_obs = None
        self.id = env.id
        self.action_dim = env.action_dim
        self.sim_step = 0
        self.reset_flag = False

        # Require env.control_freq (Hz); fail fast if missing/invalid
        if not hasattr(self.env, "control_freq"):
            raise AttributeError(f"Env: {self.id} must define 'control_freq' (Hz).")
        self.control_freq = float(self.env.control_freq)
        if self.control_freq <= 0:
            raise ValueError(f"Invalid env.control_freq: {self.control_freq}. Must be > 0.")

        # Number of frames to stack (i=0 is the most recent frame)
        self.stack_size = int(self.settings_cfg["stack_size"])
        # Ordered observation keys that will be stacked
        self.stacked_obs_order = list(self.settings_cfg["stacked_obs_order"])
        # Ordered observation keys that will NOT be stacked (single-frame)
        self.non_stacked_obs_order = list(self.settings_cfg["non_stacked_obs_order"])

        # ``reference_progress`` is a synthetic one-dimensional observation.
        # It intentionally lives in the regular observation builder (rather
        # than CommandWrapper) so reference progress can be added through the
        # same observation settings UI as ordinary locomotion inputs.
        self.obs_to_dim = dict(self.env.obs_to_dim)
        has_reference_progress = "reference_progress" in (
            self.stacked_obs_order + self.non_stacked_obs_order
        )
        self._reference_phase_clock = None
        if has_reference_progress:
            if "reference_progress" in self.stacked_obs_order:
                raise ValueError("'reference_progress' is available only as a Non-Stacked Observation.")
            if self.non_stacked_obs_order.count("reference_progress") != 1:
                raise ValueError("Add 'reference_progress' exactly once to Non-Stacked Observation.")
            progress_cfg = self.settings_cfg.get("reference_progress") or {}
            if int(progress_cfg.get("freq", 0)) != int(round(self.control_freq)) or not np.isclose(
                float(progress_cfg.get("scale", float("nan"))), 1.0
            ):
                raise ValueError(
                    "'reference_progress' must use the control frequency and scale=1.0 "
                    f"(expected {int(round(self.control_freq))} Hz)."
                )
            phase_source = str(self.settings_cfg.get("reference_progress_source", "")).strip()
            if not phase_source:
                raise ValueError(
                    "Set Reference Progress Source (.npz) in Observation Settings when using 'reference_progress'."
                )
            self._reference_phase_clock = ReferencePhaseClock(phase_source)
            self.obs_to_dim["reference_progress"] = 1

        # Cache dimensions
        self._stacked_obs_dim = sum(self.obs_to_dim[n] for n in self.stacked_obs_order)
        self._non_stacked_obs_dim = sum(self.obs_to_dim[n] for n in self.non_stacked_obs_order)
        self.state_dim = self.stack_size * self._stacked_obs_dim + self._non_stacked_obs_dim
  
        # Rolling buffer for stacked observations (shape: [stack_size, stacked_obs_dim])
        self.obs_buffer = np.zeros((self.stack_size, self._stacked_obs_dim), dtype=np.float32)

        # Cache for frequency/scale-applied observation values
        self._freq_cache = {}
        self.pretty_print_interval = max(1, int(round(self.control_freq / 2.0)))

    @staticmethod
    def _quaternion_to_euler_array(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z], dtype=np.float64)

    def _get_joint_name_lists(self):
        pos_joint_names = []
        vel_joint_names = []

        if hasattr(self.env, "joint_names_in_order"):
            pos_joint_names = list(self.env.joint_names_in_order)
        elif hasattr(self.env, "initial_joint_names"):
            vel_joint_names = list(self.env.initial_joint_names)
            qpos_dim = len(getattr(self.env, "q_indices", []))
            pos_joint_names = vel_joint_names[:qpos_dim]

        if not vel_joint_names:
            vel_joint_names = list(pos_joint_names)

        return pos_joint_names, vel_joint_names

    @staticmethod
    def _format_vector(values: np.ndarray) -> str:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(f"{v: .4f}" for v in arr) + "]"

    @staticmethod
    def _quat_wxyz_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
        if quat_wxyz.size == 4:
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        else:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if np.allclose(quat_xyzw, 0.0):
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return StateBuildWrapper._quaternion_to_euler_array(quat_xyzw)

    @staticmethod
    def _build_text_table(title: str, headers, rows) -> str:
        lines = [title, " | ".join(headers)]
        lines.append("-" * max(len(lines[1]), len(title)))
        for row in rows:
            lines.append(" | ".join(str(col) for col in row))
        return "\n".join(lines)

    def _get_robot_com_offset_in_base_frame(self):
        model = getattr(self.env, "model", None)
        data = self.env.get_data()
        if model is None or data is None:
            return None

        base_body_name = None
        for candidate in ("base_link", "pelvis_link"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id != -1:
                base_body_name = candidate
                break

        if base_body_name is None:
            return None

        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if base_body_id == -1:
            return None

    def _get_action_values(self) -> np.ndarray:
        action = getattr(self.env, "filtered_action", None)
        if action is None:
            action = getattr(self.env, "action", None)
        if action is None:
            return np.zeros((0,), dtype=np.float64)
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        scaler = getattr(self.env, "action_scaler", None)
        clip_min = getattr(self.env, "action_clip_min", None)
        clip_max = getattr(self.env, "action_clip_max", None)
        if scaler is None or clip_min is None or clip_max is None:
            return action

        scaler = np.asarray(scaler, dtype=np.float64).reshape(-1)
        clip_min = np.asarray(clip_min, dtype=np.float64).reshape(-1)
        clip_max = np.asarray(clip_max, dtype=np.float64).reshape(-1)
        if len(action) != len(scaler) or len(action) != len(clip_min) or len(action) != len(clip_max):
            return action
        return np.clip(action * scaler, clip_min, clip_max)

    def _sensor_vector_or_none(self, sensor_name: str):
        try:
            return np.asarray(self.env.get_data().sensor(sensor_name).data, dtype=np.float64).reshape(-1)
        except Exception:
            return None

    def _build_imu_rows(self, obs: dict, prefix: str):
        gyro = np.asarray(obs.get(f"{prefix}_ang_vel", []), dtype=np.float64).reshape(-1)
        projected_gravity = np.asarray(obs.get(f"{prefix}_projected_gravity", []), dtype=np.float64).reshape(-1)
        quat_wxyz = self._sensor_vector_or_none(f"{prefix}_imu_orientation")
        if quat_wxyz is None:
            return None
        euler = self._quat_wxyz_to_euler(quat_wxyz)
        return [
            ["euler angle [roll, pitch, yaw]", self._format_vector(euler)],
            ["gyro [x, y, z]", self._format_vector(gyro)],
            ["projected gravity [x, y, z]", self._format_vector(projected_gravity)],
        ]

    def _print_pretty_observation(self, obs: dict, phase: str):
        data = self.env.get_data()
        pos_joint_names, vel_joint_names = self._get_joint_name_lists()

        dof_pos = np.asarray(obs.get("dof_pos", []), dtype=np.float64).reshape(-1)
        dof_vel = np.asarray(obs.get("dof_vel", []), dtype=np.float64).reshape(-1)
        action_vals = self._get_action_values()
        gyro = np.asarray(obs.get("ang_vel", []), dtype=np.float64).reshape(-1)
        projected_gravity = np.asarray(obs.get("projected_gravity", []), dtype=np.float64).reshape(-1)
        base_height = float(data.qpos[2]) if getattr(data, "qpos", None) is not None and len(data.qpos) > 2 else float("nan")

        try:
            quat_wxyz = np.asarray(data.sensor("orientation").data, dtype=np.float64).reshape(-1)
        except Exception:
            quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64).reshape(-1)

        euler = self._quat_wxyz_to_euler(quat_wxyz)

        joint_rows = []

        max_rows = max(len(vel_joint_names), len(pos_joint_names), len(dof_pos), len(dof_vel), len(action_vals))
        for idx in range(max_rows):
            if idx < len(vel_joint_names):
                joint_name = vel_joint_names[idx]
            elif idx < len(pos_joint_names):
                joint_name = pos_joint_names[idx]
            else:
                joint_name = f"joint_{idx}"

            pos_val = f"{dof_pos[idx]: .6f}" if idx < len(dof_pos) else "-"
            vel_val = f"{dof_vel[idx]: .6f}" if idx < len(dof_vel) else "-"
            action_val = f"{action_vals[idx]: .6f}" if idx < len(action_vals) else "-"
            joint_rows.append([joint_name, pos_val, vel_val, action_val])

        imu_rows = [
            ["euler angle [roll, pitch, yaw]", self._format_vector(euler)],
            ["gyro [x, y, z]", self._format_vector(gyro)],
            ["projected gravity [x, y, z]", self._format_vector(projected_gravity)],
            ["base height", f"{base_height: .6f}"],
        ]
        lower_imu_rows = self._build_imu_rows(obs, "lower") if self.id.startswith("humanoid_light") else None
        upper_imu_rows = self._build_imu_rows(obs, "upper") if self.id.startswith("humanoid_light") else None

        if PrettyTable is not None:
            joint_table = PrettyTable()
            joint_table.title = f"{self.id} {phase} Joint States"
            joint_table.field_names = ["joint", "joint pos", "joint vel", "action"]
            joint_table.align["joint"] = "l"
            joint_table.align["joint pos"] = "r"
            joint_table.align["joint vel"] = "r"
            joint_table.align["action"] = "r"
            for row in joint_rows:
                joint_table.add_row(row)

            imu_table = PrettyTable()
            imu_table.title = f"{self.id} {phase} Base State"
            imu_table.field_names = ["signal", "value"]
            imu_table.align["signal"] = "l"
            imu_table.align["value"] = "l"
            for row in imu_rows:
                imu_table.add_row(row)

            print(joint_table)
            print(imu_table)
            for title, rows in (
                (f"{self.id} {phase} Lower IMU State", lower_imu_rows),
                (f"{self.id} {phase} Upper IMU State", upper_imu_rows),
            ):
                if rows is None:
                    continue
                imu_link_table = PrettyTable()
                imu_link_table.title = title
                imu_link_table.field_names = ["signal", "value"]
                imu_link_table.align["signal"] = "l"
                imu_link_table.align["value"] = "l"
                for row in rows:
                    imu_link_table.add_row(row)
                print(imu_link_table)
        else:
            print(self._build_text_table(f"{self.id} {phase} Joint States", ["joint", "joint pos", "joint vel", "action"], joint_rows))
            print(self._build_text_table(f"{self.id} {phase} Base State", ["signal", "value"], imu_rows))
            if lower_imu_rows is not None:
                print(self._build_text_table(f"{self.id} {phase} Lower IMU State", ["signal", "value"], lower_imu_rows))
            if upper_imu_rows is not None:
                print(self._build_text_table(f"{self.id} {phase} Upper IMU State", ["signal", "value"], upper_imu_rows))

    def _concat_obs_with_freq(self, obs, names):
        """
        Concatenate observations following frequency (Hz) and scale rules defined in
        config["observation"][<name>]. For each name:
          - If sim_step == 0: always refresh with the latest 'obs' value.
          - Else: refresh only when (control_freq / freq) step interval elapses; otherwise keep cached value.
          - Always apply 'scale' by multiplication.

        Args:
            obs (dict): {name: np.ndarray}-like observation dictionary from the env.
            names (list[str]): keys to fetch/concatenate in order.

        Returns:
            np.ndarray (float32): concatenated 1D vector with freq/scale applied.
        """
        parts = []
        for n in names:
            n_cfg = self.settings_cfg[n]
            update_freq = float(n_cfg["freq"])
            scale = float(n_cfg["scale"])

            if update_freq <= 0:
                raise ValueError(f"Invalid observation update frequency for '{n}': {update_freq}. Must be > 0.")

            # Steps between updates; at least 1 to avoid division artifacts
            update_interval = max(1, int(round(self.control_freq / update_freq)))
            need_update = (self.sim_step == 0) or (self.sim_step % update_interval == 0)

            if need_update or (n not in self._freq_cache):
                if n == "reference_progress":
                    val = self._reference_phase_clock.trajectory_progress(
                        self.sim_step, 1.0 / self.control_freq
                    ) * scale
                elif n in obs and obs[n] is not None:
                    val = np.asarray(obs[n], dtype=np.float32) * scale
                else:
                    val = np.zeros((int(self.obs_to_dim.get(n, 0)),), dtype=np.float32)
                self._freq_cache[n] = val

            parts.append(self._freq_cache[n].ravel().astype(np.float32))

        if parts:
            return np.concatenate(parts, axis=0)
        else:
            return np.zeros((0,), dtype=np.float32)

    def _push_stack(self, latest_vec, reset=False):
        """
        Push a new stacked frame into the rolling buffer.

        - If reset=True: fill the whole buffer with 'latest_vec'.
        - If reset=False: shift older frames down by one and put 'latest_vec' at index 0.
          Index 0 is the most recent; index (stack_size - 1) is the oldest.
        """
        if reset:
            self.obs_buffer[:] = latest_vec
        else:
            if self.stack_size > 1:
                self.obs_buffer[1:, :] = self.obs_buffer[:-1, :]
            self.obs_buffer[0, :] = latest_vec

    def _build_state(self, obs, reset: bool):
        """
        Build the final 1D state vector consisting of:
        - Flattened stacked observations (from the rolling buffer).
        - Concatenated non-stacked observations (also subject to freq/scale).

        Args:
            obs (dict): environment observation dict.
            reset (bool): if True, re-initialize the stack with the current observation.

        Returns:
            np.ndarray (float32): state vector of length 'state_dim'.
        """
        # 1) Gather stacked observations (with frequency/scale rules)
        obs_for_stack = self._concat_obs_with_freq(obs, self.stacked_obs_order)

        # 2) Update the rolling buffer
        self._push_stack(obs_for_stack, reset=reset)

        # 3) Flatten stacked frames and append non-stacked observations (with freq/scale)
        stacked_flat = self.obs_buffer.ravel()  # shape: (stack_size * stacked_obs_dim,)
        non_stacked_vec = self._concat_obs_with_freq(obs, self.non_stacked_obs_order)

        state = np.concatenate([stacked_flat, non_stacked_vec], axis=0)
        return state.astype(np.float32)

    def reset(self):
        """
        Reset the underlying environment and reinitialize internal counters and caches.
        Fills the stack with the initial observation.
        """
        self.reset_flag = True
        self.sim_step = 0
        self._freq_cache.clear()
        init_obs, info = self.env.reset()
        self.last_obs = init_obs
        if not self.quiet:
            self._print_pretty_observation(init_obs, phase="reset")
        init_state = self._build_state(init_obs, reset=True)
        return init_state, info

    def step(self, action: np.ndarray):
        """
        Step through the environment and build the next state.
        """
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        self.sim_step += 1
        next_obs, terminated, truncated, info = self.env.step(action)
        self.last_obs = next_obs
        if (not self.quiet) and ((self.sim_step % self.pretty_print_interval == 0) or terminated or truncated):
            self._print_pretty_observation(next_obs, phase=f"step {self.sim_step}")
        next_state = self._build_state(next_obs, reset=False)

        if terminated or truncated:
            self.reset_flag = False
        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        """Forward custom events to the wrapped environment."""
        return self.env.event(event, value)

    def get_last_obs(self):
        return self.last_obs

    def get_data(self):
        """Proxy for any data export the wrapped environment supports."""
        return self.env.get_data()

    def render(self):
        """Render via the wrapped environment."""
        self.env.render()

    def close(self):
        """Close the wrapped environment."""
        self.env.close()



class TimeLimitWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.id = env.id
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.sim_step = 0
        self.max_sim_step = int(config["env"]["max_duration"] * self.env.control_freq)
        self.reset_flag = False

    def reset(self):
        self.reset_flag = True
        self.sim_step = 0
        init_state, info = self.env.reset()
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        self.sim_step += 1
        next_state, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.reset_flag = False

        if self.sim_step == self.max_sim_step:
            truncated = True
            self.reset_flag = False

        return next_state, terminated, truncated, info
    
    def event(self, event: str, value):
        return self.env.event(event, value)

    def get_data(self):
        return self.env.get_data()

    def get_last_obs(self):
        if hasattr(self.env, "get_last_obs"):
            return self.env.get_last_obs()
        return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class ReferenceMotionResetWrapper(BaseEnv):
    """Preserve the configured physical reset pose for reference inference.

    This wrapper intentionally sits directly above ``HumanoidLightV2`` so the
    first StateBuildWrapper frame reflects the exact Initial Pose Settings.
    Reference q/dq and root targets remain part of the policy observation, but
    never teleport the simulated robot away from its configured start state.
    """

    def __init__(self, env, motion: HumanoidLightReferenceMotion, config):
        super().__init__()
        self.env = env
        self.motion = motion
        self.config = config
        self.id = env.id
        self.action_dim = env.action_dim
        self.control_freq = env.control_freq
        self.obs_to_dim = env.obs_to_dim
        self.joint_names_in_order = list(env.joint_names_in_order)
        self._q_indices = np.asarray(env.q_indices, dtype=np.int64)
        self._joint_ranges = self._controlled_joint_ranges()
        # The Isaac asset used for training declares
        # ``soft_joint_pos_limit_factor=0.9``.  Its reference-reset event
        # clamps into those soft limits, rather than the raw hard limits.
        range_center = 0.5 * (self._joint_ranges[:, 0] + self._joint_ranges[:, 1])
        range_half_width = 0.45 * (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        self._soft_joint_ranges = np.stack((range_center - range_half_width, range_center + range_half_width), axis=-1)

    def _controlled_joint_ranges(self):
        ranges = np.empty((self.action_dim, 2), dtype=np.float64)
        for index, name in enumerate(self.env.joint_names_in_order):
            joint_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Controlled joint not found in MuJoCo model: {name}")
            ranges[index] = self.env.model.jnt_range[joint_id]
        return ranges

    def _reset_noise(self):
        ref_cfg = self.config.get("reference_motion", {}) or {}
        if not bool(ref_cfg.get("reset_perturbation", False)):
            return np.zeros(2, dtype=np.float64), np.zeros(self.action_dim, dtype=np.float64)
        root_xy = np.random.uniform(-0.02, 0.02, size=2)
        joint_pos = np.random.uniform(-0.02, 0.02, size=self.action_dim)
        return root_xy, joint_pos

    def reset(self):
        # The wrapped leaf reset clears action delay/filter state and all MuJoCo
        # derived buffers, then applies Initial Pose Settings. Keep that full
        # qpos/qvel state: a real robot cannot be teleported to reference frame
        # zero before executing its first policy action.
        _, _ = self.env.reset()
        root_xy_noise, joint_noise = self._reset_noise()

        data = self.env.get_data()
        initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        initial_qvel = np.asarray(data.qvel, dtype=np.float64).copy()

        # The opt-in reference reset perturbation remains useful for stress
        # tests. With it off (the GUI default), qpos and qvel are bit-for-bit
        # the values configured in Initial Pose Settings.
        data.qpos[0:2] = root_xy_noise
        data.qpos[2:] = initial_qpos[2:]
        if np.any(joint_noise):
            data.qpos[self._q_indices] = np.clip(
                initial_qpos[self._q_indices] + joint_noise,
                self._soft_joint_ranges[:, 0],
                self._soft_joint_ranges[:, 1],
            )
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.env.model, data)

        return self.env._get_obs(), self.env._get_reset_info()

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def event(self, event: str, value):
        return self.env.event(event, value)

    def get_data(self):
        return self.env.get_data()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class ReferenceMotionTargetWrapper(BaseEnv):
    """Append Isaac's 182-D non-stack reference target to a 94-D state."""

    def __init__(self, env, motion: HumanoidLightReferenceMotion, control_freq: float, config):
        super().__init__()
        self.env = env
        self.motion = motion
        self.config = config
        self.id = env.id
        self.action_dim = env.action_dim
        self.control_freq = float(control_freq)
        self.control_dt = 1.0 / self.control_freq
        self.state_dim = env.state_dim + 182
        self.control_step = 0
        self.reset_flag = False
        self._fall_height = float((config.get("reference_motion", {}) or {}).get("fall_height", 0.35))

    def _state_with_reference(self, state):
        target = self.motion.policy_targets(self.control_step, self.control_dt)
        combined = np.concatenate((np.asarray(state, dtype=np.float32), target), dtype=np.float32)
        if combined.shape != (self.state_dim,):
            raise RuntimeError(
                f"Humanoid Light reference observation mismatch: got {combined.shape[0]}, expected {self.state_dim}."
            )
        return combined

    def _zero_command(self):
        if hasattr(self.env, "receive_user_command"):
            self.env.receive_user_command(np.zeros(4, dtype=np.float64))

    def _add_tracking_info(self, info):
        ref = self.motion.frame(self.motion.frame_id(self.control_step, self.control_dt))
        data = self.get_data()
        leaf = self._leaf_env()
        actual_dof = data.qpos[np.asarray(leaf.q_indices, dtype=np.int64)]
        actual_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        quat_dot = float(np.clip(abs(np.dot(actual_quat, ref["root_quat_wxyz"])), 0.0, 1.0))
        root_orientation_error = float(2.0 * np.arccos(quat_dot))
        root_vel = np.asarray(data.qvel[0:3], dtype=np.float64)

        info["reference_frame"] = int(ref["frame_id"])
        info["reference_joint_pos_rmse"] = float(np.sqrt(np.mean(np.square(actual_dof - ref["dof_pos"]))))
        info["reference_root_height_error"] = float(data.qpos[2] - ref["root_pos"][2])
        info["reference_root_orientation_error_rad"] = root_orientation_error
        info["reference_root_lin_vel_rmse"] = float(np.sqrt(np.mean(np.square(root_vel - ref["root_lin_vel"]))))
        info["reference_stable"] = bool(np.isfinite(data.qpos).all() and data.qpos[2] >= self._fall_height)
        return info

    def _leaf_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def reset(self):
        self.control_step = 0
        self.reset_flag = True
        self._zero_command()
        state, info = self.env.reset()
        info = self._add_tracking_info(dict(info))
        return self._state_with_reference(state), info

    def step(self, action: np.ndarray):
        assert self.reset_flag, "Call 'reset()' before calling 'step()'."
        self._zero_command()
        state, terminated, truncated, info = self.env.step(action)
        self.control_step += 1
        info = self._add_tracking_info(dict(info))
        # Unlike the generic humanoid environment, reference inference needs a
        # meaningful early stop when the free base has fallen below the floor.
        if not info["reference_stable"]:
            terminated = True
        if terminated or truncated:
            self.reset_flag = False
        return self._state_with_reference(state), terminated, truncated, info

    def receive_user_command(self, _user_command):
        # Reference PPO retained the four command slots for compatibility, but
        # trained with all of them fixed to zero.
        self._zero_command()

    def event(self, event: str, value):
        return self.env.event(event, value)

    def get_data(self):
        return self.env.get_data()

    def get_last_obs(self):
        if hasattr(self.env, "get_last_obs"):
            return self.env.get_last_obs()
        return None

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class ReferenceMotionProgressWrapper(ReferenceMotionTargetWrapper):
    """Track a reference clip while progress is built as a normal observation.

    ``StateBuildWrapper`` owns the synthetic ``reference_progress`` item.  In
    particular, the distilled student uses CommandWrapper with command_dim=0:
    its contract is the standard 90-D locomotion observation followed by this
    one progress coordinate (91-D total).
    """

    def __init__(self, env, motion: HumanoidLightReferenceMotion, control_freq: float, config):
        super().__init__(env, motion, control_freq, config)
        self.state_dim = env.state_dim
        if self.state_dim != 91:
            raise RuntimeError(
                "Distilled reference trajectory requires the 90-D locomotion observation plus "
                f"the final 1-D Reference Progress observation (91-D total); got {self.state_dim}-D."
            )
        settings_cfg = config.get("settings", config.get("observation", {})) or {}
        stacked = list(settings_cfg.get("stacked_obs_order", []) or [])
        non_stacked = list(settings_cfg.get("non_stacked_obs_order", []) or [])
        progress_cfg = settings_cfg.get("reference_progress") or {}
        if (
            int(settings_cfg.get("command_dim", -1)) != 0
            or
            "reference_progress" in stacked
            or non_stacked.count("reference_progress") != 1
            or not non_stacked
            or non_stacked[-1] != "reference_progress"
            or int(progress_cfg.get("freq", 0)) != int(round(self.control_freq))
            or not np.isclose(float(progress_cfg.get("scale", float("nan"))), 1.0)
        ):
            raise RuntimeError(
                "Distilled reference trajectory requires 'reference_progress' exactly once in "
                f"Non-Stacked Observation as its final item, Command Dim=0, and "
                f"freq={int(round(self.control_freq))} with scale=1.0."
            )
        # Keep the student-test terminal log concise enough to show whether
        # simulation continues after the clip, without printing all 90-D
        # proprioception every control tick.
        self._clip_control_steps = max(1, int(np.floor(self.motion.duration_s / self.control_dt + 1.0e-6)))
        self._command_log_interval_steps = max(1, int(round(self.control_freq / 2.0)))

    def _condition(self, step: int) -> np.ndarray:
        return self.motion.trajectory_progress(step, self.control_dt)

    def _is_post_clip(self, step: int) -> bool:
        return int(step) >= self._clip_control_steps

    def _add_progress_info(self, info: dict, condition: np.ndarray) -> dict:
        info["reference_progress"] = float(np.asarray(condition, dtype=np.float32).reshape(-1)[0])
        info["reference_control_step"] = int(self.control_step)
        info["reference_clip_control_steps"] = int(self._clip_control_steps)
        info["reference_post_clip"] = self._is_post_clip(self.control_step)
        return info

    def _log_progress(self, condition: np.ndarray, force: bool = False, end_reason: str | None = None):
        step = int(self.control_step)
        enters_post_clip = step == self._clip_control_steps
        if not force and step % self._command_log_interval_steps != 0 and not enters_post_clip:
            return
        phase = "post-clip (last condition held)" if self._is_post_clip(step) else "clip"
        progress = float(np.asarray(condition, dtype=np.float64).reshape(-1)[0])
        print(
            f"[reference-student] step={step}/{self._clip_control_steps} "
            f"t={step * self.control_dt:.2f}s phase={phase} "
            f"progress={progress:.4f}"
            + (f" end={end_reason}" if end_reason else "")
        )

    def reset(self):
        self.control_step = 0
        self.reset_flag = True
        state, info = self.env.reset()
        condition = self._condition(self.control_step)
        info = self._add_tracking_info(dict(info))
        info = self._add_progress_info(info, condition)
        self._log_progress(condition, force=True)
        return state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag, "Call 'reset()' before calling 'step()'."
        # StateBuildWrapper has already emitted the next reference-progress
        # value for this state; advance our clock for logging/tracking only.
        state, terminated, truncated, info = self.env.step(action)
        self.control_step += 1
        condition = self._condition(self.control_step)
        info = self._add_tracking_info(dict(info))
        if not info["reference_stable"]:
            terminated = True
        info = self._add_progress_info(info, condition)
        end_reason = "fall" if not info["reference_stable"] else ("time-limit" if truncated else "terminated")
        self._log_progress(condition, force=terminated or truncated, end_reason=end_reason if (terminated or truncated) else None)
        if terminated or truncated:
            self.reset_flag = False
        return state, terminated, truncated, info

    def receive_user_command(self, user_command):
        """Forward normal command events; command_dim=0 makes them a no-op."""
        if hasattr(self.env, "receive_user_command"):
            self.env.receive_user_command(user_command)


class CommandWrapper(BaseEnv):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.settings_cfg = self.config.get("settings", self.config.get("observation", {}))
        self.id = env.id
        self.action_dim = env.action_dim
        self.command_dim = int(self.settings_cfg["command_dim"])
        if self.command_dim < 0:
            raise ValueError(f"command_dim must be >= 0, got {self.command_dim}.")
        self.state_dim = env.state_dim + self.command_dim
        self.user_command = np.zeros(self.command_dim)
        self.applied_command = np.zeros(self.command_dim)
        self.reset_flag = False       

    def receive_user_command(self, user_command):
        if self.command_dim == 0:
            return
        user_command = np.asarray(user_command, dtype=np.float64).reshape(-1)
        self.user_command = user_command[:self.command_dim]
        self.applied_command[:] = self.user_command

        if self.config["env"]["position_command"] is False:
            for i in range(self.command_dim):
                self.applied_command[i] *= self.settings_cfg["command_scales"][str(i)]        
        else:
            assert self.command_dim == 2, f"Currently, position command only support 2 dimenstion, but got {self.command_dim}."
            warnings.warn("For position commands, 'command_scales' is always treated as 1.0.")

            data = self.get_data()
            robot_px, robot_py = data.qpos[0], data.qpos[1]
            target_x, target_y = self.user_command[0], self.user_command[1]

            delta_world_x = target_x - robot_px
            delta_world_y = target_y - robot_py

            w, x, y, z = data.qpos[3:7].astype(np.float64)
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cosy, siny = np.cos(-yaw), np.sin(-yaw)

            robot_x = cosy * delta_world_x - siny * delta_world_y
            robot_y = siny * delta_world_x + cosy * delta_world_y

            self.applied_command[0] = robot_x
            self.applied_command[1] = robot_y

    def reset(self):
        self.reset_flag = True
        init_state, info = self.env.reset()
        init_state = np.concatenate((init_state, self.applied_command))
        return init_state, info

    def step(self, action: np.ndarray):
        assert self.reset_flag is True, "Call 'reset()' before calling 'step()'."
        next_state, terminated, truncated, info = self.env.step(action)
        next_state = np.concatenate((next_state, self.applied_command))

        if self.command_dim >= 1:
            info["user_command_0"] = self.user_command[0]
        if self.command_dim >= 2:
            info["user_command_1"] = self.user_command[1]
        if self.command_dim >= 3:
            info["user_command_2"] = self.user_command[2]

        if terminated or truncated:
            self.reset_flag = False

        return next_state, terminated, truncated, info

    def event(self, event: str, value):
        return self.env.event(event, value)
    
    def get_data(self):
        return self.env.get_data()

    def get_last_obs(self):
        if hasattr(self.env, "get_last_obs"):
            return self.env.get_last_obs()
        return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
