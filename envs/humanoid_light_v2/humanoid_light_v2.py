from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw

from envs.humanoid_light_v2.manager.control_manager import ControlManager
from envs.humanoid_light_v2.manager.xml_manager import XMLManager
from envs.humanoid_light_v2.utils.math_utils import MathUtils
from envs.humanoid_light_v2.utils.mujoco_utils import MuJoCoUtils
from envs.humanoid_light_v2.utils.noise_generator_utils import (
    truncated_gaussian_noisy_data,
)
from envs.initial_pose import build_initial_qpos
from envs.action_utils import normalize_action_clippings, scale_and_clip_action
from envs.actuator_mode_utils import simulate_dynamic_ctrl


def _config_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


class HumanoidLightV2(MujocoEnv, utils.EzPickle):
    """
    Updated for URDF-v9 style joints:
      - torso is 3DoF: torso_yaw_joint, torso_pitch_joint, torso_roll_joint
      - elbow is 1DoF per arm: *_elbow_joint  (no elbow_pitch/yaw split)
      - wrist is 1DoF per arm: *_wrist_joint
      - head_joint included
    Action / dof_pos / dof_vel order (26) follows the policy joint order.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config, render_flag=True, render_mode="human"):
        # --- Basic properties ---
        self.id = "humanoid_light_v2"
        self.config = config
        self.render_mode = render_mode
        self.render_flag = render_flag

        self.action_dim = int(config["hardware"]["action_dim"])
        default_action_scales = np.ones(self.action_dim, dtype=np.float64) * 0.5
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple, np.ndarray)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)
        self.action_clip_min, self.action_clip_max = normalize_action_clippings(config, self.action_dim)

        # --- PD params (support both new keys and old legacy keys) ---
        hw = config["hardware"]

        # New recommended keys
        self.kp_hip_pitch = hw.get("Kp_hip_pitch", 100)
        self.kp_hip_roll = hw.get("Kp_hip_roll", 100)
        self.kp_hip_yaw = hw.get("Kp_hip_yaw", 100)
        self.kp_knee = hw.get("Kp_knee", 100)
        self.kp_ankle_pitch = hw.get("Kp_ankle_pitch", 20)
        self.kp_ankle_roll = hw.get("Kp_ankle_roll", 20)

        self.kp_torso_yaw = hw.get("Kp_torso_yaw", hw.get("Kp_torso", 100))
        self.kp_torso_pitch = hw.get("Kp_torso_pitch", hw.get("Kp_torso", 100))
        self.kp_torso_roll = hw.get("Kp_torso_roll", hw.get("Kp_torso", 100))
        self.kp_head = hw.get("Kp_head", 20)

        self.kp_shoulder_pitch = hw.get("Kp_shoulder_pitch", 100)
        self.kp_shoulder_roll = hw.get("Kp_shoulder_roll", 100)
        self.kp_shoulder_yaw = hw.get("Kp_shoulder_yaw", 25)

        # Elbow/wrist: new keys
        self.kp_elbow = hw.get("Kp_elbow", None)
        self.kp_wrist = hw.get("Kp_wrist", None)

        # Legacy fallback (old code had elbow_pitch/elbow_yaw)
        if self.kp_elbow is None:
            # If legacy exists, pick a reasonable representative
            self.kp_elbow = hw.get("Kp_elbow_pitch", hw.get("Kp_elbow_yaw", 50))
        if self.kp_wrist is None:
            self.kp_wrist = hw.get("Kp_wrist", 50)

        self.kd_hip_pitch = hw.get("Kd_hip_pitch", 1.0)
        self.kd_hip_roll = hw.get("Kd_hip_roll", 1.0)
        self.kd_hip_yaw = hw.get("Kd_hip_yaw", 1.0)
        self.kd_knee = hw.get("Kd_knee", 1.0)
        self.kd_ankle_pitch = hw.get("Kd_ankle_pitch", 0.25)
        self.kd_ankle_roll = hw.get("Kd_ankle_roll", 0.25)

        self.kd_torso_yaw = hw.get("Kd_torso_yaw", hw.get("Kd_torso", 1.0))
        self.kd_torso_pitch = hw.get("Kd_torso_pitch", hw.get("Kd_torso", 1.0))
        self.kd_torso_roll = hw.get("Kd_torso_roll", hw.get("Kd_torso", 1.0))
        self.kd_head = hw.get("Kd_head", 0.25)

        self.kd_shoulder_pitch = hw.get("Kd_shoulder_pitch", 1.0)
        self.kd_shoulder_roll = hw.get("Kd_shoulder_roll", 1.0)
        self.kd_shoulder_yaw = hw.get("Kd_shoulder_yaw", 1.0)

        self.kd_elbow = hw.get("Kd_elbow", None)
        self.kd_wrist = hw.get("Kd_wrist", None)

        if self.kd_elbow is None:
            self.kd_elbow = hw.get("Kd_elbow_pitch", hw.get("Kd_elbow_yaw", 1.0))
        if self.kd_wrist is None:
            self.kd_wrist = hw.get("Kd_wrist", 1.0)

        # --- Torque limits (support both new + legacy) ---
        # Legs
        self.max_hip_pitch = float(hw.get("hip_pitch_joint_max_torque", 120))
        self.max_hip_roll = float(hw.get("hip_roll_joint_max_torque", 60))
        self.max_hip_yaw = float(hw.get("hip_yaw_joint_max_torque", 60))
        self.max_knee = float(hw.get("knee_joint_max_torque", 120))
        self.max_ankle_pitch = float(hw.get("ankle_pitch_joint_max_torque", 14))
        self.max_ankle_roll = float(hw.get("ankle_roll_joint_max_torque", 14))

        # Arms
        self.max_shoulder_pitch = float(hw.get("shoulder_pitch_joint_max_torque", 60))
        self.max_shoulder_roll = float(hw.get("shoulder_roll_joint_max_torque", 60))
        self.max_shoulder_yaw = float(hw.get("shoulder_yaw_joint_max_torque", 17))

        # New elbow/wrist limits (fallback to legacy elbow_pitch/yaw if present)
        if "elbow_joint_max_torque" in hw:
            self.max_elbow = float(hw["elbow_joint_max_torque"])
        else:
            # legacy: take max of pitch/yaw limits if both exist, else one that exists
            self.max_elbow = float(
                max(
                    float(hw.get("elbow_pitch_joint_max_torque", 36)),
                    float(hw.get("elbow_yaw_joint_max_torque", 36)),
                )
            )
        self.max_wrist = float(hw.get("wrist_joint_max_torque", 14))

        # Torso: if you have per-axis limits, use them. Else fallback to torso_joint_max_torque.
        torso_fallback = float(hw.get("torso_joint_max_torque", 60))
        self.max_torso_yaw = float(hw.get("torso_yaw_joint_max_torque", torso_fallback))
        self.max_torso_pitch = float(hw.get("torso_pitch_joint_max_torque", torso_fallback))
        self.max_torso_roll = float(hw.get("torso_roll_joint_max_torque", torso_fallback))

        # Head
        self.max_head = float(hw.get("head_joint_max_torque", 5.5))
        self.coupled_actuator_cfg = hw.get("coupled_actuators", {})
        coupled_default = _config_bool(hw.get("coupled_enabled", False), False)
        self.coupled_observation_enabled = _config_bool(
            hw.get("coupled_observation_enabled", coupled_default), coupled_default
        )
        self.coupled_control_enabled = _config_bool(
            hw.get("coupled_control_enabled", coupled_default), coupled_default
        )

        # --- Simulation properties ---
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]

        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]
        self.control_freq = 1 / (self.dt_ * self.frame_skip)
        assert self.control_freq == 50, "Currently, only control frequency of 50 is supported."
        self.local_step = 0

        # --- Placeholders ---
        self.action = np.zeros(self.action_dim, dtype=np.float64)
        self.filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.computed_torques = np.zeros(self.action_dim, dtype=np.float64)
        self.applied_torques = np.zeros(self.action_dim, dtype=np.float64)
        self.ctrl_torques = np.zeros(self.action_dim, dtype=np.float64)
        self.viewer = None
        self.mode = None

        # --- Domain randomization XML ---
        self.xml_manager = XMLManager(config)
        self.model_path = self.xml_manager.get_model_path()

        # --- Height map ---
        if self.config["observation"].get("height_map", None) is not None:
            hm = self.config["observation"]["height_map"]
            self.size_x = hm["size_x"]
            self.size_y = hm["size_y"]
            self.res_x = hm["res_x"]
            self.res_y = hm["res_y"]
        else:
            self.size_x = 0.0
            self.size_y = 0.0
            self.res_x = 0
            self.res_y = 0

        # --- Controlled joint order (26), matching policy observation/action order ---
        self.joint_names_in_order = [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_hip_roll_joint",
            "left_ankle_pitch_joint",
            "left_hip_yaw_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_hip_roll_joint",
            "right_ankle_pitch_joint",
            "right_hip_yaw_joint",
            "right_ankle_roll_joint",
            "torso_yaw_joint",
            "head_joint",
            "torso_pitch_joint",
            "torso_roll_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_joint",
            "right_wrist_joint",
        ]
        self.actuator_joint_names_in_order = [
            "head_joint",
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            "torso_yaw_joint", "torso_roll_joint", "torso_pitch_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_elbow_joint", "right_elbow_joint",
            "left_wrist_joint", "right_wrist_joint",
        ]

        # If config action_dim mismatches, prefer the joint list length for safety.
        if self.action_dim != len(self.joint_names_in_order):
            # Keep running, but align internal dims with the joint list.
            self.action_dim = len(self.joint_names_in_order)
            self.action = np.zeros(self.action_dim, dtype=np.float64)
            self.filtered_action = np.zeros(self.action_dim, dtype=np.float64)
            self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
            self.computed_torques = np.zeros(self.action_dim, dtype=np.float64)
            self.applied_torques = np.zeros(self.action_dim, dtype=np.float64)
            self.ctrl_torques = np.zeros(self.action_dim, dtype=np.float64)
            self.action_scaler = np.ones(self.action_dim, dtype=np.float64)
            self.action_clip_min, self.action_clip_max = normalize_action_clippings(config, self.action_dim)

        # --- Observation dims (dynamic) ---
        dof_dim = len(self.joint_names_in_order)
        self.obs_to_dim = {
            "dof_pos": dof_dim,
            "dof_vel": dof_dim,
            "ang_vel": 3,
            "lower_ang_vel": 3,
            "upper_ang_vel": 3,
            "lower_imu_ang_vel": 3,
            "upper_imu_ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
            "lower_projected_gravity": 3,
            "upper_projected_gravity": 3,
            "lower_imu_projected_gravity": 3,
            "upper_imu_projected_gravity": 3,
            "last_action": self.action_dim,
            "height_map": int(self.res_x * self.res_y),
        }

        # --- MuJoCo wrapper ---
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(
                low=-np.inf,
                high=np.inf,
                shape=(sum(self.obs_to_dim.values()),),
                dtype=np.float32,
            ),
            render_mode=self.render_mode if render_flag else None,
        )

        # --- Managers/helpers ---
        self.control_manager = ControlManager(config)
        self.mujoco_utils = MuJoCoUtils(self.model)
        self.mujoco_utils.init_heightmap_visualization(self.res_x, self.res_y)

        # --- Indices in qpos/qvel for controlled joints ---
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(self.joint_names_in_order)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(self.joint_names_in_order)
        self.ctrl_from_policy_order = np.array(
            [self.joint_names_in_order.index(name) for name in self.actuator_joint_names_in_order],
            dtype=np.int64,
        )
        self.policy_from_ctrl_order = np.empty_like(self.ctrl_from_policy_order)
        self.policy_from_ctrl_order[self.ctrl_from_policy_order] = np.arange(len(self.ctrl_from_policy_order))
        self.uses_position_actuators = bool(np.any(self.model.actuator_biastype != 0))
        self.position_actuator_mask_ctrl = self.model.actuator_biastype != 0
        self.motor_actuator_mask_ctrl = ~self.position_actuator_mask_ctrl
        self.uses_hybrid_actuators = bool(
            np.any(self.position_actuator_mask_ctrl) and np.any(self.motor_actuator_mask_ctrl)
        )
        self.obs_joint_names_in_order = list(self.joint_names_in_order)
        self.coupled_pairs = self._build_coupled_pairs()
        self.kp_by_joint, self.kd_by_joint, self.max_torque_by_joint = self._build_pd_vectors()
        self._validate_position_coupling_assumptions()

    def _coupled_cfg(self, pair_name, default_g1, default_g2, default_mirror):
        pair_cfg = {}
        if isinstance(self.coupled_actuator_cfg, dict):
            pair_cfg = self.coupled_actuator_cfg.get(pair_name, {}) or {}

        hw = self.config["hardware"]
        coupled_enabled = self.coupled_observation_enabled or self.coupled_control_enabled
        prefix = pair_name
        if pair_name in ("left_ankle", "right_ankle"):
            shared_g1 = hw.get("ankle_gear_ratio_1", hw.get("ankle_gear_ratio", default_g1))
            shared_g2 = hw.get("ankle_gear_ratio_2", hw.get("ankle_gear_ratio", default_g2))
        elif pair_name == "torso_pitch_roll":
            shared_g1 = hw.get("torso_pitch_roll_gear_ratio_1", hw.get("torso_pitch_roll_gear_ratio", default_g1))
            shared_g2 = hw.get("torso_pitch_roll_gear_ratio_2", hw.get("torso_pitch_roll_gear_ratio", default_g2))
            prefix = "torso"
        else:
            shared_g1 = default_g1
            shared_g2 = default_g2

        return {
            "g1": float(pair_cfg.get("gear_ratio_1", hw.get(f"{prefix}_gear_ratio_1", shared_g1))),
            "g2": float(pair_cfg.get("gear_ratio_2", hw.get(f"{prefix}_gear_ratio_2", shared_g2))),
            "gamma": float(pair_cfg.get("gamma", hw.get(f"{prefix}_gamma", hw.get("coupled_gamma", 1.0)))),
            "mirror": _config_bool(pair_cfg.get("mirror", hw.get(f"{prefix}_mirror", default_mirror)), default_mirror),
            "enabled": coupled_enabled and _config_bool(pair_cfg.get("enabled", hw.get(f"{prefix}_coupled", True)), True),
        }

    def _build_coupled_pairs(self):
        pair_defs = [
            ("left_ankle", ("left_ankle_pitch_joint", "left_ankle_roll_joint"), -2.0, -2.0, False),
            ("right_ankle", ("right_ankle_pitch_joint", "right_ankle_roll_joint"), -2.0, -2.0, False),
            ("torso_pitch_roll", ("torso_pitch_joint", "torso_roll_joint"), 1.0, 1.0, False),
        ]
        pairs = []
        for name, joint_names, default_g1, default_g2, default_mirror in pair_defs:
            cfg = self._coupled_cfg(name, default_g1, default_g2, default_mirror)
            if not cfg["enabled"]:
                continue
            ids = [self.joint_names_in_order.index(joint_name) for joint_name in joint_names]
            if cfg["g1"] == 0.0 or cfg["g2"] == 0.0:
                raise ValueError(f"{name} coupled actuator gear ratios must be non-zero.")
            ids = np.array(ids, dtype=np.int64)
            pairs.append(
                {
                    "name": name,
                    "ids": ids,
                    "g1": cfg["g1"],
                    "g2": cfg["g2"],
                    "gamma": cfg["gamma"],
                    "mirror": cfg["mirror"],
                }
            )
        return pairs

    def _joint_to_motor(self, joint_value):
        motor_value = np.asarray(joint_value, dtype=np.float64).copy()
        for pair in self.coupled_pairs:
            ids = pair["ids"]
            pitch = joint_value[ids[0]]
            roll = joint_value[ids[1]]
            g1 = pair["g1"]
            g2 = pair["g2"]
            if pair["name"] == "left_ankle":
                motor_1 = -g1 * (roll - pitch)
                motor_2 = -g2 * (roll + pitch)
            elif pair["name"] == "right_ankle":
                motor_1 = -g1 * (roll + pitch)
                motor_2 = -g2 * (roll - pitch)
            elif pair["name"] == "torso_pitch_roll":
                motor_1 = g1 * (roll - pitch)
                motor_2 = -g2 * (roll + pitch)
            else:
                raise ValueError(f"Unhandled coupled pair: {pair['name']}")
            motor_value[ids[0]] = motor_1
            motor_value[ids[1]] = motor_2
        return motor_value

    def _motor_to_joint_position(self, motor_value):
        joint_value = np.asarray(motor_value, dtype=np.float64).copy()
        for pair in self.coupled_pairs:
            ids = pair["ids"]
            motor_1 = motor_value[ids[0]]
            motor_2 = motor_value[ids[1]]
            g1 = pair["g1"]
            g2 = pair["g2"]
            m1 = motor_1 / g1
            m2 = motor_2 / g2
            if pair["name"] == "left_ankle":
                pitch = 0.5 * (m1 - m2)
                roll = -0.5 * (m1 + m2)
            elif pair["name"] == "right_ankle":
                pitch = 0.5 * (m2 - m1)
                roll = -0.5 * (m1 + m2)
            elif pair["name"] == "torso_pitch_roll":
                pitch = -0.5 * (m1 + m2)
                roll = 0.5 * (m1 - m2)
            else:
                raise ValueError(f"Unhandled coupled pair: {pair['name']}")
            joint_value[ids[0]] = pitch
            joint_value[ids[1]] = roll
        return joint_value

    def _validate_position_coupling_assumptions(self):
        if not self.coupled_control_enabled or not self.uses_position_actuators:
            return
        for pair in self.coupled_pairs:
            ids = pair["ids"]
            name = pair["name"]
            if not np.isclose(abs(pair["g1"]), abs(pair["g2"])):
                raise ValueError(
                    f"{name} requires |gear_ratio_1| == |gear_ratio_2| for position-mode coupled control."
                )
            if not np.isclose(self.kp_by_joint[ids[0]], self.kp_by_joint[ids[1]]):
                raise ValueError(
                    f"{name} requires equal Kp for both motor slots for position-mode coupled control."
                )
            if not np.isclose(self.kd_by_joint[ids[0]], self.kd_by_joint[ids[1]]):
                raise ValueError(
                    f"{name} requires equal Kd for both motor slots for position-mode coupled control."
                )

    def _motor_to_joint_torque(self, motor_tau):
        joint_tau = np.asarray(motor_tau, dtype=np.float64).copy()
        for pair in self.coupled_pairs:
            ids = pair["ids"]
            tau_m1 = motor_tau[ids[0]]
            tau_m2 = motor_tau[ids[1]]
            g1 = pair["g1"]
            g2 = pair["g2"]
            gamma = pair["gamma"]
            if pair["name"] == "left_ankle":
                tau_pitch = g1 * tau_m1 - g2 * tau_m2
                tau_roll = -g1 * tau_m1 - g2 * tau_m2
            elif pair["name"] == "right_ankle":
                tau_pitch = -g1 * tau_m1 + g2 * tau_m2
                tau_roll = -g1 * tau_m1 - g2 * tau_m2
            elif pair["name"] == "torso_pitch_roll":
                tau_pitch = -g1 * tau_m1 - g2 * tau_m2
                tau_roll = g1 * tau_m1 - g2 * tau_m2
            else:
                raise ValueError(f"Unhandled coupled pair: {pair['name']}")
            joint_tau[ids[0]] = gamma * tau_pitch
            joint_tau[ids[1]] = gamma * tau_roll
        return joint_tau

    def _build_pd_vectors(self):
        kp = np.zeros(self.action_dim, dtype=np.float64)
        kd = np.zeros(self.action_dim, dtype=np.float64)
        max_torque = np.zeros(self.action_dim, dtype=np.float64)

        for i, joint_name in enumerate(self.joint_names_in_order):
            if "hip_pitch" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_hip_pitch, self.kd_hip_pitch, self.max_hip_pitch
            elif "hip_roll" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_hip_roll, self.kd_hip_roll, self.max_hip_roll
            elif "hip_yaw" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_hip_yaw, self.kd_hip_yaw, self.max_hip_yaw
            elif "knee" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_knee, self.kd_knee, self.max_knee
            elif "ankle_pitch" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_ankle_pitch, self.kd_ankle_pitch, self.max_ankle_pitch
            elif "ankle_roll" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_ankle_roll, self.kd_ankle_roll, self.max_ankle_roll
            elif "torso_yaw" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_torso_yaw, self.kd_torso_yaw, self.max_torso_yaw
            elif "torso_pitch" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_torso_pitch, self.kd_torso_pitch, self.max_torso_pitch
            elif "torso_roll" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_torso_roll, self.kd_torso_roll, self.max_torso_roll
            elif "shoulder_pitch" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_shoulder_pitch, self.kd_shoulder_pitch, self.max_shoulder_pitch
            elif "shoulder_roll" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_shoulder_roll, self.kd_shoulder_roll, self.max_shoulder_roll
            elif "shoulder_yaw" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_shoulder_yaw, self.kd_shoulder_yaw, self.max_shoulder_yaw
            elif "elbow" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_elbow, self.kd_elbow, self.max_elbow
            elif "wrist" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_wrist, self.kd_wrist, self.max_wrist
            elif "head" in joint_name:
                kp[i], kd[i], max_torque[i] = self.kp_head, self.kd_head, self.max_head
            else:
                raise ValueError(f"Unhandled humanoid_light joint: {joint_name}")

        return kp, kd, max_torque

    # -----------------------
    # Observation
    # -----------------------
    def _sensor_data_or(self, sensor_name, fallback_name):
        try:
            return self.data.sensor(sensor_name).data.astype(np.float64)
        except KeyError:
            return self.data.sensor(fallback_name).data.astype(np.float64)

    @staticmethod
    def _projected_gravity_from_quat(quat_wxyz):
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]].astype(np.float64)
        if np.all(quat_xyzw == 0):
            quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float64)
        return MathUtils.quat_to_base_vel(quat_xyzw, np.array([0, 0, -1], dtype=np.float64))

    def _sensor_projected_gravity(self, sensor_name, fallback_name="orientation"):
        return self._projected_gravity_from_quat(self._sensor_data_or(sensor_name, fallback_name))

    def _get_obs(self):
        # dof_pos / dof_vel follow self.obs_joint_names_in_order exactly.
        dof_pos = self.data.qpos[self.q_indices].astype(np.float64)
        dof_vel = self.data.qvel[self.qd_indices].astype(np.float64)
        if self.coupled_observation_enabled:
            dof_pos = self._joint_to_motor(dof_pos)
            dof_vel = self._joint_to_motor(dof_vel)

        ang_vel = self.data.sensor("angular-velocity").data.astype(np.float64)
        lower_imu_ang_vel = self._sensor_data_or("lower_imu_angular_velocity", "angular-velocity")
        upper_imu_ang_vel = self._sensor_data_or("upper_imu_angular_velocity", "angular-velocity")
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float64)

        projected_gravity = self._sensor_projected_gravity("orientation")
        lower_imu_projected_gravity = self._sensor_projected_gravity("lower_imu_orientation")
        upper_imu_projected_gravity = self._sensor_projected_gravity("upper_imu_orientation")

        if self.config["observation"].get("height_map", None) is not None:
            height_map = self.mujoco_utils.get_height_map(
                self.data, self.size_x, self.size_y, self.res_x, self.res_y
            ).astype(np.float64)
        else:
            height_map = None

        # --- Apply sensor noise (guard height_map) ---
        dof_pos_noisy = truncated_gaussian_noisy_data(
            dof_pos,
            mean=self.sensor_noise_map["dof_pos"]["mean"],
            std=self.sensor_noise_map["dof_pos"]["std"],
            lower=self.sensor_noise_map["dof_pos"]["lower"],
            upper=self.sensor_noise_map["dof_pos"]["upper"],
        )
        dof_vel_noisy = truncated_gaussian_noisy_data(
            dof_vel,
            mean=self.sensor_noise_map["dof_vel"]["mean"],
            std=self.sensor_noise_map["dof_vel"]["std"],
            lower=self.sensor_noise_map["dof_vel"]["lower"],
            upper=self.sensor_noise_map["dof_vel"]["upper"],
        )
        ang_vel_noisy = truncated_gaussian_noisy_data(
            ang_vel,
            mean=self.sensor_noise_map["ang_vel"]["mean"],
            std=self.sensor_noise_map["ang_vel"]["std"],
            lower=self.sensor_noise_map["ang_vel"]["lower"],
            upper=self.sensor_noise_map["ang_vel"]["upper"],
        )
        lower_imu_ang_vel_noisy = truncated_gaussian_noisy_data(
            lower_imu_ang_vel,
            mean=self.sensor_noise_map["ang_vel"]["mean"],
            std=self.sensor_noise_map["ang_vel"]["std"],
            lower=self.sensor_noise_map["ang_vel"]["lower"],
            upper=self.sensor_noise_map["ang_vel"]["upper"],
        )
        upper_imu_ang_vel_noisy = truncated_gaussian_noisy_data(
            upper_imu_ang_vel,
            mean=self.sensor_noise_map["ang_vel"]["mean"],
            std=self.sensor_noise_map["ang_vel"]["std"],
            lower=self.sensor_noise_map["ang_vel"]["lower"],
            upper=self.sensor_noise_map["ang_vel"]["upper"],
        )
        lin_vel_noisy = truncated_gaussian_noisy_data(
            lin_vel,
            mean=self.sensor_noise_map["lin_vel"]["mean"],
            std=self.sensor_noise_map["lin_vel"]["std"],
            lower=self.sensor_noise_map["lin_vel"]["lower"],
            upper=self.sensor_noise_map["lin_vel"]["upper"],
        )
        projected_gravity_noisy = truncated_gaussian_noisy_data(
            projected_gravity,
            mean=self.sensor_noise_map["projected_gravity"]["mean"],
            std=self.sensor_noise_map["projected_gravity"]["std"],
            lower=self.sensor_noise_map["projected_gravity"]["lower"],
            upper=self.sensor_noise_map["projected_gravity"]["upper"],
        )
        lower_imu_projected_gravity_noisy = truncated_gaussian_noisy_data(
            lower_imu_projected_gravity,
            mean=self.sensor_noise_map["projected_gravity"]["mean"],
            std=self.sensor_noise_map["projected_gravity"]["std"],
            lower=self.sensor_noise_map["projected_gravity"]["lower"],
            upper=self.sensor_noise_map["projected_gravity"]["upper"],
        )
        upper_imu_projected_gravity_noisy = truncated_gaussian_noisy_data(
            upper_imu_projected_gravity,
            mean=self.sensor_noise_map["projected_gravity"]["mean"],
            std=self.sensor_noise_map["projected_gravity"]["std"],
            lower=self.sensor_noise_map["projected_gravity"]["lower"],
            upper=self.sensor_noise_map["projected_gravity"]["upper"],
        )

        if height_map is not None and self.res_x * self.res_y > 0:
            height_map_noisy = truncated_gaussian_noisy_data(
                height_map,
                mean=self.sensor_noise_map["height_map"]["mean"],
                std=self.sensor_noise_map["height_map"]["std"],
                lower=self.sensor_noise_map["height_map"]["lower"],
                upper=self.sensor_noise_map["height_map"]["upper"],
            )
        else:
            height_map_noisy = np.zeros((0,), dtype=np.float64)

        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "lower_ang_vel": lower_imu_ang_vel_noisy,
            "upper_ang_vel": upper_imu_ang_vel_noisy,
            "lower_imu_ang_vel": lower_imu_ang_vel_noisy,
            "upper_imu_ang_vel": upper_imu_ang_vel_noisy,
            "lin_vel_x": float(lin_vel_noisy[0]),
            "lin_vel_y": float(lin_vel_noisy[1]),
            "lin_vel_z": float(lin_vel_noisy[2]),
            "projected_gravity": projected_gravity_noisy,
            "lower_projected_gravity": lower_imu_projected_gravity_noisy,
            "upper_projected_gravity": upper_imu_projected_gravity_noisy,
            "lower_imu_projected_gravity": lower_imu_projected_gravity_noisy,
            "upper_imu_projected_gravity": upper_imu_projected_gravity_noisy,
            "height_map": height_map_noisy,
            "last_action": self.action.astype(np.float64),
        }

    def _update_pd_torques(self, action_scaled):
        dof_pos = self.data.qpos[self.q_indices].astype(np.float64)
        dof_vel = self.data.qvel[self.qd_indices].astype(np.float64)
        if self.coupled_control_enabled:
            dof_pos = self._joint_to_motor(dof_pos)
            dof_vel = self._joint_to_motor(dof_vel)
        motor_torques = self.kp_by_joint * (action_scaled - dof_pos)
        joint_torques = self._motor_to_joint_torque(motor_torques) if self.coupled_control_enabled else motor_torques
        self.computed_torques = joint_torques.astype(np.float64)
        self.applied_torques = np.clip(
            joint_torques,
            -self.max_torque_by_joint,
            self.max_torque_by_joint,
        ).astype(np.float64)
        self.ctrl_torques = self.applied_torques[self.ctrl_from_policy_order]
        return self.ctrl_torques

    # -----------------------
    # Step
    # -----------------------
    def step(self, action):
        self.action = np.asarray(action, dtype=np.float64)
        self.filtered_action = self.control_manager.delay_filter(self.action)
        action_scaled = scale_and_clip_action(self.filtered_action, self.action_scaler, self.action_clip_min, self.action_clip_max)
        position_targets = self._motor_to_joint_position(action_scaled) if self.coupled_control_enabled else action_scaled
        if self.uses_hybrid_actuators:
            position_ctrl = position_targets[self.ctrl_from_policy_order].copy()

            def ctrl_fn():
                ctrl = position_ctrl.copy()
                motor_torques = self._update_pd_torques(action_scaled)
                ctrl[self.motor_actuator_mask_ctrl] = motor_torques[self.motor_actuator_mask_ctrl]
                return ctrl

            simulate_dynamic_ctrl(self, ctrl_fn)
            actuator_force = self.data.actuator_force.astype(np.float64)
            self.ctrl_torques = actuator_force.copy()
            self.applied_torques = actuator_force[self.policy_from_ctrl_order].copy()
        elif self.uses_position_actuators:
            self.do_simulation(position_targets[self.ctrl_from_policy_order], self.frame_skip)
            actuator_force = self.data.actuator_force.astype(np.float64)
            self.ctrl_torques = actuator_force.copy()
            self.applied_torques = actuator_force[self.policy_from_ctrl_order].copy()
        else:
            simulate_dynamic_ctrl(self, lambda: self._update_pd_torques(action_scaled))

        obs = self._get_obs()
        info = self._get_info()
        terminated = self._is_done()
        truncated = False

        self.prev_action = self.action.copy()
        self.local_step += 1

        # NOTE: Keeping legacy 4-return signature as in your original code.
        return obs, terminated, truncated, info

    # -----------------------
    # Info / reset / misc
    # -----------------------
    def _get_info(self):
        dof_pos = self.data.qpos[self.q_indices].astype(np.float64)
        ang_vel = self.data.sensor("angular-velocity").data.astype(np.float64)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float64)

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action.copy(),
            "action_diff_RMSE": float(np.sqrt(np.mean((self.action - self.prev_action) ** 2))),
            "torque": self.applied_torques.copy(),
            "lin_vel_x": float(lin_vel[0]),
            "lin_vel_y": float(lin_vel[1]),
            "ang_vel_yaw": float(ang_vel[2]),
            "set_points": scale_and_clip_action(self.filtered_action, self.action_scaler, self.action_clip_min, self.action_clip_max).copy(),
            "state": dof_pos.copy(),
        }
        return info

    def _get_reset_info(self):
        return self._get_info()

    def _is_done(self):
        return False

    def reset_model(self):
        self.local_step = 0
        self.action[:] = 0.0
        self.prev_action[:] = 0.0
        self.control_manager.reset()
        self.applied_torques[:] = 0.0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs

    def initial_qpos(self):
        env_id = self.config.get("env", {}).get("id", self.id)
        return build_initial_qpos(
            self.model,
            self.mujoco_utils,
            self.config,
            env_id=env_id,
            init_noise=self.init_noise,
            joint_names=self.joint_names_in_order,
        )

    def event(self, event: str, value):
        if event == "push":
            # value is world-frame velocity impulse [vx, vy, vz]
            raw_quat = self.data.qpos[3:7].astype(np.float64)  # [qw, qx, qy, qz]
            R = MathUtils.quat_to_rot_matrix(raw_quat).T        # world->robot
            world_vel = np.array(value, dtype=np.float64).reshape(3,)
            robot_vel = R.dot(world_vel)

            # xy in robot frame, z in world frame (legacy behavior)
            self.data.qvel[:2] = robot_vel[:2]  
            self.data.qvel[2] = world_vel[2]
        else:
            raise NotImplementedError(f"event:{event} is not supported.")

    def get_data(self):
        return self.data

    def close(self):
        if self.viewer is not None:
            if glfw.get_current_context() == self.viewer.window:
                glfw.make_context_current(None)
            glfw.destroy_window(self.viewer.window)
            glfw.terminate()
            self.viewer = None
            print("Viewer closed")
        super().close()
