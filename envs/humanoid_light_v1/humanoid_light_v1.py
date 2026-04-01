from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw

from envs.humanoid_light_v1.manager.control_manager import ControlManager
from envs.humanoid_light_v1.manager.xml_manager import XMLManager
from envs.humanoid_light_v1.utils.math_utils import MathUtils
from envs.humanoid_light_v1.utils.mujoco_utils import MuJoCoUtils
from envs.humanoid_light_v1.utils.noise_generator_utils import (
    truncated_gaussian_noisy_data,
)
from envs.initial_pose import build_initial_qpos


class HumanoidLightV1(MujocoEnv, utils.EzPickle):
    """
    Updated for URDF-v9 style joints:
      - torso is 3DoF: torso_yaw_joint, torso_pitch_joint, torso_roll_joint
      - elbow is 1DoF per arm: *_elbow_joint  (no elbow_pitch/yaw split)
      - wrist is 1DoF per arm: *_wrist_joint
      - head_joint included
    Action order (26):
      [0:2]   hip_pitch (L,R)
      [2:5]   torso (yaw,pitch,roll)
      [5:7]   hip_roll (L,R)
      [7:9]   shoulder_pitch (L,R)
      [9:11]  hip_yaw (L,R)
      [11:13] shoulder_roll (L,R)
      [13:15] knee (L,R)
      [15:17] shoulder_yaw (L,R)
      [17:19] ankle_pitch (L,R)
      [19:21] elbow (L,R)
      [21:23] ankle_roll (L,R)
      [23:25] wrist (L,R)
      [25:26] head
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config, render_flag=True, render_mode="human"):
        # --- Basic properties ---
        self.id = "humanoid_light_v1"
        self.config = config
        self.render_mode = render_mode
        self.render_flag = render_flag

        self.action_dim = int(config["hardware"]["action_dim"])
        default_action_scales = np.ones(self.action_dim, dtype=np.float64) * 0.5
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple, np.ndarray)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)

        # --- PD params (support both new keys and old legacy keys) ---
        hw = config["hardware"]

        # New recommended keys
        self.kp_hip_pitch = hw.get("Kp_hip_pitch", 200)
        self.kp_hip_roll = hw.get("Kp_hip_roll", 150)
        self.kp_hip_yaw = hw.get("Kp_hip_yaw", 150)
        self.kp_knee = hw.get("Kp_knee", 200)
        self.kp_ankle_pitch = hw.get("Kp_ankle_pitch", 40)
        self.kp_ankle_roll = hw.get("Kp_ankle_roll", 40)

        self.kp_torso = hw.get("Kp_torso", 300)
        self.kp_head = hw.get("Kp_head", 50)

        self.kp_shoulder_pitch = hw.get("Kp_shoulder_pitch", 100)
        self.kp_shoulder_roll = hw.get("Kp_shoulder_roll", 100)
        self.kp_shoulder_yaw = hw.get("Kp_shoulder_yaw", 50)

        # Elbow/wrist: new keys
        self.kp_elbow = hw.get("Kp_elbow", None)
        self.kp_wrist = hw.get("Kp_wrist", None)

        # Legacy fallback (old code had elbow_pitch/elbow_yaw)
        if self.kp_elbow is None:
            # If legacy exists, pick a reasonable representative
            self.kp_elbow = hw.get("Kp_elbow_pitch", hw.get("Kp_elbow_yaw", 50))
        if self.kp_wrist is None:
            self.kp_wrist = hw.get("Kp_wrist", 50)

        self.kd_hip_pitch = hw.get("Kd_hip_pitch", 2)
        self.kd_hip_roll = hw.get("Kd_hip_roll", 2)
        self.kd_hip_yaw = hw.get("Kd_hip_yaw", 2)
        self.kd_knee = hw.get("Kd_knee", 4)
        self.kd_ankle_pitch = hw.get("Kd_ankle_pitch", 2)
        self.kd_ankle_roll = hw.get("Kd_ankle_roll", 2)

        self.kd_torso = hw.get("Kd_torso", 6)
        self.kd_head = hw.get("Kd_head", 2)

        self.kd_shoulder_pitch = hw.get("Kd_shoulder_pitch", 2)
        self.kd_shoulder_roll = hw.get("Kd_shoulder_roll", 2)
        self.kd_shoulder_yaw = hw.get("Kd_shoulder_yaw", 2)

        self.kd_elbow = hw.get("Kd_elbow", None)
        self.kd_wrist = hw.get("Kd_wrist", None)

        if self.kd_elbow is None:
            self.kd_elbow = hw.get("Kd_elbow_pitch", hw.get("Kd_elbow_yaw", 2))
        if self.kd_wrist is None:
            self.kd_wrist = hw.get("Kd_wrist", 2)

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
        self.applied_torques = np.zeros(self.action_dim, dtype=np.float64)
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

        # --- Controlled joint order (26) ---
        self.joint_names_in_order = [
            # hips pitch
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            # torso (3DoF)
            "torso_yaw_joint",
            # hips roll
            "left_hip_roll_joint", "right_hip_roll_joint",
            # torso pitch
            "torso_pitch_joint",
            # hips yaw
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            # torso roll
            "torso_roll_joint",
            # knees
            "left_knee_joint", "right_knee_joint",
            # head
            "head_joint",
            # shoulders pitch
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            # ankles pitch
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            # shoulders roll
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            # ankles roll
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            # shoulders yaw
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            # elbows (1DoF)
            "left_elbow_joint", "right_elbow_joint",
            # wrists
            "left_wrist_joint", "right_wrist_joint",
        ]

        # If config action_dim mismatches, prefer the joint list length for safety.
        if self.action_dim != len(self.joint_names_in_order):
            # Keep running, but align internal dims with the joint list.
            self.action_dim = len(self.joint_names_in_order)
            self.action = np.zeros(self.action_dim, dtype=np.float64)
            self.filtered_action = np.zeros(self.action_dim, dtype=np.float64)
            self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
            self.applied_torques = np.zeros(self.action_dim, dtype=np.float64)
            self.action_scaler = np.ones(self.action_dim, dtype=np.float64)

        # --- Observation dims (dynamic) ---
        dof_dim = len(self.joint_names_in_order)
        self.obs_to_dim = {
            "dof_pos": dof_dim,
            "dof_vel": dof_dim,
            "ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
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

    # -----------------------
    # Observation
    # -----------------------
    def _get_obs(self):
        dof_pos = self.data.qpos[self.q_indices].astype(np.float64)
        dof_vel = self.data.qvel[self.qd_indices].astype(np.float64)

        ang_vel = self.data.sensor("angular-velocity").data.astype(np.float64)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float64)

        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64)  # xyzw -> wxyz-like usage below
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1], dtype=np.float64)

        projected_gravity = MathUtils.quat_to_base_vel(quat, np.array([0, 0, -1], dtype=np.float64))

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
            "lin_vel_x": float(lin_vel_noisy[0]),
            "lin_vel_y": float(lin_vel_noisy[1]),
            "lin_vel_z": float(lin_vel_noisy[2]),
            "projected_gravity": projected_gravity_noisy,
            "height_map": height_map_noisy,
            "last_action": self.action.astype(np.float64),
        }

    # -----------------------
    # Step
    # -----------------------
    def step(self, action):
        self.action = np.asarray(action, dtype=np.float64)
        self.filtered_action = self.control_manager.delay_filter(self.action)
        print(f"Step {self.local_step}: action={self.action}, filtered_action={self.filtered_action}")
        
        # Pull joint positions/velocities (controlled joints only, in our order)
        dof_pos = self.data.qpos[self.q_indices].astype(np.float64)
        dof_vel = self.data.qvel[self.qd_indices].astype(np.float64)

        # --- Slice helper ---
        def sl(a, b):
            return slice(a, b)

        s_hip_pitch   = sl(0, 2)    # L,R
        s_torso_yaw   = sl(2, 3)    # 1
        s_hip_roll    = sl(3, 5)    # L,R
        s_torso_pitch = sl(5, 6)    # 1
        s_hip_yaw     = sl(6, 8)    # L,R
        s_torso_roll  = sl(8, 9)    # 1
        s_knee        = sl(9, 11)   # L,R
        s_head        = sl(11, 12)  # 1
        s_sh_pitch    = sl(12, 14)  # L,R
        s_ank_pitch   = sl(14, 16)  # L,R
        s_sh_roll     = sl(16, 18)  # L,R
        s_ank_roll    = sl(18, 20)  # L,R
        s_sh_yaw      = sl(20, 22)  # L,R
        s_elbow       = sl(22, 24)  # L,R
        s_wrist       = sl(24, 26)  # L,R

        # Targets (scaled actions)
        tgt_hip_pitch   = self.filtered_action[s_hip_pitch]   * self.action_scaler[s_hip_pitch]
        tgt_torso_yaw   = self.filtered_action[s_torso_yaw]   * self.action_scaler[s_torso_yaw]
        tgt_hip_roll    = self.filtered_action[s_hip_roll]    * self.action_scaler[s_hip_roll]
        tgt_torso_pitch = self.filtered_action[s_torso_pitch] * self.action_scaler[s_torso_pitch]
        tgt_hip_yaw     = self.filtered_action[s_hip_yaw]     * self.action_scaler[s_hip_yaw]
        tgt_torso_roll  = self.filtered_action[s_torso_roll]  * self.action_scaler[s_torso_roll]
        tgt_knee        = self.filtered_action[s_knee]        * self.action_scaler[s_knee]
        tgt_head        = self.filtered_action[s_head]        * self.action_scaler[s_head]
        tgt_sh_pitch    = self.filtered_action[s_sh_pitch]    * self.action_scaler[s_sh_pitch]
        tgt_ank_pitch   = self.filtered_action[s_ank_pitch]   * self.action_scaler[s_ank_pitch]
        tgt_sh_roll     = self.filtered_action[s_sh_roll]     * self.action_scaler[s_sh_roll]
        tgt_ank_roll    = self.filtered_action[s_ank_roll]    * self.action_scaler[s_ank_roll]
        tgt_sh_yaw      = self.filtered_action[s_sh_yaw]      * self.action_scaler[s_sh_yaw]
        tgt_elbow       = self.filtered_action[s_elbow]       * self.action_scaler[s_elbow]
        tgt_wrist       = self.filtered_action[s_wrist]       * self.action_scaler[s_wrist]

        # Current states
        pos_hip_pitch,   vel_hip_pitch   = dof_pos[s_hip_pitch],   dof_vel[s_hip_pitch]
        pos_torso_yaw,   vel_torso_yaw   = dof_pos[s_torso_yaw],   dof_vel[s_torso_yaw]
        pos_hip_roll,    vel_hip_roll    = dof_pos[s_hip_roll],    dof_vel[s_hip_roll]
        pos_torso_pitch, vel_torso_pitch = dof_pos[s_torso_pitch], dof_vel[s_torso_pitch]
        pos_hip_yaw,     vel_hip_yaw     = dof_pos[s_hip_yaw],     dof_vel[s_hip_yaw]
        pos_torso_roll,  vel_torso_roll  = dof_pos[s_torso_roll],  dof_vel[s_torso_roll]
        pos_knee,        vel_knee        = dof_pos[s_knee],        dof_vel[s_knee]
        pos_head,        vel_head        = dof_pos[s_head],        dof_vel[s_head]
        pos_sh_pitch,    vel_sh_pitch    = dof_pos[s_sh_pitch],    dof_vel[s_sh_pitch]
        pos_ank_pitch,   vel_ank_pitch   = dof_pos[s_ank_pitch],   dof_vel[s_ank_pitch]
        pos_sh_roll,     vel_sh_roll     = dof_pos[s_sh_roll],     dof_vel[s_sh_roll]
        pos_ank_roll,    vel_ank_roll    = dof_pos[s_ank_roll],    dof_vel[s_ank_roll]
        pos_sh_yaw,      vel_sh_yaw      = dof_pos[s_sh_yaw],      dof_vel[s_sh_yaw]
        pos_elbow,       vel_elbow       = dof_pos[s_elbow],       dof_vel[s_elbow]
        pos_wrist,       vel_wrist       = dof_pos[s_wrist],       dof_vel[s_wrist]

        # PD torques
        hip_pitch_t = self.control_manager.pd_controller(
            self.kp_hip_pitch, tgt_hip_pitch, pos_hip_pitch,
            self.kd_hip_pitch, 0.0, vel_hip_pitch
        )

        torso_yaw_t = self.control_manager.pd_controller(
            self.kp_torso, tgt_torso_yaw, pos_torso_yaw,
            self.kd_torso, 0.0, vel_torso_yaw
        )
        hip_roll_t = self.control_manager.pd_controller(
            self.kp_hip_roll, tgt_hip_roll, pos_hip_roll,
            self.kd_hip_roll, 0.0, vel_hip_roll
        )
        torso_pitch_t = self.control_manager.pd_controller(
            self.kp_torso, tgt_torso_pitch, pos_torso_pitch,
            self.kd_torso, 0.0, vel_torso_pitch
        )
        hip_yaw_t = self.control_manager.pd_controller(
            self.kp_hip_yaw, tgt_hip_yaw, pos_hip_yaw,
            self.kd_hip_yaw, 0.0, vel_hip_yaw
        )
        torso_roll_t = self.control_manager.pd_controller(
            self.kp_torso, tgt_torso_roll, pos_torso_roll,
            self.kd_torso, 0.0, vel_torso_roll
        )

        knee_t = self.control_manager.pd_controller(
            self.kp_knee, tgt_knee, pos_knee,
            self.kd_knee, 0.0, vel_knee
        )
        head_t = self.control_manager.pd_controller(
            self.kp_head, tgt_head, pos_head,
            self.kd_head, 0.0, vel_head
        )

        sh_pitch_t = self.control_manager.pd_controller(
            self.kp_shoulder_pitch, tgt_sh_pitch, pos_sh_pitch,
            self.kd_shoulder_pitch, 0.0, vel_sh_pitch
        )
        ank_pitch_t = self.control_manager.pd_controller(
            self.kp_ankle_pitch, tgt_ank_pitch, pos_ank_pitch,
            self.kd_ankle_pitch, 0.0, vel_ank_pitch
        )
        sh_roll_t = self.control_manager.pd_controller(
            self.kp_shoulder_roll, tgt_sh_roll, pos_sh_roll,
            self.kd_shoulder_roll, 0.0, vel_sh_roll
        )
        ank_roll_t = self.control_manager.pd_controller(
            self.kp_ankle_roll, tgt_ank_roll, pos_ank_roll,
            self.kd_ankle_roll, 0.0, vel_ank_roll
        )
        sh_yaw_t = self.control_manager.pd_controller(
            self.kp_shoulder_yaw, tgt_sh_yaw, pos_sh_yaw,
            self.kd_shoulder_yaw, 0.0, vel_sh_yaw
        )
        elbow_t = self.control_manager.pd_controller(
            self.kp_elbow, tgt_elbow, pos_elbow,
            self.kd_elbow, 0.0, vel_elbow
        )
        wrist_t = self.control_manager.pd_controller(
            self.kp_wrist, tgt_wrist, pos_wrist,
            self.kd_wrist, 0.0, vel_wrist
        )

        # Clip torques
        hip_pitch_t = np.clip(hip_pitch_t, -self.max_hip_pitch, self.max_hip_pitch)
        hip_roll_t  = np.clip(hip_roll_t,  -self.max_hip_roll,  self.max_hip_roll)
        hip_yaw_t   = np.clip(hip_yaw_t,   -self.max_hip_yaw,   self.max_hip_yaw)
        knee_t      = np.clip(knee_t,      -self.max_knee,      self.max_knee)

        ank_pitch_t = np.clip(ank_pitch_t, -self.max_ankle_pitch, self.max_ankle_pitch)
        ank_roll_t  = np.clip(ank_roll_t,  -self.max_ankle_roll,  self.max_ankle_roll)

        sh_pitch_t  = np.clip(sh_pitch_t,  -self.max_shoulder_pitch, self.max_shoulder_pitch)
        sh_roll_t   = np.clip(sh_roll_t,   -self.max_shoulder_roll,  self.max_shoulder_roll)
        sh_yaw_t    = np.clip(sh_yaw_t,    -self.max_shoulder_yaw,   self.max_shoulder_yaw)

        elbow_t = np.clip(elbow_t, -self.max_elbow, self.max_elbow)
        wrist_t = np.clip(wrist_t, -self.max_wrist, self.max_wrist)
        head_t  = np.clip(head_t,  -self.max_head,  self.max_head)

        torso_yaw_t   = np.clip(torso_yaw_t,   -self.max_torso_yaw,   self.max_torso_yaw)
        torso_pitch_t = np.clip(torso_pitch_t, -self.max_torso_pitch, self.max_torso_pitch)
        torso_roll_t  = np.clip(torso_roll_t,  -self.max_torso_roll,  self.max_torso_roll)

        # Concatenate EXACTLY in joint_names_in_order
        self.applied_torques = np.concatenate(
            [
                hip_pitch_t,     # 2
                torso_yaw_t,     # 1
                hip_roll_t,      # 2
                torso_pitch_t,   # 1
                hip_yaw_t,       # 2
                torso_roll_t,    # 1
                knee_t,          # 2
                head_t,          # 1
                sh_pitch_t,      # 2
                ank_pitch_t,     # 2
                sh_roll_t,       # 2
                ank_roll_t,      # 2
                sh_yaw_t,        # 2
                elbow_t,         # 2
                wrist_t,         # 2
            ],
            axis=0,
        ).astype(np.float64)

        # Simulate
        self.do_simulation(self.applied_torques, self.frame_skip)

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
            "set_points": (self.action * self.action_scaler).copy(),
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
