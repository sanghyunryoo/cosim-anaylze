from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw

from envs.wheeldog_p_v2.manager.control_manager import ControlManager
from envs.wheeldog_p_v2.manager.xml_manager import XMLManager
from envs.wheeldog_p_v2.utils.math_utils import MathUtils
from envs.wheeldog_p_v2.utils.mujoco_utils import MuJoCoUtils
from envs.wheeldog_p_v2.utils.noise_generator_utils import (
    truncated_gaussian_noisy_data,
)
from envs.initial_pose import build_initial_qpos


class WheelDogPV2(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config, render_flag=True, render_mode='human'):
        self.id = "wheeldog_p_v2"
        self.config = config
        self.action_dim = int(config["hardware"]["action_dim"])

        default_action_scales = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 40.0, 40.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 40.0, 40.0
        ]
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)
        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD parameters
        self.kp_hip = config["hardware"]["Kp_hip"]
        self.kp_shoulder = config["hardware"]["Kp_shoulder"]
        self.kp_leg = config["hardware"]["Kp_leg"]

        self.kd_hip = config["hardware"]["Kd_hip"]
        self.kd_shoulder = config["hardware"]["Kd_shoulder"]
        self.kd_leg = config["hardware"]["Kd_leg"]
        self.kd_wheel = config["hardware"]["Kd_wheel"]

        # Gear settings
        self.gear_ratio = config["hardware"]["gear_ratio"]
        self.gamma = config["hardware"]["gamma"]
        self.use_gear = (self.gear_ratio != 1.0)

        # Simulation properties
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]
        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]
        self.control_freq = 1 / (self.dt_ * self.frame_skip)
        assert self.control_freq == 50, "Currently, only control frequency of 50 is supported."
        self.local_step = 0

        # Placeholders
        self.action = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)
        self.applied_torques = np.zeros(self.action_dim)
        self.viewer = None
        self.mode = None

        # Domain randomization / model path
        self.xml_manager = XMLManager(config)
        self.model_path = self.xml_manager.get_model_path()

        # Height map
        if self.config["observation"]["height_map"] is not None:
            self.size_x = self.config["observation"]["height_map"]["size_x"]
            self.size_y = self.config["observation"]["height_map"]["size_y"]
            self.res_x = self.config["observation"]["height_map"]["res_x"]
            self.res_y = self.config["observation"]["height_map"]["res_y"]
        else:
            self.res_x = 0
            self.res_y = 0

        self.obs_to_dim = {
            "dof_pos": 12,
            "dof_vel": 16,
            "ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
            "last_action": self.action_dim,
            "height_map": int(self.res_x * self.res_y),
        }

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

        # Managers / helpers
        self.control_manager = ControlManager(config)
        self.mujoco_utils = MuJoCoUtils(self.model)
        self.mujoco_utils.init_heightmap_visualization(self.res_x, self.res_y)

        # Joint indices
        qpos_joint_names = [
            'FL_hip_joint', 'FR_hip_joint',
            'FL_shoulder_joint', 'FR_shoulder_joint',
            'FL_leg_joint', 'FR_leg_joint',
            'RL_hip_joint', 'RR_hip_joint',
            'RL_shoulder_joint', 'RR_shoulder_joint',
            'RL_leg_joint', 'RR_leg_joint',
        ]
        qvel_joint_names = [
            'FL_hip_joint', 'FR_hip_joint',
            'FL_shoulder_joint', 'FR_shoulder_joint',
            'FL_leg_joint', 'FR_leg_joint',
            'FL_wheel_joint', 'FR_wheel_joint',
            'RL_hip_joint', 'RR_hip_joint',
            'RL_shoulder_joint', 'RR_shoulder_joint',
            'RL_leg_joint', 'RR_leg_joint',
            'RL_wheel_joint', 'RR_wheel_joint',
        ]
        self.initial_joint_names = list(qvel_joint_names)
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(qpos_joint_names)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(qvel_joint_names)

    def _build_full_qpos_vector(self, dof_pos_12):
        """
        Convert 12-dim qpos vector into 16-dim control-order vector.

        Control order:
        [FL_hip, FR_hip,
         FL_shoulder, FR_shoulder,
         FL_leg, FR_leg,
         FL_wheel, FR_wheel,
         RL_hip, RR_hip,
         RL_shoulder, RR_shoulder,
         RL_leg, RR_leg,
         RL_wheel, RR_wheel]
        """
        q_full = np.zeros(self.action_dim, dtype=np.float64)

        q_full[0:2] = dof_pos_12[0:2]
        q_full[2:4] = dof_pos_12[2:4]
        q_full[4:6] = dof_pos_12[4:6]
        q_full[6:8] = 0.0

        q_full[8:10] = dof_pos_12[6:8]
        q_full[10:12] = dof_pos_12[8:10]
        q_full[12:14] = dof_pos_12[10:12]
        q_full[14:16] = 0.0

        return q_full

    def _build_control_vectors(self, action_scaled):
        kp_vec = np.zeros(self.action_dim, dtype=np.float64)
        kd_vec = np.zeros(self.action_dim, dtype=np.float64)
        td_vec = np.zeros(self.action_dim, dtype=np.float64)

        # hip
        kp_vec[[0, 1, 8, 9]] = self.kp_hip
        kd_vec[[0, 1, 8, 9]] = self.kd_hip

        # shoulder / leg are overwritten inside ControlManager, but fill for completeness
        kp_vec[[2, 3, 10, 11]] = self.kp_shoulder
        kd_vec[[2, 3, 10, 11]] = self.kd_shoulder

        kp_vec[[4, 5, 12, 13]] = self.kp_leg
        kd_vec[[4, 5, 12, 13]] = self.kd_leg

        # wheel: velocity control
        kd_vec[[6, 7, 14, 15]] = self.kd_wheel
        td_vec[[6, 7, 14, 15]] = action_scaled[[6, 7, 14, 15]]

        return kp_vec, kd_vec, td_vec

    def _clip_group_torques(self, tau):
        tau = np.asarray(tau, dtype=np.float64).copy()

        tau[[0, 1, 8, 9]] = np.clip(
            tau[[0, 1, 8, 9]],
            -self.config['hardware']['hip_max_torque'],
            self.config['hardware']['hip_max_torque'],
        )

        tau[[2, 3, 10, 11]] = np.clip(
            tau[[2, 3, 10, 11]],
            -self.config['hardware']['shoulder_max_torque'],
            self.config['hardware']['shoulder_max_torque'],
        )

        tau[[4, 5, 12, 13]] = np.clip(
            tau[[4, 5, 12, 13]],
            -self.config['hardware']['leg_max_torque'],
            self.config['hardware']['leg_max_torque'],
        )

        tau[[6, 7, 14, 15]] = np.clip(
            tau[[6, 7, 14, 15]],
            -self.config['hardware']['wheel_max_torque'],
            self.config['hardware']['wheel_max_torque'],
        )

        return tau

    def _get_obs(self):
        dof_pos = self.data.qpos[self.q_indices].copy()
        leg_pos = [4, 5, 10, 11]
        if self.use_gear:
            dof_pos[leg_pos] = dof_pos[leg_pos] * self.gear_ratio  # joint -> motor

        dof_vel = self.data.qvel[self.qd_indices].copy()
        leg_vel = [4, 5, 12, 13]
        if self.use_gear:
            dof_vel[leg_vel] = dof_vel[leg_vel] * self.gear_ratio  # joint -> motor (gamma X)

        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])
        projected_gravity = MathUtils.quat_to_base_vel(quat, np.array([0, 0, -1], dtype=np.double))

        if self.config["observation"]["height_map"] is not None:
            height_map = self.mujoco_utils.get_height_map(
                self.data, self.size_x, self.size_y, self.res_x, self.res_y
            )
        else:
            height_map = None

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
        projected_gravity_noisy = truncated_gaussian_noisy_data(
            projected_gravity,
            mean=self.sensor_noise_map["projected_gravity"]["mean"],
            std=self.sensor_noise_map["projected_gravity"]["std"],
            lower=self.sensor_noise_map["projected_gravity"]["lower"],
            upper=self.sensor_noise_map["projected_gravity"]["upper"],
        )

        if height_map is not None:
            height_map_noisy = truncated_gaussian_noisy_data(
                height_map,
                mean=self.sensor_noise_map["height_map"]["mean"],
                std=self.sensor_noise_map["height_map"]["std"],
                lower=self.sensor_noise_map["height_map"]["lower"],
                upper=self.sensor_noise_map["height_map"]["upper"],
            )
        else:
            height_map_noisy = np.zeros((0,), dtype=np.float32)

        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "projected_gravity": projected_gravity_noisy,
            "height_map": height_map_noisy,
            "last_action": self.action,
        }

    def step(self, action):
        self.action = np.asarray(action, dtype=np.float64)

        dof_pos = self.data.qpos[self.q_indices].copy()   # 12-dim
        dof_vel = self.data.qvel[self.qd_indices].copy()  # 16-dim

        q_full = self._build_full_qpos_vector(dof_pos)
        d_full = dof_vel.copy()
        if self.use_gear:
            # Convert leg joint state to motor space for controller input.
            leg_pos_idx = [4, 5, 12, 13]
            leg_vel_idx = [4, 5, 12, 13]
            q_full[leg_pos_idx] = q_full[leg_pos_idx] * self.gear_ratio
            d_full[leg_vel_idx] = d_full[leg_vel_idx] * self.gear_ratio
        action_scaled = self.action * self.action_scaler

        kp_vec, kd_vec, td_vec = self._build_control_vectors(action_scaled)

        # Default actuator behavior is PD unless overridden in actuator config.
        tau = self.control_manager.compute_torque(
            kp=kp_vec,
            tq=action_scaled,
            q=q_full,
            kd=kd_vec,
            td=td_vec,
            d=d_full,
        )

        self.applied_torques = self._clip_group_torques(tau)
        self.do_simulation(self.applied_torques, self.frame_skip)

        obs = self._get_obs()
        info = self._get_info()
        terminated = self._is_done()
        truncated = False

        self.prev_action = self.action.copy()
        self.local_step += 1

        return obs, terminated, truncated, info

    def _get_info(self):
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float32)
        joint_state = [
            dof_pos[0], dof_pos[1], dof_pos[2], dof_pos[3], dof_pos[4], dof_pos[5], dof_vel[6], dof_vel[7],
            dof_pos[6], dof_pos[7], dof_pos[8], dof_pos[9], dof_pos[10], dof_pos[11], dof_vel[14], dof_vel[15],
        ]

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "action_diff_RMSE": np.sqrt(np.mean((self.action - self.prev_action) ** 2)),
            "torque": self.applied_torques,
            "lin_vel_x": lin_vel[0],
            "lin_vel_y": lin_vel[1],
            "ang_vel_yaw": ang_vel[2],
            "set_points": self.action * self.action_scaler,
            "state": joint_state,
        }
        return info

    def _get_reset_info(self):
        return self._get_info()

    def _is_done(self):
        return False

    def reset_model(self):
        self.local_step = 0
        self.action = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)
        self.control_manager.reset()
        self.applied_torques = np.zeros(self.action_dim)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0

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
            joint_names=self.initial_joint_names,
        )

    def event(self, event: str, value):
        if event == 'push':
            raw_quat = self.data.qpos[3:7].astype(np.float64)
            R = MathUtils.quat_to_rot_matrix(raw_quat).T
            world_vel = np.array(value, dtype=np.float64).reshape(3,)
            robot_vel = R.dot(world_vel)
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
