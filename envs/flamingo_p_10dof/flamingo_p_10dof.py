from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw
from envs.flamingo_p_10dof.manager.control_manager import ControlManager
from envs.flamingo_p_10dof.manager.xml_manager import XMLManager
from envs.flamingo_p_10dof.utils.math_utils import MathUtils
from envs.flamingo_p_10dof.utils.mujoco_utils import MuJoCoUtils
from envs.flamingo_p_10dof.utils.noise_generator_utils import truncated_gaussian_noisy_data
from envs.initial_pose import build_initial_qpos


class FlamingoP10dof(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "flamingo_p_10dof"
        self.config = config
        self.action_dim = int(config["hardware"]["action_dim"])
        self.has_wheels = (self.action_dim >= 10)
        self.leg_joint_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_leg_joint", "right_leg_joint",
        ]
        self.wheel_joint_names = ["left_wheel_joint", "right_wheel_joint"]
        self.joint_names_in_order = list(self.leg_joint_names) + (list(self.wheel_joint_names) if self.has_wheels else [])
        self.actuator_joint_order = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_leg_joint",
            "left_wheel_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_leg_joint",
            "right_wheel_joint",
        ]
        self._joint_order_index = {name: idx for idx, name in enumerate(self.joint_names_in_order)}

        # action scaler: 8 leg joints + optional 2 wheels
        leg_scaler = [1.0] * len(self.leg_joint_names)
        wheel_scaler = [40.0, 40.0] if self.has_wheels else []
        default_action_scales = leg_scaler + wheel_scaler
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)

        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD control parameters
        self.kp_hip_pitch = config["hardware"]["Kp_hip_pitch"]
        self.kp_hip_roll = config["hardware"]["Kp_hip_roll"]
        self.kp_hip_yaw = config["hardware"]["Kp_hip_yaw"]
        self.kp_leg = config["hardware"]["Kp_leg"]

        self.kd_hip_pitch = config["hardware"]["Kd_hip_pitch"]
        self.kd_hip_roll = config["hardware"]["Kd_hip_roll"]
        self.kd_hip_yaw = config["hardware"]["Kd_hip_yaw"]
        self.kd_leg = config["hardware"]["Kd_leg"]
        self.kd_wheel = config["hardware"]["Kd_wheel"]
        self.leg_torque_limits = np.asarray(
            [
                config["hardware"]["left_hip_pitch_joint_max_torque"],
                config["hardware"]["right_hip_pitch_joint_max_torque"],
                config["hardware"]["left_hip_roll_joint_max_torque"],
                config["hardware"]["right_hip_roll_joint_max_torque"],
                config["hardware"]["left_hip_yaw_joint_max_torque"],
                config["hardware"]["right_hip_yaw_joint_max_torque"],
                config["hardware"]["left_leg_joint_max_torque"],
                config["hardware"]["right_leg_joint_max_torque"],
            ],
            dtype=np.float64,
        )
        self.wheel_torque_limits = np.asarray(
            [
                config["hardware"]["left_wheel_joint_max_torque"],
                config["hardware"]["right_wheel_joint_max_torque"],
            ],
            dtype=np.float64,
        ) if self.has_wheels else np.zeros(0, dtype=np.float64)

        # Set Simulation Properties
        precision_level = self.config["random"]["precision"]
        sensor_noise_level = self.config["random"]["sensor_noise"]
        self.init_noise = self.config["random"]["init_noise"]
        self.dt_ = config["random_table"]["precision"][precision_level]["timestep"]
        self.frame_skip = config["random_table"]["precision"][precision_level]["frame_skip"]
        self.sensor_noise_map = config["random_table"]["sensor_noise"][sensor_noise_level]
        self.control_freq = 1 / (self.dt_ * self.frame_skip)
        assert self.control_freq == 50, "Currently, only control frequency of 50 is supported."
        self.local_step = 0

        # Set Placeholders
        self.action = np.zeros(self.action_dim)
        self.filtered_action = np.zeros(self.action_dim)
        self.prev_action = np.zeros(self.action_dim)
        self.applied_torques = np.zeros(self.action_dim)
        self.viewer = None
        self.mode = None

        # Domain Randomization
        self.xml_manager = XMLManager(config, has_wheels=self.has_wheels, use_gear=False)
        self.model_path = self.xml_manager.get_model_path()

        # Height Map
        if self.config["observation"]["height_map"] is not None:
            self.size_x = self.config["observation"]["height_map"]["size_x"]
            self.size_y = self.config["observation"]["height_map"]["size_y"]
            self.res_x = self.config["observation"]["height_map"]["res_x"]
            self.res_y = self.config["observation"]["height_map"]["res_y"]
        else:
            self.res_x = 0
            self.res_y = 0

        # Set dimensions of observations
        self.obs_to_dim = {
            "dof_pos": len(self.leg_joint_names),
            "dof_vel": len(self.joint_names_in_order),
            "ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
            "last_action": self.action_dim,
            "height_map": int(self.res_x * self.res_y),
        }

        # Set MuJoCo Wrapper
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(low=-np.inf, high=np.inf, shape=(sum(self.obs_to_dim.values()),), dtype=np.float32,),
            render_mode=self.render_mode if render_flag else None,
        )

        # Set other Managers and Helpers
        self.control_manager = ControlManager(config)
        self.mujoco_utils = MuJoCoUtils(self.model)
        self.mujoco_utils.init_heightmap_visualization(self.res_x, self.res_y)

        self.initial_joint_names = list(self.joint_names_in_order)
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(self.leg_joint_names)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(self.joint_names_in_order)

    def _to_actuator_order(self, values):
        ordered = []
        for joint_name in self.actuator_joint_order:
            if not self.has_wheels and "wheel" in joint_name:
                continue
            ordered.append(values[self._joint_order_index[joint_name]])
        return np.asarray(ordered, dtype=np.float64)

    def _get_obs(self):
        dof_pos = self.data.qpos[self.q_indices].copy()
        dof_vel = self.data.qvel[self.qd_indices].copy()

        ang_vel = self.data.sensor("angular_velocity").data.astype(np.double)
        lin_vel = self.data.sensor("linear_velocity").data.astype(np.float32)
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])
        projected_gravity = MathUtils.quat_to_base_vel(quat, np.array([0, 0, -1], dtype=np.double))

        if self.config["observation"]["height_map"] is not None:
            height_map = self.mujoco_utils.get_height_map(self.data, self.size_x, self.size_y, self.res_x, self.res_y)
        else:
            height_map = None

        dof_pos_noisy = truncated_gaussian_noisy_data(dof_pos, **self.sensor_noise_map["dof_pos"])
        dof_vel_noisy = truncated_gaussian_noisy_data(dof_vel, **self.sensor_noise_map["dof_vel"])
        ang_vel_noisy = truncated_gaussian_noisy_data(ang_vel, **self.sensor_noise_map["ang_vel"])
        lin_vel_noisy = truncated_gaussian_noisy_data(lin_vel, **self.sensor_noise_map["lin_vel"])
        projected_gravity_noisy = truncated_gaussian_noisy_data(projected_gravity, **self.sensor_noise_map["projected_gravity"])
        height_map_noisy = truncated_gaussian_noisy_data(height_map, **self.sensor_noise_map["height_map"])
        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "lin_vel_x": lin_vel_noisy[0],
            "lin_vel_y": lin_vel_noisy[1],
            "lin_vel_z": lin_vel_noisy[2],
            "projected_gravity": projected_gravity_noisy,
            "height_map": height_map_noisy,
            "last_action": self.action,
        }

    def step(self, action):
        self.action = action
        self.filtered_action = self.control_manager.delay_filter(action)

        # Current joint states
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]
        pos_hip_pitch = dof_pos[0:2]
        pos_hip_roll = dof_pos[2:4]
        pos_hip_yaw = dof_pos[4:6]
        pos_leg = dof_pos[6:8]

        vel_hip_pitch = dof_vel[0:2]
        vel_hip_roll = dof_vel[2:4]
        vel_hip_yaw = dof_vel[4:6]
        vel_leg = dof_vel[6:8]

        hip_pitch_action = self.filtered_action[0:2] * self.action_scaler[0:2]
        hip_roll_action = self.filtered_action[2:4] * self.action_scaler[2:4]
        hip_yaw_action = self.filtered_action[4:6] * self.action_scaler[4:6]
        leg_action = self.filtered_action[6:8] * self.action_scaler[6:8]

        hip_pitch_torques = self.control_manager.pd_controller(self.kp_hip_pitch, hip_pitch_action, pos_hip_pitch, self.kd_hip_pitch, 0.0, vel_hip_pitch)
        hip_roll_torques = self.control_manager.pd_controller(self.kp_hip_roll, hip_roll_action, pos_hip_roll, self.kd_hip_roll, 0.0, vel_hip_roll)
        hip_yaw_torques = self.control_manager.pd_controller(self.kp_hip_yaw, hip_yaw_action, pos_hip_yaw, self.kd_hip_yaw, 0.0, vel_hip_yaw)
        leg_torques = self.control_manager.pd_controller(self.kp_leg, leg_action, pos_leg, self.kd_leg, 0.0, vel_leg)

        leg_torques_all = np.concatenate([
            hip_pitch_torques,
            hip_roll_torques,
            hip_yaw_torques,
            leg_torques,
        ])
        leg_torques_clipped = np.clip(leg_torques_all, -self.leg_torque_limits, self.leg_torque_limits)

        torques_list = [leg_torques_clipped]

        # If wheels exist, add wheel control (pure D + FF)
        if self.has_wheels:
            vel_wheel = dof_vel[8:10]
            wheel_action_scaled = self.filtered_action[8:10] * self.action_scaler[8:10]
            wheel_torques = self.control_manager.pd_controller(0.0, 0.0, 0.0, self.kd_wheel, wheel_action_scaled, vel_wheel)
            wheel_torques_clipped = np.clip(wheel_torques, -self.wheel_torque_limits, self.wheel_torque_limits)
            torques_list.append(wheel_torques_clipped)

        self.applied_torques = np.concatenate(torques_list)
        sim_torques = self._to_actuator_order(self.applied_torques)
        self.do_simulation(sim_torques, self.frame_skip)

        obs = self._get_obs()
        info = self._get_info()
        terminated = self._is_done()
        truncated = False

        self.prev_action = self.action
        self.local_step += 1
        return obs, terminated, truncated, info

    def _get_info(self):
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]
        ang_vel = self.data.sensor("angular_velocity").data.astype(np.double)
        lin_vel = self.data.sensor("linear_velocity").data.astype(np.float32)
        joint_state = list(dof_pos[:8])
        if self.has_wheels:
            joint_state.extend(list(dof_vel[8:10]))

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
        contact_forces = self.data.cfrc_ext[1:12]  # External contact forces
        base_contact = contact_forces[0] > 1.0
        hip_l_contact = contact_forces[1] > 1.0
        hip_r_contact = contact_forces[6] > 1.0
        hip_roll_l_contact = contact_forces[2] > 1.0
        hip_roll_r_contact = contact_forces[7] > 1.0
        contact = (
            base_contact.any()
            or hip_l_contact.any()
            or hip_r_contact.any()
            or hip_roll_l_contact.any()
            or hip_roll_r_contact.any()
        )
        return contact

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

        return self._get_obs()

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
            raw_quat = self.data.qpos[3:7].astype(np.float64)           # [w, x, y, z]
            R = MathUtils.quat_to_rot_matrix(raw_quat).T                # World-to-local rotation matrix (3×3)
            world_vel = np.array(value, dtype=np.float64).reshape(3,)   # Velocity in world frame
            robot_vel = R.dot(world_vel)                                # Transform to robot-frame velocity
            self.data.qvel[:2] = robot_vel[:2]  # xy: robot frame
            self.data.qvel[2] = world_vel[2]    #  z: world frame
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
