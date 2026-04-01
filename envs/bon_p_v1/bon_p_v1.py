from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw
from envs.bon_p_v1.manager.control_manager import ControlManager
from envs.bon_p_v1.manager.xml_manager import XMLManager
from envs.bon_p_v1.utils.math_utils import MathUtils
from envs.bon_p_v1.utils.mujoco_utils import MuJoCoUtils
from envs.bon_p_v1.utils.noise_generator_utils import truncated_gaussian_noisy_data
from envs.initial_pose import build_initial_qpos


class BonPV1(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "bon_p_v1"
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
        
        # PD control parameters
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
        self.xml_manager = XMLManager(config)
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
            "dof_pos": 12,
            "dof_vel": 16,
            "ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
            "last_action": self.action_dim,
            "height_map": int(self.res_x * self.res_y)
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

        # Set Indices of q and qd
        qpos_joint_names = [
        'FL_hip_joint', 'FR_hip_joint',
        'FL_shoulder_joint', 'FR_shoulder_joint',
        'FL_leg_joint', 'FR_leg_joint',
        'RL_hip_joint', 'RR_hip_joint',
        'RL_shoulder_joint', 'RR_shoulder_joint',
        'RL_leg_joint', 'RR_leg_joint'
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
        
    def _get_obs(self):
        dof_pos = self.data.qpos[self.q_indices].copy()
        leg_pos = [4,5,10,11]
        dof_pos[leg_pos] = dof_pos[leg_pos] * self.gear_ratio if self.use_gear else dof_pos[leg_pos]  # Joint space -> Motor space
        
        dof_vel = self.data.qvel[self.qd_indices]
        leg_vel=[4,5,12,13]
        dof_vel[leg_vel] = dof_vel[leg_vel] * self.gear_ratio * self.gamma if self.use_gear else dof_vel[leg_vel]  # Joint space -> Motor space
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        if np.all(quat == 0):
            quat = np.array([0, 0, 0, 1])
        projected_gravity = MathUtils.quat_to_base_vel(quat, np.array([0, 0, -1], dtype=np.double))

        if self.config["observation"]["height_map"] is not None:
            height_map = self.mujoco_utils.get_height_map(self.data, self.size_x, self.size_y, self.res_x, self.res_y) 
        else:
            height_map = None

        dof_pos_noisy = truncated_gaussian_noisy_data(dof_pos, mean=self.sensor_noise_map["dof_pos"]["mean"], std=self.sensor_noise_map["dof_pos"]["std"], lower=self.sensor_noise_map["dof_pos"]["lower"], upper=self.sensor_noise_map["dof_pos"]["upper"])
        dof_vel_noisy = truncated_gaussian_noisy_data(dof_vel, mean=self.sensor_noise_map["dof_vel"]["mean"], std=self.sensor_noise_map["dof_vel"]["std"], lower=self.sensor_noise_map["dof_vel"]["lower"], upper=self.sensor_noise_map["dof_vel"]["upper"])
        ang_vel_noisy = truncated_gaussian_noisy_data(ang_vel, mean=self.sensor_noise_map["ang_vel"]["mean"], std=self.sensor_noise_map["ang_vel"]["std"], lower=self.sensor_noise_map["ang_vel"]["lower"], upper=self.sensor_noise_map["ang_vel"]["upper"])
        projected_gravity_noisy = truncated_gaussian_noisy_data(projected_gravity, mean=self.sensor_noise_map["projected_gravity"]["mean"], std=self.sensor_noise_map["projected_gravity"]["std"], lower=self.sensor_noise_map["projected_gravity"]["lower"], upper=self.sensor_noise_map["projected_gravity"]["upper"])
        height_map_noisy = truncated_gaussian_noisy_data(height_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"], lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        
        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "projected_gravity": projected_gravity_noisy,
            "height_map": height_map_noisy,
            "last_action": self.action
        }

    def step(self, action):
        self.action = action
        self.filtered_action = self.control_manager.delay_filter(action)

        # Pull the current joint positions and velocities
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]

        # Extract joint positions and velocities from observation
        f_pos_hip = dof_pos[0:2]
        f_pos_shoulder = dof_pos[2:4]
        f_pos_leg = dof_pos[4:6] * self.gear_ratio if self.use_gear else dof_pos[4:6]  # Joint space -> Motor space
        r_pos_hip = dof_pos[6:8]
        r_pos_shoulder = dof_pos[8:10]
        r_pos_leg = dof_pos[10:12] * self.gear_ratio if self.use_gear else dof_pos[10:12]  # Joint space -> Motor space

        f_vel_hip = dof_vel[0:2]
        f_vel_shoulder = dof_vel[2:4]
        f_vel_leg = dof_vel[4:6] * self.gear_ratio * self.gamma if self.use_gear else dof_vel[4:6]  # Joint space -> Motor space
        f_vel_wheel = dof_vel[6:8]
        r_vel_hip = dof_vel[8:10]
        r_vel_shoulder = dof_vel[10:12]
        r_vel_leg = dof_vel[12:14] * self.gear_ratio * self.gamma if self.use_gear else dof_vel[12:14]  # Joint space -> Motor space
        r_vel_wheel = dof_vel[14:16]

        f_hip_action_scaled = self.filtered_action[0:2] * self.action_scaler[0:2]
        f_shoulder_action_scaled = self.filtered_action[2:4] * self.action_scaler[2:4]
        f_leg_action_scaled = self.filtered_action[4:6] * self.action_scaler[4:6]
        f_wheel_action_scaled = self.filtered_action[6:8] * self.action_scaler[6:8]
        r_hip_action_scaled = self.filtered_action[8:10] * self.action_scaler[8:10]
        r_shoulder_action_scaled = self.filtered_action[10:12] * self.action_scaler[10:12]
        r_leg_action_scaled = self.filtered_action[12:14] * self.action_scaler[12:14]
        r_wheel_action_scaled = self.filtered_action[14:16] * self.action_scaler[14:16]


        f_hip_torques = self.control_manager.pd_controller(self.kp_hip, f_hip_action_scaled, f_pos_hip, self.kd_hip, 0.0, f_vel_hip)
        r_hip_torques = self.control_manager.pd_controller(self.kp_hip, r_hip_action_scaled, r_pos_hip, self.kd_hip, 0.0, r_vel_hip)
        f_shoulder_torques = self.control_manager.pd_controller(self.kp_shoulder, f_shoulder_action_scaled, f_pos_shoulder, self.kd_shoulder, 0.0, f_vel_shoulder)
        r_shoulder_torques = self.control_manager.pd_controller(self.kp_shoulder, r_shoulder_action_scaled, r_pos_shoulder, self.kd_shoulder, 0.0, r_vel_shoulder)
        f_leg_torques = self.control_manager.pd_controller(self.kp_leg, f_leg_action_scaled, f_pos_leg, self.kd_leg, 0.0, f_vel_leg)
        r_leg_torques = self.control_manager.pd_controller(self.kp_leg, r_leg_action_scaled, r_pos_leg, self.kd_leg, 0.0, r_vel_leg)
        f_leg_torques = f_leg_torques * np.full(2, self.gamma, dtype=np.float64) if self.use_gear else f_leg_torques
        r_leg_torques = r_leg_torques * np.full(2, self.gamma, dtype=np.float64) if self.use_gear else r_leg_torques
        f_wheel_torques = self.control_manager.pd_controller(0.0, 0.0, 0.0, self.kd_wheel, f_wheel_action_scaled, f_vel_wheel)
        r_wheel_torques = self.control_manager.pd_controller(0.0, 0.0, 0.0, self.kd_wheel, r_wheel_action_scaled, r_vel_wheel)

        f_hip_torques_clipped = np.clip(f_hip_torques, -self.config['hardware']['hip_max_torque'], self.config['hardware']['hip_max_torque'])
        r_hip_torques_clipped = np.clip(r_hip_torques, -self.config['hardware']['hip_max_torque'], self.config['hardware']['hip_max_torque'])
        f_shoulder_torques_clipped = np.clip(f_shoulder_torques, -self.config['hardware']['shoulder_max_torque'], self.config['hardware']['shoulder_max_torque'])
        r_shoulder_torques_clipped = np.clip(r_shoulder_torques, -self.config['hardware']['shoulder_max_torque'], self.config['hardware']['shoulder_max_torque'])
        f_leg_torques_clipped = np.clip(f_leg_torques, -self.config['hardware']['leg_max_torque'], self.config['hardware']['leg_max_torque'])
        r_leg_torques_clipped = np.clip(r_leg_torques, -self.config['hardware']['leg_max_torque'], self.config['hardware']['leg_max_torque'])
        f_wheel_torques_clipped = np.clip(f_wheel_torques, -self.config['hardware']['wheel_max_torque'], self.config['hardware']['wheel_max_torque'])
        r_wheel_torques_clipped = np.clip(r_wheel_torques, -self.config['hardware']['wheel_max_torque'], self.config['hardware']['wheel_max_torque'])

        torques_list = [f_hip_torques_clipped, f_shoulder_torques_clipped, f_leg_torques_clipped, f_wheel_torques_clipped, r_hip_torques_clipped, r_shoulder_torques_clipped, r_leg_torques_clipped, r_wheel_torques_clipped]
        
        self.applied_torques = np.concatenate(torques_list)
        self.do_simulation(self.applied_torques, self.frame_skip)

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
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float32)
        joint_state = [dof_pos[0], dof_pos[1], dof_pos[2], dof_pos[3], dof_pos[4], dof_pos[5], dof_vel[6], dof_vel[7], dof_pos[6], dof_pos[7], dof_pos[8], dof_pos[9], dof_pos[10], dof_pos[11], dof_vel[14], dof_vel[15]
        ]

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "action_diff_RMSE": np.sqrt(np.mean((self.action - self.prev_action)**2)),
            "torque": self.applied_torques,
            "lin_vel_x": lin_vel[0],
            "lin_vel_y": lin_vel[1],
            "ang_vel_yaw": ang_vel[2],
            "set_points": self.action * self.action_scaler,
            "state": joint_state
        }
        return info

    def _get_reset_info(self):
        info = self._get_info()
        return info

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
            # Assume value is given in world frame
            # Convert this to robot-frame
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
        super().close()  # Call the parent class's close method to ensure everything is properly closed

