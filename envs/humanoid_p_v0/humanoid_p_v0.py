from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw
from envs.humanoid_p_v0.manager.control_manager import ControlManager
from envs.humanoid_p_v0.manager.xml_manager import XMLManager
from envs.humanoid_p_v0.utils.math_utils import MathUtils
from envs.humanoid_p_v0.utils.mujoco_utils import MuJoCoUtils
from envs.humanoid_p_v0.utils.noise_generator_utils import truncated_gaussian_noisy_data
from envs.initial_pose import build_initial_qpos


class HumanoidPV0(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "humanoid_p_v0"
        self.config = config
        self.action_dim = int(config["hardware"]["action_dim"])
        default_action_scales = np.ones(self.action_dim, dtype=np.float64)
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple, np.ndarray)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)
        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD control parameters
        self.kp_hip_pitch = config["hardware"]["Kp_hip_pitch"]
        self.kp_torso = config["hardware"]["Kp_torso"]
        self.kp_hip_roll = config["hardware"]["Kp_hip_roll"]
        self.kp_shoulder_pitch = config["hardware"]["Kp_shoulder_pitch"]
        self.kp_hip_yaw = config["hardware"]["Kp_hip_yaw"]
        self.kp_shoulder_roll = config["hardware"]["Kp_shoulder_roll"]
        self.kp_knee = config["hardware"]["Kp_knee"]
        self.kp_shoulder_yaw = config["hardware"]["Kp_shoulder_yaw"]
        self.kp_ankle_pitch = config["hardware"]["Kp_ankle_pitch"]
        self.kp_elbow_pitch = config["hardware"]["Kp_elbow_pitch"]    
        self.kp_ankle_roll = config["hardware"]["Kp_ankle_roll"]  
        self.kp_elbow_yaw = config["hardware"]["Kp_elbow_yaw"]  

        self.kd_hip_pitch = config["hardware"]["Kd_hip_pitch"]
        self.kd_torso = config["hardware"]["Kd_torso"]
        self.kd_hip_roll = config["hardware"]["Kd_hip_roll"]
        self.kd_shoulder_pitch = config["hardware"]["Kd_shoulder_pitch"]
        self.kd_hip_yaw = config["hardware"]["Kd_hip_yaw"]
        self.kd_shoulder_roll = config["hardware"]["Kd_shoulder_roll"]
        self.kd_knee = config["hardware"]["Kd_knee"]
        self.kd_shoulder_yaw = config["hardware"]["Kd_shoulder_yaw"]
        self.kd_ankle_pitch = config["hardware"]["Kd_ankle_pitch"]
        self.kd_elbow_pitch = config["hardware"]["Kd_elbow_pitch"]    
        self.kd_ankle_roll = config["hardware"]["Kd_ankle_roll"]  
        self.kd_elbow_yaw = config["hardware"]["Kd_elbow_yaw"]       

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
            "dof_pos": 23,
            "dof_vel": 23,
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
        self.joint_names_in_order = ["left_hip_pitch_joint", "right_hip_pitch_joint",
                            "torso_joint",
                            "left_hip_roll_joint", "right_hip_roll_joint",
                            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                            "left_hip_yaw_joint", "right_hip_yaw_joint",
                            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                            "left_knee_joint", "right_knee_joint",
                            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                            "left_elbow_pitch_joint", "right_elbow_pitch_joint",
                            "left_ankle_roll_joint", "right_ankle_roll_joint",
                            "left_elbow_yaw_joint", "right_elbow_yaw_joint",]

        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(self.joint_names_in_order)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(self.joint_names_in_order)

    def _get_obs(self):
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float32)
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
        lin_vel_noisy = truncated_gaussian_noisy_data(lin_vel, mean=self.sensor_noise_map["lin_vel"]["mean"], std=self.sensor_noise_map["lin_vel"]["std"], lower=self.sensor_noise_map["lin_vel"]["lower"], upper=self.sensor_noise_map["lin_vel"]["upper"])
        projected_gravity_noisy = truncated_gaussian_noisy_data(projected_gravity, mean=self.sensor_noise_map["projected_gravity"]["mean"], std=self.sensor_noise_map["projected_gravity"]["std"], lower=self.sensor_noise_map["projected_gravity"]["lower"], upper=self.sensor_noise_map["projected_gravity"]["upper"])
        height_map_noisy = truncated_gaussian_noisy_data(height_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"], lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        
        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "lin_vel_x": lin_vel_noisy[0],
            "lin_vel_y": lin_vel_noisy[1],
            "lin_vel_z": lin_vel_noisy[2],
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

        # Extract joint positions and velocities from observation (order from Isaac Lab)
        pos_hip_pitch, vel_hip_pitch = dof_pos[0:2], dof_vel[0:2]
        pos_torso, vel_torso = dof_pos[2:3], dof_vel[2:3]
        pos_hip_roll, vel_hip_roll = dof_pos[3:5], dof_vel[3:5]
        pos_shoulder_pitch, vel_shoulder_pitch = dof_pos[5:7], dof_vel[5:7]
        pos_hip_yaw, vel_hip_yaw = dof_pos[7:9], dof_vel[7:9]
        pos_shoulder_roll, vel_shoulder_roll = dof_pos[9:11], dof_vel[9:11]
        pos_knee, vel_knee = dof_pos[11:13], dof_vel[11:13]
        pos_shoulder_yaw, vel_shoulder_yaw = dof_pos[13:15], dof_vel[13:15]
        pos_ankle_pitch, vel_ankle_pitch = dof_pos[15:17], dof_vel[15:17]
        pos_elbow_pitch, vel_elbow_pitch = dof_pos[17:19], dof_vel[17:19]
        pos_ankle_roll, vel_ankle_roll = dof_pos[19:21], dof_vel[19:21]
        pos_elbow_yaw, vel_elbow_yaw = dof_pos[21:23], dof_vel[21:23]

        # Get the scaled action
        hip_pitch_action_scaled = self.filtered_action[0:2] * self.action_scaler[0:2]
        torso_action_scaled = self.filtered_action[2:3] * self.action_scaler[2:3]
        hip_roll_action_scaled = self.filtered_action[3:5] * self.action_scaler[3:5]
        shoulder_pitch_action_scaled = self.filtered_action[5:7] * self.action_scaler[5:7]
        hip_yaw_action_scaled = self.filtered_action[7:9] * self.action_scaler[7:9]
        shoulder_roll_action_scaled = self.filtered_action[9:11] * self.action_scaler[9:11]
        knee_action_scaled = self.filtered_action[11:13] * self.action_scaler[11:13]
        shoulder_yaw_action_scaled = self.filtered_action[13:15] * self.action_scaler[13:15]
        ankle_pitch_action_scaled = self.filtered_action[15:17] * self.action_scaler[15:17]
        elbow_pitch_action_scaled = self.filtered_action[17:19] * self.action_scaler[17:19]
        ankle_roll_action_scaled = self.filtered_action[19:21] * self.action_scaler[19:21]
        elbow_yaw_action_scaled = self.filtered_action[21:23] * self.action_scaler[21:23]

        hip_pitch_torques = self.control_manager.pd_controller(self.kp_hip_pitch, hip_pitch_action_scaled, pos_hip_pitch, self.kd_hip_pitch, 0.0, vel_hip_pitch)
        torso_torques = self.control_manager.pd_controller(self.kp_torso, torso_action_scaled, pos_torso, self.kd_torso, 0.0, vel_torso)
        hip_roll_torques = self.control_manager.pd_controller(self.kp_hip_roll, hip_roll_action_scaled, pos_hip_roll, self.kd_hip_roll, 0.0, vel_hip_roll)
        shoulder_pitch_torques = self.control_manager.pd_controller(self.kp_shoulder_pitch, shoulder_pitch_action_scaled, pos_shoulder_pitch, self.kd_shoulder_pitch, 0.0, vel_shoulder_pitch)
        hip_yaw_torques = self.control_manager.pd_controller(self.kp_hip_yaw, hip_yaw_action_scaled, pos_hip_yaw, self.kd_hip_yaw, 0.0, vel_hip_yaw)
        shoulder_roll_torques = self.control_manager.pd_controller(self.kp_shoulder_roll, shoulder_roll_action_scaled, pos_shoulder_roll, self.kd_shoulder_roll, 0.0, vel_shoulder_roll)
        knee_torques = self.control_manager.pd_controller(self.kp_knee, knee_action_scaled, pos_knee, self.kd_knee, 0.0, vel_knee)
        shoulder_yaw_torques = self.control_manager.pd_controller(self.kp_shoulder_yaw, shoulder_yaw_action_scaled, pos_shoulder_yaw, self.kd_shoulder_yaw, 0.0, vel_shoulder_yaw)
        ankle_pitch_torques = self.control_manager.pd_controller(self.kp_ankle_pitch, ankle_pitch_action_scaled, pos_ankle_pitch, self.kd_ankle_pitch, 0.0, vel_ankle_pitch)
        elbow_pitch_torques = self.control_manager.pd_controller(self.kp_elbow_pitch, elbow_pitch_action_scaled, pos_elbow_pitch, self.kd_elbow_pitch, 0.0, vel_elbow_pitch)
        ankle_roll_torques = self.control_manager.pd_controller(self.kp_ankle_roll, ankle_roll_action_scaled, pos_ankle_roll, self.kd_ankle_roll, 0.0, vel_ankle_roll)
        elbow_yaw_torques = self.control_manager.pd_controller(self.kp_elbow_yaw, elbow_yaw_action_scaled, pos_elbow_yaw, self.kd_elbow_yaw, 0.0, vel_elbow_yaw)

        hip_pitch_torques_clipped = np.clip(hip_pitch_torques, -self.config['hardware']['hip_pitch_joint_max_torque'], self.config['hardware']['hip_pitch_joint_max_torque'])
        torso_torques_clipped = np.clip(torso_torques, -self.config['hardware']['torso_joint_max_torque'], self.config['hardware']['torso_joint_max_torque'])
        hip_roll_torques_clipped = np.clip(hip_roll_torques, -self.config['hardware']['hip_roll_joint_max_torque'], self.config['hardware']['hip_roll_joint_max_torque'])
        shoulder_pitch_torques_clipped = np.clip(shoulder_pitch_torques, -self.config['hardware']['shoulder_pitch_joint_max_torque'], self.config['hardware']['shoulder_pitch_joint_max_torque'])
        hip_yaw_torques_clipped = np.clip(hip_yaw_torques, -self.config['hardware']['hip_yaw_joint_max_torque'], self.config['hardware']['hip_yaw_joint_max_torque'])
        shoulder_roll_torques_clipped = np.clip(shoulder_roll_torques, -self.config['hardware']['shoulder_roll_joint_max_torque'], self.config['hardware']['shoulder_roll_joint_max_torque'])
        knee_torques_clipped = np.clip(knee_torques, -self.config['hardware']['knee_joint_max_torque'], self.config['hardware']['knee_joint_max_torque'])
        shoulder_yaw_torques_clipped = np.clip(shoulder_yaw_torques, -self.config['hardware']['shoulder_yaw_joint_max_torque'], self.config['hardware']['shoulder_yaw_joint_max_torque'])
        ankle_pitch_torques_clipped = np.clip(ankle_pitch_torques, -self.config['hardware']['ankle_pitch_joint_max_torque'], self.config['hardware']['ankle_pitch_joint_max_torque'])
        elbow_pitch_torques_clipped = np.clip(elbow_pitch_torques, -self.config['hardware']['elbow_pitch_joint_max_torque'], self.config['hardware']['elbow_pitch_joint_max_torque'])
        ankle_roll_torques_clipped = np.clip(ankle_roll_torques, -self.config['hardware']['ankle_roll_joint_max_torque'], self.config['hardware']['ankle_roll_joint_max_torque'])
        elbow_yaw_torques_clipped = np.clip(elbow_yaw_torques, -self.config['hardware']['elbow_yaw_joint_max_torque'], self.config['hardware']['elbow_yaw_joint_max_torque'])

        self.applied_torques = np.concatenate([hip_pitch_torques_clipped, torso_torques_clipped, hip_roll_torques_clipped, shoulder_pitch_torques_clipped,
        hip_yaw_torques_clipped, shoulder_roll_torques_clipped,  knee_torques_clipped, shoulder_yaw_torques_clipped, ankle_pitch_torques_clipped,
        elbow_pitch_torques_clipped, ankle_roll_torques_clipped, elbow_yaw_torques_clipped])
        
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
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        lin_vel = self.data.sensor("linear-velocity").data.astype(np.float32)
        joint_state = dof_pos

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
            joint_names=self.joint_names_in_order,
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
