from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
import glfw
from envs.flamingo_light_p_v3.manager.control_manager import ControlManager
from envs.flamingo_light_p_v3.manager.xml_manager import XMLManager
from envs.flamingo_light_p_v3.utils.math_utils import MathUtils
from envs.flamingo_light_p_v3.utils.mujoco_utils import MuJoCoUtils
from envs.flamingo_light_p_v3.utils.noise_generator_utils import truncated_gaussian_noisy_data
from envs.initial_pose import build_initial_qpos
from envs.action_utils import normalize_action_clippings, scale_and_clip_action
from envs.camera_height_map import build_camera_height_map
from envs.masked_height_map import masked_height_map, parse_camera_fovs_from_xml


class FlamingoLightPV3(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(self, config, render_flag=True, render_mode='human'):
        # Set Basic Properties
        self.id = "flamingo_light_p_v3"
        self.config = config
        
        self.action_dim = int(config["hardware"]["action_dim"])

        default_action_scales = [1.0, 1.0, 20.0, 20.0]
        cfg_action_scales = config.get("action_scales", default_action_scales)
        if not isinstance(cfg_action_scales, (list, tuple)) or len(cfg_action_scales) != self.action_dim:
            cfg_action_scales = default_action_scales
        self.action_scaler = np.array(cfg_action_scales, dtype=np.float64)
        self.action_clip_min, self.action_clip_max = normalize_action_clippings(config, self.action_dim)
        self.render_mode = render_mode
        self.render_flag = render_flag

        # PD control parameters
        self.kp_shoulder = config["hardware"]["Kp_shoulder"]

        self.kd_shoulder = config["hardware"]["Kd_shoulder"]
        self.kd_wheel = config["hardware"]["Kd_wheel"]

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
        self.height_map_cameras = parse_camera_fovs_from_xml(self.model_path)

        # Height Map
        if self.config["observation"]["height_map"] is not None:
            self.size_x = self.config["observation"]["height_map"]["size_x"]
            self.size_y = self.config["observation"]["height_map"]["size_y"]
            self.res_x = self.config["observation"]["height_map"]["res_x"]
            self.res_y = self.config["observation"]["height_map"]["res_y"]
            self.height_map_target_height = float(
                self.config["observation"]["height_map"].get("target_height", 0.33)
            )
            self.height_map_clipping_min = float(
                self.config["observation"]["height_map"].get("clipping_min", 0.0)
            )
            self.height_map_clipping_max = float(
                self.config["observation"]["height_map"].get("clipping_max", 0.33)
            )
            self.camera_height_map_point_stride = int(
                self.config["observation"]["height_map"].get("point_stride", 16)
            )
            self.camera_height_map_max_range = float(
                self.config["observation"]["height_map"].get("max_range", 2.5)
            )
            self.camera_height_map_update_freq = float(
                self.config["observation"]["height_map"].get("camera_update_freq", 10.0)
            )
            self.height_map_debug_print = bool(
                self.config["observation"]["height_map"].get("debug_print", False)
            )
        else:
            self.res_x = 0
            self.res_y = 0
            self.height_map_target_height = 0.33
            self.height_map_clipping_min = 0.0
            self.height_map_clipping_max = 0.33
            self.camera_height_map_point_stride = 16
            self.camera_height_map_max_range = 2.5
            self.camera_height_map_update_freq = 10.0
            self.height_map_debug_print = False
        if self.height_map_clipping_min > self.height_map_clipping_max:
            self.height_map_clipping_min, self.height_map_clipping_max = (
                self.height_map_clipping_max,
                self.height_map_clipping_min,
            )
        self.camera_height_map_update_interval = max(
            1,
            int(round(self.control_freq / max(1e-6, self.camera_height_map_update_freq))),
        )
        self._camera_height_map_cache = None
        self._camera_height_map_cache_step = -1

        # Set dimensions of observations
        self.obs_to_dim = {
            "dof_pos": 2,
            "dof_vel": 4,
            "ang_vel": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "projected_gravity": 3,
            "last_action": self.action_dim,
            "height_map": int(self.res_x * self.res_y),
            "masked_height_map": int(self.res_x * self.res_y),
            "camera_height_map": int(self.res_x * self.res_y),
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
        qpos_joint_names = ["left_shoulder_joint", "right_shoulder_joint"]
        qvel_joint_names = ["left_shoulder_joint", "right_shoulder_joint", "left_wheel_joint", "right_wheel_joint"]
        self.initial_joint_names = list(qvel_joint_names)
        self.q_indices = self.mujoco_utils.get_qpos_joint_indices_by_name(qpos_joint_names)
        self.qd_indices = self.mujoco_utils.get_qvel_joint_indices_by_name(qvel_joint_names)

    def _debug_print_height_maps(self, height_map, masked_height_map, fov_valid_mask=None):
        if not self.height_map_debug_print:
            return
        interval = max(1, int(round(self.control_freq)))
        if self.local_step % interval != 0:
            return
        raw = np.asarray(height_map, dtype=np.float64).reshape(self.res_y, self.res_x)
        masked = np.asarray(masked_height_map, dtype=np.float64).reshape(self.res_y, self.res_x)
        print(f"[{self.id}] height_map debug t={self.local_step / self.control_freq:.2f}s")
        print(
            "settings: "
            f"target_height={self.height_map_target_height:.3f}, "
            f"clipping_min={self.height_map_clipping_min:.3f}, "
            f"clipping_max={self.height_map_clipping_max:.3f}"
        )
        if fov_valid_mask is not None:
            valid_count = int(np.count_nonzero(fov_valid_mask))
            print(f"fov_valid_count: {valid_count}/{int(self.res_x * self.res_y)}")
        print("raw height_map:")
        print(np.array2string(raw, precision=3, suppress_small=True))
        print("masked_height_map:")
        print(np.array2string(masked, precision=3, suppress_small=True))

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
            height_map, height_points_w = self.mujoco_utils.get_height_map(
                self.data, self.size_x, self.size_y, self.res_x, self.res_y, return_points=True
            )
            raw_height_map = height_map
            height_map = np.clip(raw_height_map, self.height_map_clipping_min, self.height_map_clipping_max)
            masked_height_source = np.clip(raw_height_map, 0.0, self.height_map_target_height)
            masked_map, fov_valid_mask = masked_height_map(
                self.model,
                self.data,
                masked_height_source,
                height_points_w,
                self.height_map_cameras,
                base_height=float(self.data.qpos[2]),
                offset=0.5,
                fill_value=self.height_map_target_height,
                return_valid_mask=True,
            )
            masked_map = np.clip(masked_map, 0.0, self.height_map_target_height)
            if (
                self._camera_height_map_cache is None
                or self.local_step - self._camera_height_map_cache_step >= self.camera_height_map_update_interval
            ):
                self._camera_height_map_cache = build_camera_height_map(
                    self.model,
                    self.data,
                    camera_name="depth_camera",
                    camera_body_name="F_camera_link",
                    grid_body_name="base_link",
                    size_x=self.size_x,
                    size_y=self.size_y,
                    res_x=self.res_x,
                    res_y=self.res_y,
                    target_height=self.height_map_target_height,
                    clipping_min=self.height_map_clipping_min,
                    clipping_max=self.height_map_clipping_max,
                    max_range=self.camera_height_map_max_range,
                    point_stride=self.camera_height_map_point_stride,
                )
                self._camera_height_map_cache_step = self.local_step
            camera_map = self._camera_height_map_cache
            self.mujoco_utils.color_heightmap_by_mask(fov_valid_mask, self.res_x, self.res_y)
        else:
            height_map = None
            masked_map = None
            camera_map = None

        dof_pos_noisy = truncated_gaussian_noisy_data(dof_pos, mean=self.sensor_noise_map["dof_pos"]["mean"], std=self.sensor_noise_map["dof_pos"]["std"], lower=self.sensor_noise_map["dof_pos"]["lower"], upper=self.sensor_noise_map["dof_pos"]["upper"])
        dof_vel_noisy = truncated_gaussian_noisy_data(dof_vel, mean=self.sensor_noise_map["dof_vel"]["mean"], std=self.sensor_noise_map["dof_vel"]["std"], lower=self.sensor_noise_map["dof_vel"]["lower"], upper=self.sensor_noise_map["dof_vel"]["upper"])
        ang_vel_noisy = truncated_gaussian_noisy_data(ang_vel, mean=self.sensor_noise_map["ang_vel"]["mean"], std=self.sensor_noise_map["ang_vel"]["std"], lower=self.sensor_noise_map["ang_vel"]["lower"], upper=self.sensor_noise_map["ang_vel"]["upper"])
        lin_vel_noisy = truncated_gaussian_noisy_data(lin_vel, mean=self.sensor_noise_map["lin_vel"]["mean"], std=self.sensor_noise_map["lin_vel"]["std"], lower=self.sensor_noise_map["lin_vel"]["lower"], upper=self.sensor_noise_map["lin_vel"]["upper"])
        projected_gravity_noisy = truncated_gaussian_noisy_data(projected_gravity, mean=self.sensor_noise_map["projected_gravity"]["mean"], std=self.sensor_noise_map["projected_gravity"]["std"], lower=self.sensor_noise_map["projected_gravity"]["lower"], upper=self.sensor_noise_map["projected_gravity"]["upper"])
        height_map_noisy = truncated_gaussian_noisy_data(height_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"], lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        masked_height_map_noisy = truncated_gaussian_noisy_data(masked_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"], lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        camera_height_map_noisy = truncated_gaussian_noisy_data(camera_map, mean=self.sensor_noise_map["height_map"]["mean"], std=self.sensor_noise_map["height_map"]["std"], lower=self.sensor_noise_map["height_map"]["lower"], upper=self.sensor_noise_map["height_map"]["upper"])
        if height_map is not None and masked_map is not None:
            self._debug_print_height_maps(height_map, masked_map, fov_valid_mask)
        
        return {
            "dof_pos": dof_pos_noisy,
            "dof_vel": dof_vel_noisy,
            "ang_vel": ang_vel_noisy,
            "lin_vel_x": lin_vel_noisy[0],
            "lin_vel_y": lin_vel_noisy[1],
            "lin_vel_z": lin_vel_noisy[2],
            "projected_gravity": projected_gravity_noisy,
            "height_map": height_map_noisy,
            "masked_height_map": masked_height_map_noisy,
            "camera_height_map": camera_height_map_noisy,
            "last_action": self.action
        }
    
    def step(self, action):
        self.action = action
        self.filtered_action = self.control_manager.delay_filter(action)
        # Pull the current joint positions and velocities
        dof_pos = self.data.qpos[self.q_indices]
        dof_vel = self.data.qvel[self.qd_indices]

        # Extract joint positions and velocities from observation
        pos_shoulder = dof_pos[0:2]
        vel_shoulder = dof_vel[0:2]
        vel_wheel = dof_vel[2:4]
        
        action_scaled = scale_and_clip_action(self.action, self.action_scaler, self.action_clip_min, self.action_clip_max)
        shoulder_action_scaled = action_scaled[0:2]
        wheel_action_scaled = action_scaled[2:4]

        shoulder_torques = self.control_manager.pd_controller(self.kp_shoulder, shoulder_action_scaled, pos_shoulder, self.kd_shoulder, 0.0, vel_shoulder)
        wheel_torques = self.control_manager.pd_controller(0.0, 0.0, 0.0, self.kd_wheel, wheel_action_scaled, vel_wheel)

        shoulder_torques_clipped = np.clip(shoulder_torques, -self.config['hardware']['leg_max_torque'], self.config['hardware']['leg_max_torque'])
        wheel_torques_clipped = np.clip(wheel_torques, -self.config['hardware']['wheel_max_torque'], self.config['hardware']['wheel_max_torque'])

        self.applied_torques = np.concatenate([shoulder_torques_clipped, wheel_torques_clipped])

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
        joint_state = [dof_pos[0], dof_pos[1], dof_vel[2], dof_vel[3]]

        info = {
            "dt": self.dt_ * self.frame_skip,
            "action": self.action,
            "action_diff_RMSE": np.sqrt(np.mean((self.action - self.prev_action)**2)),
            "torque": self.applied_torques,
            "lin_vel_x": lin_vel[0],
            "lin_vel_y": lin_vel[1],
            "ang_vel_yaw": ang_vel[2],
            "set_points": scale_and_clip_action(self.action, self.action_scaler, self.action_clip_min, self.action_clip_max),
            "state": joint_state
        }
        return info

    def _get_reset_info(self):
        info = self._get_info()
        return info

    def _is_done(self):
        # Get the IDs of the bodies of interest
        body_ids = self.mujoco_utils.get_body_indices_by_name([]) # "base_link", "left_leg_link", "right_leg_link"

        # Iterate through all active contacts in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get the body indices from the geometry indices involved in the contact
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]

            # Check if either of the bodies in this contact is one of the bodies of interest
            if body1_id in body_ids or body2_id in body_ids:
                # If a contact involves one of the interested bodies, return True to indicate simulation should be reset
                return True

        # If no relevant contact is found, return False
        return False

    def reset_model(self):
        self.local_step = 0
        self._camera_height_map_cache = None
        self._camera_height_map_cache_step = -1
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
