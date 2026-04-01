import warnings

import mujoco
import numpy as np
from envs.flamingo_p_v3.utils.math_utils import MathUtils


class MuJoCoUtils:
    def __init__(self, model):
        self.model = model
        self.hf_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.site_ids = None

    def get_body_indices_by_name(self, body_names):
        """
        Get the indices of bodies for given body names.

        Args:
            model: MuJoCo mjModel instance.
            body_names: List of body names to fetch indices for.

        Returns:
            body_indices: List of body indices corresponding to body names.
        """
        body_indices = []
        for body_name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"Body name '{body_name}' not found in the model.")
            body_indices.append(body_id)
        return body_indices

    def get_qpos_joint_indices_by_name(self, joint_names):
        """
        Get the qpos indices for the given joint names.

        Args:
            model: MuJoCo mjModel instance.
            joint_names: List of joint names to look up.

        Returns:
            qpos_indices: List of qpos indices corresponding to the given joints.
        """
        
        qpos_indices = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint name '{joint_name}' not found in the model.")
            # Fetch qpos and qvel indices
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
        return qpos_indices

    def get_qvel_joint_indices_by_name(self, joint_names):
        """
        Get the qvel indices for the given joint names.

        Args:
            model: MuJoCo mjModel instance.
            joint_names: List of joint names to look up.

        Returns:
            qvel_indices: List of qvel indices corresponding to the given joints.
        """
        qvel_indices = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint name '{joint_name}' not found in the model.")
            # Fetch qpos and qvel indices
            qvel_indices.append(self.model.jnt_dofadr[joint_id])
        return qvel_indices
    
    def init_heightmap_visualization(self, res_x, res_y):
        """
        Initialize site IDs for heightmap visualization.

        This method should be called after the MuJoCo model is loaded (and before the first simulation step).
        It finds all site IDs whose names follow the pattern "heightmap_site_i_j" for i in [0, res_y) and j in [0, res_x).
        The result is stored in self.site_ids as a 2D list of shape [res_x][res_x], where each entry is the integer site ID
        corresponding to that grid cell.

        Args:
            res_x (int): Number of columns in the heightmap grid.
            res_y (int): Number of rows in the heightmap grid.

        Raises:
            ValueError: If any expected site name is not found in the model's XML.
        """
        self.site_ids = [[None for _ in range(res_x)] for _ in range(res_y)]
        for i in range(res_y):
            for j in range(res_x):
                name = f"heightmap_site_{i}_{j}"
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid == -1:
                    raise ValueError(f"Site '{name}' not found in model. Check that the XML defines this site.")
                self.site_ids[i][j] = sid

    def get_height_map(self, data, size_x, size_y, res_x, res_y):
        """
        Generate a heightmap by raycasting from the robot's base frame onto the ground.

        For each grid cell in a (res_x × res_y) window centered on the robot, this function:
          1. Computes the 3D position P_world of the grid point in world coordinates (using the robot's pose).
          2. Casts a ray straight downward from height z_max_world above P_world.
          3. Uses mj_rayHfield to measure distance to the heightfield (ground).
          4. Computes the terrain height and calculates the difference relative to the robot's base height.
          5. Issues a warning if no intersection is found (assigning a fallback value z_min_world).
          6. Updates the corresponding visualization site’s position and appearance.

        Args:
            data: MuJoCo mjData instance containing the current simulation state (including qpos).
            size_x (float): Width of the heightmap window in meters (robot’s local frame).
            size_y (float): Depth of the heightmap window in meters (robot’s local frame).
            res_x (int): Number of columns (sampling points) along the x dimension.
            res_y (int): Number of rows (sampling points) along the y dimension.

        Returns:
            numpy.ndarray: A 1D array of length (res_x * res_y), containing the height difference
                           (robot_z − terrain_z) for each grid cell, flattened row-major.

        Raises:
            RuntimeError: If init_heightmap_visualization has not been called (self.site_ids is None).
        """
        if self.site_ids is None:
            raise RuntimeError(
                "Heightmap visualization sites not initialized. Call init_heightmap_visualization(res_x, res_x) first."
            )

        # Extract robot base position (x, y, z) and orientation quaternion [w, x, y, z]
        robot_pos = data.qpos[0:3].astype(np.float64)
        raw_quat = data.qpos[3:7].astype(np.float64)
        R = MathUtils.quat_to_rot_matrix(raw_quat)  # 3×3 rotation matrix

        # Define the local window in the robot's frame
        x_min_robot, x_max_robot = -size_x / 2, size_x / 2
        y_min_robot, y_max_robot = -size_y / 2, size_y / 2
        num_x, num_y = res_x, res_y

        # Ray parameters
        z_max_world = 10.0  # Starting height for ray above the ground
        z_min_world = -1.0  # Fallback height if no intersection

        # Create meshgrid in robot's local XY plane
        x_robot = np.linspace(x_min_robot, x_max_robot, num_x, dtype=np.float64)
        y_robot = np.linspace(y_min_robot, y_max_robot, num_y, dtype=np.float64)
        XX_robot, YY_robot = np.meshgrid(x_robot, y_robot)
        heightmap = np.zeros((num_y, num_x), dtype=np.float64)

        for i in range(num_y):
            for j in range(num_x):
                # Local point in robot frame
                P_robot = np.array([XX_robot[i, j], YY_robot[i, j], 0.0], dtype=np.float64)
                # Transform to world coordinates
                P_world = robot_pos + R.dot(P_robot)

                # Ray origin: above the terrain point by z_max_world
                pnt = np.array(
                    [
                        [P_world[0]],
                        [P_world[1]],
                        [P_world[2] + z_max_world],
                    ],
                    dtype=np.float64,
                )
                # Ray direction: straight down
                vec = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)

                # Perform raycast against heightfield
                dist = mujoco.mj_rayHfield(self.model, data, self.hf_geom_id, pnt, vec)

                if dist >= 0.0:
                    # Terrain height = ray_origin_z − dist
                    terrain_height = pnt[2, 0] - dist
                    heightmap[i, j] = robot_pos[2] - terrain_height
                else:
                    # No intersection → fallback value + warning
                    terrain_height = z_min_world
                    heightmap[i, j] = robot_pos[2] - z_min_world
                    warnings.warn("No intersection with heightfield!")

                # Update visualization site to the terrain contact point
                sid = self.site_ids[i][j]
                data.site_xpos[sid][0] = P_world[0]
                data.site_xpos[sid][1] = P_world[1]
                data.site_xpos[sid][2] = terrain_height
                self.model.site_size[sid][0] = 0.01
                self.model.site_rgba[sid][3] = 0.6

        return heightmap.flatten()