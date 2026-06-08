import warnings
import mujoco
import numpy as np
from envs.wheeldog_p_v2.utils.math_utils import MathUtils


class MuJoCoUtils:
    def __init__(self, model):
        self.model = model
        self.hf_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.ground_geom_type = (
            int(self.model.geom_type[self.hf_geom_id]) if self.hf_geom_id != -1 else -1
        )
        self.site_ids_by_prefix = {}

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
    
    def init_heightmap_visualization(self, res_x, res_y, prefix="heightmap_site"):
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
        site_ids = [[None for _ in range(res_x)] for _ in range(res_y)]
        for i in range(res_y):
            for j in range(res_x):
                name = f"{prefix}_{i}_{j}"
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid == -1:
                    raise ValueError(f"Site '{name}' not found in model. Check that the XML defines this site.")
                site_ids[i][j] = sid
        self.site_ids_by_prefix[prefix] = site_ids

    def color_heightmap_by_mask(
        self,
        valid_mask,
        res_x,
        res_y,
        prefix="heightmap_site",
        valid_rgba=(0.0, 1.0, 0.0, 0.7),
        invalid_rgba=(1.0, 1.0, 1.0, 0.7),
    ):
        site_ids = self.site_ids_by_prefix.get(prefix)
        if site_ids is None:
            raise RuntimeError(
                f"Heightmap visualization sites not initialized for prefix '{prefix}'."
            )

        mask = np.asarray(valid_mask, dtype=bool).reshape(int(res_y), int(res_x))
        for i in range(int(res_y)):
            for j in range(int(res_x)):
                sid = site_ids[i][j]
                self.model.site_rgba[sid][:] = valid_rgba if mask[i, j] else invalid_rgba

    def get_height_map(
        self,
        data,
        x_forward,
        x_backward,
        y_left,
        y_right,
        res_x,
        res_y,
        frame_body_name="base_link",
        site_prefix="heightmap_site",
        axis_body_name=None,
        return_points=False,
    ):
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
            x_forward (float): Forward extent from the origin in meters.
            x_backward (float): Backward extent from the origin in meters.
            y_left (float): Left extent from the origin in meters.
            y_right (float): Right extent from the origin in meters.
            res_x (int): Number of columns (sampling points) along the x dimension.
            res_y (int): Number of rows (sampling points) along the y dimension.

        Returns:
            numpy.ndarray: A 1D array of length (res_x * res_y), containing the height difference
                           (robot_z − terrain_z) for each grid cell, flattened row-major.

        Raises:
            RuntimeError: If init_heightmap_visualization has not been called (self.site_ids is None).
        """
        site_ids = self.site_ids_by_prefix.get(site_prefix)
        if site_ids is None:
            raise RuntimeError(
                f"Heightmap visualization sites not initialized for prefix '{site_prefix}'."
            )
        if self.hf_geom_id == -1:
            heightmap = np.zeros((int(res_y), int(res_x)), dtype=np.float64)
            hit_points = np.zeros((int(res_y), int(res_x), 3), dtype=np.float64)
            if return_points:
                return heightmap.flatten(), hit_points.reshape(-1, 3)
            return heightmap.flatten()

        origin_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_body_name)
        if origin_body_id == -1:
            raise ValueError(f"Body '{frame_body_name}' not found in model.")
        frame_pos = np.asarray(data.xpos[origin_body_id], dtype=np.float64)

        axis_body_id = -1
        if axis_body_name:
            axis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, axis_body_name)
        if axis_body_id == -1:
            axis_body_id = origin_body_id
        R_axes = np.asarray(data.xmat[axis_body_id], dtype=np.float64).reshape(3, 3)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        forward_axis = R_axes[:, 0].copy()
        forward_axis[2] = 0.0
        forward_norm = np.linalg.norm(forward_axis)
        if forward_norm < 1e-8:
            forward_axis = R_axes[:, 1].copy()
            forward_axis[2] = 0.0
            forward_norm = np.linalg.norm(forward_axis)
        if forward_norm < 1e-8:
            forward_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward_axis /= forward_norm

        lateral_axis = np.cross(world_up, forward_axis)
        lateral_norm = np.linalg.norm(lateral_axis)
        if lateral_norm < 1e-8:
            lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            lateral_axis /= lateral_norm

        # Define the local window in the robot's frame
        x_min_robot, x_max_robot = -float(x_backward), float(x_forward)
        y_min_robot, y_max_robot = -float(y_right), float(y_left)
        num_x, num_y = res_x, res_y

        # Ray parameters
        z_max_world = 10.0  # Starting height for ray above the ground
        z_min_world = -1.0  # Fallback height if no intersection

        # Create meshgrid in robot's local XY plane
        x_robot = np.linspace(x_min_robot, x_max_robot, num_x, dtype=np.float64)
        y_robot = np.linspace(y_min_robot, y_max_robot, num_y, dtype=np.float64)
        XX_robot, YY_robot = np.meshgrid(x_robot, y_robot)
        heightmap = np.zeros((num_y, num_x), dtype=np.float64)
        hit_points = np.zeros((num_y, num_x, 3), dtype=np.float64)

        for i in range(num_y):
            for j in range(num_x):
                # Local point in robot frame
                P_world = frame_pos + (forward_axis * XX_robot[i, j]) + (lateral_axis * YY_robot[i, j])

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

                if self.ground_geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                    dist = mujoco.mj_rayHfield(self.model, data, self.hf_geom_id, pnt, vec)
                    if dist >= 0.0:
                        terrain_height = pnt[2, 0] - dist
                        heightmap[i, j] = frame_pos[2] - terrain_height
                    else:
                        terrain_height = z_min_world
                        heightmap[i, j] = frame_pos[2] - z_min_world
                        warnings.warn("No intersection with heightfield!")
                elif self.ground_geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    terrain_height = 0.0
                    heightmap[i, j] = frame_pos[2] - terrain_height
                else:
                    terrain_height = z_min_world
                    heightmap[i, j] = frame_pos[2] - z_min_world
                    warnings.warn("Unsupported ground geom type for height map visualization.")

                # Update visualization site to the terrain contact point
                sid = site_ids[i][j]
                data.site_xpos[sid][0] = P_world[0]
                data.site_xpos[sid][1] = P_world[1]
                data.site_xpos[sid][2] = terrain_height
                hit_points[i, j] = [P_world[0], P_world[1], terrain_height]
                self.model.site_size[sid][0] = 0.01
                self.model.site_rgba[sid][3] = 0.6

        if return_points:
            return heightmap.flatten(), hit_points.reshape(-1, 3)
        return heightmap.flatten()
