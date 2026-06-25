import math
from functools import lru_cache

import mujoco
import numpy as np


D435I_DEPTH_WIDTH = 640
D435I_DEPTH_HEIGHT = 480
D435I_H_FOV_DEG = 87.0
D435I_V_FOV_DEG = 58.0


@lru_cache(maxsize=32)
def _camera_ray_directions(depth_width, depth_height, point_stride, h_fov_deg, v_fov_deg):
    depth_width = int(depth_width)
    depth_height = int(depth_height)
    point_stride = max(1, int(point_stride))
    fx = (float(depth_width) * 0.5) / math.tan(math.radians(float(h_fov_deg)) * 0.5)
    fy = (float(depth_height) * 0.5) / math.tan(math.radians(float(v_fov_deg)) * 0.5)
    cx = (float(depth_width) - 1.0) * 0.5
    cy = (float(depth_height) - 1.0) * 0.5

    u = np.arange(0, depth_width, point_stride, dtype=np.float64)
    v = np.arange(0, depth_height, point_stride, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    x = (uu.reshape(-1) - cx) / fx
    y = -((vv.reshape(-1) - cy) / fy)
    directions = np.column_stack((x, y, -np.ones_like(x)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def _raycast_ground(model, data, ground_geom_id, ground_geom_type, origin, direction, max_range):
    origin_col = np.asarray(origin, dtype=np.float64).reshape(3, 1)
    direction_col = np.asarray(direction, dtype=np.float64).reshape(3, 1)

    if ground_geom_id == -1:
        return None

    if ground_geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
        dist = mujoco.mj_rayHfield(model, data, ground_geom_id, origin_col, direction_col)
        if dist < 0.0 or dist > max_range:
            return None
        return origin + direction * dist

    if ground_geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        denom = direction[2]
        if abs(denom) < 1e-8:
            return None
        dist = -origin[2] / denom
        if dist < 0.0 or dist > max_range:
            return None
        return origin + direction * dist

    return None


def _camera_pose(model, data, camera_name, camera_body_name):
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id != -1 and hasattr(data, "cam_xpos") and hasattr(data, "cam_xmat"):
        return (
            np.asarray(data.cam_xpos[camera_id], dtype=np.float64),
            np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3),
        )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera_body_name)
    if body_id == -1:
        raise ValueError(f"Camera '{camera_name}' and body '{camera_body_name}' were not found.")
    return (
        np.asarray(data.xpos[body_id], dtype=np.float64),
        np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
    )


def build_camera_height_map(
    model,
    data,
    *,
    camera_name="depth_camera",
    camera_body_name="camera_link",
    grid_body_name="base_link",
    size_x=1.1,
    size_y=1.1,
    res_x=12,
    res_y=12,
    target_height=0.33,
    clipping_min=0.0,
    clipping_max=0.33,
    max_range=2.5,
    depth_width=D435I_DEPTH_WIDTH,
    depth_height=D435I_DEPTH_HEIGHT,
    point_stride=16,
    h_fov_deg=D435I_H_FOV_DEG,
    v_fov_deg=D435I_V_FOV_DEG,
    ground_geom_name="ground",
    return_valid_mask=False,
):
    """Build a D435i-like height map from camera rays projected onto the terrain."""
    res_x = int(res_x)
    res_y = int(res_y)
    point_stride = max(1, int(point_stride))
    target_height = float(target_height)
    clip_min = min(float(clipping_min), float(clipping_max))
    clip_max = max(float(clipping_min), float(clipping_max))

    camera_pos, R_world_camera = _camera_pose(model, data, camera_name, camera_body_name)

    grid_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grid_body_name)
    if grid_body_id == -1:
        raise ValueError(f"Body '{grid_body_name}' not found in model.")
    grid_pos = np.asarray(data.xpos[grid_body_id], dtype=np.float64)
    R_world_grid = np.asarray(data.xmat[grid_body_id], dtype=np.float64).reshape(3, 3)

    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name)
    ground_geom_type = int(model.geom_type[ground_geom_id]) if ground_geom_id != -1 else -1

    grid_min = np.full((res_y, res_x), np.inf, dtype=np.float64)
    valid_mask = np.zeros((res_y, res_x), dtype=bool)

    x_min = -float(size_x) * 0.5
    y_min = -float(size_y) * 0.5
    cell_x = float(size_x) / float(res_x)
    cell_y = float(size_y) / float(res_y)
    directions_camera = _camera_ray_directions(
        int(depth_width),
        int(depth_height),
        point_stride,
        float(h_fov_deg),
        float(v_fov_deg),
    )
    directions_world = directions_camera @ R_world_camera.T

    if ground_geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        denom = directions_world[:, 2]
        finite = np.abs(denom) >= 1e-8
        dist = np.full((directions_world.shape[0],), np.nan, dtype=np.float64)
        dist[finite] = -camera_pos[2] / denom[finite]
        valid = np.isfinite(dist) & (dist >= 0.0) & (dist <= float(max_range))
        hit_world = camera_pos[None, :] + directions_world[valid] * dist[valid, None]
        hit_grid = (R_world_grid.T @ (hit_world - grid_pos).T).T
        col = np.floor((hit_grid[:, 0] - x_min) / cell_x).astype(np.int64)
        row = np.floor((hit_grid[:, 1] - y_min) / cell_y).astype(np.int64)
        in_grid = (col >= 0) & (col < res_x) & (row >= 0) & (row < res_y)
        if np.any(in_grid):
            flat_idx = row[in_grid] * res_x + col[in_grid]
            height = -hit_grid[in_grid, 2]
            grid_flat = grid_min.reshape(-1)
            np.minimum.at(grid_flat, flat_idx, height)
            valid_mask.reshape(-1)[np.unique(flat_idx)] = True
    else:
        for direction_world in directions_world:
            hit_world = _raycast_ground(
                model,
                data,
                ground_geom_id,
                ground_geom_type,
                camera_pos,
                direction_world,
                float(max_range),
            )
            if hit_world is None:
                continue

            hit_grid = R_world_grid.T @ (hit_world - grid_pos)
            col = int(math.floor((hit_grid[0] - x_min) / cell_x))
            row = int(math.floor((hit_grid[1] - y_min) / cell_y))
            if col < 0 or col >= res_x or row < 0 or row >= res_y:
                continue

            height = -float(hit_grid[2])
            if height < grid_min[row, col]:
                grid_min[row, col] = height
                valid_mask[row, col] = True

    height_map = np.full((res_y, res_x), target_height, dtype=np.float64)
    height_map[valid_mask] = grid_min[valid_mask]
    height_map = np.clip(height_map, clip_min, clip_max)

    if return_valid_mask:
        return height_map.reshape(-1), valid_mask.reshape(-1)
    return height_map.reshape(-1)
