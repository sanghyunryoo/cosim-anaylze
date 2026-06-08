import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class CameraFOV:
    role: str
    camera_name: str
    body_name: str
    h_fov_deg: float
    v_fov_deg: float
    min_depth: float
    max_depth: float
    h_fov_margin_deg: float
    v_fov_margin_deg: float


def _role_from_body_name(body_name: str, index: int) -> str:
    lowered = body_name.lower()
    if lowered.startswith(("f_", "front")) or "front" in lowered:
        return "front"
    if lowered.startswith(("r_", "rear")) or "rear" in lowered:
        return "rear"
    if "camera" in lowered and index == 0:
        return "front"
    return f"camera_{index}"


def parse_camera_fovs_from_xml(
    xml_path: str,
    *,
    default_h_fov_deg: float = 87.0,
    default_min_depth: float = 0.1,
    default_max_depth: float = 2.5,
    default_h_fov_margin_deg: float = 0.0,
    default_v_fov_margin_deg: float = 9.0,
) -> List[CameraFOV]:
    """Parse MuJoCo camera bodies into Isaac-style camera FOV metadata."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)

    root = ET.parse(xml_path).getroot()
    cameras = []
    for body in root.iter("body"):
        body_name = body.attrib.get("name", "")
        for camera in body.findall("camera"):
            camera_name = camera.attrib.get("name", f"{body_name}_camera")
            v_fov_deg = float(camera.attrib.get("fovy", 58.0))
            idx = len(cameras)
            cameras.append(
                CameraFOV(
                    role=_role_from_body_name(body_name, idx),
                    camera_name=camera_name,
                    body_name=body_name,
                    h_fov_deg=float(camera.attrib.get("h_fov", default_h_fov_deg)),
                    v_fov_deg=v_fov_deg,
                    min_depth=float(camera.attrib.get("min_depth", default_min_depth)),
                    max_depth=float(camera.attrib.get("max_depth", default_max_depth)),
                    h_fov_margin_deg=float(camera.attrib.get("h_fov_margin", default_h_fov_margin_deg)),
                    v_fov_margin_deg=float(camera.attrib.get("v_fov_margin", default_v_fov_margin_deg)),
                )
            )
    return cameras


def camera_fov_valid_mask(model, data, hit_points_w: np.ndarray, cameras: Iterable[CameraFOV]) -> np.ndarray:
    """Return True where any XML camera can observe the world-space hit point."""
    import mujoco

    points = np.asarray(hit_points_w, dtype=np.float64).reshape(-1, 3)
    valid = np.zeros((points.shape[0],), dtype=bool)

    for camera in cameras:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.camera_name)
        if camera_id != -1 and hasattr(data, "cam_xpos") and hasattr(data, "cam_xmat"):
            cam_pos_w = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
            R_world_camera = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        else:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera.body_name)
            if body_id == -1:
                continue
            cam_pos_w = np.asarray(data.xpos[body_id], dtype=np.float64)
            R_world_camera = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)

        if not np.all(np.isfinite(cam_pos_w)) or not np.all(np.isfinite(R_world_camera)):
            continue

        points_c = (R_world_camera.T @ (points - cam_pos_w).T).T

        # MuJoCo cameras render along local -Z; use the actual camera element pose,
        # including the camera-local quat, so the FOV follows the rendered depth image.
        depth = -points_c[:, 2]
        in_depth = (depth >= camera.min_depth) & (depth <= camera.max_depth)
        safe_depth = np.maximum(depth, 1e-8)

        half_h = math.radians(camera.h_fov_deg) * 0.5 + math.radians(camera.h_fov_margin_deg)
        half_v = math.radians(camera.v_fov_deg) * 0.5 + math.radians(camera.v_fov_margin_deg)
        in_h = np.abs(np.arctan2(points_c[:, 0], safe_depth)) <= half_h
        in_v = np.abs(np.arctan2(points_c[:, 1], safe_depth)) <= half_v
        valid |= in_depth & in_h & in_v

    return valid


def masked_height_map(
    model,
    data,
    height_map: np.ndarray,
    hit_points_w: np.ndarray,
    cameras: Iterable[CameraFOV],
    *,
    base_height: float,
    offset: float = 0.5,
    fill_value: float | None = None,
    return_valid_mask: bool = False,
) -> np.ndarray:
    """Apply an Isaac RayCasterFOV-style camera visibility mask to a height map."""
    height = np.asarray(height_map, dtype=np.float64).reshape(-1)
    hit_points = np.asarray(hit_points_w, dtype=np.float64).reshape(-1, 3)
    if height.shape[0] != hit_points.shape[0]:
        raise ValueError("height_map and hit_points_w must have the same number of cells.")

    valid = camera_fov_valid_mask(model, data, hit_points, cameras)
    mask_fill_value = float(base_height) - float(offset) if fill_value is None else float(fill_value)
    masked = np.where(valid, height, mask_fill_value).astype(np.float64)
    if return_valid_mask:
        return masked, valid
    return masked
