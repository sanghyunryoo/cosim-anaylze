import multiprocessing as mp
import queue
import math
from typing import Optional

import mujoco
import numpy as np


def _downsample_nearest(image: np.ndarray, magnitude: int) -> np.ndarray:
    if magnitude <= 1:
        return image
    return image[::magnitude, ::magnitude]


def _upsample_nearest(image: np.ndarray, target_shape) -> np.ndarray:
    target_h, target_w = target_shape
    if image.shape == (target_h, target_w):
        return image
    scale_y = max(1, int(np.ceil(target_h / image.shape[0])))
    scale_x = max(1, int(np.ceil(target_w / image.shape[1])))
    up = np.repeat(np.repeat(image, scale_y, axis=0), scale_x, axis=1)
    return up[:target_h, :target_w]


def _apply_spatial_filter(depth_mm: np.ndarray, magnitude: int, alpha: float, delta_mm: float) -> np.ndarray:
    filtered = depth_mm.copy()
    for _ in range(max(0, int(magnitude))):
        acc = filtered.copy()
        weight = np.ones_like(filtered, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(filtered, shift=(dy, dx), axis=(0, 1))
            similar = np.abs(shifted - filtered) <= delta_mm
            acc += np.where(similar, shifted, 0.0)
            weight += similar.astype(np.float32)
        neighborhood = acc / np.maximum(weight, 1.0)
        filtered = alpha * neighborhood + (1.0 - alpha) * filtered
    return filtered


def _apply_temporal_filter(current_mm: np.ndarray, previous_mm: Optional[np.ndarray], alpha: float, delta_mm: float) -> np.ndarray:
    if previous_mm is None or previous_mm.shape != current_mm.shape:
        return current_mm
    similar = np.abs(current_mm - previous_mm) <= delta_mm
    blended = alpha * current_mm + (1.0 - alpha) * previous_mm
    return np.where(similar, blended, current_mm)


def _normalize_depth_to_u8(depth_m: np.ndarray, max_range_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_range_m)
    if not np.any(valid):
        return np.zeros_like(depth_m, dtype=np.uint8)
    clipped = np.where(valid, depth_m, max_range_m)
    normalized = np.clip(clipped / max_range_m, 0.0, 1.0)
    return ((1.0 - normalized) * 255.0).astype(np.uint8)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
            (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
            (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
            (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half = angle_rad * 0.5
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _rotate_image_nearest(image: np.ndarray, angle_deg: float, fill_value: float) -> np.ndarray:
    if abs(float(angle_deg)) < 1e-6:
        return image
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    x = xx - cx
    y = yy - cy
    src_x = (cos_a * x) + (sin_a * y) + cx
    src_y = (-sin_a * x) + (cos_a * y) + cy
    src_x_nn = np.rint(src_x).astype(np.int32)
    src_y_nn = np.rint(src_y).astype(np.int32)
    valid = (src_x_nn >= 0) & (src_x_nn < w) & (src_y_nn >= 0) & (src_y_nn < h)
    rotated = np.full_like(image, fill_value=fill_value)
    rotated[valid] = image[src_y_nn[valid], src_x_nn[valid]]
    return rotated


def _apply_edge_noise(depth_m: np.ndarray, ratio: float, max_range_m: float, rng) -> np.ndarray:
    gx = np.abs(np.diff(depth_m, axis=1, append=depth_m[:, -1:]))
    gy = np.abs(np.diff(depth_m, axis=0, append=depth_m[-1:, :]))
    grad = np.maximum(gx, gy)
    threshold = max(0.02, float(np.nanpercentile(grad, 85)))
    edge_mask = grad >= threshold
    if not np.any(edge_mask):
        return depth_m
    perturb_mask = edge_mask & (rng.random(depth_m.shape) < max(0.0, float(ratio)))
    noisy = depth_m.copy()
    delta = rng.uniform(-0.08, 0.08, size=depth_m.shape).astype(np.float32)
    noisy[perturb_mask] = np.clip(noisy[perturb_mask] + delta[perturb_mask], 0.0, max_range_m)
    return noisy


def _apply_small_objects(depth_m: np.ndarray, area_ratio: float, count: int, max_range_m: float, rng) -> np.ndarray:
    h, w = depth_m.shape
    noisy = depth_m.copy()
    max_count = max(0, int(count))
    if max_count <= 0:
        return noisy
    budget = max(1, int(h * w * max(0.0, float(area_ratio))))
    remaining = budget
    for _ in range(rng.integers(1, max_count + 1)):
        if remaining <= 0:
            break
        rect_h = int(rng.integers(2, max(3, h // 8 + 1)))
        rect_w = int(rng.integers(2, max(3, w // 8 + 1)))
        rect_h = min(rect_h, h)
        rect_w = min(rect_w, w)
        if rect_h * rect_w > remaining:
            scale = math.sqrt(remaining / max(1, rect_h * rect_w))
            rect_h = max(1, int(rect_h * scale))
            rect_w = max(1, int(rect_w * scale))
        y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
        x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
        patch = noisy[y0:y0 + rect_h, x0:x0 + rect_w]
        if patch.size == 0:
            continue
        if rng.random() < 0.5:
            value = float(rng.uniform(0.05, min(max_range_m, 0.8)))
        else:
            value = max_range_m
        noisy[y0:y0 + rect_h, x0:x0 + rect_w] = value
        remaining -= rect_h * rect_w
    return noisy


def _apply_spot_noise(depth_m: np.ndarray, ratio: float, max_range_m: float, rng) -> np.ndarray:
    h, w = depth_m.shape
    noisy = depth_m.copy()
    count = max(1, int(h * w * max(0.0, float(ratio))))
    ys = rng.integers(0, h, size=count)
    xs = rng.integers(0, w, size=count)
    values = rng.uniform(0.0, max_range_m, size=count).astype(np.float32)
    noisy[ys, xs] = values
    return noisy


def _apply_depth_randomization(depth_m: np.ndarray, max_range_m: float, randomization, rng) -> np.ndarray:
    if not bool(randomization.get("enabled", False)):
        return depth_m

    randomized = depth_m.copy()

    if rng.random() < float(randomization.get("gaussian_prob", 0.0)):
        stddev = max(0.0, float(randomization.get("gaussian_stddev", 0.0)))
        randomized += rng.normal(0.0, stddev, size=randomized.shape).astype(np.float32)

    if rng.random() < float(randomization.get("rotation_prob", 0.0)):
        max_deg = abs(float(randomization.get("rotation_deg", 0.0)))
        randomized = _rotate_image_nearest(
            randomized,
            angle_deg=float(rng.uniform(-max_deg, max_deg)),
            fill_value=max_range_m,
        )

    if rng.random() < float(randomization.get("edge_noise_prob", 0.0)):
        randomized = _apply_edge_noise(
            randomized,
            ratio=float(randomization.get("edge_noise_ratio", 0.0)),
            max_range_m=max_range_m,
            rng=rng,
        )

    if rng.random() < float(randomization.get("small_object_prob", 0.0)):
        randomized = _apply_small_objects(
            randomized,
            area_ratio=float(randomization.get("small_object_ratio", 0.0)),
            count=int(randomization.get("small_object_count", 0)),
            max_range_m=max_range_m,
            rng=rng,
        )

    if rng.random() < float(randomization.get("spot_noise_prob", 0.0)):
        randomized = _apply_spot_noise(
            randomized,
            ratio=float(randomization.get("spot_noise_ratio", 0.0)),
            max_range_m=max_range_m,
            rng=rng,
        )

    randomized = np.where(np.isfinite(randomized), randomized, max_range_m)
    return np.clip(randomized, 0.0, max_range_m)


def _apply_camera_randomization(model, camera_name: str, randomization, rng):
    if not bool(randomization.get("enabled", False)):
        return
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        return

    xyz_shift = abs(float(randomization.get("camera_xyz_shift_m", 0.0)))
    pitch_shift_deg = abs(float(randomization.get("camera_pitch_shift_deg", 0.0)))
    fov_shift_deg = abs(float(randomization.get("camera_fov_shift_deg", 0.0)))

    if xyz_shift > 0.0:
        model.cam_pos[camera_id] = np.asarray(model.cam_pos[camera_id], dtype=np.float64) + rng.uniform(
            -xyz_shift, xyz_shift, size=3
        )

    if pitch_shift_deg > 0.0:
        base_quat = np.asarray(model.cam_quat[camera_id], dtype=np.float64)
        delta_quat = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), math.radians(rng.uniform(-pitch_shift_deg, pitch_shift_deg)))
        model.cam_quat[camera_id] = _quat_mul(base_quat, delta_quat)

    if fov_shift_deg > 0.0 and hasattr(model, "cam_fovy"):
        model.cam_fovy[camera_id] = float(model.cam_fovy[camera_id]) + float(rng.uniform(-fov_shift_deg, fov_shift_deg))


def _depth_worker(
    input_queue,
    output_queue,
    model_path: str,
    camera_name: str,
    frame_size,
    processing,
    randomization,
    preserve_all_frames: bool,
):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    width, height = frame_size
    rng = np.random.default_rng()
    randomization = dict(randomization or {})
    _apply_camera_randomization(model, camera_name=camera_name, randomization=randomization, rng=rng)
    renderer = mujoco.Renderer(model, height=height, width=width)
    prev_filtered_mm = None

    try:
        while True:
            item = input_queue.get()
            if item is None:
                break

            if not preserve_all_frames:
                # Drain queued states so depth always follows the latest snapshot.
                while True:
                    try:
                        newer = input_queue.get_nowait()
                    except queue.Empty:
                        break
                    if newer is None:
                        item = None
                        break
                    item = newer

            if item is None:
                break

            qpos = np.asarray(item["qpos"], dtype=np.float64)
            qvel = np.asarray(item["qvel"], dtype=np.float64)
            env_id = str(item.get("env_id", "robot"))

            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera_name)
            renderer.enable_depth_rendering()
            depth = np.asarray(renderer.render(), dtype=np.float32)
            renderer.disable_depth_rendering()

            max_range_m = float(processing["max_range_m"])
            depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, max_range_m)
            depth = np.clip(depth, 0.0, max_range_m)

            decimation_magnitude = int(processing["decimation_magnitude"])
            spatial_magnitude = int(processing["spatial_magnitude"])
            spatial_alpha = float(processing["spatial_alpha"])
            spatial_delta = float(processing["spatial_delta"])
            temporal_alpha = float(processing["temporal_alpha"])
            temporal_delta = float(processing["temporal_delta"])

            raw_depth_mm = depth * 1000.0
            decimated_mm = _downsample_nearest(raw_depth_mm, decimation_magnitude)
            spatial_mm = _apply_spatial_filter(
                decimated_mm,
                magnitude=spatial_magnitude,
                alpha=spatial_alpha,
                delta_mm=spatial_delta,
            )
            temporal_mm = _apply_temporal_filter(
                spatial_mm,
                previous_mm=prev_filtered_mm,
                alpha=temporal_alpha,
                delta_mm=temporal_delta,
            )
            prev_filtered_mm = temporal_mm.copy()
            filtered_mm = _upsample_nearest(temporal_mm, raw_depth_mm.shape)
            filtered_m = filtered_mm / 1000.0
            filtered_m = _apply_depth_randomization(
                filtered_m,
                max_range_m=max_range_m,
                randomization=randomization,
                rng=rng,
            )
            depth_u8 = _normalize_depth_to_u8(filtered_m, max_range_m=max_range_m)

            payload = {
                "env_id": env_id,
                "camera_name": camera_name,
                "image": depth_u8,
                "resolution": f"{width}x{height}@60",
                "max_range_m": max_range_m,
                "randomized": bool(randomization.get("enabled", False)),
            }
            if not preserve_all_frames:
                try:
                    while True:
                        output_queue.get_nowait()
                except queue.Empty:
                    pass
            output_queue.put(payload)
    finally:
        try:
            renderer.close()
        except Exception:
            pass


class DepthStreamClient:
    def __init__(
        self,
        model_path: str,
        camera_name: str = "depth_camera",
        frame_size=(640, 480),
        processing=None,
        randomization=None,
        preserve_all_frames: bool = False,
        queue_size: int = 2,
    ):
        ctx = mp.get_context("spawn")
        queue_size = max(2, int(queue_size))
        self._input_queue = ctx.Queue(maxsize=queue_size)
        self._output_queue = ctx.Queue(maxsize=queue_size)
        self._preserve_all_frames = bool(preserve_all_frames)
        processing = dict(processing or {})
        randomization = dict(randomization or {})
        self._process = ctx.Process(
            target=_depth_worker,
            args=(
                self._input_queue,
                self._output_queue,
                model_path,
                camera_name,
                frame_size,
                processing,
                randomization,
                self._preserve_all_frames,
            ),
            daemon=True,
        )
        self._process.start()

    def submit(self, env_id: str, qpos, qvel):
        payload = {
            "env_id": str(env_id),
            "qpos": np.asarray(qpos, dtype=np.float64),
            "qvel": np.asarray(qvel, dtype=np.float64),
        }
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full:
            if self._preserve_all_frames:
                return False
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(payload)
            except queue.Full:
                return False
        return True

    def poll(self) -> Optional[dict]:
        latest = None
        while True:
            try:
                latest = self._output_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def poll_all(self):
        payloads = []
        while True:
            try:
                payloads.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return payloads

    def close(self):
        try:
            self._input_queue.put_nowait(None)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
