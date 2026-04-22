import multiprocessing as mp
import queue
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


def _depth_worker(
    input_queue,
    output_queue,
    model_path: str,
    camera_name: str,
    frame_size,
    processing,
    preserve_all_frames: bool,
):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    width, height = frame_size
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
            depth_u8 = _normalize_depth_to_u8(filtered_m, max_range_m=max_range_m)

            payload = {
                "env_id": env_id,
                "camera_name": camera_name,
                "image": depth_u8,
                "resolution": f"{width}x{height}@60",
                "max_range_m": max_range_m,
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
        preserve_all_frames: bool = False,
        queue_size: int = 2,
    ):
        ctx = mp.get_context("spawn")
        queue_size = max(2, int(queue_size))
        self._input_queue = ctx.Queue(maxsize=queue_size)
        self._output_queue = ctx.Queue(maxsize=queue_size)
        self._preserve_all_frames = bool(preserve_all_frames)
        processing = dict(processing or {})
        self._process = ctx.Process(
            target=_depth_worker,
            args=(
                self._input_queue,
                self._output_queue,
                model_path,
                camera_name,
                frame_size,
                processing,
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
