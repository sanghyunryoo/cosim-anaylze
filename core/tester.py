import os
from collections import deque

import glfw
import numpy as np
import mujoco
from core.policy import build_policy
from core.depth_streamer import DepthStreamClient
from core.heightmap_dataset_writer import HeightMapDatasetWriter
from core.reporter import Reporter
from envs.build import build_env
from PyQt5.QtCore import QObject, pyqtSignal


class Tester(QObject):
    finished = pyqtSignal()
    stepFinished = pyqtSignal()
    overlayUpdated = pyqtSignal(dict)
    depthUpdated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.user_command = None
        self._push_event = False
        self._render_window_patched = False
        self._stop = False
        self._had_error = False
        self.policy = None
        self.policy_path = None
        self.encoder_path = None
        self._pending_fine_tune_bias = None
        self._pending_fine_tune_enabled = None
        self._pending_fine_tune_max_samples = None
        self._monitor_joint_names = []
        self._monitor_history = {}
        self._monitor_session_history = {}
        self._monitor_history_len = 90
        self._depth_enabled = False
        self._dataset_enabled = False
        self._height_map_enabled = False
        self._depth_stream = None
        self._dataset_writer = None
        self._dataset_warned = False
        self._height_map_cfg = {}
        self._depth_camera_name = "depth_camera"
        self._depth_update_interval = 30
        self._depth_gui_emit_interval = 30
        self._depth_base_frame_size = (640, 480)
        self._depth_frame_size = (80, 60)
        self._depth_scale = 8
        self._depth_processing = {
            "max_range_m": 2.5,
            "decimation_magnitude": 3,
            "spatial_magnitude": 2,
            "spatial_alpha": 0.5,
            "spatial_delta": 20.0,
            "temporal_alpha": 0.4,
            "temporal_delta": 20.0,
        }

    def load_config(self, config):
        self.config = config
        monitoring_cfg = self.config.get("monitoring", {}) or {}
        self.set_monitor_joints(monitoring_cfg.get("selected_joints", []))
        self._dataset_enabled = bool(monitoring_cfg.get("dataset_enabled", False))
        self._height_map_cfg = monitoring_cfg.get("height_map", {}) if isinstance(monitoring_cfg.get("height_map", {}), dict) else {}
        self._height_map_enabled = bool(self._height_map_cfg.get("enabled", False))
        self._depth_enabled = bool(monitoring_cfg.get("depth_enabled", False)) or self._dataset_enabled
        self._depth_scale = max(1, int(monitoring_cfg.get("depth_scale", 8) or 8))
        base_w, base_h = self._depth_base_frame_size
        self._depth_frame_size = (
            max(1, int(base_w // self._depth_scale)),
            max(1, int(base_h // self._depth_scale)),
        )
        self._depth_update_interval = 1 if self._dataset_enabled else 30
        self._depth_gui_emit_interval = 30

    def load_policy(self, policy_path):
        self.policy_path = policy_path

    def load_encoder(self, encoder_path):
        self.encoder_path = encoder_path

    def init_user_command(self):
        settings_cfg = self.config.get("settings", self.config.get("observation", {}))
        self.user_command = np.zeros(settings_cfg["command_dim"])

    def receive_user_command(self):
        if self.user_command is None:
            self.init_user_command()
        self.env.receive_user_command(self.user_command)

    def update_command(self, index, value):
        if self.user_command is None:
            self.init_user_command()
        settings_cfg = self.config.get("settings", self.config.get("observation", {}))
        if index < settings_cfg["command_dim"]:
            self.user_command[index] = value

    def activate_push_event(self, push_vel):
        self._push_event = True
        self._push_vel = push_vel

    def deactivate_push_event(self):
        self._push_event = False

    def _apply_pending_policy_controls(self):
        if self.policy is None:
            return
        if self._pending_fine_tune_enabled is not None and hasattr(self.policy, "set_fine_tune_enabled"):
            self.policy.set_fine_tune_enabled(self._pending_fine_tune_enabled)
        if self._pending_fine_tune_max_samples is not None and hasattr(self.policy, "set_max_samples"):
            self.policy.set_max_samples(self._pending_fine_tune_max_samples)
        if self._pending_fine_tune_bias is not None and hasattr(self.policy, "set_manual_bias"):
            self.policy.set_manual_bias(self._pending_fine_tune_bias)

    def set_fine_tune_enabled(self, enabled: bool):
        self._pending_fine_tune_enabled = bool(enabled)
        if self.policy is not None and hasattr(self.policy, "set_fine_tune_enabled"):
            self.policy.set_fine_tune_enabled(enabled)

    def set_fine_tune_max_samples(self, max_samples: int):
        self._pending_fine_tune_max_samples = int(max_samples)
        if self.policy is not None and hasattr(self.policy, "set_max_samples"):
            self.policy.set_max_samples(max_samples)

    def set_fine_tune_bias(self, bias):
        arr = np.asarray(bias, dtype=np.float32).reshape(-1)
        self._pending_fine_tune_bias = arr
        if self.policy is not None and hasattr(self.policy, "set_manual_bias"):
            self.policy.set_manual_bias(arr)

    def clear_fine_tune_bias(self):
        self._pending_fine_tune_bias = None
        if self.policy is not None and hasattr(self.policy, "clear_manual_bias"):
            self.policy.clear_manual_bias()

    def set_monitor_joints(self, joint_names):
        names = []
        for joint_name in joint_names or []:
            key = str(joint_name).strip()
            if key and key not in names:
                names.append(key)
        self._monitor_joint_names = names
        selected = set(names)
        self._monitor_history = {
            joint_name: self._monitor_history.get(joint_name, deque(maxlen=self._monitor_history_len))
            for joint_name in selected
        }
        self._monitor_session_history = {
            joint_name: list(self._monitor_session_history.get(joint_name, []))
            for joint_name in selected
        }
        if not names:
            self.overlayUpdated.emit({})

    def get_fine_tune_status(self):
        if self.policy is not None and hasattr(self.policy, "get_fine_tune_status"):
            return self.policy.get_fine_tune_status()
        return {
            "enabled": bool(self._pending_fine_tune_enabled),
            "samples": 0,
            "trained": False,
            "state_dim": 0,
            "action_dim": 0,
            "max_samples": int(self._pending_fine_tune_max_samples or 0),
            "manual_bias": np.asarray(self._pending_fine_tune_bias if self._pending_fine_tune_bias is not None else [], dtype=np.float32),
        }

    def fit_fine_tune_head(self, ridge_lambda=None):
        if self.policy is None or not hasattr(self.policy, "fit_residual_head"):
            raise RuntimeError("Fine-tune policy is not initialized yet.")
        return self.policy.fit_residual_head(ridge_lambda=ridge_lambda)

    def export_fine_tuned_policy(self, output_path: str):
        if self.policy is None or not hasattr(self.policy, "export_merged_onnx"):
            raise RuntimeError("Fine-tune policy is not initialized yet.")
        return self.policy.export_merged_onnx(output_path)

    def test(self):
        report_path = os.path.join(os.path.dirname(self.policy_path), 'report.pdf')
        self.reporter = Reporter(report_path=report_path, config=self.config)
        self.policy = build_policy(
            self.config,
            policy_path=os.path.join(self.policy_path),
            encoder_path=self.encoder_path if hasattr(self, 'encoder_path') else None,
        )
        self._apply_pending_policy_controls()
        self.env = build_env(self.config)
        self._init_depth_stream()
        self._init_height_map_visualization()
        self._init_dataset_writer()
        self._monitor_history = {joint_name: deque(maxlen=self._monitor_history_len) for joint_name in self._monitor_joint_names}
        self._monitor_session_history = {joint_name: [] for joint_name in self._monitor_joint_names}
        state, info = self.env.reset()
        self._update_height_map_visualization()
        self.env.render()
        self._emit_overlay_payload()
        self._emit_depth_payload(force=True)
        done = False
        print(state.shape)
        while not done and not self._stop:
            self.receive_user_command()
            try:
                action = self.policy.get_action(state)
            except Exception:
                self.close()
                self._had_error = True
                raise RuntimeError(
                    "Failed to run inference with the selected ONNX policy.\n"
                    "Please check if you have chosen a valid ONNX file."
                )

            if self._push_event:
                self.env.event(event="push", value=self._push_vel)

            self.env.render()
            if not self._render_window_patched:
                self._patch_render_window()
                self._render_window_patched = True

            assert self.user_command is not None, "user_command must not be None."
            next_state, terminated, truncated, info = self.env.step(action)
            self.reporter.write_info(info)
            self._update_height_map_visualization()
            self._emit_overlay_payload()
            self._emit_depth_payload()
            self.stepFinished.emit()

            done = terminated or truncated
            state = next_state

        if not self._had_error:
            try:
                self.reporter.generate_report()
            except RuntimeError as exc:
                print(f"[WARN] Report generation skipped: {exc}")
        self.overlayUpdated.emit({})
        self.depthUpdated.emit({})
        self.close()
        self.finished.emit()

    def stop(self):
        self._stop = True

    def close(self):
        try:
            if self._depth_stream is not None:
                self._depth_stream.close()
                self._depth_stream = None
        except Exception:
            pass
        try:
            if self._dataset_writer is not None:
                print(f"[dataset] saved {self._dataset_writer.sample_count} samples.")
                self._dataset_writer.close()
                self._dataset_writer = None
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass

    def _patch_render_window(self):
        try:
            glfw.init()
            win = glfw.get_current_context()
            if not win:
                return

            def _on_close(w):
                self._stop = True
                glfw.set_window_should_close(w, False)

            glfw.set_window_close_callback(win, _on_close)

        except Exception as e:
            print(f"[WARN] close-button handler failed: {e}")

    def _get_leaf_env(self):
        env = getattr(self, "env", None)
        visited = set()
        while env is not None and hasattr(env, "env") and id(env) not in visited:
            visited.add(id(env))
            env = env.env
        return env

    def _get_joint_names_for_monitoring(self):
        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return []
        if hasattr(leaf_env, "joint_names_in_order"):
            return list(leaf_env.joint_names_in_order)
        if hasattr(leaf_env, "initial_joint_names"):
            return list(leaf_env.initial_joint_names)
        return []

    def _get_observed_motor_torques(self, leaf_env, joint_names):
        data = getattr(leaf_env, "data", None)
        model = getattr(leaf_env, "model", None)
        if data is None or model is None:
            return np.asarray(getattr(leaf_env, "applied_torques", []), dtype=np.float64).reshape(-1)   

        actuator_force = np.asarray(getattr(data, "actuator_force", []), dtype=np.float64).reshape(-1)
        if actuator_force.size == 0:
            return np.asarray(getattr(leaf_env, "applied_torques", []), dtype=np.float64).reshape(-1)

        actuator_names = []
        for idx in range(actuator_force.size):
            try:
                actuator_names.append(str(model.actuator(idx).name))
            except Exception:
                actuator_names.append("")

        torque_by_joint = {}
        for name, force in zip(actuator_names, actuator_force):
            if not name:
                continue
            key = str(name)
            torque_by_joint[key] = float(force)
            if key.endswith("_ctrl"):
                base_key = key[:-5]
                if base_key and base_key not in torque_by_joint:
                    torque_by_joint[base_key] = float(force)

        observed = []
        matched = 0
        for joint_name in joint_names:
            key = str(joint_name)
            if key in torque_by_joint:
                observed.append(torque_by_joint[key])
                matched += 1
            else:
                observed.append(np.nan)

        observed = np.asarray(observed, dtype=np.float64)
        if matched > 0:
            return observed

        return np.asarray(getattr(leaf_env, "applied_torques", []), dtype=np.float64).reshape(-1)

    def _get_monitor_snapshot(self):
        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return []

        joint_names = self._get_joint_names_for_monitoring()
        if not joint_names:
            return []

        qd_indices = getattr(leaf_env, "qd_indices", [])
        torques = self._get_observed_motor_torques(leaf_env, joint_names)
        if len(qd_indices) == 0 or torques.size == 0:
            return []

        velocities = np.asarray(leaf_env.data.qvel[list(qd_indices)], dtype=np.float64).reshape(-1)
        max_len = min(len(joint_names), velocities.size, torques.size)
        if max_len <= 0:
            return []

        selected = list(self._monitor_joint_names)
        if not selected:
            return []
        selected_set = set(selected)
        velocity_limits = self._resolve_velocity_limits(joint_names[:max_len], velocities[:max_len])
        torque_limits = self._resolve_torque_limits(joint_names[:max_len], torques[:max_len])

        snapshot = []
        for idx, joint_name in enumerate(joint_names[:max_len]):
            if joint_name not in selected_set:
                continue
            vel = float(velocities[idx])
            tau_raw = torques[idx]
            if np.isnan(tau_raw):
                continue
            tau = float(tau_raw)
            history = self._monitor_history.setdefault(joint_name, deque(maxlen=self._monitor_history_len))
            history.append((vel, tau))
            session_history = self._monitor_session_history.setdefault(joint_name, [])
            session_history.append((vel, tau))
            snapshot.append(
                {
                    "joint": joint_name,
                    "velocity": vel,
                    "torque": tau,
                    "torque_limit": float(torque_limits.get(joint_name, max(abs(tau), 1.0))),
                    "velocity_limit": float(velocity_limits.get(joint_name, max(abs(vel), 1.0))),
                    "history": list(history),
                }
            )

        ordered = {name: i for i, name in enumerate(selected)}
        snapshot.sort(key=lambda item: ordered.get(item["joint"], 10**9))
        return snapshot

    def _get_peak_velocity_limit(self):
        peak_velocity = 0.0
        for history in self._monitor_history.values():
            for vel, _ in history:
                peak_velocity = max(peak_velocity, abs(float(vel)))
        return max(peak_velocity * 1.15, 1.0)

    def _resolve_velocity_limits(self, joint_names, velocities):
        hardware_cfg = self.config.get("hardware", {}) or {}
        limits = {}
        fallback_limit = self._get_peak_velocity_limit()
        for joint_name, vel in zip(joint_names, velocities):
            exact_key = f"{joint_name}_max_vel"
            if exact_key in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg[exact_key]))
                continue

            lower_name = joint_name.lower()
            if "wheel" in lower_name and "wheel_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["wheel_max_vel"]))
            elif "shoulder" in lower_name and "shoulder_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["shoulder_max_vel"]))
            elif "hip" in lower_name and "hip_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["hip_max_vel"]))
            elif "torso" in lower_name and "torso_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["torso_max_vel"]))
            elif "ankle" in lower_name and "ankle_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["ankle_max_vel"]))
            elif "elbow" in lower_name and "elbow_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["elbow_max_vel"]))
            elif "wrist" in lower_name and "wrist_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["wrist_max_vel"]))
            elif "head" in lower_name and "head_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["head_max_vel"]))
            elif ("leg" in lower_name or "knee" in lower_name) and "leg_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["leg_max_vel"]))
            elif "knee" in lower_name and "knee_max_vel" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["knee_max_vel"]))
            else:
                limits[joint_name] = max(abs(float(vel)) * 1.15, fallback_limit, 1.0)
        return limits

    def _resolve_torque_limits(self, joint_names, torques):
        hardware_cfg = self.config.get("hardware", {}) or {}
        limits = {}
        for joint_name, tau in zip(joint_names, torques):
            exact_key = f"{joint_name}_max_torque"
            if exact_key in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg[exact_key]))
                continue

            lower_name = joint_name.lower()
            if "wheel" in lower_name and "wheel_max_torque" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["wheel_max_torque"]))
            elif "shoulder" in lower_name and "shoulder_max_torque" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["shoulder_max_torque"]))
            elif "hip" in lower_name and "hip_max_torque" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["hip_max_torque"]))
            elif ("leg" in lower_name or "knee" in lower_name) and "leg_max_torque" in hardware_cfg:
                limits[joint_name] = abs(float(hardware_cfg["leg_max_torque"]))
            else:
                limits[joint_name] = max(abs(float(tau)) * 1.25, 1.0)
        return limits

    def _emit_overlay_payload(self):
        snapshot = self._get_monitor_snapshot()
        if not snapshot:
            self.overlayUpdated.emit({})
            return
        env_id = (self.config.get("env", {}) or {}).get("id", "")
        leaf_env = self._get_leaf_env()
        dt = 1.0 / float(getattr(leaf_env, "control_freq", 50.0)) if leaf_env is not None else 0.02
        self.overlayUpdated.emit({"env_id": env_id, "dt": dt, "joints": snapshot})

    def _init_depth_stream(self):
        self._depth_stream = None
        if not self._depth_enabled:
            return

        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return
        model = getattr(leaf_env, "model", None)
        if model is None:
            return
        model_path = getattr(leaf_env, "model_path", "")
        if not model_path:
            return

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._depth_camera_name)
        if camera_id == -1:
            return

        self._depth_stream = DepthStreamClient(
            model_path=model_path,
            camera_name=self._depth_camera_name,
            frame_size=self._depth_frame_size,
            processing=self._depth_processing,
            preserve_all_frames=self._dataset_enabled,
            queue_size=2048 if self._dataset_enabled else 2,
        )

    def _init_dataset_writer(self):
        self._dataset_writer = None
        self._dataset_warned = False
        if not self._dataset_enabled:
            return

        res_x = int(self._height_map_cfg.get("res_x", 0) or 0)
        res_y = int(self._height_map_cfg.get("res_y", 0) or 0)
        if res_x <= 0 or res_y <= 0:
            print("[WARN] Dataset saving requested but height_map observation is disabled for this env.")
            return

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env_id = (self.config.get("env", {}) or {}).get("id", "")
        output_root = os.path.join(repo_root, "envs", env_id, "dataset")
        self._dataset_writer = HeightMapDatasetWriter(
            env_id=env_id,
            output_root=output_root,
            height_map_shape=(res_y, res_x),
            depth_shape=(self._depth_frame_size[1], self._depth_frame_size[0]),
            depth_scale=self._depth_scale,
        )
        print(f"[dataset] saving samples to {self._dataset_writer.run_dir}")

    def _init_height_map_visualization(self):
        if not bool(self._height_map_cfg.get("enabled", False)):
            return
        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return
        mujoco_utils = getattr(leaf_env, "mujoco_utils", None)
        if mujoco_utils is None:
            return
        res_x = int(self._height_map_cfg.get("res_x", 0) or 0)
        res_y = int(self._height_map_cfg.get("res_y", 0) or 0)
        if res_x <= 0 or res_y <= 0:
            return
        mujoco_utils.init_heightmap_visualization(res_x, res_y, prefix="dataset_heightmap_site")

    def _compute_projected_gravity(self, leaf_env):
        model = getattr(leaf_env, "model", None)
        data = getattr(leaf_env, "data", None)
        if model is None or data is None:
            return None

        for body_name in ("base_link", "pelvis_link"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)
        return None

    def _get_height_map_axis_body_name(self, leaf_env):
        model = getattr(leaf_env, "model", None)
        if model is None:
            return str(self._height_map_cfg.get("frame_body", "camera_link"))
        for body_name in ("base_link", "pelvis_link"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                return body_name
        return str(self._height_map_cfg.get("frame_body", "camera_link"))

    def _compute_height_map_target(self, leaf_env):
        res_x = int(self._height_map_cfg.get("res_x", 0) or 0)
        res_y = int(self._height_map_cfg.get("res_y", 0) or 0)
        if res_x <= 0 or res_y <= 0:
            return None
        mujoco_utils = getattr(leaf_env, "mujoco_utils", None)
        data = getattr(leaf_env, "data", None)
        if mujoco_utils is None or data is None:
            return None
        return mujoco_utils.get_height_map(
            data,
            float(self._height_map_cfg.get("x_forward", float(self._height_map_cfg.get("size_x", 1.0)) / 2.0)),
            float(self._height_map_cfg.get("x_backward", float(self._height_map_cfg.get("size_x", 1.0)) / 2.0)),
            float(self._height_map_cfg.get("y_left", float(self._height_map_cfg.get("size_y", 0.6)) / 2.0)),
            float(self._height_map_cfg.get("y_right", float(self._height_map_cfg.get("size_y", 0.6)) / 2.0)),
            res_x,
            res_y,
            frame_body_name=str(self._height_map_cfg.get("frame_body", "camera_link")),
            site_prefix="dataset_heightmap_site",
            axis_body_name=self._get_height_map_axis_body_name(leaf_env),
        ).astype(np.float32)

    def _maybe_record_dataset(self, payload):
        if self._dataset_writer is None or not payload or payload.get("image") is None:
            return

        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return

        projected_gravity = self._compute_projected_gravity(leaf_env)
        height_map = self._compute_height_map_target(leaf_env)
        if projected_gravity is None or height_map is None:
            if not self._dataset_warned:
                print("[WARN] Dataset writer could not fetch projected_gravity or height_map; skipping samples.")
                self._dataset_warned = True
            return

        self._dataset_writer.add_sample(
            depth_frame=payload["image"],
            projected_gravity=projected_gravity,
            height_map=height_map,
        )

    def _update_height_map_visualization(self):
        if not bool(self._height_map_cfg.get("visualize", False)):
            return
        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            return
        try:
            self._compute_height_map_target(leaf_env)
        except Exception as exc:
            if not self._dataset_warned:
                print(f"[WARN] Height map visualization skipped: {exc}")
                self._dataset_warned = True

    def _emit_depth_payload(self, force=False):
        if self._depth_stream is None:
            if force:
                self.depthUpdated.emit({})
            return

        leaf_env = self._get_leaf_env()
        if leaf_env is None:
            if force:
                self.depthUpdated.emit({})
            return

        local_step = int(getattr(leaf_env, "local_step", 0))
        if force or (local_step % max(1, self._depth_update_interval)) == 0:
            self._depth_stream.submit(
                env_id=(self.config.get("env", {}) or {}).get("id", ""),
                qpos=np.asarray(leaf_env.data.qpos, dtype=np.float64),
                qvel=np.asarray(leaf_env.data.qvel, dtype=np.float64),
            )

        try:
            if self._dataset_enabled:
                payloads = self._depth_stream.poll_all()
                if payloads:
                    for payload in payloads:
                        self._maybe_record_dataset(payload)
                    if force or (local_step % max(1, self._depth_gui_emit_interval)) == 0:
                        self.depthUpdated.emit(payloads[-1])
            else:
                payload = self._depth_stream.poll()
                if payload is not None:
                    self._maybe_record_dataset(payload)
                    self.depthUpdated.emit(payload)
        except Exception as exc:
            print(f"[WARN] Depth stream skipped: {exc}")
            self.depthUpdated.emit({})

    def get_monitor_export_payload(self):
        env_id = (self.config.get("env", {}) or {}).get("id", "")
        leaf_env = self._get_leaf_env()
        dt = 1.0 / float(getattr(leaf_env, "control_freq", 50.0)) if leaf_env is not None else 0.02
        joints = []
        velocity_limits = self._resolve_velocity_limits(
            list(self._monitor_session_history.keys()),
            [history[-1][0] if history else 0.0 for history in self._monitor_session_history.values()],
        )
        torque_limits = self._resolve_torque_limits(
            list(self._monitor_session_history.keys()),
            [0.0] * len(self._monitor_session_history),
        )
        for joint_name in self._monitor_joint_names:
            history = list(self._monitor_session_history.get(joint_name, []))
            if not history:
                continue
            last_vel, last_tau = history[-1]
            joints.append(
                {
                    "joint": joint_name,
                    "velocity": float(last_vel),
                    "torque": float(last_tau),
                    "torque_limit": float(torque_limits.get(joint_name, 1.0)),
                    "velocity_limit": float(velocity_limits.get(joint_name, max(abs(float(last_vel)), 1.0))),
                    "history": history,
                }
            )
        return {"env_id": env_id, "dt": dt, "joints": joints}
