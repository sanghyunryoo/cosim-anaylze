import os
import sys
import yaml
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QMessageBox, QMainWindow,
    QFileDialog, QGroupBox, QScrollArea, QLineEdit, QCheckBox, QDialog,
    QTextEdit
)
from PyQt5.QtCore import QThread, Qt, QEvent, QUrl, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QDesktopServices, QFont, QFontDatabase, QIcon, QColor, QTextCharFormat, QTextCursor
from core.tester import Tester
from ui.utils import to_float, to_int, normalize_numkey_float_values
from ui.custom_widgets import AlphaOverlayWidget, DepthImageWidget, MujocoOverlayWidget, NoWheelComboBox, NoWheelSlider, NonClickableButton
from ui.dialogs.action_scale_settings import ActionScaleSettingsDialog
from ui.dialogs.actuator_settings import ActuatorSettingsDialog
from ui.dialogs.hardware_settings import HardwareSettingsDialog
from ui.dialogs.observation_settings import ObservationSettingsDialog
from ui.dialogs.initial_pose_settings import InitialPoseSettingsDialog
from ui.dialogs.final_pose_settings import FinalPoseSettingsDialog
from ui.dialogs.command_range_settings import CommandRangeSettingsDialog
from ui.dialogs.fine_tune_bias_editor import FineTuneBiasEditorDialog
from ui.dialogs.depth_randomization_settings import DepthRandomizationSettingsDialog
from ui.dialogs.vision_train_dialog import VisionTrainDialog
from ui.dialogs.moe_train_dialog import MoETrainDialog
from ui.dialogs.moe_manual_dialog import MoEManualDialog
from ui.dialogs.homing_train_dialog import HomingTrainDialog
from ui.dialogs.ctbc_train_dialog import CtbcTrainDialog
from ui.workers import TesterWorker, VisionTrainerWorker, MoEWorker, HomingWorker, CtbcWorker
from PyQt5.QtWidgets import QSizePolicy
from envs.initial_pose import get_default_initial_pose, get_initial_pose_joint_names


class _QtLogEmitter(QObject):
    messageWritten = pyqtSignal(str)


class _TeeStream:
    def __init__(self, emitter: _QtLogEmitter, original_stream):
        self._emitter = emitter
        self._original_stream = original_stream

    def write(self, message):
        if not isinstance(message, str):
            message = str(message)
        if message:
            self._emitter.messageWritten.emit(message)
            if self._original_stream is not None:
                self._original_stream.write(message)
        return len(message)

    def flush(self):
        if self._original_stream is not None:
            self._original_stream.flush()

    def isatty(self):
        if self._original_stream is not None and hasattr(self._original_stream, 'isatty'):
            return self._original_stream.isatty()
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        cur_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(cur_file_path), "../config/env_table.yaml")
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            self.env_config = yaml.full_load(f)

        self.obs_types = [
            "dof_pos", "dof_vel",
            "lin_vel_x", "lin_vel_y", "lin_vel_z",
            "ang_vel", "projected_gravity",
            "lower_ang_vel", "upper_ang_vel",
            "lower_projected_gravity", "upper_projected_gravity",
            "lower_imu_ang_vel", "upper_imu_ang_vel",
            "lower_imu_projected_gravity", "upper_imu_projected_gravity",
            "height_map", "masked_height_map", "camera_height_map",
            "last_action",
        ]

        # Per-environment observation settings cache
        self.obs_settings_by_env = {}

        self._init_window()
        self._init_variables()
        self._setup_ui()
        self._log_emitter.messageWritten.connect(self._append_log)
        self._init_default_command_values()
        self.status_label.setText("Waiting ...")
        self.env_id_cb.currentTextChanged.connect(self.update_defaults)
        self.update_defaults(self.env_id_cb.currentText())
        self._last_run_had_error = False

    def _init_window(self):
        app_icon_path = os.path.join(os.path.dirname(__file__), "icon", "window_icon.png")
        self.setWindowIcon(QIcon(app_icon_path))
        self.setWindowTitle("cosim - act_net")
        self.resize(1440, 1020)
        self.setMinimumSize(1320, 920)
        self.installEventFilter(self)
        
    def _init_variables(self):
        self.key_mapping = {}
        self.active_keys = {}
        self._pending_key_release_timers = {}
        self.thread = None
        self.worker = None
        self.vision_train_thread = None
        self.vision_train_worker = None
        self.vision_train_dialog = None
        self.moe_thread = None
        self.moe_worker = None
        self.moe_dialog = None
        self.moe_worker_mode = None
        self.moe_manual_dialog = None
        self.homing_thread = None
        self.homing_worker = None
        self.homing_dialog = None
        self.ctbc_dialog = None
        self.homing_worker_mode = None
        self.ctbc_thread = None
        self.ctbc_worker = None
        self.ctbc_worker_mode = None
        self.homing_command_timer = None
        self.tester = None
        self.current_command_values = [0.0] * 6
        self.command_sensitivity_le_list = []
        self.min_command_value_le_list = []
        self.max_command_value_le_list = []
        self.command_initial_value_le_list = []
        self.command_discrete_cb_list = []
        self.command_timer = None
        self.actuator_settings = {}
        self.actuator_settings_by_env = {}
        self.action_scales = []
        self.action_scales_by_env = {}
        self.action_clippings = []
        self.action_clippings_by_env = {}
        self.hardware_settings = {}
        self.hardware_settings_by_env = {}
        self.initial_pose_settings = {}
        self.initial_pose_settings_by_env = {}
        self.final_pose_settings = {}
        self.final_pose_settings_by_env = {}
        self.homing_command_ranges = {}
        self.homing_command_ranges_by_env = {}
        self.monitor_settings = {}
        self.monitor_settings_by_env = {}
        self.dataset_height_map_settings = {}
        self.dataset_height_map_settings_by_env = {}
        self.depth_randomization_settings = {}
        self.depth_randomization_settings_by_env = {}
        self.monitor_joint_checkboxes = {}
        self.fine_tune_settings = {}
        self.fine_tune_settings_by_env = {}
        self.fine_tune_bias_dialog = None
        self.vision_train_settings = {}
        self.vision_train_settings_by_env = {}
        self._vision_last_summary = None
        self._vision_last_summary_by_env = {}
        self.moe_settings = {}
        self.moe_settings_by_env = {}
        self.moe_manual_settings = {}
        self.moe_manual_settings_by_env = {}
        self._moe_last_summary = None
        self._moe_last_summary_by_env = {}
        self.homing_settings = {}
        self.homing_settings_by_env = {}
        self.ctbc_settings = {}
        self.ctbc_settings_by_env = {}
        self._homing_last_summary = None
        self._homing_last_summary_by_env = {}
        self.mujoco_overlay = MujocoOverlayWidget()
        self.mujoco_overlay.closed.connect(self._on_monitor_overlay_closed)
        self.alpha_overlay = AlphaOverlayWidget()
        self.alpha_overlay.closed.connect(self._on_alpha_overlay_closed)
        self.depth_image_widget = DepthImageWidget()
        self.depth_image_widget.closed.connect(self._on_depth_widget_closed)
        self._log_emitter = _QtLogEmitter()
        self._stdout_stream = None
        self._stderr_stream = None
        self._original_stdout = None
        self._original_stderr = None
        self._log_buffer = ""
        self._rainbow_palette = [
            "#ff595e", "#ff924c", "#ffca3a", "#8ac926",
            "#52a675", "#1982c4", "#6a4c93", "#f15bb5"
        ]
        self._log_color_index = 0
        self._joint_color_map = {}
        self._signal_color_map = {
            "euler angle [roll, pitch, yaw]": "#4cc9f0",
            "gyro [x, y, z]": "#f72585",
            "projected gravity [x, y, z]": "#b8f35d",
        }
        self._pending_log_chunks = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(50)
        self._log_flush_timer.timeout.connect(self._flush_log_output)

        # Whether the user manually changed observation settings via dialog (kept for reference; cache now used)
        self.observation_overridden_by_user = False

        # Initial observation_settings (will be overridden by update_defaults for the first env)
        self.observation_settings = {
            "stacked_obs_order": [],
            "non_stacked_obs_order": [],
            "stack_size": 3,
            "command_dim": 6,
            "command_scales": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
            "height_map": {
                "size_x": 1.0,
                "size_y": 0.6,
                "res_x": 10,
                "res_y": 6,
                "freq": 50,
                "scale": 1.0,
                "target_height": 0.5,
                "clipping_min": 0.0,
                "clipping_max": 0.33,
            },
            "dof_pos": None,
            "dof_vel": None,
            "lin_vel_x": None,
            "lin_vel_y": None,
            "lin_vel_z": None,
            "ang_vel": None,
            "projected_gravity": None,
            "lower_imu_ang_vel": None,
            "upper_imu_ang_vel": None,
            "lower_imu_projected_gravity": None,
            "upper_imu_projected_gravity": None,
            "last_action": None,
        }

    def _init_default_command_values(self):
        """Initialize current_command_values from the UI 'Initial Value' fields."""
        try:
            vals = []
            for widget in self.command_initial_value_le_list:
                if isinstance(widget, QLineEdit):
                    vals.append(float(widget.text()))
                elif isinstance(widget, QLabel):
                    vals.append(float(widget.text()))
                else:
                    vals.append(0.0)
            self.current_command_values = vals if len(vals) == 6 else [0.0] * 6
        except Exception:
            self.current_command_values = [0.0] * 6

    # -------- observation defaults/caching --------
    def _make_observation_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        settings_cfg = env_cfg.get("settings", env_cfg) if isinstance(env_cfg, dict) else {}
        cmd_cfg_raw = settings_cfg.get("command", {}) if isinstance(settings_cfg.get("command", {}), dict) else {}
        obs_scales = settings_cfg.get("obs_scales", {}) or {}
        command_scales_cfg = normalize_numkey_float_values(settings_cfg.get("command_scales", {}))
        stacked_list = settings_cfg.get("stacked_obs_order", []) or []
        non_stacked_list = settings_cfg.get("non_stacked_obs_order", []) or []
        stack_size_yaml = to_int(settings_cfg.get("stack_size", 3), 3)

        # Apply default frequency and scale
        obs_dict = {}
        for obs in stacked_list:
            obs_dict[obs] = {"freq": 50, "scale": to_float(obs_scales.get(obs, 1.0), 1.0)}

        for obs in non_stacked_list:
            obs_dict[obs] = {"freq": 50, "scale": to_float(obs_scales.get(obs, 1.0), 1.0)}

        for obs in self.obs_types:
            if obs not in obs_dict:
                obs_dict[obs] = None

        cmd_dim = to_int(cmd_cfg_raw.get("command_dim", 6), 6)

        merged_command_scales = {}
        for i in range(cmd_dim):
            key = str(i)
            merged_command_scales[key] = to_float(command_scales_cfg.get(key, 1.0), 1.0)

        height_in_order = any(
            name in stacked_list or name in non_stacked_list
            for name in ("height_map", "masked_height_map", "camera_height_map")
        )
        if height_in_order:
            height_map_yaml = settings_cfg.get("height_map", {}) if isinstance(settings_cfg.get("height_map", {}), dict) else {}
            height_map_val = {
                "size_x": to_float(height_map_yaml.get("size_x", 1.0)),
                "size_y": to_float(height_map_yaml.get("size_y", 0.6)),
                "res_x": to_int(height_map_yaml.get("res_x", 15)),
                "res_y": to_int(height_map_yaml.get("res_y", 9)),
                "freq": 50,
                "scale": 1.0,
                "target_height": to_float(height_map_yaml.get("target_height", 0.5), 0.5),
                "clipping_min": to_float(height_map_yaml.get("clipping_min", 0.0), 0.0),
                "clipping_max": to_float(height_map_yaml.get("clipping_max", 0.33), 0.33),
                "point_stride": to_int(height_map_yaml.get("point_stride", 16), 16),
                "max_range": to_float(height_map_yaml.get("max_range", 2.5), 2.5),
                "camera_update_freq": to_float(height_map_yaml.get("camera_update_freq", 10.0), 10.0),
                "debug_print": bool(height_map_yaml.get("debug_print", False)),
            }
        else:
            height_map_val = None

        return {
            "stacked_obs_order": stacked_list,
            "non_stacked_obs_order": non_stacked_list,
            "stack_size": stack_size_yaml,
            "command_dim": cmd_dim,
            "command_scales": merged_command_scales,
            "height_map": height_map_val,
            **obs_dict
        }

    def _ensure_observation_defaults(self):
        # If not in cache, create defaults for the current env
        env_id = self.env_id_cb.currentText()
        if env_id not in self.obs_settings_by_env:
            self.obs_settings_by_env[env_id] = self._make_observation_defaults(env_id)
        # Sync current observation_settings with latest cache
        self.observation_settings = (self.obs_settings_by_env[env_id]).copy()

    # ---------------- per-env action scale helpers ----------------
    def _make_action_scale_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        action_dim = to_int((env_cfg.get("hardware", {}) or {}).get("action_dim", 0), 0)
        raw = env_cfg.get("action_scales", [])
        scales = [to_float(v, 1.0) for v in raw] if isinstance(raw, list) else []
        if action_dim > 0 and len(scales) != action_dim:
            if len(scales) == 0:
                scales = [1.0] * action_dim
            elif len(scales) < action_dim:
                scales = scales + [1.0] * (action_dim - len(scales))
            else:
                scales = scales[:action_dim]
        return scales

    def _make_action_clipping_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        action_dim = to_int((env_cfg.get("hardware", {}) or {}).get("action_dim", 0), 0)
        raw = env_cfg.get("action_clippings", [])
        clippings = []
        if isinstance(raw, list):
            for item in raw[:action_dim]:
                if isinstance(item, dict):
                    enabled = bool(item.get("enabled", False))
                    min_value = to_float(item.get("min", -1.0), -1.0)
                    max_value = to_float(item.get("max", 1.0), 1.0)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    enabled = True
                    min_value = to_float(item[0], -1.0)
                    max_value = to_float(item[1], 1.0)
                else:
                    enabled = False
                    min_value = -1.0
                    max_value = 1.0
                if min_value > max_value:
                    min_value, max_value = max_value, min_value
                clippings.append({"enabled": enabled, "min": min_value, "max": max_value})

        if action_dim > 0 and len(clippings) != action_dim:
            clippings.extend(
                {"enabled": False, "min": -1.0, "max": 1.0}
                for _ in range(action_dim - len(clippings))
            )
            clippings = clippings[:action_dim]
        return clippings

    def _ensure_action_scale_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.action_scales_by_env:
            self.action_scales_by_env[env_id] = self._make_action_scale_defaults(env_id)
        self.action_scales = list(self.action_scales_by_env[env_id])
        if env_id not in self.action_clippings_by_env:
            self.action_clippings_by_env[env_id] = self._make_action_clipping_defaults(env_id)
        self.action_clippings = [dict(item) for item in self.action_clippings_by_env[env_id]]

    # ---------------- per-env actuator helpers ----------------
    def _detect_actuator_control_axis(self, raw: dict) -> str:
        return "group"

    def _normalize_actuator_settings(self, merged: dict) -> dict:
        merged = dict(merged or {})
        merged["control_axis"] = "group"

        # group 모드 (기존 호환)
        units = ("hip", "shoulder", "leg", "wheel")

        global_mode = str(merged.get("mode", "")).strip().lower()
        if global_mode:
            for unit in units:
                merged.setdefault(f"{unit}_mode", global_mode)

        for unit in units:
            mode_key = f"{unit}_mode"
            path_key = f"{unit}_net_path"
            mode = str(merged.get(mode_key, "pd")).strip().lower()
            merged[mode_key] = "actuator_net" if mode == "actuator_net" else "pd"
            merged[path_key] = str(merged.get(path_key, "")).strip()

        return merged
    
    def _make_actuator_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        raw = env_cfg.get("actuator", {}) if isinstance(env_cfg.get("actuator", {}), dict) else {}

        default_net_mode = "pd"

        shoulder_default_path = "act_net/shoulder/pos_vel.pt"
        leg_default_path = (
            "/home/sanghyunryoo/Documents/4w4l/Isaac-RL-Two-wheel-Legged-Bot_joint/"
            "lab/flamingo/assets/data/ActuatorNets/Flamingo/mlp/geared_leg/pos_vel_joint.pt"
        )

        # 기존 group 단위 기본값
        defaults = {
            "control_axis": "group",

            "hip_mode": "pd",
            "hip_net_path": "",

            "shoulder_mode": default_net_mode,
            "shoulder_net_path": shoulder_default_path,

            "leg_mode": default_net_mode,
            "leg_net_path": leg_default_path,

            "wheel_mode": "pd",
            "wheel_net_path": "",
        }

        merged = {**defaults, **raw}
        return self._normalize_actuator_settings(merged)
        
    def _ensure_actuator_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.actuator_settings_by_env:
            self.actuator_settings_by_env[env_id] = self._make_actuator_defaults(env_id)
        self.actuator_settings = (self.actuator_settings_by_env[env_id]).copy()

    # ---------------- per-env hardware helpers (like observation) ----------------
    def _make_hardware_defaults(self, env_id: str):
        """Build default hardware settings for the env from YAML (shallow copy)."""
        env_cfg = self.env_config.get(env_id, {}) or {}
        hw = env_cfg.get("hardware", {}) or {}
        # Keep string values (editable in dialog). Numeric conversion is done in _gather_config.
        return hw.copy()

    def _ensure_hardware_defaults(self):
        """Ensure current env has cached hardware settings and sync self.hardware_settings."""
        env_id = self.env_id_cb.currentText()
        defaults = self._make_hardware_defaults(env_id)
        if env_id not in self.hardware_settings_by_env:
            self.hardware_settings_by_env[env_id] = defaults
        else:
            merged = defaults.copy()
            merged.update(self.hardware_settings_by_env[env_id])
            self.hardware_settings_by_env[env_id] = merged
        self.hardware_settings = (self.hardware_settings_by_env[env_id]).copy()

    @staticmethod
    def _normalize_hardware_value(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return text
            try:
                number = float(text)
            except ValueError:
                return text
            if number.is_integer() and not any(ch in text.lower() for ch in (".", "e")):
                return int(number)
            return number
        return value

    def _get_current_action_dim(self, env_id=None):
        target_env_id = env_id or self.env_id_cb.currentText()
        env_cfg = self.env_config.get(target_env_id, {}) or {}
        hardware_cfg = env_cfg.get("hardware", {}) if isinstance(env_cfg.get("hardware", {}), dict) else {}
        return max(0, to_int(hardware_cfg.get("action_dim", 0), 0))

    def _make_fine_tune_defaults(self, env_id: str):
        action_dim = self._get_current_action_dim(env_id)
        return {
            "enabled": False,
            "ridge_lambda": "1e-4",
            "max_samples": "5000",
            "bias": [0.0] * action_dim,
        }

    def _make_vision_train_defaults(self, env_id: str):
        _ = env_id
        return {
            "epochs": "100",
            "batch_size": "64",
            "learning_rate": "1e-3",
            "latent_dim": "128",
            "hidden_dim": "128",
            "val_ratio": "0.15",
            "seed": "42",
            "selected_datasets": [],
        }

    def _ensure_vision_train_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.vision_train_settings_by_env:
            self.vision_train_settings_by_env[env_id] = self._make_vision_train_defaults(env_id)
        self.vision_train_settings = dict(self.vision_train_settings_by_env[env_id])

    def _sync_vision_train_controls_from_cache(self):
        self._ensure_vision_train_defaults()
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.set_env_id(self.env_id_cb.currentText())
            self.vision_train_dialog.set_settings(self.vision_train_settings)
        self._update_vision_train_status_label()

    def _collect_vision_train_ui_settings(self, source_settings=None):
        self._ensure_vision_train_defaults()
        source = dict(source_settings or self.vision_train_settings)
        settings = {
            "epochs": str(source.get("epochs", self.vision_train_settings.get("epochs", "10"))).strip(),
            "batch_size": str(source.get("batch_size", self.vision_train_settings.get("batch_size", "64"))).strip(),
            "learning_rate": str(source.get("learning_rate", self.vision_train_settings.get("learning_rate", "1e-3"))).strip(),
            "latent_dim": str(source.get("latent_dim", self.vision_train_settings.get("latent_dim", "128"))).strip(),
            "hidden_dim": str(source.get("hidden_dim", self.vision_train_settings.get("hidden_dim", "128"))).strip(),
            "val_ratio": str(source.get("val_ratio", self.vision_train_settings.get("val_ratio", "0.1"))).strip(),
            "seed": str(source.get("seed", self.vision_train_settings.get("seed", "42"))).strip(),
            "selected_datasets": list(source.get("selected_datasets", self.vision_train_settings.get("selected_datasets", []))),
        }
        env_id = self.env_id_cb.currentText()
        self.vision_train_settings = settings
        self.vision_train_settings_by_env[env_id] = dict(settings)
        return settings

    def _repo_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _vision_dataset_root(self, env_id: str):
        return os.path.join(self._repo_root(), "envs", env_id, "dataset", "height_map_supervision")

    def _list_vision_train_datasets(self, env_id: str):
        dataset_root = self._vision_dataset_root(env_id)
        if not os.path.isdir(dataset_root):
            return []

        datasets = []
        run_names = sorted(
            [
                name for name in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, name))
            ],
            reverse=True,
        )
        for run_name in run_names:
            dataset_path = os.path.join(dataset_root, run_name, "dataset.npz")
            if not os.path.isfile(dataset_path):
                continue
            try:
                with np.load(dataset_path) as payload:
                    samples = int(payload["height_map"].shape[0])
                    depth_shape = tuple(int(v) for v in payload["depth_history"].shape[1:])
                    height_shape = tuple(int(v) for v in payload["height_map"].shape[1:])
            except Exception:
                samples = 0
                depth_shape = ()
                height_shape = ()
            shape_text = ""
            if depth_shape and height_shape:
                shape_text = f" | depth {depth_shape} -> hm {height_shape}"
            datasets.append({
                "path": dataset_path,
                "label": f"{run_name} | {samples} samples{shape_text}",
            })
        return datasets

    def _terrain_ids(self):
        return [
            'flat', 'rocky_easy', 'rocky_hard',
            'slope_easy', 'slope_hard',
            'stairs_up_easy', 'stairs_up_normal', 'stairs_up_hard', 'stairs_up_extrme'
        ]

    def _make_moe_defaults(self, env_id: str):
        action_scales = ",".join(str(v) for v in self._make_action_scale_defaults(env_id))
        return {
            "env_id": env_id,
            "policy_a_path": "",
            "policy_b_path": "",
            "policy_a_action_scales": action_scales,
            "policy_b_action_scales": action_scales,
            "output_action_scales": action_scales,
            "terrains": ["flat", "rocky_easy", "rocky_hard", "stairs_up_easy", "stairs_up_normal", "stairs_up_hard"],
            "samples": "200000",
            "rollout_steps": "1000",
            "boundary_m": "8.0",
            "command_min": "-1.0",
            "command_max": "1.0",
            "seed": "42",
            "lambda_smooth": "0",
            "cmd_label_threshold": "0.2",
            "cmd_label_alpha": "0",
            "epochs": "30",
            "batch_size": "256",
            "learning_rate": "1e-3",
            "val_ratio": "0.1",
            "selected_datasets": [],
        }

    def _make_moe_manual_defaults(self, env_id: str):
        output_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "moe_manual")
        action_scales = ",".join(str(v) for v in self._make_action_scale_defaults(env_id))
        return {
            "env_id": env_id,
            "policy_a_path": "",
            "policy_b_path": "",
            "policy_a_action_scales": action_scales,
            "policy_b_action_scales": action_scales,
            "output_action_scales": action_scales,
            "output_path": os.path.join(output_dir, "manual_moe_policy.onnx"),
        }

    def _make_homing_defaults(self, env_id: str):
        output_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "homing", "latest")
        self._ensure_final_pose_defaults_for_env(env_id)
        self._ensure_homing_command_ranges_for_env(env_id)
        final_pos = self._final_pose_csv(env_id, "joints")
        final_vel = self._final_pose_csv(env_id, "velocities")
        final_priority = self._final_pose_csv(env_id, "priorities")
        command_mins = self._homing_command_range_csv(env_id, "mins")
        command_maxs = self._homing_command_range_csv(env_id, "maxs")
        final_pose = self.final_pose_settings_by_env.get(env_id, {})
        return {
            "env_id": env_id,
            "policy_path": "",
            "terrains": ["flat"],
            "samples": "50000",
            "rollout_steps": "1000",
            "homing_trajectory_seconds": "3.0",
            "homing_stand_warmup_steps": "200",
            "homing_balance_blend": "0.0",
            "command_min": "-1.0",
            "command_max": "1.0",
            "command_mins": command_mins,
            "command_maxs": command_maxs,
            "seed": "42",
            "final_pos": final_pos,
            "final_vel": final_vel,
            "final_pose_same": "1" if final_pose.get("same", True) else "0",
            "final_pose_priorities": final_priority,
            "epochs": "30",
            "batch_size": "256",
            "learning_rate": "1e-3",
            "val_ratio": "0.1",
            "hidden_dim": "256",
            "ppo_total_steps": "1000000",
            "ppo_num_envs": "32",
            "ppo_rollout_steps": "512",
            "ppo_epochs": "4",
            "ppo_learning_rate": "5e-5",
            "ppo_domain_randomize": "0.1",
            "ppo_supervised_init": "1",
            "ppo_use_trajectory_reward": "1",
            "ppo_mask_wheel_actions": "1",
            "ppo_strategy_preset": "light",
            "reward_track": "6.0",
            "reward_base_acc": "0.002",
            "reward_upright": "2.0",
            "reward_action_rate": "0.04",
            "reward_contact": "0.0005",
            "selected_datasets": [],
            "checkpoint_path": os.path.join(output_dir, "homing_policy_supervised.pt"),
            "output_path": os.path.join(output_dir, "homing_policy.onnx"),
        }

    def _make_ctbc_defaults(self, env_id: str):
        output_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "ctbc", "latest")
        self._ensure_homing_command_ranges_for_env(env_id)
        return {
            "task_mode": "ctbc",
            "env_id": env_id,
            "policy_path": "",
            "ctbc_terrain": "stairs_up_easy",
            "ctbc_contact_threshold": "30.0",
            "ctbc_contact_window": "3",
            "ctbc_lift_amplitude": "0.90",
            "ctbc_lift_period": "0.75",
            "ctbc_anneal_ratio": "0.70",
            "ctbc_episode_steps": "1024",
            "ctbc_residual_limit": "4.0",
            "ctbc_gate_height_threshold": "0.06",
            "ctbc_gate_height_softness": "0.025",
            "ctbc_gate_rise": "0.35",
            "ctbc_gate_fall": "0.08",
            "ctbc_gate_lift_threshold": "0.25",
            "ctbc_gate_reward_threshold": "0.35",
            "ctbc_assist_trigger_gate": "0.12",
            "ctbc_assist_gate_floor": "0.85",
            "ctbc_assist_min": "0.0",
            "ctbc_gate_residual_runtime": "0",
            "ctbc_anneal_bc_with_assist": "1",
            "ctbc_distill_primitive": "1",
            "ctbc_bc_weight_min": "0.15",
            "ctbc_reflex_only": "1",
            "ctbc_controller_candidates": "64",
            "ctbc_reflex_samples": "8192",
            "ctbc_reflex_epochs": "12",
            "ctbc_reflex_batch": "256",
            "ctbc_reflex_lr": "3e-4",
            "ctbc_reflex_flat_ratio": "0.35",
            "ctbc_reflex_gain": "1.0",
            "ctbc_reflex_segment_steps": "128",
            "ctbc_fast_teacher_steps": "4096",
            "ctbc_fast_teacher_epochs": "6",
            "ctbc_fast_teacher_batch": "256",
            "ctbc_fast_teacher_lr": "2e-4",
            "ctbc_fast_teacher_gain": "1.0",
            "ctbc_fast_teacher_stair_height": "0.12",
            "ctbc_safe_tilt": "0.22",
            "ctbc_emergency_tilt": "0.34",
            "ctbc_terminate_tilt": "0.42",
            "ctbc_tilt_guard_penalty": "8.0",
            "ctbc_bad_contact_threshold": "1.0",
            "ctbc_bad_contact_penalty": "20.0",
            "ctbc_lift_cooldown": "0.35",
            "ctbc_contact_baseline_alpha": "0.02",
            "ctbc_contact_spike_threshold": "80.0",
            "ctbc_force_alternating_lift": "1",
            "ctbc_curriculum_enabled": "1",
            "ctbc_stair_height_min": "0.025",
            "ctbc_stair_height_max": "0.20",
            "ctbc_curriculum_ratio": "0.60",
            "ctbc_select_after_ratio": "0.70",
            "ctbc_shoulder_gain": "0.50",
            "ctbc_leg_gain": "0.0",
            "ctbc_leg_push_gain": "1.75",
            "ctbc_hip_gain": "0.0",
            "ctbc_stance_gain": "0.30",
            "ctbc_wheel_push_gain": "0.0",
            "ctbc_ff_clip": "4.0",
            "ctbc_action_clip": "4.0",
            "ctbc_compensate_action_scale": "1",
            "ctbc_clearance_target": "0.14",
            "ctbc_base_height_target": "0.14",
            "ctbc_clearance_stair_ratio": "0.90",
            "ctbc_climb_stair_ratio": "0.75",
            "ctbc_reward_lift": "2.0",
            "ctbc_reward_clearance": "1.0",
            "ctbc_reward_wheel_clearance": "4.0",
            "ctbc_reward_base_height": "4.0",
            "ctbc_reward_stair_success": "5.0",
            "ctbc_hard_stair_threshold": "0.14",
            "ctbc_hard_stair_fail_penalty": "1.5",
            "ctbc_reward_forward_progress": "35.0",
            "ctbc_min_forward_progress": "0.010",
            "ctbc_reward_stair_forward": "2.0",
            "ctbc_reward_stair_motion": "4.0",
            "ctbc_no_progress_penalty": "1.0",
            "ctbc_reward_height_progress": "30.0",
            "ctbc_reward_balance_on_stair": "0.7",
            "ctbc_min_climb_height": "0.015",
            "ctbc_no_climb_penalty": "0.12",
            "ctbc_base_imitation": "0.5",
            "ctbc_non_wheel_contact_penalty": "4.0",
            "ctbc_command_x_min": "0.35",
            "ctbc_command_x_max": "0.70",
            "ctbc_command_y_abs": "0.03",
            "ctbc_command_yaw_abs": "0.05",
            "reward_track": "1.2",
            "reward_upright": "2.0",
            "reward_action_rate": "0.04",
            "ppo_total_steps": "1000000",
            "ppo_num_envs": "32",
            "ppo_rollout_steps": "512",
            "ppo_epochs": "4",
            "ppo_learning_rate": "5e-5",
            "ppo_domain_randomize": "0.05",
            "hidden_dim": "256",
            "seed": "42",
            "command_min": "-1.0",
            "command_max": "1.0",
            "command_mins": self._homing_command_range_csv(env_id, "mins"),
            "command_maxs": self._homing_command_range_csv(env_id, "maxs"),
            "checkpoint_path": os.path.join(output_dir, "ctbc_policy_ppo.pt"),
            "output_path": os.path.join(output_dir, "ctbc_policy.onnx"),
        }

    def _ensure_moe_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.moe_settings_by_env:
            self.moe_settings_by_env[env_id] = self._make_moe_defaults(env_id)
        self.moe_settings = dict(self.moe_settings_by_env[env_id])

    def _ensure_moe_manual_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.moe_manual_settings_by_env:
            self.moe_manual_settings_by_env[env_id] = self._make_moe_manual_defaults(env_id)
        self.moe_manual_settings = dict(self.moe_manual_settings_by_env[env_id])

    def _ensure_homing_defaults(self, env_id=None):
        env_id = str(env_id or self.env_id_cb.currentText())
        if env_id not in self.homing_settings_by_env:
            self.homing_settings_by_env[env_id] = self._make_homing_defaults(env_id)
        self.homing_settings = dict(self.homing_settings_by_env[env_id])

    def _ensure_ctbc_defaults(self, env_id=None):
        env_id = str(env_id or self.env_id_cb.currentText())
        if env_id not in self.ctbc_settings_by_env:
            self.ctbc_settings_by_env[env_id] = self._make_ctbc_defaults(env_id)
        else:
            settings = dict(self.ctbc_settings_by_env[env_id])
            defaults = self._make_ctbc_defaults(env_id)
            old_defaults = {
                "ctbc_lift_amplitude": {"0.60"},
                "ctbc_lift_period": {"0.60"},
                "ctbc_residual_limit": {"0.65", "0.85"},
                "ctbc_gate_height_threshold": {"0.10"},
                "ctbc_gate_height_softness": {"0.03"},
                "ctbc_gate_lift_threshold": {"0.55"},
                "ctbc_shoulder_gain": {"1.25", "1.60"},
                "ctbc_leg_gain": {"1.45", "-1.35"},
                "ctbc_leg_push_gain": {"0.75"},
                "ctbc_hip_gain": {"0.35", "0.45"},
                "ctbc_stance_gain": {"0.25"},
                "ctbc_ff_clip": {"1.0"},
                "ctbc_reward_wheel_clearance": {"1.5", "2.5"},
                "ctbc_assist_min": {"0.35"},
                "ctbc_curriculum_ratio": {"1.0"},
                "ctbc_select_after_ratio": {"0.65"},
                "ctbc_clearance_target": {"0.08"},
                "ctbc_base_height_target": {"0.06"},
                "ctbc_reward_base_height": {"2.0"},
                "ppo_total_steps": {"200000"},
                "ppo_num_envs": {"16"},
                "ppo_rollout_steps": {"256"},
                "ppo_domain_randomize": {"0.3"},
            }
            for key, value in defaults.items():
                if key not in settings or str(settings.get(key, "")) in old_defaults.get(key, set()):
                    settings[key] = value
            self.ctbc_settings_by_env[env_id] = settings
        self.ctbc_settings = dict(self.ctbc_settings_by_env[env_id])

    def _collect_moe_ui_settings(self, source_settings=None):
        self._ensure_moe_defaults()
        source = dict(source_settings or self.moe_settings)
        env_id = str(source.get("env_id", self.env_id_cb.currentText()))
        settings = dict(self.moe_settings)
        settings.update({
            "env_id": env_id,
            "policy_a_path": str(source.get("policy_a_path", "")).strip(),
            "policy_b_path": str(source.get("policy_b_path", "")).strip(),
            "policy_a_action_scales": str(source.get("policy_a_action_scales", settings.get("policy_a_action_scales", ""))).strip(),
            "policy_b_action_scales": str(source.get("policy_b_action_scales", settings.get("policy_b_action_scales", ""))).strip(),
            "output_action_scales": str(source.get("output_action_scales", settings.get("output_action_scales", ""))).strip(),
            "terrains": list(source.get("terrains", settings.get("terrains", []))),
            "samples": str(source.get("samples", settings.get("samples", "200000"))).strip(),
            "rollout_steps": str(source.get("rollout_steps", settings.get("rollout_steps", "1000"))).strip(),
            "boundary_m": str(source.get("boundary_m", settings.get("boundary_m", "8.0"))).strip(),
            "command_min": str(source.get("command_min", settings.get("command_min", "-1.0"))).strip(),
            "command_max": str(source.get("command_max", settings.get("command_max", "1.0"))).strip(),
            "seed": str(source.get("seed", settings.get("seed", "42"))).strip(),
            "epochs": str(source.get("epochs", settings.get("epochs", "30"))).strip(),
            "batch_size": str(source.get("batch_size", settings.get("batch_size", "256"))).strip(),
            "learning_rate": str(source.get("learning_rate", settings.get("learning_rate", "1e-3"))).strip(),
            "lambda_smooth": str(source.get("lambda_smooth", settings.get("lambda_smooth", "0"))).strip(),
            "cmd_label_threshold": str(source.get("cmd_label_threshold", settings.get("cmd_label_threshold", "0.2"))).strip(),
            "cmd_label_alpha": str(source.get("cmd_label_alpha", settings.get("cmd_label_alpha", "0"))).strip(),
            "val_ratio": str(source.get("val_ratio", settings.get("val_ratio", "0.1"))).strip(),
            "selected_datasets": list(source.get("selected_datasets", settings.get("selected_datasets", []))),
        })
        self.moe_settings = settings
        self.moe_settings_by_env[env_id] = dict(settings)
        return settings

    def _collect_moe_manual_ui_settings(self, source_settings=None):
        self._ensure_moe_manual_defaults()
        source = dict(source_settings or self.moe_manual_settings)
        env_id = self.env_id_cb.currentText()
        settings = dict(self.moe_manual_settings)
        settings.update({
            "env_id": env_id,
            "policy_a_path": str(source.get("policy_a_path", "")).strip(),
            "policy_b_path": str(source.get("policy_b_path", "")).strip(),
            "policy_a_action_scales": str(source.get("policy_a_action_scales", settings.get("policy_a_action_scales", ""))).strip(),
            "policy_b_action_scales": str(source.get("policy_b_action_scales", settings.get("policy_b_action_scales", ""))).strip(),
            "output_action_scales": str(source.get("output_action_scales", settings.get("output_action_scales", ""))).strip(),
            "output_path": str(source.get("output_path", settings.get("output_path", ""))).strip(),
        })
        self.moe_manual_settings = settings
        self.moe_manual_settings_by_env[env_id] = dict(settings)
        return settings

    def _collect_homing_ui_settings(self, source_settings=None):
        source = dict(source_settings or self.homing_settings)
        env_id = str(source.get("env_id", self.env_id_cb.currentText()))
        self._ensure_homing_defaults(env_id)
        self._ensure_final_pose_defaults_for_env(env_id)
        self._ensure_homing_command_ranges_for_env(env_id)
        settings = dict(self.homing_settings)
        settings.update({
            "env_id": env_id,
            "policy_path": str(source.get("policy_path", "")).strip(),
            "terrains": ["flat"],
            "samples": str(source.get("samples", settings.get("samples", "50000"))).strip(),
            "rollout_steps": str(source.get("rollout_steps", settings.get("rollout_steps", "1000"))).strip(),
            "homing_trajectory_seconds": str(source.get("homing_trajectory_seconds", settings.get("homing_trajectory_seconds", "3.0"))).strip(),
            "homing_stand_warmup_steps": str(source.get("homing_stand_warmup_steps", settings.get("homing_stand_warmup_steps", "200"))).strip(),
            "homing_balance_blend": str(source.get("homing_balance_blend", settings.get("homing_balance_blend", "0.0"))).strip(),
            "command_min": str(source.get("command_min", settings.get("command_min", "-1.0"))).strip(),
            "command_max": str(source.get("command_max", settings.get("command_max", "1.0"))).strip(),
            "command_mins": self._homing_command_range_csv(env_id, "mins"),
            "command_maxs": self._homing_command_range_csv(env_id, "maxs"),
            "seed": str(source.get("seed", settings.get("seed", "42"))).strip(),
            "final_pos": self._final_pose_csv(env_id, "joints"),
            "final_vel": self._final_pose_csv(env_id, "velocities"),
            "final_pose_same": "1" if self.final_pose_settings_by_env[env_id].get("same", True) else "0",
            "final_pose_priorities": self._final_pose_csv(env_id, "priorities"),
            "epochs": str(source.get("epochs", settings.get("epochs", "30"))).strip(),
            "batch_size": str(source.get("batch_size", settings.get("batch_size", "256"))).strip(),
            "learning_rate": str(source.get("learning_rate", settings.get("learning_rate", "1e-3"))).strip(),
            "val_ratio": str(source.get("val_ratio", settings.get("val_ratio", "0.1"))).strip(),
            "hidden_dim": str(source.get("hidden_dim", settings.get("hidden_dim", "256"))).strip(),
            "ppo_total_steps": str(source.get("ppo_total_steps", settings.get("ppo_total_steps", "20000"))).strip(),
            "ppo_num_envs": str(source.get("ppo_num_envs", settings.get("ppo_num_envs", "4"))).strip(),
            "ppo_rollout_steps": str(source.get("ppo_rollout_steps", settings.get("ppo_rollout_steps", "256"))).strip(),
            "ppo_epochs": str(source.get("ppo_epochs", settings.get("ppo_epochs", "4"))).strip(),
            "ppo_learning_rate": str(source.get("ppo_learning_rate", settings.get("ppo_learning_rate", "3e-4"))).strip(),
            "ppo_domain_randomize": str(source.get("ppo_domain_randomize", settings.get("ppo_domain_randomize", "0.3"))).strip(),
            "ppo_supervised_init": str(source.get("ppo_supervised_init", settings.get("ppo_supervised_init", "1"))).strip(),
            "ppo_use_trajectory_reward": str(source.get("ppo_use_trajectory_reward", settings.get("ppo_use_trajectory_reward", "1"))).strip(),
            "ppo_mask_wheel_actions": str(source.get("ppo_mask_wheel_actions", settings.get("ppo_mask_wheel_actions", "1"))).strip(),
            "ppo_strategy_preset": str(source.get("ppo_strategy_preset", settings.get("ppo_strategy_preset", "light"))).strip(),
            "reward_track": str(source.get("reward_track", settings.get("reward_track", "6.0"))).strip(),
            "reward_base_acc": str(source.get("reward_base_acc", settings.get("reward_base_acc", "0.002"))).strip(),
            "reward_upright": str(source.get("reward_upright", settings.get("reward_upright", "2.0"))).strip(),
            "reward_action_rate": str(source.get("reward_action_rate", settings.get("reward_action_rate", "0.04"))).strip(),
            "reward_contact": str(source.get("reward_contact", settings.get("reward_contact", "0.0005"))).strip(),
            "selected_datasets": list(source.get("selected_datasets", settings.get("selected_datasets", []))),
            "checkpoint_path": str(source.get("checkpoint_path", settings.get("checkpoint_path", ""))).strip(),
            "output_path": str(source.get("output_path", settings.get("output_path", ""))).strip(),
        })
        self.homing_settings = settings
        self.homing_settings_by_env[env_id] = dict(settings)
        return settings

    def _collect_ctbc_ui_settings(self, source_settings=None):
        source = dict(source_settings or self.ctbc_settings)
        env_id = str(source.get("env_id", self.env_id_cb.currentText()))
        self._ensure_ctbc_defaults(env_id)
        self._ensure_homing_command_ranges_for_env(env_id)
        settings = dict(self.ctbc_settings)
        settings.update({
            "task_mode": "ctbc",
            "env_id": env_id,
            "policy_path": str(source.get("policy_path", settings.get("policy_path", ""))).strip(),
            "checkpoint_path": str(source.get("checkpoint_path", settings.get("checkpoint_path", ""))).strip(),
            "output_path": str(source.get("output_path", settings.get("output_path", ""))).strip(),
            "ctbc_terrain": str(source.get("ctbc_terrain", settings.get("ctbc_terrain", "stairs_up_easy"))).strip(),
            "ctbc_contact_threshold": str(source.get("ctbc_contact_threshold", settings.get("ctbc_contact_threshold", "30.0"))).strip(),
            "ctbc_contact_window": str(source.get("ctbc_contact_window", settings.get("ctbc_contact_window", "3"))).strip(),
            "ctbc_lift_amplitude": str(source.get("ctbc_lift_amplitude", settings.get("ctbc_lift_amplitude", "0.90"))).strip(),
            "ctbc_lift_period": str(source.get("ctbc_lift_period", settings.get("ctbc_lift_period", "0.75"))).strip(),
            "ctbc_anneal_ratio": str(source.get("ctbc_anneal_ratio", settings.get("ctbc_anneal_ratio", "0.70"))).strip(),
            "ctbc_episode_steps": str(source.get("ctbc_episode_steps", settings.get("ctbc_episode_steps", "1024"))).strip(),
            "ctbc_residual_limit": str(source.get("ctbc_residual_limit", settings.get("ctbc_residual_limit", "4.0"))).strip(),
            "ctbc_gate_height_threshold": str(source.get("ctbc_gate_height_threshold", settings.get("ctbc_gate_height_threshold", "0.06"))).strip(),
            "ctbc_gate_height_softness": str(source.get("ctbc_gate_height_softness", settings.get("ctbc_gate_height_softness", "0.025"))).strip(),
            "ctbc_gate_rise": str(source.get("ctbc_gate_rise", settings.get("ctbc_gate_rise", "0.35"))).strip(),
            "ctbc_gate_fall": str(source.get("ctbc_gate_fall", settings.get("ctbc_gate_fall", "0.08"))).strip(),
            "ctbc_gate_lift_threshold": str(source.get("ctbc_gate_lift_threshold", settings.get("ctbc_gate_lift_threshold", "0.25"))).strip(),
            "ctbc_gate_reward_threshold": str(source.get("ctbc_gate_reward_threshold", settings.get("ctbc_gate_reward_threshold", "0.35"))).strip(),
            "ctbc_assist_trigger_gate": str(source.get("ctbc_assist_trigger_gate", settings.get("ctbc_assist_trigger_gate", "0.12"))).strip(),
            "ctbc_assist_gate_floor": str(source.get("ctbc_assist_gate_floor", settings.get("ctbc_assist_gate_floor", "0.85"))).strip(),
            "ctbc_assist_min": str(source.get("ctbc_assist_min", settings.get("ctbc_assist_min", "0.0"))).strip(),
            "ctbc_gate_residual_runtime": str(source.get("ctbc_gate_residual_runtime", settings.get("ctbc_gate_residual_runtime", "0"))).strip(),
            "ctbc_anneal_bc_with_assist": str(source.get("ctbc_anneal_bc_with_assist", settings.get("ctbc_anneal_bc_with_assist", "1"))).strip(),
            "ctbc_distill_primitive": str(source.get("ctbc_distill_primitive", settings.get("ctbc_distill_primitive", "1"))).strip(),
            "ctbc_bc_weight_min": str(source.get("ctbc_bc_weight_min", settings.get("ctbc_bc_weight_min", "0.15"))).strip(),
            "ctbc_reflex_only": str(source.get("ctbc_reflex_only", settings.get("ctbc_reflex_only", "1"))).strip(),
            "ctbc_controller_candidates": str(source.get("ctbc_controller_candidates", settings.get("ctbc_controller_candidates", "64"))).strip(),
            "ctbc_reflex_samples": str(source.get("ctbc_reflex_samples", settings.get("ctbc_reflex_samples", "8192"))).strip(),
            "ctbc_reflex_epochs": str(source.get("ctbc_reflex_epochs", settings.get("ctbc_reflex_epochs", "12"))).strip(),
            "ctbc_reflex_batch": str(source.get("ctbc_reflex_batch", settings.get("ctbc_reflex_batch", "256"))).strip(),
            "ctbc_reflex_lr": str(source.get("ctbc_reflex_lr", settings.get("ctbc_reflex_lr", "3e-4"))).strip(),
            "ctbc_reflex_flat_ratio": str(source.get("ctbc_reflex_flat_ratio", settings.get("ctbc_reflex_flat_ratio", "0.35"))).strip(),
            "ctbc_reflex_gain": str(source.get("ctbc_reflex_gain", settings.get("ctbc_reflex_gain", "1.0"))).strip(),
            "ctbc_reflex_segment_steps": str(source.get("ctbc_reflex_segment_steps", settings.get("ctbc_reflex_segment_steps", "128"))).strip(),
            "ctbc_fast_teacher_steps": str(source.get("ctbc_fast_teacher_steps", settings.get("ctbc_fast_teacher_steps", "4096"))).strip(),
            "ctbc_fast_teacher_epochs": str(source.get("ctbc_fast_teacher_epochs", settings.get("ctbc_fast_teacher_epochs", "6"))).strip(),
            "ctbc_fast_teacher_batch": str(source.get("ctbc_fast_teacher_batch", settings.get("ctbc_fast_teacher_batch", "256"))).strip(),
            "ctbc_fast_teacher_lr": str(source.get("ctbc_fast_teacher_lr", settings.get("ctbc_fast_teacher_lr", "2e-4"))).strip(),
            "ctbc_fast_teacher_gain": str(source.get("ctbc_fast_teacher_gain", settings.get("ctbc_fast_teacher_gain", "1.0"))).strip(),
            "ctbc_fast_teacher_stair_height": str(source.get("ctbc_fast_teacher_stair_height", settings.get("ctbc_fast_teacher_stair_height", "0.12"))).strip(),
            "ctbc_safe_tilt": str(source.get("ctbc_safe_tilt", settings.get("ctbc_safe_tilt", "0.22"))).strip(),
            "ctbc_emergency_tilt": str(source.get("ctbc_emergency_tilt", settings.get("ctbc_emergency_tilt", "0.34"))).strip(),
            "ctbc_terminate_tilt": str(source.get("ctbc_terminate_tilt", settings.get("ctbc_terminate_tilt", "0.42"))).strip(),
            "ctbc_tilt_guard_penalty": str(source.get("ctbc_tilt_guard_penalty", settings.get("ctbc_tilt_guard_penalty", "8.0"))).strip(),
            "ctbc_bad_contact_threshold": str(source.get("ctbc_bad_contact_threshold", settings.get("ctbc_bad_contact_threshold", "1.0"))).strip(),
            "ctbc_bad_contact_penalty": str(source.get("ctbc_bad_contact_penalty", settings.get("ctbc_bad_contact_penalty", "20.0"))).strip(),
            "ctbc_lift_cooldown": str(source.get("ctbc_lift_cooldown", settings.get("ctbc_lift_cooldown", "0.35"))).strip(),
            "ctbc_contact_baseline_alpha": str(source.get("ctbc_contact_baseline_alpha", settings.get("ctbc_contact_baseline_alpha", "0.02"))).strip(),
            "ctbc_contact_spike_threshold": str(source.get("ctbc_contact_spike_threshold", settings.get("ctbc_contact_spike_threshold", "80.0"))).strip(),
            "ctbc_force_alternating_lift": str(source.get("ctbc_force_alternating_lift", settings.get("ctbc_force_alternating_lift", "1"))).strip(),
            "ctbc_curriculum_enabled": str(source.get("ctbc_curriculum_enabled", settings.get("ctbc_curriculum_enabled", "1"))).strip(),
            "ctbc_stair_height_min": str(source.get("ctbc_stair_height_min", settings.get("ctbc_stair_height_min", "0.025"))).strip(),
            "ctbc_stair_height_max": str(source.get("ctbc_stair_height_max", settings.get("ctbc_stair_height_max", "0.20"))).strip(),
            "ctbc_curriculum_ratio": str(source.get("ctbc_curriculum_ratio", settings.get("ctbc_curriculum_ratio", "0.60"))).strip(),
            "ctbc_select_after_ratio": str(source.get("ctbc_select_after_ratio", settings.get("ctbc_select_after_ratio", "0.70"))).strip(),
            "ctbc_shoulder_gain": str(source.get("ctbc_shoulder_gain", settings.get("ctbc_shoulder_gain", "0.50"))).strip(),
            "ctbc_leg_gain": str(source.get("ctbc_leg_gain", settings.get("ctbc_leg_gain", "0.0"))).strip(),
            "ctbc_leg_push_gain": str(source.get("ctbc_leg_push_gain", settings.get("ctbc_leg_push_gain", "1.75"))).strip(),
            "ctbc_hip_gain": str(source.get("ctbc_hip_gain", settings.get("ctbc_hip_gain", "0.0"))).strip(),
            "ctbc_stance_gain": str(source.get("ctbc_stance_gain", settings.get("ctbc_stance_gain", "0.30"))).strip(),
            "ctbc_wheel_push_gain": str(source.get("ctbc_wheel_push_gain", settings.get("ctbc_wheel_push_gain", "0.0"))).strip(),
            "ctbc_ff_clip": str(source.get("ctbc_ff_clip", settings.get("ctbc_ff_clip", "4.0"))).strip(),
            "ctbc_action_clip": str(source.get("ctbc_action_clip", settings.get("ctbc_action_clip", "4.0"))).strip(),
            "ctbc_compensate_action_scale": str(source.get("ctbc_compensate_action_scale", settings.get("ctbc_compensate_action_scale", "1"))).strip(),
            "ctbc_clearance_target": str(source.get("ctbc_clearance_target", settings.get("ctbc_clearance_target", "0.14"))).strip(),
            "ctbc_base_height_target": str(source.get("ctbc_base_height_target", settings.get("ctbc_base_height_target", "0.14"))).strip(),
            "ctbc_clearance_stair_ratio": str(source.get("ctbc_clearance_stair_ratio", settings.get("ctbc_clearance_stair_ratio", "0.90"))).strip(),
            "ctbc_climb_stair_ratio": str(source.get("ctbc_climb_stair_ratio", settings.get("ctbc_climb_stair_ratio", "0.75"))).strip(),
            "ctbc_reward_lift": str(source.get("ctbc_reward_lift", settings.get("ctbc_reward_lift", "2.0"))).strip(),
            "ctbc_reward_clearance": str(source.get("ctbc_reward_clearance", settings.get("ctbc_reward_clearance", "1.0"))).strip(),
            "ctbc_reward_wheel_clearance": str(source.get("ctbc_reward_wheel_clearance", settings.get("ctbc_reward_wheel_clearance", "4.0"))).strip(),
            "ctbc_reward_base_height": str(source.get("ctbc_reward_base_height", settings.get("ctbc_reward_base_height", "4.0"))).strip(),
            "ctbc_reward_stair_success": str(source.get("ctbc_reward_stair_success", settings.get("ctbc_reward_stair_success", "5.0"))).strip(),
            "ctbc_hard_stair_threshold": str(source.get("ctbc_hard_stair_threshold", settings.get("ctbc_hard_stair_threshold", "0.14"))).strip(),
            "ctbc_hard_stair_fail_penalty": str(source.get("ctbc_hard_stair_fail_penalty", settings.get("ctbc_hard_stair_fail_penalty", "1.5"))).strip(),
            "ctbc_reward_forward_progress": str(source.get("ctbc_reward_forward_progress", settings.get("ctbc_reward_forward_progress", "35.0"))).strip(),
            "ctbc_min_forward_progress": str(source.get("ctbc_min_forward_progress", settings.get("ctbc_min_forward_progress", "0.010"))).strip(),
            "ctbc_reward_stair_forward": str(source.get("ctbc_reward_stair_forward", settings.get("ctbc_reward_stair_forward", "2.0"))).strip(),
            "ctbc_reward_stair_motion": str(source.get("ctbc_reward_stair_motion", settings.get("ctbc_reward_stair_motion", "4.0"))).strip(),
            "ctbc_no_progress_penalty": str(source.get("ctbc_no_progress_penalty", settings.get("ctbc_no_progress_penalty", "1.0"))).strip(),
            "ctbc_reward_height_progress": str(source.get("ctbc_reward_height_progress", settings.get("ctbc_reward_height_progress", "30.0"))).strip(),
            "ctbc_reward_balance_on_stair": str(source.get("ctbc_reward_balance_on_stair", settings.get("ctbc_reward_balance_on_stair", "0.7"))).strip(),
            "ctbc_min_climb_height": str(source.get("ctbc_min_climb_height", settings.get("ctbc_min_climb_height", "0.015"))).strip(),
            "ctbc_no_climb_penalty": str(source.get("ctbc_no_climb_penalty", settings.get("ctbc_no_climb_penalty", "0.12"))).strip(),
            "ctbc_base_imitation": str(source.get("ctbc_base_imitation", settings.get("ctbc_base_imitation", "0.5"))).strip(),
            "ctbc_non_wheel_contact_penalty": str(source.get("ctbc_non_wheel_contact_penalty", settings.get("ctbc_non_wheel_contact_penalty", "4.0"))).strip(),
            "ctbc_command_x_min": str(source.get("ctbc_command_x_min", settings.get("ctbc_command_x_min", "0.35"))).strip(),
            "ctbc_command_x_max": str(source.get("ctbc_command_x_max", settings.get("ctbc_command_x_max", "0.70"))).strip(),
            "ctbc_command_y_abs": str(source.get("ctbc_command_y_abs", settings.get("ctbc_command_y_abs", "0.03"))).strip(),
            "ctbc_command_yaw_abs": str(source.get("ctbc_command_yaw_abs", settings.get("ctbc_command_yaw_abs", "0.05"))).strip(),
            "reward_track": str(source.get("reward_track", settings.get("reward_track", "1.2"))).strip(),
            "reward_upright": str(source.get("reward_upright", settings.get("reward_upright", "2.0"))).strip(),
            "reward_action_rate": str(source.get("reward_action_rate", settings.get("reward_action_rate", "0.04"))).strip(),
            "ppo_total_steps": str(source.get("ppo_total_steps", settings.get("ppo_total_steps", "1000000"))).strip(),
            "ppo_num_envs": str(source.get("ppo_num_envs", settings.get("ppo_num_envs", "32"))).strip(),
            "ppo_rollout_steps": str(source.get("ppo_rollout_steps", settings.get("ppo_rollout_steps", "512"))).strip(),
            "ppo_epochs": str(source.get("ppo_epochs", settings.get("ppo_epochs", "4"))).strip(),
            "ppo_learning_rate": str(source.get("ppo_learning_rate", settings.get("ppo_learning_rate", "5e-5"))).strip(),
            "ppo_domain_randomize": str(source.get("ppo_domain_randomize", settings.get("ppo_domain_randomize", "0.05"))).strip(),
            "hidden_dim": str(source.get("hidden_dim", settings.get("hidden_dim", "256"))).strip(),
            "seed": str(source.get("seed", settings.get("seed", "42"))).strip(),
            "command_min": str(source.get("command_min", settings.get("command_min", "-1.0"))).strip(),
            "command_max": str(source.get("command_max", settings.get("command_max", "1.0"))).strip(),
            "command_mins": self._homing_command_range_csv(env_id, "mins"),
            "command_maxs": self._homing_command_range_csv(env_id, "maxs"),
        })
        if not settings.get("ctbc_terrain"):
            settings["ctbc_terrain"] = "stairs_up_easy"
        try:
            if float(settings.get("ctbc_reward_forward_progress", 35.0)) <= 8.0:
                settings["ctbc_reward_forward_progress"] = "35.0"
        except Exception:
            settings["ctbc_reward_forward_progress"] = "35.0"
        self.ctbc_settings = settings
        self.ctbc_settings_by_env[env_id] = dict(settings)
        return settings

    def _moe_dataset_root(self, env_id: str):
        return os.path.join(self._repo_root(), "envs", env_id, "dataset", "moe_gate")

    def _homing_dataset_root(self, env_id: str):
        return os.path.join(self._repo_root(), "envs", env_id, "dataset", "homing")

    def _moe_alpha_onnx_path(self, env_id: str):
        latest_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "moe_gate", "latest")
        alpha_path = os.path.join(latest_dir, "moe_alpha.onnx")
        if os.path.isfile(alpha_path):
            return alpha_path
        return os.path.join(latest_dir, "moe_gate.onnx")

    def _list_moe_datasets(self, env_id: str):
        dataset_root = self._moe_dataset_root(env_id)
        if not os.path.isdir(dataset_root):
            return []
        datasets = []
        for run_name in sorted(os.listdir(dataset_root), reverse=True):
            run_dir = os.path.join(dataset_root, run_name)
            dataset_path = os.path.join(run_dir, "dataset.npz")
            if not os.path.isfile(dataset_path):
                continue
            try:
                with np.load(dataset_path) as payload:
                    samples = int(payload["obs"].shape[0])
                    obs_dim = int(payload["obs"].shape[1])
            except Exception:
                samples = 0
                obs_dim = 0
            datasets.append({
                "path": dataset_path,
                "label": f"{run_name} | {samples} samples | obs {obs_dim}",
            })
        return datasets

    def _list_homing_datasets(self, env_id: str):
        dataset_root = self._homing_dataset_root(env_id)
        if not os.path.isdir(dataset_root):
            return []
        datasets = []
        for run_name in sorted(os.listdir(dataset_root), reverse=True):
            run_dir = os.path.join(dataset_root, run_name)
            dataset_path = os.path.join(run_dir, "dataset.npz")
            if not os.path.isfile(dataset_path):
                continue
            try:
                with np.load(dataset_path) as payload:
                    samples = int(payload["input"].shape[0])
                    input_dim = int(payload["input"].shape[1])
                    action_dim = int(payload["action_label"].shape[1])
            except Exception:
                samples = 0
                input_dim = 0
                action_dim = 0
            datasets.append({
                "path": dataset_path,
                "label": f"{run_name} | {samples} samples | input {input_dim} -> action {action_dim}",
            })
        return datasets

    def _refresh_moe_dialog(self):
        if self.moe_dialog is None:
            return
        self._ensure_moe_defaults()
        settings = self.moe_settings
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        self.moe_dialog.set_settings(settings)
        self.moe_dialog.set_available_datasets(
            self._list_moe_datasets(env_id),
            settings.get("selected_datasets", []),
        )
        running = self.moe_thread is not None and self.moe_thread.isRunning()
        self.moe_dialog.set_running(running)
        if self._moe_last_summary:
            self.moe_dialog.set_status(f"last: {self._moe_last_summary.get('samples', self._moe_last_summary.get('best_val_loss', 'done'))}")

    def _refresh_homing_dialog(self):
        if self.homing_dialog is None:
            return
        dialog_env = self.homing_dialog.get_settings().get("env_id", "") if self.homing_dialog is not None else ""
        self._ensure_homing_defaults(dialog_env or self.env_id_cb.currentText())
        settings = self.homing_settings
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        self.homing_dialog.set_settings(settings)
        self.homing_dialog.set_available_datasets(
            self._list_homing_datasets(env_id),
            settings.get("selected_datasets", []),
        )
        running = self.homing_thread is not None and self.homing_thread.isRunning()
        self.homing_dialog.set_running(running)
        if self._homing_last_summary:
            self.homing_dialog.set_status(f"last: {self._homing_last_summary.get('samples', self._homing_last_summary.get('onnx_path', 'done'))}")

    def _refresh_ctbc_dialog(self):
        if self.ctbc_dialog is None:
            return
        dialog_env = self.ctbc_dialog.get_settings().get("env_id", "") if self.ctbc_dialog is not None else ""
        self._ensure_ctbc_defaults(dialog_env or self.env_id_cb.currentText())
        settings = self.ctbc_settings
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        self.ctbc_dialog.set_settings(settings)
        self.ctbc_dialog.set_available_datasets([], [])
        running = self.ctbc_thread is not None and self.ctbc_thread.isRunning()
        self.ctbc_dialog.set_running(running)
        if self._homing_last_summary and self._homing_last_summary.get("mode") == "ctbc_fine_tune":
            self.ctbc_dialog.set_status(f"last: {self._homing_last_summary.get('onnx_path', 'done')}")

    def _on_homing_env_changed(self, previous_env_id, new_env_id):
        if self.homing_dialog is not None and previous_env_id:
            previous_settings = self.homing_dialog.get_settings()
            previous_settings["env_id"] = str(previous_env_id)
            self._collect_homing_ui_settings(previous_settings)
        self._ensure_homing_defaults(str(new_env_id))
        self._refresh_homing_dialog()

    def _on_ctbc_env_changed(self, previous_env_id, new_env_id):
        if self.ctbc_dialog is not None and previous_env_id:
            previous_settings = self.ctbc_dialog.get_settings()
            previous_settings["env_id"] = str(previous_env_id)
            self._collect_ctbc_ui_settings(previous_settings)
        self._ensure_ctbc_defaults(str(new_env_id))
        self._refresh_ctbc_dialog()

    def _refresh_vision_train_dialog(self):
        if self.vision_train_dialog is None:
            return
        env_id = self.env_id_cb.currentText()
        self._ensure_vision_train_defaults()
        self.vision_train_dialog.set_env_id(env_id)
        self.vision_train_dialog.set_settings(self.vision_train_settings)
        self.vision_train_dialog.set_available_datasets(
            self._list_vision_train_datasets(env_id),
            self.vision_train_settings.get("selected_datasets", []),
        )
        self.vision_train_dialog.set_running(
            self.vision_train_thread is not None and self.vision_train_thread.isRunning()
        )
        self._update_vision_train_status_label()

    def _ensure_fine_tune_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.fine_tune_settings_by_env:
            self.fine_tune_settings_by_env[env_id] = self._make_fine_tune_defaults(env_id)
        cached = dict(self.fine_tune_settings_by_env[env_id])
        action_dim = self._get_current_action_dim(env_id)
        raw_bias = cached.get("bias", [])
        bias = [to_float(v, 0.0) for v in raw_bias] if isinstance(raw_bias, list) else []
        if len(bias) < action_dim:
            bias = bias + [0.0] * (action_dim - len(bias))
        elif len(bias) > action_dim:
            bias = bias[:action_dim]
        self.fine_tune_settings = {
            "enabled": bool(cached.get("enabled", False)),
            "ridge_lambda": str(cached.get("ridge_lambda", "1e-4")),
            "max_samples": str(cached.get("max_samples", "5000")),
            "bias": bias,
        }
        self.fine_tune_settings_by_env[env_id] = dict(self.fine_tune_settings)

    def _sync_fine_tune_controls_from_cache(self):
        if not hasattr(self, "fine_tune_enable_cb"):
            return
        self._ensure_fine_tune_defaults()
        self.fine_tune_enable_cb.blockSignals(True)
        self.fine_tune_enable_cb.setChecked(bool(self.fine_tune_settings.get("enabled", False)))
        self.fine_tune_enable_cb.blockSignals(False)
        self.fine_tune_ridge_lambda_le.setText(str(self.fine_tune_settings.get("ridge_lambda", "1e-4")))
        self.fine_tune_max_samples_le.setText(str(self.fine_tune_settings.get("max_samples", "5000")))
        if self.fine_tune_bias_dialog is not None:
            self.fine_tune_bias_dialog.close()
            self.fine_tune_bias_dialog = None
        self._update_fine_tune_status_label()

    def _collect_fine_tune_ui_settings(self):
        self._ensure_fine_tune_defaults()
        settings = {
            "enabled": bool(self.fine_tune_enable_cb.isChecked()) if hasattr(self, "fine_tune_enable_cb") else bool(self.fine_tune_settings.get("enabled", False)),
            "ridge_lambda": self.fine_tune_ridge_lambda_le.text().strip() if hasattr(self, "fine_tune_ridge_lambda_le") else str(self.fine_tune_settings.get("ridge_lambda", "1e-4")),
            "max_samples": self.fine_tune_max_samples_le.text().strip() if hasattr(self, "fine_tune_max_samples_le") else str(self.fine_tune_settings.get("max_samples", "5000")),
            "bias": list(self.fine_tune_settings.get("bias", [])),
        }
        env_id = self.env_id_cb.currentText()
        self.fine_tune_settings = settings
        self.fine_tune_settings_by_env[env_id] = dict(settings)
        return settings

    def _apply_fine_tune_settings_to_tester(self):
        if not self.tester:
            return
        settings = self._collect_fine_tune_ui_settings()
        self.tester.set_fine_tune_enabled(settings["enabled"])
        self.tester.set_fine_tune_max_samples(to_int(settings["max_samples"], 5000))
        self.tester.set_fine_tune_bias(settings["bias"])

    def _make_initial_pose_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        raw = env_cfg.get("initial_positions", env_cfg.get("initial_pose", env_cfg.get("joint_offsets", {}))) or {}
        pose_defaults = get_default_initial_pose(env_id)
        joint_defaults = pose_defaults["joints"]
        base_z = pose_defaults["base_z"]
        joints_raw = raw.get("joints", raw) if isinstance(raw, dict) else {}
        if isinstance(raw, dict):
            base_z = raw.get("base_z", raw.get("z", base_z))
        if isinstance(joints_raw, dict):
            for joint_name in joint_defaults:
                if joint_name in joints_raw:
                    joint_defaults[joint_name] = str(joints_raw[joint_name])
                else:
                    joint_defaults[joint_name] = str(joint_defaults[joint_name])
        else:
            for joint_name in joint_defaults:
                joint_defaults[joint_name] = str(joint_defaults[joint_name])
        return {"base_z": str(base_z), "joints": joint_defaults}

    def _ensure_initial_pose_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.initial_pose_settings_by_env:
            self.initial_pose_settings_by_env[env_id] = self._make_initial_pose_defaults(env_id)
        self.initial_pose_settings = {
            "base_z": str((self.initial_pose_settings_by_env[env_id]).get("base_z", self._make_initial_pose_defaults(env_id).get("base_z", "0.3"))),
            "joints": dict((self.initial_pose_settings_by_env[env_id]).get("joints", {}))
        }

    def _make_final_pose_defaults(self, env_id: str):
        initial = self._make_initial_pose_defaults(env_id).get("joints", {})
        priorities = {}
        group_order = {}
        for joint_name in initial.keys():
            group_key = self._final_pose_priority_group_key(joint_name)
            if group_key not in group_order:
                group_order[group_key] = len(group_order) + 1
            priorities[joint_name] = str(group_order[group_key])
        return {
            "joints": dict(initial),
            "velocities": {joint_name: "0.0" for joint_name in initial.keys()},
            "same": True,
            "priorities": priorities,
        }

    def _ensure_final_pose_defaults_for_env(self, env_id: str):
        if env_id not in self.final_pose_settings_by_env:
            self.final_pose_settings_by_env[env_id] = self._make_final_pose_defaults(env_id)
        cached = self.final_pose_settings_by_env[env_id]
        joint_names = list(get_initial_pose_joint_names(env_id))
        joints = dict(cached.get("joints", {}))
        velocities = dict(cached.get("velocities", {}))
        priorities = dict(cached.get("priorities", {}))
        same = cached.get("same", True)
        if not isinstance(same, bool):
            same = str(same).strip().lower() not in ("0", "false", "no", "off")
        group_order = {}
        for joint_name in joint_names:
            group_key = self._final_pose_priority_group_key(joint_name)
            if group_key not in group_order:
                group_order[group_key] = len(group_order) + 1
            joints.setdefault(joint_name, "0.0")
            velocities.setdefault(joint_name, "0.0")
            priorities.setdefault(joint_name, str(group_order[group_key]))
        grouped_priorities = {}
        for joint_name in joint_names:
            group_key = self._final_pose_priority_group_key(joint_name)
            grouped_priorities.setdefault(group_key, str(priorities.get(joint_name, group_order[group_key])))
        self.final_pose_settings_by_env[env_id] = {
            "joints": {joint_name: str(joints.get(joint_name, "0.0")) for joint_name in joint_names},
            "velocities": {joint_name: str(velocities.get(joint_name, "0.0")) for joint_name in joint_names},
            "same": bool(same),
            "priorities": {
                joint_name: str(grouped_priorities[self._final_pose_priority_group_key(joint_name)])
                for joint_name in joint_names
            },
        }
        if env_id == self.env_id_cb.currentText():
            self.final_pose_settings = {
                "joints": dict(self.final_pose_settings_by_env[env_id]["joints"]),
                "velocities": dict(self.final_pose_settings_by_env[env_id]["velocities"]),
                "same": bool(self.final_pose_settings_by_env[env_id]["same"]),
                "priorities": dict(self.final_pose_settings_by_env[env_id]["priorities"]),
            }

    def _ensure_final_pose_defaults(self):
        self._ensure_final_pose_defaults_for_env(self.env_id_cb.currentText())

    @staticmethod
    def _final_pose_priority_group_key(joint_name: str):
        name = str(joint_name)
        for prefix in ("left_", "right_", "FL_", "FR_", "RL_", "RR_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _final_pose_csv(self, env_id: str, key: str):
        self._ensure_final_pose_defaults_for_env(env_id)
        values = self.final_pose_settings_by_env[env_id].get(key, {})
        return ",".join(str(values.get(joint_name, "0.0")) for joint_name in get_initial_pose_joint_names(env_id))

    def _command_dim_for_env(self, env_id: str):
        if env_id in self.obs_settings_by_env:
            return max(1, to_int(self.obs_settings_by_env[env_id].get("command_dim", 6), 6))
        return max(1, to_int(self._make_observation_defaults(env_id).get("command_dim", 6), 6))

    def _make_homing_command_range_defaults(self, env_id: str):
        cmd_dim = self._command_dim_for_env(env_id)
        mins = ["-1.0"] * cmd_dim
        maxs = ["1.0"] * cmd_dim
        for index in (1, 3):
            if index < cmd_dim:
                mins[index] = "0.0"
                maxs[index] = "0.0"
        return {
            "mins": mins,
            "maxs": maxs,
        }

    def _ensure_homing_command_ranges_for_env(self, env_id: str):
        if env_id not in self.homing_command_ranges_by_env:
            self.homing_command_ranges_by_env[env_id] = self._make_homing_command_range_defaults(env_id)
        cmd_dim = self._command_dim_for_env(env_id)
        cached = self.homing_command_ranges_by_env[env_id]
        mins = list(cached.get("mins", []))
        maxs = list(cached.get("maxs", []))
        while len(mins) < cmd_dim:
            mins.append("-1.0")
        while len(maxs) < cmd_dim:
            maxs.append("1.0")
        for index in (1, 3):
            if index < cmd_dim and str(mins[index]) == "-1.0" and str(maxs[index]) == "1.0":
                mins[index] = "0.0"
                maxs[index] = "0.0"
        self.homing_command_ranges_by_env[env_id] = {
            "mins": [str(value) for value in mins[:cmd_dim]],
            "maxs": [str(value) for value in maxs[:cmd_dim]],
        }
        if env_id == self.env_id_cb.currentText() or not self.homing_command_ranges:
            self.homing_command_ranges = {
                "mins": list(self.homing_command_ranges_by_env[env_id]["mins"]),
                "maxs": list(self.homing_command_ranges_by_env[env_id]["maxs"]),
            }

    def _homing_command_range_csv(self, env_id: str, key: str):
        self._ensure_homing_command_ranges_for_env(env_id)
        return ",".join(str(value) for value in self.homing_command_ranges_by_env[env_id].get(key, []))

    def _make_monitor_defaults(self, env_id: str):
        joint_names = list(get_initial_pose_joint_names(env_id))
        default_selected = joint_names[: min(4, len(joint_names))]
        return {
            "available_joints": joint_names,
            "selected_joints": list(default_selected),
        }

    def _default_height_map_frame_body(self, env_id: str):
        _ = env_id
        return "base_link"

    def _make_dataset_height_map_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        settings_cfg = env_cfg.get("settings", env_cfg) if isinstance(env_cfg, dict) else {}
        hm_cfg = settings_cfg.get("height_map", {}) if isinstance(settings_cfg.get("height_map", {}), dict) else {}
        size_x = to_float(hm_cfg.get("size_x", 1.0), 1.0)
        size_y = to_float(hm_cfg.get("size_y", 0.6), 0.6)
        x_forward = to_float(hm_cfg.get("x_forward", size_x / 2.0), size_x / 2.0)
        x_backward = to_float(hm_cfg.get("x_backward", size_x / 2.0), size_x / 2.0)
        y_left = to_float(hm_cfg.get("y_left", size_y / 2.0), size_y / 2.0)
        y_right = to_float(hm_cfg.get("y_right", size_y / 2.0), size_y / 2.0)
        return {
            "x_forward": str(x_forward),
            "x_backward": str(x_backward),
            "y_left": str(y_left),
            "y_right": str(y_right),
            "resolution": "0.1",
            "visualize": False,
            "inference_visualize": False,
            "inference_onnx_path": "",
            "frame_body": self._default_height_map_frame_body(env_id),
            "depth_scale": "8",
        }

    def _make_depth_randomization_defaults(self, env_id: str):
        _ = env_id
        return {
            "enabled": False,
            "camera_xyz_shift_m": "0.01",
            "camera_pitch_shift_deg": "1.0",
            "camera_fov_shift_deg": "1.0",
            "gaussian_prob": "0.3",
            "gaussian_stddev": "0.01",
            "rotation_prob": "0.3",
            "rotation_deg": "2.0",
            "edge_noise_prob": "0.3",
            "edge_noise_ratio": "0.03",
            "small_object_prob": "0.3",
            "small_object_ratio": "0.02",
            "small_object_count": "6",
            "spot_noise_prob": "0.3",
            "spot_noise_ratio": "0.03",
        }

    def _ensure_dataset_height_map_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.dataset_height_map_settings_by_env:
            self.dataset_height_map_settings_by_env[env_id] = self._make_dataset_height_map_defaults(env_id)
        self.dataset_height_map_settings = dict(self.dataset_height_map_settings_by_env[env_id])

    def _ensure_depth_randomization_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.depth_randomization_settings_by_env:
            self.depth_randomization_settings_by_env[env_id] = self._make_depth_randomization_defaults(env_id)
        self.depth_randomization_settings = dict(self.depth_randomization_settings_by_env[env_id])

    def _compute_height_map_grid(
        self,
        x_forward: float,
        x_backward: float,
        y_left: float,
        y_right: float,
        resolution: float,
    ):
        resolution = max(float(resolution), 1e-6)
        size_x = max(0.0, float(x_forward)) + max(0.0, float(x_backward))
        size_y = max(0.0, float(y_left)) + max(0.0, float(y_right))
        res_x = max(1, int(np.floor((size_x / resolution) + 1e-9)))
        res_y = max(1, int(np.floor((size_y / resolution) + 1e-9)))
        return res_x, res_y

    def _ensure_monitor_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.monitor_settings_by_env:
            self.monitor_settings_by_env[env_id] = self._make_monitor_defaults(env_id)
        cached = self.monitor_settings_by_env[env_id]
        available = list(cached.get("available_joints", get_initial_pose_joint_names(env_id)))
        selected = [name for name in cached.get("selected_joints", []) if name in available]
        self.monitor_settings = {
            "available_joints": available,
            "selected_joints": selected,
        }
        self.monitor_settings_by_env[env_id] = dict(self.monitor_settings)

    def _refresh_monitor_joint_checkboxes(self):
        self._ensure_monitor_defaults()
        selected = set(self.monitor_settings.get("selected_joints", []))
        count = len(selected)
        if hasattr(self, "monitor_summary_label"):
            self.monitor_summary_label.setText(f"{count} selected")
        if hasattr(self, "monitor_window_toggle_cb"):
            self.monitor_window_toggle_cb.blockSignals(True)
            self.monitor_window_toggle_cb.setChecked(self.mujoco_overlay.isVisible())
            self.monitor_window_toggle_cb.blockSignals(False)

    def _set_monitor_selection(self, selected):
        env_id = self.env_id_cb.currentText()
        available = self.monitor_settings.get("available_joints", [])
        filtered = [joint_name for joint_name in selected if joint_name in available]
        self.monitor_settings = {
            "available_joints": list(available),
            "selected_joints": filtered,
        }
        self.monitor_settings_by_env[env_id] = dict(self.monitor_settings)
        self._refresh_monitor_joint_checkboxes()
        if not filtered:
            self.mujoco_overlay.clear_overlay()
        if self.tester is not None:
            self.tester.set_monitor_joints(filtered)

    def _on_monitor_window_toggled(self, checked):
        if not checked:
            self.mujoco_overlay.clear_overlay()

    def _update_monitor_overlay(self, payload):
        if not hasattr(self, "monitor_window_toggle_cb") or not self.monitor_window_toggle_cb.isChecked():
            return
        self.mujoco_overlay.update_overlay(payload)

    def _on_monitor_overlay_closed(self):
        if hasattr(self, "monitor_window_toggle_cb"):
            self.monitor_window_toggle_cb.blockSignals(True)
            self.monitor_window_toggle_cb.setChecked(False)
            self.monitor_window_toggle_cb.blockSignals(False)

    def _on_alpha_vis_toggled(self, checked):
        if not checked:
            self.alpha_overlay.clear_overlay()

    def _update_alpha_overlay(self, payload):
        if not hasattr(self, "moe_alpha_vis_cb") or not self.moe_alpha_vis_cb.isChecked():
            return
        self.alpha_overlay.update_overlay(payload if isinstance(payload, dict) else {})

    def _on_alpha_overlay_closed(self):
        if hasattr(self, "moe_alpha_vis_cb"):
            self.moe_alpha_vis_cb.blockSignals(True)
            self.moe_alpha_vis_cb.setChecked(False)
            self.moe_alpha_vis_cb.blockSignals(False)

    def _env_has_depth_camera(self, env_id: str) -> bool:
        xml_map = {
            "wheeldog_p_v2": os.path.join("envs", "wheeldog_p_v2", "assets", "xml", "wheeldog_p_v2.xml"),
            "flamingo_p_v3_1": os.path.join("envs", "flamingo_p_v3_1", "assets", "xml", "flamingo_p_v3.xml"),
        }
        rel_path = xml_map.get(str(env_id).strip())
        if not rel_path:
            return False
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
        if not os.path.isfile(xml_path):
            return False
        with open(xml_path, "r", encoding="utf-8") as handle:
            return 'camera name="depth_camera"' in handle.read()

    def _refresh_depth_controls(self, env_id: str):
        has_depth = self._env_has_depth_camera(env_id)
        self.depth_window_toggle_cb.blockSignals(True)
        self.depth_dataset_save_cb.blockSignals(True)
        self.depth_window_toggle_cb.setEnabled(has_depth)
        self.depth_dataset_save_cb.setEnabled(has_depth)
        self.depth_scale_le.setEnabled(has_depth)
        self.hm_x_fwd_le.setEnabled(has_depth)
        self.hm_x_bwd_le.setEnabled(has_depth)
        self.hm_y_left_le.setEnabled(has_depth)
        self.hm_y_right_le.setEnabled(has_depth)
        self.hm_resolution_le.setEnabled(has_depth)
        self.hm_visualize_cb.setEnabled(has_depth)
        self.hm_inference_cb.setEnabled(has_depth)
        self.hm_infer_btn.setEnabled(has_depth)
        if not has_depth:
            self.depth_window_toggle_cb.setChecked(False)
            self.depth_dataset_save_cb.setChecked(False)
            self.hm_visualize_cb.setChecked(False)
            self.hm_inference_cb.setChecked(False)
            self.depth_status_label.setText("Unavailable")
            self.depth_image_widget.clear_frame()
        else:
            self.depth_status_label.setText("")
        self.depth_window_toggle_cb.blockSignals(False)
        self.depth_dataset_save_cb.blockSignals(False)

    def _on_depth_window_toggled(self, checked):
        if not checked:
            self.depth_image_widget.clear_frame()

    def _update_depth_overlay(self, payload):
        if not hasattr(self, "depth_window_toggle_cb") or not self.depth_window_toggle_cb.isChecked():
            return
        self.depth_image_widget.update_depth(payload if isinstance(payload, dict) else {})

    def _on_depth_widget_closed(self):
        if hasattr(self, "depth_window_toggle_cb"):
            self.depth_window_toggle_cb.blockSignals(True)
            self.depth_window_toggle_cb.setChecked(False)
            self.depth_window_toggle_cb.blockSignals(False)

    def _show_monitor_plot_if_enabled(self):
        if not hasattr(self, "monitor_save_cb") or not self.monitor_save_cb.isChecked():
            return
        if not self.tester:
            return
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            payload = self.tester.get_monitor_export_payload()
            if not payload.get("joints"):
                return

            joints = list(payload.get("joints", []))
            dt = float(payload.get("dt", 0.02))
            env_id = str(payload.get("env_id", "env") or "env")
            if hasattr(self, "_monitor_summary_dialog") and self._monitor_summary_dialog is not None:
                self._monitor_summary_dialog.close()

            fig = Figure(figsize=(14, max(4.5, len(joints) * 3.6)))
            fig.patch.set_facecolor("black")
            axes = fig.subplots(len(joints), 2, squeeze=False)

            def compute_plot_range(values, fallback_limit):
                data = [float(v) for v in values]
                if not data:
                    return -fallback_limit, fallback_limit
                vmin = min(data)
                vmax = max(data)
                span = vmax - vmin
                if span < 1e-6:
                    margin = max(abs(vmax) * 0.15, fallback_limit * 0.08, 0.5)
                    return vmin - margin, vmax + margin
                margin = max(span * 0.12, fallback_limit * 0.05, 0.2)
                lower = vmin - margin
                upper = vmax + margin
                if lower > 0.0:
                    lower = min(0.0, lower - margin * 0.35)
                if upper < 0.0:
                    upper = max(0.0, upper + margin * 0.35)
                return lower, upper

            def short_joint_label(name):
                label = str(name).replace("_joint", "")
                if label.startswith("left_"):
                    label = "L " + label[5:]
                elif label.startswith("right_"):
                    label = "R " + label[6:]
                return label.replace("_", " ")

            for row_idx, joint in enumerate(joints):
                joint_name = str(joint.get("joint", f"joint_{row_idx}"))
                short_name = short_joint_label(joint_name)
                history = list(joint.get("history", []))
                torque_values = [float(tau) for _, tau in history]
                velocity_values = [float(vel) for vel, _ in history]
                duration = [(idx * dt) for idx in range(len(history))]
                torque_limit = max(abs(float(joint.get("torque_limit", 1.0))), 1.0)
                velocity_limit = max(abs(float(joint.get("velocity_limit", 1.0))), 1.0)
                torque_min, torque_max = compute_plot_range(torque_values, torque_limit)
                velocity_min, velocity_max = compute_plot_range(velocity_values, velocity_limit)

                torque_ax = axes[row_idx][0]
                velocity_ax = axes[row_idx][1]
                for ax in (torque_ax, velocity_ax):
                    ax.set_facecolor("black")
                    ax.tick_params(colors="white", labelsize=10)
                    for spine in ax.spines.values():
                        spine.set_color("#666666")
                    ax.grid(True, color="#444444", linestyle="--", linewidth=0.7, alpha=0.8)
                    ax.axhline(0.0, color="#BBBBBB", linewidth=0.9)
                    ax.set_xlim(left=0.0, right=max(duration[-1], dt) if duration else dt)
                    ax.set_xlabel("time (s)", color="white", fontsize=11)
                    ax.margins(x=0.02, y=0.08)

                torque_ax.plot(duration, torque_values, color="#7DD3FC", linewidth=2.0)
                torque_ax.set_ylim(torque_min, torque_max)
                torque_ax.set_ylabel("torque (Nm)", color="white", fontsize=11)
                torque_ax.text(
                    0.02, 0.96, f"- {short_name}",
                    transform=torque_ax.transAxes,
                    ha="left", va="top",
                    color="white", fontsize=10,
                    bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
                )

                velocity_ax.plot(duration, velocity_values, color="#F59E0B", linewidth=2.0)
                velocity_ax.set_ylim(velocity_min, velocity_max)
                velocity_ax.set_ylabel("velocity (rad/s)", color="white", fontsize=11)
                velocity_ax.text(
                    0.02, 0.96, f"- {short_name}",
                    transform=velocity_ax.transAxes,
                    ha="left", va="top",
                    color="white", fontsize=10,
                    bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
                )

            fig.suptitle(f"Motor Monitor Summary | {env_id}", color="white", fontsize=16)
            fig.tight_layout(rect=[0.02, 0.02, 1, 0.965], h_pad=2.0, w_pad=1.6)

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Motor Monitor Summary | {env_id}")
            dialog.resize(1400, max(700, min(1200, 280 + len(joints) * 260)))
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(8, 8, 8, 8)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            canvas.draw()

            self._monitor_summary_dialog = dialog
            dialog.show()
            dialog.raise_()
            self._append_log("[monitor] displayed end-of-test summary window.\n")
        except Exception as exc:
            QMessageBox.warning(self, "Monitor Plot", str(exc))

    def open_monitor_selector(self):
        self._ensure_monitor_defaults()
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Monitor Joints")
        dialog.resize(360, 460)
        layout = QVBoxLayout(dialog)

        info = QLabel("Choose the motor channels to stream into the detached monitor window.")
        info.setWordWrap(True)
        layout.addWidget(info)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)

        selected = set(self.monitor_settings.get("selected_joints", []))
        checkboxes = {}
        for joint_name in self.monitor_settings.get("available_joints", []):
            checkbox = QCheckBox(joint_name)
            checkbox.setChecked(joint_name in selected)
            body_layout.addWidget(checkbox)
            checkboxes[joint_name] = checkbox
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        actions = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        apply_btn = QPushButton("Apply")
        cancel_btn.clicked.connect(dialog.reject)
        apply_btn.clicked.connect(dialog.accept)
        actions.addStretch()
        actions.addWidget(cancel_btn)
        actions.addWidget(apply_btn)
        layout.addLayout(actions)

        if dialog.exec_() == QDialog.Accepted:
            chosen = [joint_name for joint_name, checkbox in checkboxes.items() if checkbox.isChecked()]
            self._set_monitor_selection(chosen)

    def update_defaults(self, new_env_id):
        settings = self.env_config.get(new_env_id, {}) or {}
        if new_env_id in self.action_scales_by_env:
            self.action_scales = list(self.action_scales_by_env[new_env_id])
        else:
            self.action_scales = self._make_action_scale_defaults(new_env_id)
            self.action_scales_by_env[new_env_id] = list(self.action_scales)
        if new_env_id in self.action_clippings_by_env:
            self.action_clippings = [dict(item) for item in self.action_clippings_by_env[new_env_id]]
        else:
            self.action_clippings = self._make_action_clipping_defaults(new_env_id)
            self.action_clippings_by_env[new_env_id] = [dict(item) for item in self.action_clippings]

        if new_env_id in self.actuator_settings_by_env:
            self.actuator_settings = (self.actuator_settings_by_env[new_env_id]).copy()
        else:
            self.actuator_settings = self._make_actuator_defaults(new_env_id)
            self.actuator_settings_by_env[new_env_id] = (self.actuator_settings).copy()

        hardware_defaults = self._make_hardware_defaults(new_env_id)
        if new_env_id in self.hardware_settings_by_env:
            self.hardware_settings = hardware_defaults.copy()
            self.hardware_settings.update(self.hardware_settings_by_env[new_env_id])
        else:
            self.hardware_settings = hardware_defaults
        self.hardware_settings_by_env[new_env_id] = (self.hardware_settings).copy()

        if new_env_id in self.initial_pose_settings_by_env:
            self.initial_pose_settings = {
                "base_z": str((self.initial_pose_settings_by_env[new_env_id]).get("base_z", self._make_initial_pose_defaults(new_env_id).get("base_z", "0.3"))),
                "joints": dict((self.initial_pose_settings_by_env[new_env_id]).get("joints", {}))
            }
        else:
            self.initial_pose_settings = self._make_initial_pose_defaults(new_env_id)
            self.initial_pose_settings_by_env[new_env_id] = {
                "base_z": str((self.initial_pose_settings).get("base_z", "0.3")),
                "joints": dict((self.initial_pose_settings).get("joints", {}))
            }
        self._ensure_final_pose_defaults_for_env(new_env_id)
        self._ensure_homing_command_ranges_for_env(new_env_id)

        if new_env_id in self.monitor_settings_by_env:
            self.monitor_settings = dict(self.monitor_settings_by_env[new_env_id])
        else:
            self.monitor_settings = self._make_monitor_defaults(new_env_id)
            self.monitor_settings_by_env[new_env_id] = dict(self.monitor_settings)

        if new_env_id in self.dataset_height_map_settings_by_env:
            self.dataset_height_map_settings = dict(self.dataset_height_map_settings_by_env[new_env_id])
        else:
            self.dataset_height_map_settings = self._make_dataset_height_map_defaults(new_env_id)
            self.dataset_height_map_settings_by_env[new_env_id] = dict(self.dataset_height_map_settings)

        if new_env_id in self.depth_randomization_settings_by_env:
            self.depth_randomization_settings = dict(self.depth_randomization_settings_by_env[new_env_id])
        else:
            self.depth_randomization_settings = self._make_depth_randomization_defaults(new_env_id)
            self.depth_randomization_settings_by_env[new_env_id] = dict(self.depth_randomization_settings)

        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}

        # UI upper bounds (example retained)
        command_0_max = "1.5"
        command_2_max = "1.5"
        command_0_min = "-1.5"
        command_2_min = "-1.5"
        if self.min_command_value_le_list:
            self.min_command_value_le_list[0].setText(command_0_min)
            self.min_command_value_le_list[2].setText(command_2_min)
        if self.max_command_value_le_list:
            self.max_command_value_le_list[0].setText(command_0_max)
            self.max_command_value_le_list[2].setText(command_2_max)

        # command[3] initial value (accept float/int)
        if self.command_initial_value_le_list and isinstance(self.command_initial_value_le_list[3], QLineEdit):
            c3 = cmd_cfg.get("command_3_initial", 0.0)
            self.command_initial_value_le_list[3].setText(str(to_float(c3, 0.0)))

        # On environment change: observation settings via cache or defaults
        if new_env_id in self.obs_settings_by_env:
            self.observation_settings = (self.obs_settings_by_env[new_env_id]).copy()
        else:
            self.observation_settings = self._make_observation_defaults(new_env_id)
            self.obs_settings_by_env[new_env_id] = (self.observation_settings).copy()

        if hasattr(self, "hm_x_fwd_le"):
            self.hm_x_fwd_le.setText(str(self.dataset_height_map_settings.get("x_forward", "0.5")))
            self.hm_x_bwd_le.setText(str(self.dataset_height_map_settings.get("x_backward", "0.5")))
            self.hm_y_left_le.setText(str(self.dataset_height_map_settings.get("y_left", "0.3")))
            self.hm_y_right_le.setText(str(self.dataset_height_map_settings.get("y_right", "0.3")))
            self.hm_resolution_le.setText(str(self.dataset_height_map_settings.get("resolution", "0.1")))
            self.depth_scale_le.setText(str(self.dataset_height_map_settings.get("depth_scale", "8")))
            self.hm_visualize_cb.blockSignals(True)
            self.hm_visualize_cb.setChecked(bool(self.dataset_height_map_settings.get("visualize", False)))
            self.hm_visualize_cb.blockSignals(False)
            self.hm_inference_cb.blockSignals(True)
            self.hm_inference_cb.setChecked(bool(self.dataset_height_map_settings.get("inference_visualize", False)))
            self.hm_inference_cb.blockSignals(False)
            self._sync_height_map_inference_button()

        self._refresh_monitor_joint_checkboxes()
        self._refresh_depth_controls(new_env_id)
        self._sync_fine_tune_controls_from_cache()
        self._vision_last_summary = self._vision_last_summary_by_env.get(new_env_id)
        self._sync_vision_train_controls_from_cache()
        if self.vision_train_dialog is not None and self.vision_train_dialog.isVisible():
            self._refresh_vision_train_dialog()

    def _sync_height_map_inference_button(self):
        path = str(self.dataset_height_map_settings.get("inference_onnx_path", "")).strip()
        self.hm_infer_btn.setToolTip(path)

    def select_height_map_inference_onnx(self):
        env_id = self.env_id_cb.currentText()
        current_path = str(self.dataset_height_map_settings.get("inference_onnx_path", "")).strip()
        default_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "vision_heightmap", "latest")
        start_dir = current_path if current_path else default_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Height-Map Inference ONNX",
            start_dir,
            "ONNX Files (*.onnx)"
        )
        if not file_path:
            return
        self._ensure_dataset_height_map_defaults()
        self.dataset_height_map_settings["inference_onnx_path"] = file_path
        self.dataset_height_map_settings_by_env[env_id] = dict(self.dataset_height_map_settings)
        self._sync_height_map_inference_button()

    def showEvent(self, event):
        self.centralWidget().setFocus()
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.handle_key_press(event)
            return True
        elif event.type() == QEvent.KeyRelease:
            self.handle_key_release(event)
            return True
        return super().eventFilter(obj, event)

    def handle_key_press(self, event):
        key = event.key()
        pending_timer = self._pending_key_release_timers.pop(key, None)
        if pending_timer is not None:
            pending_timer.stop()
            pending_timer.deleteLater()
        if event.isAutoRepeat():
            return
        if key in self.key_mapping and key not in self.active_keys:
            btn, cmd_index, direction = self.key_mapping[key]
            btn.setChecked(True)
            is_discrete = self._is_discrete_command(cmd_index)
            if is_discrete:
                self._apply_discrete_command_delta(cmd_index, direction)
            self.active_keys[key] = {"cmd_index": cmd_index, "direction": direction, "discrete": is_discrete}

    def handle_key_release(self, event):
        key = event.key()
        if key not in self.key_mapping:
            return
        if event.isAutoRepeat():
            return
        pending_timer = self._pending_key_release_timers.get(key)
        if pending_timer is not None:
            pending_timer.stop()
            pending_timer.deleteLater()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda key=key: self._finalize_key_release(key))
        self._pending_key_release_timers[key] = timer
        timer.start(35)

    def _finalize_key_release(self, key):
        timer = self._pending_key_release_timers.pop(key, None)
        if timer is not None:
            timer.deleteLater()
        if key not in self.key_mapping:
            return
        btn, cmd_index, _ = self.key_mapping[key]
        btn.setChecked(False)
        if key in self.active_keys:
            self.active_keys.pop(key)
        same_command_still_active = any(
            key_info.get("cmd_index") == cmd_index
            for key_info in self.active_keys.values()
        )
        if same_command_still_active:
            return
        if self._is_discrete_command(cmd_index):
            return
        default_value = self._get_default_command_value(cmd_index)
        self.current_command_values[cmd_index] = default_value
        self._update_command_button(cmd_index, default_value)

    def _get_default_command_value(self, index):
        try:
            widget = self.command_initial_value_le_list[index]
            if isinstance(widget, (QLineEdit, QLabel)):
                return float(widget.text())
            return 0.0
        except Exception:
            return 0.0

    def _update_status_label(self):
        html_text = (
            "<html><head><style>"
            "h3 { margin: 0 0 8px 0; }"
            "table { border-collapse: collapse; }"
            "td { padding: 4px 8px; border: 1px solid #ddd; }"
            "</style></head><body>"
            "<h4> Current Command Values</h4><table>"
        )
        for i, value in enumerate(self.current_command_values):
            if i % 6 == 0:
                if i != 0:
                    html_text += "</tr>"
                html_text += "<tr>"
            html_text += f"<td>[{i}] = {value:.3f}</td>"
        html_text += "</tr></table></body></html>"
        self.status_label.setText(html_text)

    def _update_command_button(self, index, value):
        self.current_command_values[index] = value
        self._update_status_label()

    def _is_discrete_command(self, index):
        try:
            return self.command_discrete_cb_list[index].isChecked()
        except Exception:
            return False

    def _get_command_bounds(self, index):
        min_value = self._parse_float(self.min_command_value_le_list[index].text(), -2.0)
        max_value = self._parse_float(self.max_command_value_le_list[index].text(), 2.0)
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return min_value, max_value

    def _clamp_command_value(self, index, value):
        min_value, max_value = self._get_command_bounds(index)
        return min(max(value, min_value), max_value)

    def _push_current_command_to_tester(self):
        if self.tester:
            for i, value in enumerate(self.current_command_values):
                self.tester.update_command(i, value)
        if (
            self.homing_worker is not None
            and self.homing_worker_mode == "test_policy"
            and hasattr(self.homing_worker, "update_command_values")
        ):
            self.homing_worker.update_command_values(self.current_command_values)
        if (
            self.ctbc_worker is not None
            and self.ctbc_worker_mode == "test_policy"
            and hasattr(self.ctbc_worker, "update_command_values")
        ):
            self.ctbc_worker.update_command_values(self.current_command_values)

    def _start_homing_command_timer(self):
        self._stop_homing_command_timer()
        self._init_default_command_values()
        self._push_current_command_to_tester()
        self.homing_command_timer = QTimer(self)
        self.homing_command_timer.timeout.connect(self.send_current_command)
        self.homing_command_timer.start(20)

    def _stop_homing_command_timer(self):
        timer = getattr(self, "homing_command_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        self.homing_command_timer = None

    def _apply_discrete_command_delta(self, index, direction):
        step = self._parse_float(self.command_sensitivity_le_list[index].text(), 0.1)
        new_value = self.current_command_values[index] + direction * step
        new_value = self._clamp_command_value(index, new_value)
        self.current_command_values[index] = new_value
        self._update_command_button(index, new_value)
        self._push_current_command_to_tester()

    def send_current_command(self):
        # Apply key-driven deltas within bounds, update tester and status
        for key_info in self.active_keys.values():
            if key_info.get("discrete", False):
                continue
            cmd_index = key_info["cmd_index"]
            direction = key_info["direction"]
            step = self._parse_float(self.command_sensitivity_le_list[cmd_index].text(), 0.1)
            current_value = self.current_command_values[cmd_index]
            new_value = current_value + direction * step
            new_value = self._clamp_command_value(cmd_index, new_value)
            self.current_command_values[cmd_index] = new_value
            self._update_command_button(cmd_index, new_value)
        self._push_current_command_to_tester()
        self._update_status_label()
        self._update_fine_tune_status_label()

    def _parse_float(self, text, default):
        try:
            return float(text)
        except Exception:
            return default

    # ---------------- UI SETUP ----------------

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QScrollArea.NoFrame)
        main_layout.addWidget(content_scroll, 1)

        content_widget = QWidget()
        content_scroll.setWidget(content_widget)
        top_h_layout = QHBoxLayout(content_widget)
        top_h_layout.setContentsMargins(0, 0, 0, 0)
        top_h_layout.setSpacing(15)

        # Left: scroll area (Policy placed below Environment)
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumWidth(500)
        top_h_layout.addWidget(config_scroll, 4)
        config_widget = QWidget()
        config_scroll.setWidget(config_widget)
        self.config_layout = QVBoxLayout(config_widget)
        self.config_layout.setContentsMargins(10, 10, 10, 10)
        self.config_layout.setSpacing(15)

        # Vertical: Policy under Environment
        self._create_env_group(self.config_layout)
        self._create_policy_group(self.config_layout)

        # Random Settings group
        self._create_random_group()

        # Place Event Input on the left (under Random Settings)
        self._create_event_input_group(self.config_layout)

        # Right: Command Settings / Command Input
        right_v_layout = QVBoxLayout()
        right_v_layout.setSpacing(10)
        top_h_layout.addLayout(right_v_layout, 2)
        self._create_command_settings_group(right_v_layout)
        self._create_fine_tune_group(right_v_layout)
        self._setup_key_visual_buttons(right_v_layout)

        # Far right: Terminal Log
        log_v_layout = QVBoxLayout()
        log_v_layout.setSpacing(10)
        top_h_layout.addLayout(log_v_layout, 3)
        self._create_log_group(log_v_layout)

        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_button = QPushButton("Start Test")
        self.start_button.setFixedWidth(120)
        self.start_button.clicked.connect(self.start_test)
        btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Test")
        self.stop_button.setFixedWidth(120)
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)
        main_layout.addLayout(btn_layout)
        self._apply_styles()

    def _create_event_input_group(self, parent_layout):
        event_group = QGroupBox("Event Input")
        event_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        event_layout = QFormLayout()
        event_layout.setLabelAlignment(Qt.AlignRight)
        event_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        event_layout.setSpacing(8)
        event_group.setLayout(event_layout)
        push_vel_layout = QHBoxLayout()
        self.push_vel_x_le = QLineEdit("0.0")
        self.push_vel_x_le.setPlaceholderText("x")
        self.push_vel_x_le.setFixedWidth(50)
        self.push_vel_y_le = QLineEdit("0.0")
        self.push_vel_y_le.setPlaceholderText("y")
        self.push_vel_y_le.setFixedWidth(50)
        self.push_vel_z_le = QLineEdit("0.0")
        self.push_vel_z_le.setPlaceholderText("z")
        self.push_vel_z_le.setFixedWidth(50)
        push_vel_layout.addWidget(self.push_vel_x_le)
        push_vel_layout.addWidget(self.push_vel_y_le)
        push_vel_layout.addWidget(self.push_vel_z_le)
        event_layout.addRow("Push Velocity (x, y, z):", push_vel_layout)
        self.push_button = QPushButton("Push")
        self.push_button.pressed.connect(self.activate_push_trigger)
        self.push_button.released.connect(self.deactivate_push_trigger)
        event_layout.addRow(self.push_button)
        parent_layout.addWidget(event_group)

    def activate_push_trigger(self):
        if self.tester:
            try:
                push_vel = [
                    float(self.push_vel_x_le.text()),
                    float(self.push_vel_y_le.text()),
                    float(self.push_vel_z_le.text())
                ]
                self.tester.activate_push_event(push_vel)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Push velocity must be numeric values.")

    def deactivate_push_trigger(self):
        if self.tester:
            self.tester.deactivate_push_event()

    # --------- CONFIG GROUPS ---------

    def _create_env_group(self, parent_layout):
        env_group = QGroupBox("Environment Settings")
        env_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        env_group.setMinimumWidth(460)
        env_layout = QFormLayout()
        env_layout.setLabelAlignment(Qt.AlignRight)
        env_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        env_layout.setSpacing(8)
        env_group.setLayout(env_layout)
        self.env_id_cb = NoWheelComboBox()
        self.env_id_cb.addItems(self.env_config.keys())

        default_env = list(self.env_config.keys())[0]
        self.env_id_cb.setCurrentText(default_env)
        env_layout.addRow("ID:", self.env_id_cb)

        self.max_duration_le = QLineEdit("120.0")
        env_layout.addRow("Max Duration (s):", self.max_duration_le)

        actuator_btn = QPushButton("Actuator Settings")
        actuator_btn.clicked.connect(self.open_actuator_settings)
        env_layout.addRow("Actuator:", actuator_btn)

        action_scale_btn = QPushButton("Action Scale Settings")
        action_scale_btn.clicked.connect(self.open_action_scale_settings)
        env_layout.addRow("Action Scale:", action_scale_btn)

        settings_btn = QPushButton("Hardware Settings")
        settings_btn.clicked.connect(self.open_hardware_settings)
        env_layout.addRow("Hardware:", settings_btn)

        initial_pose_btn = QPushButton("Initial Pose Settings")
        initial_pose_btn.clicked.connect(self.open_initial_pose_settings)
        env_layout.addRow("Initial Pose:", initial_pose_btn)

        obs_settings_btn = QPushButton("Observation Settings")
        obs_settings_btn.clicked.connect(self.open_observation_settings)
        env_layout.addRow("Settings:", obs_settings_btn)

        depth_randomize_btn = QPushButton("Depth Randomize")
        depth_randomize_btn.clicked.connect(self.open_depth_randomization_settings)
        env_layout.addRow("Depth Aug:", depth_randomize_btn)

        monitor_row = QWidget()
        monitor_row_layout = QHBoxLayout(monitor_row)
        monitor_row_layout.setContentsMargins(0, 0, 0, 0)
        monitor_row_layout.setSpacing(6)
        self.monitor_config_btn = QPushButton("Joints")
        self.monitor_config_btn.setFixedWidth(72)
        self.monitor_config_btn.clicked.connect(self.open_monitor_selector)
        self.monitor_window_toggle_cb = QCheckBox("Window")
        self.monitor_window_toggle_cb.toggled.connect(self._on_monitor_window_toggled)
        self.monitor_save_cb = QCheckBox("Show End")
        self.monitor_summary_label = QLabel("0 selected")
        self.monitor_summary_label.setStyleSheet("color: #64748B;")
        monitor_row_layout.addWidget(self.monitor_config_btn)
        monitor_row_layout.addWidget(self.monitor_window_toggle_cb)
        monitor_row_layout.addWidget(self.monitor_save_cb)
        monitor_row_layout.addWidget(self.monitor_summary_label, 1)
        env_layout.addRow("Monitor:", monitor_row)

        depth_row = QWidget()
        depth_row_layout = QHBoxLayout(depth_row)
        depth_row_layout.setContentsMargins(0, 0, 0, 0)
        depth_row_layout.setSpacing(6)
        self.depth_window_toggle_cb = QCheckBox("Window")
        self.depth_window_toggle_cb.toggled.connect(self._on_depth_window_toggled)
        self.depth_dataset_save_cb = QCheckBox("Save")
        self.depth_scale_le = QLineEdit("8")
        self.depth_scale_le.setFixedWidth(40)
        self.hm_train_btn = QPushButton("Train")
        self.hm_train_btn.setFixedWidth(72)
        self.hm_train_btn.clicked.connect(self.open_vision_train_dialog)
        self.hm_infer_btn = QPushButton("Inference")
        self.hm_infer_btn.setFixedWidth(84)
        self.hm_infer_btn.clicked.connect(self.select_height_map_inference_onnx)
        self.depth_status_label = QLabel("Unavailable")
        self.depth_status_label.setStyleSheet("color: #64748B;")
        depth_row_layout.addWidget(self.depth_window_toggle_cb)
        depth_row_layout.addWidget(self.depth_dataset_save_cb)
        depth_row_layout.addWidget(QLabel("Scale"))
        depth_row_layout.addWidget(self.depth_scale_le)
        depth_row_layout.addWidget(self.hm_train_btn)
        depth_row_layout.addWidget(self.hm_infer_btn)
        depth_row_layout.addWidget(self.depth_status_label, 1)
        env_layout.addRow("Depth:", depth_row)

        hm_row = QWidget()
        hm_row_layout = QHBoxLayout(hm_row)
        hm_row_layout.setContentsMargins(0, 0, 0, 0)
        hm_row_layout.setSpacing(6)
        self.hm_x_fwd_le = QLineEdit("0.5")
        self.hm_x_fwd_le.setFixedWidth(48)
        self.hm_x_bwd_le = QLineEdit("0.5")
        self.hm_x_bwd_le.setFixedWidth(48)
        self.hm_y_left_le = QLineEdit("0.3")
        self.hm_y_left_le.setFixedWidth(48)
        self.hm_y_right_le = QLineEdit("0.3")
        self.hm_y_right_le.setFixedWidth(48)
        self.hm_resolution_le = QLineEdit("0.1")
        self.hm_resolution_le.setFixedWidth(48)
        self.hm_visualize_cb = QCheckBox("Viz")
        self.hm_inference_cb = QCheckBox("Inference")
        self.vision_status_inline_label = QLabel("")
        self.vision_status_inline_label.setStyleSheet("color: #64748B;")
        hm_row_layout.addWidget(QLabel("X"))
        hm_row_layout.addWidget(self.hm_x_fwd_le)
        hm_row_layout.addWidget(QLabel("-X"))
        hm_row_layout.addWidget(self.hm_x_bwd_le)
        hm_row_layout.addWidget(QLabel("Y"))
        hm_row_layout.addWidget(self.hm_y_left_le)
        hm_row_layout.addWidget(QLabel("-Y"))
        hm_row_layout.addWidget(self.hm_y_right_le)
        hm_row_layout.addWidget(QLabel("Res"))
        hm_row_layout.addWidget(self.hm_resolution_le)
        hm_row_layout.addWidget(self.hm_visualize_cb)
        hm_row_layout.addWidget(self.hm_inference_cb)
        hm_row_layout.addWidget(self.vision_status_inline_label, 1)
        env_layout.addRow("Height Map:", hm_row)

        # === MoE row: independent from Depth / Height Map ===
        moe_row = QWidget()
        moe_row_layout = QHBoxLayout(moe_row)
        moe_row_layout.setContentsMargins(0, 0, 0, 0)
        moe_row_layout.setSpacing(6)

        self.moe_train_inline_btn = QPushButton("MoE Training")
        self.moe_train_inline_btn.setFixedWidth(120)
        self.moe_train_inline_btn.clicked.connect(self.open_moe_train_dialog)
        self.moe_manual_btn = QPushButton("Manual")
        self.moe_manual_btn.setFixedWidth(80)
        self.moe_manual_btn.clicked.connect(self.open_moe_manual_dialog)
        self.moe_alpha_vis_cb = QCheckBox("Alpha Vis")
        self.moe_alpha_vis_cb.toggled.connect(self._on_alpha_vis_toggled)

        self.moe_status_inline_label = QLabel("")
        self.moe_status_inline_label.setStyleSheet("color: #64748B;")

        moe_row_layout.addWidget(self.moe_train_inline_btn)
        moe_row_layout.addWidget(self.moe_manual_btn)
        moe_row_layout.addWidget(self.moe_alpha_vis_cb)
        moe_row_layout.addWidget(self.moe_status_inline_label, 1)

        env_layout.addRow("MoE:", moe_row)

        homing_row = QWidget()
        homing_row_layout = QHBoxLayout(homing_row)
        homing_row_layout.setContentsMargins(0, 0, 0, 0)
        homing_row_layout.setSpacing(6)
        self.homing_train_inline_btn = QPushButton("Homing Training")
        self.homing_train_inline_btn.setFixedWidth(130)
        self.homing_train_inline_btn.clicked.connect(self.open_homing_train_dialog)
        homing_row_layout.addWidget(self.homing_train_inline_btn)
        homing_row_layout.addStretch()
        env_layout.addRow("Homing:", homing_row)

        self.terrain_id_cb = NoWheelComboBox()
        self.terrain_id_cb.addItems([
            'flat', 'rocky_easy', 'rocky_hard',
            'slope_easy', 'slope_hard',
            'stairs_up_easy', 'stairs_up_normal', 'stairs_up_hard', 'stairs_up_extrme'
        ])

        self.terrain_id_cb = NoWheelComboBox()
        self.terrain_id_cb.addItems([
            'flat', 'rocky_easy', 'rocky_hard',
            'slope_easy', 'slope_hard',
            'stairs_up_easy', 'stairs_up_normal', 'stairs_up_hard', 'stairs_up_extrme'
        ])
        self.terrain_id_cb.setCurrentText("flat")
        env_layout.addRow("Terrain:", self.terrain_id_cb)
        parent_layout.addWidget(env_group, 1)

    def _create_policy_group(self, parent_layout):
        policy_group = QGroupBox("Policy Settings")
        policy_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        policy_group.setMinimumWidth(460)
        policy_layout = QFormLayout()
        policy_layout.setLabelAlignment(Qt.AlignRight)
        policy_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        policy_layout.setSpacing(8)
        # ▶ 필드 영역이 가로로 잘 늘어나도록
        policy_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        policy_group.setLayout(policy_layout)

        # Policy type
        self.policy_type_cb = NoWheelComboBox()
        self.policy_type_cb.addItems(["MLP", "LSTM", "Encoder+MLP"])
        self.policy_type_cb.setCurrentText("MLP")
        policy_layout.addRow("Policy Type:", self.policy_type_cb)

        # Dims
        self.h_in_dim_le = QLineEdit("256")
        policy_layout.addRow("h_in Dim:", self.h_in_dim_le)
        self.c_in_dim_le = QLineEdit("256")
        policy_layout.addRow("c_in Dim:", self.c_in_dim_le)

        # === Policy File (기본) ===
        self.policy_file_le = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_policy_file)

        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)    # ▶ 동일 마진
        file_layout.setSpacing(6)
        file_layout.addWidget(self.policy_file_le, 1) # ▶ LineEdit에 stretch=1
        file_layout.addWidget(browse_btn)

        file_row = QWidget()
        file_row.setLayout(file_layout)
        file_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # ▶ 동일 SizePolicy

        policy_layout.addRow("Policy File:", file_row)

        # === Encoder File (조건부 표시) ===
        self.encoder_file_le = QLineEdit()
        enc_browse_btn = QPushButton("Browse")
        enc_browse_btn.clicked.connect(self.browse_encoder_file)

        enc_file_layout = QHBoxLayout()
        enc_file_layout.setContentsMargins(0, 0, 0, 0)    # ▶ 동일 마진
        enc_file_layout.setSpacing(6)
        enc_file_layout.addWidget(self.encoder_file_le, 1) # ▶ LineEdit에 stretch=1
        enc_file_layout.addWidget(enc_browse_btn)

        self.encoder_row_widget = QWidget()
        self.encoder_row_widget.setLayout(enc_file_layout)
        self.encoder_row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.encoder_label = QLabel("Encoder File:")
        policy_layout.addRow(self.encoder_label, self.encoder_row_widget)

        # 기본은 숨김 + 콤보 변경 시 토글
        self.encoder_label.setVisible(False)
        self.encoder_row_widget.setVisible(False)
        self.policy_type_cb.currentTextChanged.connect(self._update_policy_fields)

        # 그룹을 한 번만 추가
        parent_layout.addWidget(policy_group, 0)

    def _create_random_group(self):
        random_group = QGroupBox("Random Settings")
        random_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        random_group.setMinimumWidth(460)
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(8)
        random_group.setLayout(form_layout)

        self.precision_cb = NoWheelComboBox()
        self.precision_cb.addItems(["low", "medium", "high", "ultra", "extreme"])
        self.precision_cb.setCurrentText("medium")
        form_layout.addRow("Precision:", self.precision_cb)

        self.sensor_noise_cb = NoWheelComboBox()
        self.sensor_noise_cb.addItems(["none", "low", "medium", "high", "ultra", "extreme"])
        self.sensor_noise_cb.setCurrentText("none")
        form_layout.addRow("Sensor Noise:", self.sensor_noise_cb)

        def create_slider_row(slider, min_val, max_val, init_val, scale, decimals):
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(init_val)
            value_label = QLabel(f"{init_val / scale:.{decimals}f}")
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v / scale:.{decimals}f}"))
            h_layout = QHBoxLayout()
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return h_layout

        self.init_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Init Noise:", create_slider_row(self.init_noise_slider, 0, 100, 5, 100, 2))
        self.sliding_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Sliding Friction:", create_slider_row(self.sliding_friction_slider, 0, 100, 100, 100, 2))
        self.torsional_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Torsional Friction:", create_slider_row(self.torsional_friction_slider, 0, 100, 50, 10000, 4))
        self.rolling_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Rolling Friction:", create_slider_row(self.rolling_friction_slider, 0, 100, 1, 10000, 4))
        self.friction_loss_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Friction Loss:", create_slider_row(self.friction_loss_slider, 0, 100, 0, 100, 2))
        self.action_delay_prob_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Action Delay Prob.:", create_slider_row(self.action_delay_prob_slider, 0, 100, 5, 100, 2))
        self.mass_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Mass Noise:", create_slider_row(self.mass_noise_slider, 0, 50, 5, 100, 2))
        self.load_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Load:", create_slider_row(self.load_slider, 0, 200, 0, 10, 1))
        self.config_layout.addWidget(random_group)

    def _create_fine_tune_group(self, parent_layout):
        fine_tune_group = QGroupBox("Fine-tune")
        fine_tune_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        fine_layout = QFormLayout(fine_tune_group)
        fine_layout.setLabelAlignment(Qt.AlignRight)
        fine_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        fine_layout.setSpacing(8)

        self.fine_tune_enable_cb = QCheckBox("Enable residual fine-tune")
        self.fine_tune_enable_cb.toggled.connect(self._on_fine_tune_controls_changed)
        fine_layout.addRow(self.fine_tune_enable_cb)

        self.fine_tune_ridge_lambda_le = QLineEdit("1e-4")
        self.fine_tune_ridge_lambda_le.editingFinished.connect(self._on_fine_tune_controls_changed)
        fine_layout.addRow("Ridge lambda:", self.fine_tune_ridge_lambda_le)

        self.fine_tune_max_samples_le = QLineEdit("5000")
        self.fine_tune_max_samples_le.editingFinished.connect(self._on_fine_tune_controls_changed)
        fine_layout.addRow("Max samples:", self.fine_tune_max_samples_le)

        self.fine_tune_bias_btn = QPushButton("Action Bias Editor")
        self.fine_tune_bias_btn.clicked.connect(self.open_fine_tune_bias_editor)
        fine_layout.addRow("Manual Bias:", self.fine_tune_bias_btn)

        self.fine_tune_fit_btn = QPushButton("Fit Residual")
        self.fine_tune_fit_btn.clicked.connect(self.fit_fine_tune_residual)
        fine_layout.addRow("Train:", self.fine_tune_fit_btn)

        self.fine_tune_export_btn = QPushButton("Export Merged ONNX")
        self.fine_tune_export_btn.clicked.connect(self.export_fine_tuned_onnx)
        fine_layout.addRow("Export:", self.fine_tune_export_btn)

        self.fine_tune_status_label = QLabel("Fine-tune idle")
        self.fine_tune_status_label.setWordWrap(True)
        fine_layout.addRow("Status:", self.fine_tune_status_label)

        parent_layout.addWidget(fine_tune_group)

    def _create_vision_train_group(self, parent_layout):
        vision_group = QGroupBox("Vision Train")
        vision_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        vision_layout = QFormLayout(vision_group)
        vision_layout.setLabelAlignment(Qt.AlignRight)
        vision_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        vision_layout.setSpacing(8)

        self.vision_epochs_le = QLineEdit("10")
        vision_layout.addRow("Epochs:", self.vision_epochs_le)

        self.vision_batch_size_le = QLineEdit("64")
        vision_layout.addRow("Batch size:", self.vision_batch_size_le)

        self.vision_lr_le = QLineEdit("1e-3")
        vision_layout.addRow("Learning rate:", self.vision_lr_le)

        self.vision_latent_dim_le = QLineEdit("128")
        vision_layout.addRow("Latent dim:", self.vision_latent_dim_le)

        self.vision_hidden_dim_le = QLineEdit("128")
        vision_layout.addRow("Hidden dim:", self.vision_hidden_dim_le)

        self.vision_val_ratio_le = QLineEdit("0.1")
        vision_layout.addRow("Val ratio:", self.vision_val_ratio_le)

        self.vision_seed_le = QLineEdit("42")
        vision_layout.addRow("Seed:", self.vision_seed_le)

        self.vision_train_btn = QPushButton("Train Predictor")
        self.vision_train_btn.clicked.connect(self.train_vision_predictor)
        vision_layout.addRow("Train:", self.vision_train_btn)

        self.vision_export_btn = QPushButton("Export Predictor ONNX")
        self.vision_export_btn.clicked.connect(self.export_vision_predictor_onnx)
        vision_layout.addRow("Export:", self.vision_export_btn)

        moe_tools_row = QWidget()
        moe_tools_layout = QHBoxLayout(moe_tools_row)
        moe_tools_layout.setContentsMargins(0, 0, 0, 0)
        self.moe_train_btn = QPushButton("MoE Training")
        self.moe_train_btn.clicked.connect(self.open_moe_train_dialog)
        self.moe_manual_vision_btn = QPushButton("Manual")
        self.moe_manual_vision_btn.clicked.connect(self.open_moe_manual_dialog)
        moe_tools_layout.addWidget(self.moe_train_btn)
        moe_tools_layout.addWidget(self.moe_manual_vision_btn)
        vision_layout.addRow("MoE:", moe_tools_row)

        self.vision_status_label = QLabel("idle")
        self.vision_status_label.setWordWrap(True)
        vision_layout.addRow("Status:", self.vision_status_label)

        parent_layout.addWidget(vision_group)

    def _create_command_settings_group(self, parent_layout):
        command_group = QGroupBox("Command Settings")
        command_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        grid_layout = QGridLayout(command_group)
        grid_layout.addWidget(QLabel("Index"), 0, 0)
        grid_layout.addWidget(QLabel("Sensitivity"), 0, 1)
        grid_layout.addWidget(QLabel("Min Value"), 0, 2)
        grid_layout.addWidget(QLabel("Max Value"), 0, 3)
        grid_layout.addWidget(QLabel("Initial Value"), 0, 4)
        grid_layout.addWidget(QLabel("Discrete"), 0, 5)

        # command[3] initial value is taken from the env's 'command' section
        settings = self.env_config.get(self.env_id_cb.currentText(), {}) or {}
        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}
        cmd3_init = str(to_float(cmd_cfg.get("command_3_initial", 0.0), 0.0))

        for i in range(6):  # indices 0~5
            label = QLabel(f"command[{i}]")
            sensitivity_le = QLineEdit("0.02")
            min_value_le = QLineEdit("-1.5" if i in [0, 1, 2] else "-1")
            max_value_le = QLineEdit("1.5" if i in [0, 1, 2] else "1")
            init_value_widget = QLineEdit(cmd3_init) if i == 3 else QLabel("0.0")
            discrete_cb = QCheckBox()
            grid_layout.addWidget(label, i + 1, 0)
            grid_layout.addWidget(sensitivity_le, i + 1, 1)
            grid_layout.addWidget(min_value_le, i + 1, 2)
            grid_layout.addWidget(max_value_le, i + 1, 3)
            grid_layout.addWidget(init_value_widget, i + 1, 4)
            grid_layout.addWidget(discrete_cb, i + 1, 5, Qt.AlignCenter)
            self.command_sensitivity_le_list.append(sensitivity_le)
            self.min_command_value_le_list.append(min_value_le)
            self.max_command_value_le_list.append(max_value_le)
            self.command_initial_value_le_list.append(init_value_widget)
            self.command_discrete_cb_list.append(discrete_cb)
        self.position_command_cb = QCheckBox("Position Command")
        self.position_command_cb.setChecked(False)
        row_position = 6 + 1
        grid_layout.addWidget(self.position_command_cb, row_position, 0, 1, 6, Qt.AlignLeft)
        parent_layout.addWidget(command_group)

    def _setup_key_visual_buttons(self, parent_layout):
        button_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 10px; border-radius: 10px; min-width: 36px; min-height: 36px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        key_group = QGroupBox("Command Input")
        key_layout = QVBoxLayout(key_group)
        key_layout.setSpacing(8)

        dir_group = QGroupBox("command[0], command[2]")
        dir_layout = QGridLayout(dir_group)
        self.btn_up = NonClickableButton("W"); self.btn_up.setStyleSheet(button_style); self.btn_up.setCheckable(True); dir_layout.addWidget(self.btn_up, 0, 1)
        self.btn_left = NonClickableButton("A"); self.btn_left.setStyleSheet(button_style); self.btn_left.setCheckable(True); dir_layout.addWidget(self.btn_left, 1, 0)
        self.btn_right = NonClickableButton("D"); self.btn_right.setStyleSheet(button_style); self.btn_right.setCheckable(True); dir_layout.addWidget(self.btn_right, 1, 2)
        self.btn_down = NonClickableButton("S"); self.btn_down.setStyleSheet(button_style); self.btn_down.setCheckable(True); dir_layout.addWidget(self.btn_down, 1, 1)
        key_layout.addWidget(dir_group)

        other_group = QGroupBox("command[3], command[4], command[5]")
        other_layout = QGridLayout(other_group)
        self.btn_i = NonClickableButton("I"); self.btn_i.setStyleSheet(button_style); self.btn_i.setCheckable(True); other_layout.addWidget(self.btn_i, 0, 0)
        self.btn_o = NonClickableButton("O"); self.btn_o.setStyleSheet(button_style); self.btn_o.setCheckable(True); other_layout.addWidget(self.btn_o, 0, 1)
        self.btn_p = NonClickableButton("P"); self.btn_p.setStyleSheet(button_style); self.btn_p.setCheckable(True); other_layout.addWidget(self.btn_p, 0, 2)
        self.btn_j = NonClickableButton("J"); self.btn_j.setStyleSheet(button_style); self.btn_j.setCheckable(True); other_layout.addWidget(self.btn_j, 1, 0)
        self.btn_k = NonClickableButton("K"); self.btn_k.setStyleSheet(button_style); self.btn_k.setCheckable(True); other_layout.addWidget(self.btn_k, 1, 1)
        self.btn_l = NonClickableButton("L"); self.btn_l.setStyleSheet(button_style); self.btn_l.setCheckable(True); other_layout.addWidget(self.btn_l, 1, 2)
        key_layout.addWidget(other_group)

        zx_group = QGroupBox("command[1]")
        zx_layout = QHBoxLayout(zx_group)
        zx_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 4px; border-radius: 10px; min-width: 22px; min-height: 22px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        self.btn_z = NonClickableButton("Z"); self.btn_z.setStyleSheet(zx_style); self.btn_z.setCheckable(True); zx_layout.addWidget(self.btn_z)
        self.btn_x = NonClickableButton("X"); self.btn_x.setStyleSheet(zx_style); self.btn_x.setCheckable(True); zx_layout.addWidget(self.btn_x)
        key_layout.addWidget(zx_group)
        parent_layout.addWidget(key_group, 1)

        self.key_mapping = {
            Qt.Key_W: (self.btn_up, 0, +1.0),
            Qt.Key_S: (self.btn_down, 0, -1.0),
            Qt.Key_A: (self.btn_left, 2, +1.0),
            Qt.Key_D: (self.btn_right, 2, -1.0),
            Qt.Key_Z: (self.btn_z, 1, -1.0),
            Qt.Key_X: (self.btn_x, 1, +1.0),
            Qt.Key_I: (self.btn_i, 3, +1.0),
            Qt.Key_J: (self.btn_j, 3, -1.0),
            Qt.Key_O: (self.btn_o, 4, +1.0),
            Qt.Key_K: (self.btn_k, 4, -1.0),
            Qt.Key_P: (self.btn_p, 5, +1.0),
            Qt.Key_L: (self.btn_l, 5, -1.0)
        }

    def _create_log_group(self, parent_layout):
        log_group = QGroupBox("Terminal Log")
        log_group.setMinimumWidth(420)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.setMinimumHeight(180)
        self.log_output.document().setMaximumBlockCount(5000)
        self.log_output.setPlaceholderText("Runtime logs will appear here.")
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setStyleHint(QFont.Monospace)
        self.log_output.setFont(fixed_font)
        self.log_output.setStyleSheet(
            "QTextEdit { background-color: #000000; color: #f5f5f5; border: 1px solid #333333; }"
        )
        log_layout.addWidget(self.log_output)

        parent_layout.addWidget(log_group, 1)

    def _next_rainbow_color(self) -> str:
        color = self._rainbow_palette[self._log_color_index % len(self._rainbow_palette)]
        self._log_color_index += 1
        return color

    def _get_joint_color(self, joint_name: str) -> str:
        key = joint_name.strip()
        if key not in self._joint_color_map:
            self._joint_color_map[key] = self._next_rainbow_color()
        return self._joint_color_map[key]

    @staticmethod
    def _parse_table_columns(line: str):
        stripped = line.strip()
        if not (stripped.startswith('|') and stripped.endswith('|')):
            return []
        return [col.strip() for col in stripped[1:-1].split('|')]

    def _pick_log_color(self, line: str) -> str:
        stripped = line.strip()
        lowered = stripped.lower()

        if not stripped:
            return "#f5f5f5"
        if "error" in lowered or "traceback" in lowered or "failed" in lowered:
            return "#ff4d6d"
        if "warn" in lowered:
            return "#ffb703"
        if stripped.startswith('+') and stripped.endswith('+'):
            return "#8d99ae"
        if "joint states" in lowered or "base state" in lowered or "step" in lowered:
            return "#f5f5f5"

        cols = self._parse_table_columns(line)
        if cols:
            first_col = cols[0].lower()
            if first_col in {"joint", "signal"}:
                return "#f5f5f5"
            if len(cols) > 1 and cols[1].strip().lower() == "value":
                return "#f5f5f5"

            signal_name = cols[0].strip().lower()
            for key, color in self._signal_color_map.items():
                if signal_name == key.lower():
                    return color

            joint_name = cols[0].strip()
            if joint_name and joint_name != '-':
                return self._get_joint_color(joint_name)

        if "base_height" in lowered or "viewer closed" in lowered or "report successfully saved" in lowered:
            return "#f5f5f5"
        return "#f5f5f5"

    def _insert_log_line(self, line: str):
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._pick_log_color(line)))
        cursor.insertText(line, fmt)

        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _flush_log_output(self):
        if not hasattr(self, 'log_output') or self.log_output is None or not self._pending_log_chunks:
            return

        self._log_buffer += "".join(self._pending_log_chunks)
        self._pending_log_chunks.clear()

        while True:
            newline_idx = self._log_buffer.find("\n")
            if newline_idx < 0:
                break
            line = self._log_buffer[:newline_idx + 1]
            self._log_buffer = self._log_buffer[newline_idx + 1:]
            self._insert_log_line(line)

        if self._log_buffer and ("\r" in self._log_buffer):
            self._insert_log_line(self._log_buffer)
            self._log_buffer = ""

        if not self._pending_log_chunks:
            self._log_flush_timer.stop()

    def _append_log(self, message: str):
        if not hasattr(self, 'log_output') or self.log_output is None:
            return
        self._pending_log_chunks.append(message)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _redirect_log_streams(self):
        if self._stdout_stream is not None:
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_stream = _TeeStream(self._log_emitter, self._original_stdout)
        self._stderr_stream = _TeeStream(self._log_emitter, self._original_stderr)
        sys.stdout = self._stdout_stream
        sys.stderr = self._stderr_stream

    def _restore_log_streams(self):
        if getattr(self, '_stdout_stream', None) is None and getattr(self, '_stderr_stream', None) is None:
            return
        if sys.stdout is self._stdout_stream and self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if sys.stderr is self._stderr_stream and self._original_stderr is not None:
            sys.stderr = self._original_stderr
        self._stdout_stream = None
        self._stderr_stream = None
        self._original_stdout = None
        self._original_stderr = None
        if hasattr(self, '_pending_log_chunks'):
            self._pending_log_chunks.clear()
        if hasattr(self, '_log_flush_timer'):
            self._log_flush_timer.stop()

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QLineEdit, QComboBox, QSlider {
                padding: 4px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #4E94D4;
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
            }
            QPushButton:hover:!disabled {
                background-color: #005999;
            }
        """)

    def _update_policy_fields(self, text: str):
        is_encoder = text.strip().lower() == "encoder+mlp"
        # Encoder 파일 행 토글
        self.encoder_label.setVisible(is_encoder)
        self.encoder_row_widget.setVisible(is_encoder)

    def browse_policy_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Policy ONNX File", os.path.join(os.getcwd(), "weights"),
            "ONNX Files (*.onnx)"
        )
        if file_path:
            self.policy_file_le.setText(file_path)

    def browse_encoder_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Encoder ONNX File", os.path.join(os.getcwd(), "weights"),
            "ONNX Files (*.onnx)"
        )
        if file_path:
            self.encoder_file_le.setText(file_path)

    def _on_fine_tune_controls_changed(self):
        self._collect_fine_tune_ui_settings()
        self._apply_fine_tune_settings_to_tester()
        self._update_fine_tune_status_label()

    def open_fine_tune_bias_editor(self):
        self._ensure_fine_tune_defaults()
        action_dim = len(self.fine_tune_settings.get("bias", []))
        if action_dim <= 0:
            QMessageBox.warning(self, "Fine-tune", "Current environment has no action dimensions to edit.")
            return
        if self.fine_tune_bias_dialog is not None:
            try:
                self.fine_tune_bias_dialog.biasChanged.disconnect(self.on_fine_tune_bias_changed)
            except Exception:
                pass
            self.fine_tune_bias_dialog.close()
        self.fine_tune_bias_dialog = FineTuneBiasEditorDialog(action_dim, self.fine_tune_settings.get("bias", []), self)
        self.fine_tune_bias_dialog.biasChanged.connect(self.on_fine_tune_bias_changed)
        self.fine_tune_bias_dialog.show()
        self.fine_tune_bias_dialog.raise_()
        self.fine_tune_bias_dialog.activateWindow()

    def on_fine_tune_bias_changed(self, bias):
        self._ensure_fine_tune_defaults()
        self.fine_tune_settings["bias"] = [to_float(v, 0.0) for v in list(bias)]
        self.fine_tune_settings_by_env[self.env_id_cb.currentText()] = dict(self.fine_tune_settings)
        if self.tester:
            self.tester.set_fine_tune_bias(self.fine_tune_settings["bias"])
        self._update_fine_tune_status_label()

    def _update_fine_tune_status_label(self):
        if not hasattr(self, "fine_tune_status_label"):
            return
        settings = self._collect_fine_tune_ui_settings() if hasattr(self, "fine_tune_enable_cb") else self.fine_tune_settings
        if self.tester:
            status = self.tester.get_fine_tune_status()
            samples = status.get("samples", 0)
            trained = status.get("trained", False)
            max_samples = status.get("max_samples", 0)
        else:
            samples = 0
            trained = False
            max_samples = to_int(settings.get("max_samples", 5000), 5000)
        state = "enabled" if settings.get("enabled", False) else "disabled"
        trained_text = "trained" if trained else "untrained"
        bias_norm = np.linalg.norm(np.asarray(settings.get("bias", []), dtype=np.float32)) if settings.get("bias") else 0.0
        self.fine_tune_status_label.setText(
            f"{state} | samples: {samples}/{max_samples} | {trained_text} | bias norm: {bias_norm:.4f}"
        )

    def fit_fine_tune_residual(self):
        if not self.tester:
            QMessageBox.warning(self, "Fine-tune", "Start a test first so samples can be collected.")
            return
        ridge_lambda = to_float(self.fine_tune_ridge_lambda_le.text().strip(), 1e-4)
        try:
            fit_info = self.tester.fit_fine_tune_head(ridge_lambda=ridge_lambda)
        except Exception as e:
            QMessageBox.critical(self, "Fine-tune", str(e))
            return

        self.fine_tune_settings["bias"] = [0.0] * len(self.fine_tune_settings.get("bias", []))
        self.fine_tune_settings_by_env[self.env_id_cb.currentText()] = dict(self.fine_tune_settings)
        if self.fine_tune_bias_dialog is not None:
            self.fine_tune_bias_dialog.set_bias(self.fine_tune_settings["bias"])
        self._update_fine_tune_status_label()
        QMessageBox.information(
            self,
            "Fine-tune",
            f"Residual layer fitted with {fit_info['samples']} samples.\n"
            f"RMSE: {fit_info['rmse']:.6f}",
        )

    def export_fine_tuned_onnx(self):
        if not self.tester:
            QMessageBox.warning(self, "Fine-tune", "Run or load a policy first before exporting.")
            return
        policy_file_path = self.policy_file_le.text().strip()
        default_dir = os.path.dirname(policy_file_path) if policy_file_path else os.getcwd()
        base_name = os.path.splitext(os.path.basename(policy_file_path))[0] if policy_file_path else "policy"
        default_path = os.path.join(default_dir, f"{base_name}_merged_finetuned.onnx")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Merged ONNX",
            default_path,
            "ONNX Files (*.onnx)"
        )
        if not output_path:
            return
        try:
            exported = self.tester.export_fine_tuned_policy(output_path)
        except Exception as e:
            QMessageBox.critical(self, "Fine-tune", str(e))
            return
        self._update_fine_tune_status_label()
        QMessageBox.information(self, "Fine-tune", f"Merged ONNX exported to:\n{exported}")

    def _update_vision_train_status_label(self):
        settings = self.vision_train_settings if self.vision_train_settings else self._collect_vision_train_ui_settings()
        if self.vision_train_thread is not None and self.vision_train_thread.isRunning():
            status_text = "training..."
            if hasattr(self, "vision_status_inline_label"):
                self.vision_status_inline_label.setText(status_text)
            if self.vision_train_dialog is not None:
                self.vision_train_dialog.set_status(status_text)
            return
        if isinstance(self._vision_last_summary, dict):
            status_text = (
                f"trained | samples: {self._vision_last_summary.get('samples', 0)} | "
                f"val: {self._vision_last_summary.get('best_val_loss', 0.0):.6f}"
            )
            if hasattr(self, "vision_status_inline_label"):
                self.vision_status_inline_label.setText(status_text)
            if self.vision_train_dialog is not None:
                self.vision_train_dialog.set_status(status_text)
            return
        status_text = ""
        if hasattr(self, "vision_status_inline_label"):
            self.vision_status_inline_label.setText(status_text)
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.set_status(status_text)

    def open_vision_train_dialog(self):
        self._ensure_vision_train_defaults()
        if self.vision_train_dialog is None:
            self.vision_train_dialog = VisionTrainDialog(self)
            self.vision_train_dialog.trainRequested.connect(self.train_vision_predictor)
            self.vision_train_dialog.exportRequested.connect(self.export_vision_predictor_onnx)
            self.vision_train_dialog.refreshRequested.connect(self._refresh_vision_train_dialog)
            self.vision_train_dialog.stopRequested.connect(self.stop_vision_train)
        self._refresh_vision_train_dialog()
        self.vision_train_dialog.show()
        self.vision_train_dialog.raise_()
        self.vision_train_dialog.activateWindow()

    def train_vision_predictor(self):
        if self.vision_train_thread is not None and self.vision_train_thread.isRunning():
            QMessageBox.warning(self, "Vision Train", "Vision predictor training is already running.")
            return

        dialog_settings = self.vision_train_dialog.get_settings() if self.vision_train_dialog is not None else None
        settings = self._collect_vision_train_ui_settings(dialog_settings)
        env_id = self.env_id_cb.currentText()
        dataset_paths = list(settings.get("selected_datasets", []))
        if not dataset_paths:
            QMessageBox.warning(self, "Vision Train", "Select at least one dataset for training.")
            return
        run_name = "latest"
        repo_root = self._repo_root()
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.clear_log()
            self.vision_train_dialog.set_running(True)
        self.vision_train_thread = QThread()
        self.vision_train_worker = VisionTrainerWorker(
            repo_root=repo_root,
            env_id=env_id,
            dataset_paths=dataset_paths,
            settings={
                "epochs": to_int(settings.get("epochs", 10), 10),
                "batch_size": to_int(settings.get("batch_size", 64), 64),
                "learning_rate": to_float(settings.get("learning_rate", 1e-3), 1e-3),
                "latent_dim": to_int(settings.get("latent_dim", 128), 128),
                "hidden_dim": to_int(settings.get("hidden_dim", 128), 128),
                "val_ratio": to_float(settings.get("val_ratio", 0.1), 0.1),
                "seed": to_int(settings.get("seed", 42), 42),
                "run_name": run_name,
            },
        )
        self.vision_train_worker.moveToThread(self.vision_train_thread)
        self.vision_train_thread.started.connect(self.vision_train_worker.run)
        self.vision_train_worker.log.connect(self.on_vision_train_log)
        self.vision_train_worker.finished.connect(self.on_vision_train_finished)
        self.vision_train_worker.error.connect(self.on_vision_train_error)
        self.vision_train_worker.finished.connect(self.vision_train_thread.quit)
        self.vision_train_worker.error.connect(self.vision_train_thread.quit)
        self.vision_train_worker.finished.connect(self.vision_train_worker.deleteLater)
        self.vision_train_worker.error.connect(self.vision_train_worker.deleteLater)
        self.vision_train_thread.finished.connect(self._on_vision_train_thread_finished)
        self.vision_train_thread.finished.connect(self.vision_train_thread.deleteLater)
        self.vision_train_thread.start()
        self._vision_last_summary = None
        self._update_vision_train_status_label()

    def on_vision_train_log(self, message):
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.append_log(message)

    def stop_vision_train(self):
        if self.vision_train_worker is not None:
            self.vision_train_worker.request_stop()
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.append_log("[vision-train] stop requested by user.\n")
            self.vision_train_dialog.set_status("stopping")

    def on_vision_train_finished(self, summary):
        self._vision_last_summary = dict(summary or {})
        self._vision_last_summary_by_env[self.env_id_cb.currentText()] = dict(self._vision_last_summary)
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.set_running(False)
        self._update_vision_train_status_label()
        QMessageBox.information(
            self,
            "Vision Train",
            "Training stopped." if self._vision_last_summary.get("stopped", False) else (
                f"Training finished.\nBest val loss: {self._vision_last_summary.get('best_val_loss', 0.0):.6f}\n"
                f"ONNX: {self._vision_last_summary.get('onnx_path', '')}"
            ),
        )

    def on_vision_train_error(self, error_msg):
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.set_running(False)
            self.vision_train_dialog.append_log(f"[vision-train] ERROR: {error_msg}\n")
        self._update_vision_train_status_label()
        QMessageBox.critical(self, "Vision Train", error_msg)

    def _on_vision_train_thread_finished(self):
        self.vision_train_thread = None
        self.vision_train_worker = None
        if self.vision_train_dialog is not None:
            self.vision_train_dialog.set_running(False)
        self._update_vision_train_status_label()

    def export_vision_predictor_onnx(self):
        env_id = self.env_id_cb.currentText()
        default_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "vision_heightmap", "latest")
        default_path = os.path.join(default_dir, "vision_heightmap_predictor.onnx")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Vision Predictor ONNX",
            default_path,
            "ONNX Files (*.onnx)"
        )
        if not output_path:
            return
        checkpoint_path = os.path.join(default_dir, "vision_heightmap_predictor.pt")
        try:
            from core.vision_heightmap_trainer import VisionHeightMapTrainer
            trainer = VisionHeightMapTrainer(
                repo_root=self._repo_root(),
                env_id=env_id,
                settings={},
            )
            exported = trainer.export_onnx_from_checkpoint(checkpoint_path, output_path)
        except Exception as e:
            QMessageBox.critical(self, "Vision Train", str(e))
            return
        QMessageBox.information(self, "Vision Train", f"Predictor ONNX exported to:\n{exported}")

    def open_moe_train_dialog(self):
        self._ensure_moe_defaults()
        if self.moe_dialog is None:
            self.moe_dialog = MoETrainDialog(self.env_config.keys(), self._terrain_ids(), self)
            self.moe_dialog.collectRequested.connect(self.collect_moe_data)
            self.moe_dialog.trainRequested.connect(self.train_moe_gate)
            self.moe_dialog.exportRequested.connect(self.export_moe_onnx)
            self.moe_dialog.refreshRequested.connect(self._refresh_moe_dialog)
            self.moe_dialog.stopRequested.connect(self.stop_moe_job)
        self._refresh_moe_dialog()
        self.moe_dialog.show()
        self.moe_dialog.raise_()
        self.moe_dialog.activateWindow()

    def open_moe_manual_dialog(self):
        self._ensure_moe_manual_defaults()
        if self.moe_manual_dialog is None:
            self.moe_manual_dialog = MoEManualDialog(self)
            self.moe_manual_dialog.exportRequested.connect(self.export_manual_moe_onnx)
        self.moe_manual_dialog.set_settings(self.moe_manual_settings)
        running = self.moe_thread is not None and self.moe_thread.isRunning()
        self.moe_manual_dialog.set_running(running)
        self.moe_manual_dialog.show()
        self.moe_manual_dialog.raise_()
        self.moe_manual_dialog.activateWindow()

    def _start_moe_worker(self, mode, settings):
        if self.moe_thread is not None and self.moe_thread.isRunning():
            QMessageBox.warning(self, "MoE Training", "A MoE job is already running.")
            return False
        if self.moe_dialog is not None:
            self.moe_dialog.set_running(True)
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.set_running(True)
        self.moe_worker_mode = mode
        self.moe_thread = QThread()
        self.moe_worker = MoEWorker(self._repo_root(), settings, mode)
        self.moe_worker.moveToThread(self.moe_thread)
        self.moe_thread.started.connect(self.moe_worker.run)
        self.moe_worker.log.connect(self.on_moe_log)
        self.moe_worker.finished.connect(self.on_moe_finished)
        self.moe_worker.error.connect(self.on_moe_error)
        self.moe_worker.finished.connect(self.moe_thread.quit)
        self.moe_worker.error.connect(self.moe_thread.quit)
        self.moe_worker.finished.connect(self.moe_worker.deleteLater)
        self.moe_worker.error.connect(self.moe_worker.deleteLater)
        self.moe_thread.finished.connect(self._on_moe_thread_finished)
        self.moe_thread.finished.connect(self.moe_thread.deleteLater)
        self.moe_thread.start()
        return True

    def export_manual_moe_onnx(self):
        dialog_settings = self.moe_manual_dialog.get_settings() if self.moe_manual_dialog is not None else None
        settings = self._collect_moe_manual_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_a_path", "")) or not os.path.isfile(settings.get("policy_b_path", "")):
            QMessageBox.warning(self, "Manual MoE Export", "Select valid Policy A and Policy B ONNX files.")
            return
        output_path = settings.get("output_path", "")
        if not output_path:
            default_dir = os.path.join(self._repo_root(), "envs", settings.get("env_id", self.env_id_cb.currentText()), "weights", "moe_manual")
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Manual MoE ONNX",
                os.path.join(default_dir, "manual_moe_policy.onnx"),
                "ONNX Files (*.onnx)"
            )
            if not output_path:
                return
            settings["output_path"] = output_path
        self.moe_manual_settings = dict(settings)
        self.moe_manual_settings_by_env[settings.get("env_id", self.env_id_cb.currentText())] = dict(settings)
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.set_settings(settings)
            self.moe_manual_dialog.set_status("exporting")
            self.moe_manual_dialog.append_log("[moe-manual] export requested with alpha mapped to the final command")
        self._start_moe_worker("manual_export", settings)

    def collect_moe_data(self):
        dialog_settings = self.moe_dialog.get_settings() if self.moe_dialog is not None else None
        settings = self._collect_moe_ui_settings(dialog_settings)
        if not settings.get("terrains"):
            QMessageBox.warning(self, "MoE Training", "Select at least one terrain.")
            return
        if not os.path.isfile(settings.get("policy_a_path", "")) or not os.path.isfile(settings.get("policy_b_path", "")):
            QMessageBox.warning(self, "MoE Training", "Select valid Policy A and Policy B ONNX files.")
            return
        if self.moe_dialog is not None:
            self.moe_dialog.clear_log()
            self.moe_dialog.set_status("collecting")
        self._start_moe_worker("collect", {
            **settings,
            "samples": to_int(settings.get("samples", 200000), 200000),
            "rollout_steps": to_int(settings.get("rollout_steps", 1000), 1000),
            "boundary_m": to_float(settings.get("boundary_m", 8.0), 8.0),
            "command_min": to_float(settings.get("command_min", -1.0), -1.0),
            "command_max": to_float(settings.get("command_max", 1.0), 1.0),
            "seed": to_int(settings.get("seed", 42), 42),
            "cmd_label_threshold": to_float(settings.get("cmd_label_threshold", 0.2), 0.2),
            "cmd_label_alpha": to_float(settings.get("cmd_label_alpha", 0.0), 0.0),
        })

    def train_moe_gate(self):
        dialog_settings = self.moe_dialog.get_settings() if self.moe_dialog is not None else None
        settings = self._collect_moe_ui_settings(dialog_settings)
        if not settings.get("selected_datasets"):
            QMessageBox.warning(self, "MoE Training", "Select at least one collected MoE dataset.")
            return
        if self.moe_dialog is not None:
            self.moe_dialog.clear_log()
            self.moe_dialog.set_status("training")
        self._start_moe_worker("train", {
            **settings,
            "epochs": to_int(settings.get("epochs", 30), 30),
            "batch_size": to_int(settings.get("batch_size", 256), 256),
            "learning_rate": to_float(settings.get("learning_rate", 1e-3), 1e-3),
            "lambda_smooth": to_float(settings.get("lambda_smooth", 0.0), 0.0),
            "cmd_label_threshold": to_float(settings.get("cmd_label_threshold", 0.2), 0.2),
            "cmd_label_alpha": to_float(settings.get("cmd_label_alpha", 0.0), 0.0),
            "val_ratio": to_float(settings.get("val_ratio", 0.1), 0.1),
            "seed": to_int(settings.get("seed", 42), 42),
        })

    def export_moe_onnx(self):
        settings = self._collect_moe_ui_settings(self.moe_dialog.get_settings() if self.moe_dialog is not None else None)
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        default_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "moe_gate", "latest")
        checkpoint_path = os.path.join(default_dir, "moe_gate.pt")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MoE Policy ONNX",
            os.path.join(default_dir, "moe_policy.onnx"),
            "ONNX Files (*.onnx)"
        )
        if not output_path:
            return
        if self.moe_dialog is not None:
            self.moe_dialog.set_status("exporting")
        self._start_moe_worker("export", {
            **settings,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
        })

    def on_moe_log(self, message):
        if self.moe_dialog is not None:
            self.moe_dialog.append_log(message)
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.append_log(message)

    def stop_moe_job(self):
        if self.moe_worker is not None:
            self.moe_worker.request_stop()
        if self.moe_dialog is not None:
            self.moe_dialog.append_log("[moe] stop requested by user.\n")
            self.moe_dialog.set_status("stopping")

    def on_moe_finished(self, summary):
        self._moe_last_summary = dict(summary or {})
        env_id = self._moe_last_summary.get("env_id", self.moe_settings.get("env_id", self.env_id_cb.currentText()))
        self._moe_last_summary_by_env[env_id] = dict(self._moe_last_summary)
        if self.moe_dialog is not None:
            self.moe_dialog.set_running(False)
            self.moe_dialog.set_status("done")
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.set_running(False)
            if self.moe_worker_mode == "manual_export":
                self.moe_manual_dialog.set_status(f"exported: {self._moe_last_summary.get('onnx_path', '')}")
        self._refresh_moe_dialog()
        if self._moe_last_summary.get("stopped", False):
            QMessageBox.information(self, "MoE Training", "MoE job stopped.")
        else:
            QMessageBox.information(self, "MoE Training", "MoE job finished.")

    def on_moe_error(self, error_msg):
        if self.moe_dialog is not None:
            self.moe_dialog.set_running(False)
            self.moe_dialog.append_log(f"[moe] ERROR: {error_msg}\n")
            self.moe_dialog.set_status("error")
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.set_running(False)
            self.moe_manual_dialog.append_log(f"[moe] ERROR: {error_msg}\n")
            self.moe_manual_dialog.set_status("error")
        QMessageBox.critical(self, "MoE Training", error_msg)

    def _on_moe_thread_finished(self):
        self.moe_thread = None
        self.moe_worker = None
        self.moe_worker_mode = None
        if self.moe_dialog is not None:
            self.moe_dialog.set_running(False)
        if self.moe_manual_dialog is not None:
            self.moe_manual_dialog.set_running(False)

    def open_homing_train_dialog(self):
        self._ensure_homing_defaults()
        if self.homing_dialog is None:
            self.homing_dialog = HomingTrainDialog(self.env_config.keys(), self._terrain_ids(), self)
            self.homing_dialog.collectRequested.connect(self.collect_homing_data)
            self.homing_dialog.trainRequested.connect(self.train_homing_policy)
            self.homing_dialog.rlTrainRequested.connect(self.rl_fine_tune_homing_policy)
            self.homing_dialog.exportRequested.connect(self.export_homing_onnx)
            self.homing_dialog.refreshRequested.connect(self._refresh_homing_dialog)
            self.homing_dialog.stopRequested.connect(self.stop_homing_job)
            self.homing_dialog.finalPoseRequested.connect(self.open_final_pose_settings)
            self.homing_dialog.testTeacherRequested.connect(self.test_homing_teacher)
            self.homing_dialog.testPolicyRequested.connect(self.test_homing_export_policy)
            self.homing_dialog.switchPolicyRequested.connect(self.switch_homing_export_policy)
            self.homing_dialog.commandRangeRequested.connect(self.open_homing_command_range_settings)
            self.homing_dialog.envChanged.connect(self._on_homing_env_changed)
            self.homing_dialog.set_settings(self.homing_settings)
        self._refresh_homing_dialog()
        self.homing_dialog.show()
        self.homing_dialog.raise_()
        self.homing_dialog.activateWindow()

    def open_ctbc_train_dialog(self):
        self._ensure_ctbc_defaults()
        if self.ctbc_dialog is None:
            self.ctbc_dialog = CtbcTrainDialog(self.env_config.keys(), self._terrain_ids(), self)
            self.ctbc_dialog.rlTrainRequested.connect(self.rl_fine_tune_ctbc_policy)
            self.ctbc_dialog.refreshRequested.connect(self._refresh_ctbc_dialog)
            self.ctbc_dialog.stopRequested.connect(self.stop_ctbc_job)
            self.ctbc_dialog.testPolicyRequested.connect(self.test_ctbc_export_policy)
            self.ctbc_dialog.testPrimitiveRequested.connect(self.test_ctbc_primitive)
            self.ctbc_dialog.envChanged.connect(self._on_ctbc_env_changed)
            self.ctbc_dialog.set_settings(self.ctbc_settings)
        self._refresh_ctbc_dialog()
        self.ctbc_dialog.show()
        self.ctbc_dialog.raise_()
        self.ctbc_dialog.activateWindow()

    def _start_homing_worker(self, mode, settings):
        if self.homing_thread is not None and self.homing_thread.isRunning():
            QMessageBox.warning(self, "Homing Training", "A Homing job is already running.")
            return False
        if self.homing_dialog is not None:
            self.homing_dialog.set_running(True)
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_running(True)
        self.homing_worker_mode = mode
        self.homing_thread = QThread()
        self.homing_worker = HomingWorker(self._repo_root(), settings, mode)
        self.homing_worker.moveToThread(self.homing_thread)
        self.homing_thread.started.connect(self.homing_worker.run)
        self.homing_worker.log.connect(self.on_homing_log)
        self.homing_worker.finished.connect(self.on_homing_finished)
        self.homing_worker.error.connect(self.on_homing_error)
        self.homing_worker.finished.connect(self.homing_thread.quit)
        self.homing_worker.error.connect(self.homing_thread.quit)
        self.homing_worker.finished.connect(self.homing_worker.deleteLater)
        self.homing_worker.error.connect(self.homing_worker.deleteLater)
        self.homing_thread.finished.connect(self._on_homing_thread_finished)
        self.homing_thread.finished.connect(self.homing_thread.deleteLater)
        self.homing_thread.start()
        return True

    def _start_ctbc_worker(self, mode, settings):
        if self.ctbc_thread is not None and self.ctbc_thread.isRunning():
            QMessageBox.warning(self, "CTBC Stair Reflex", "A CTBC job is already running.")
            return False
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_running(True)
        self.ctbc_worker_mode = mode
        self.ctbc_thread = QThread()
        self.ctbc_worker = CtbcWorker(self._repo_root(), settings, mode)
        self.ctbc_worker.moveToThread(self.ctbc_thread)
        self.ctbc_thread.started.connect(self.ctbc_worker.run)
        self.ctbc_worker.log.connect(self.on_ctbc_log)
        self.ctbc_worker.finished.connect(self.on_ctbc_finished)
        self.ctbc_worker.error.connect(self.on_ctbc_error)
        self.ctbc_worker.finished.connect(self.ctbc_thread.quit)
        self.ctbc_worker.error.connect(self.ctbc_thread.quit)
        self.ctbc_worker.finished.connect(self.ctbc_worker.deleteLater)
        self.ctbc_worker.error.connect(self.ctbc_worker.deleteLater)
        self.ctbc_thread.finished.connect(self._on_ctbc_thread_finished)
        self.ctbc_thread.finished.connect(self.ctbc_thread.deleteLater)
        self.ctbc_thread.start()
        return True

    def _homing_env_overrides(self, env_id):
        env_id = str(env_id or self.env_id_cb.currentText())
        hardware = self.hardware_settings_by_env.get(env_id)
        if hardware is None:
            hardware = self._make_hardware_defaults(env_id)
        hardware_numeric = {
            k: (v if isinstance(v, bool) else to_float(v, v))
            for k, v in dict(hardware).items()
        }

        action_scales = self.action_scales_by_env.get(env_id)
        if action_scales is None:
            action_scales = self._make_action_scale_defaults(env_id)

        action_clippings = self.action_clippings_by_env.get(env_id)
        if action_clippings is None:
            action_clippings = self._make_action_clipping_defaults(env_id)

        actuator = self.actuator_settings_by_env.get(env_id)
        if actuator is None:
            actuator = self._make_actuator_defaults(env_id)

        initial_pose = self.initial_pose_settings_by_env.get(env_id)
        if initial_pose is None:
            initial_pose = self._make_initial_pose_defaults(env_id)
        initial_positions = {
            "base_z": to_float(initial_pose.get("base_z", self._make_initial_pose_defaults(env_id).get("base_z", 0.3)), 0.3),
            "joints": {
                joint_name: to_float(value, 0.0)
                for joint_name, value in dict(initial_pose.get("joints", {})).items()
            }
        }

        return {
            "hardware": hardware_numeric,
            "action_scales": [to_float(value, 1.0) for value in list(action_scales or [])],
            "action_clippings": [dict(item) for item in list(action_clippings or [])],
            "actuator": dict(actuator or {}),
            "initial_positions": initial_positions,
            "joint_offsets": initial_positions,
            "random": {
                "precision": self.precision_cb.currentText(),
                "sensor_noise": self.sensor_noise_cb.currentText(),
                "init_noise": self.init_noise_slider.value() / 100.0,
                "sliding_friction": self.sliding_friction_slider.value() / 100.0,
                "torsional_friction": self.torsional_friction_slider.value() / 100.0,
                "rolling_friction": self.rolling_friction_slider.value() / 100.0,
                "friction_loss": self.friction_loss_slider.value() / 100.0,
                "action_delay_prob": self.action_delay_prob_slider.value() / 100.0,
                "mass_noise": self.mass_noise_slider.value() / 100.0,
                "load": self.load_slider.value() / 10.0,
            },
        }

    def collect_homing_data(self):
        dialog_settings = self.homing_dialog.get_settings() if self.homing_dialog is not None else None
        settings = self._collect_homing_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "Homing Training", "Select a valid stand-drive ONNX policy.")
            return
        if self.homing_dialog is not None:
            self.homing_dialog.clear_log()
            self.homing_dialog.set_status("collecting")
        self._start_homing_worker("collect", {
            **settings,
            "samples": to_int(settings.get("samples", 50000), 50000),
            "rollout_steps": to_int(settings.get("rollout_steps", 1000), 1000),
            "homing_trajectory_seconds": to_float(settings.get("homing_trajectory_seconds", 3.0), 3.0),
            "homing_stand_warmup_steps": to_int(settings.get("homing_stand_warmup_steps", 200), 200),
            "homing_balance_blend": to_float(settings.get("homing_balance_blend", 0.0), 0.0),
            "command_min": to_float(settings.get("command_min", -1.0), -1.0),
            "command_max": to_float(settings.get("command_max", 1.0), 1.0),
            "seed": to_int(settings.get("seed", 42), 42),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def test_homing_teacher(self):
        dialog_settings = self.homing_dialog.get_settings() if self.homing_dialog is not None else None
        settings = self._collect_homing_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "Homing Training", "Select a valid stand-drive ONNX policy.")
            return
        if self.homing_dialog is not None:
            self.homing_dialog.set_status("testing teacher")
            self.homing_dialog.append_log("[homing-test] launching render test on flat terrain.\n")
        self._start_homing_worker("test_teacher", {
            **settings,
            "command_min": to_float(settings.get("command_min", -1.0), -1.0),
            "command_max": to_float(settings.get("command_max", 1.0), 1.0),
            "seed": to_int(settings.get("seed", 42), 42),
            "test_warmup_steps": 200,
            "test_steps": to_int(settings.get("rollout_steps", 600), 600),
            "homing_trajectory_seconds": to_float(settings.get("homing_trajectory_seconds", 3.0), 3.0),
            "homing_balance_blend": to_float(settings.get("homing_balance_blend", 0.0), 0.0),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def test_homing_export_policy(self):
        dialog_settings = self.homing_dialog.get_settings() if self.homing_dialog is not None else None
        settings = self._collect_homing_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "Homing Training", "Select a valid stand-drive ONNX policy.")
            return
        if not os.path.isfile(settings.get("output_path", "")):
            QMessageBox.warning(self, "Homing Training", "Select a valid exported Homing ONNX file.")
            return
        if self.homing_dialog is not None:
            self.homing_dialog.set_status("testing export policy")
            self.homing_dialog.append_log("[homing-policy-test] launching render test. Use keyboard commands, then click Switch Policy.\n")
        self._start_homing_command_timer()
        self._start_homing_worker("test_policy", {
            **settings,
            "test_steps": to_int(settings.get("rollout_steps", 1000), 1000),
            "command_values": list(self.current_command_values),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def switch_homing_export_policy(self):
        if self.homing_worker is None or self.homing_worker_mode != "test_policy":
            if self.homing_dialog is not None:
                self.homing_dialog.append_log("[homing-policy-test] Start Test Export before switching policy.\n")
            return
        self.homing_worker.request_policy_switch()
        if self.homing_dialog is not None:
            self.homing_dialog.append_log("[homing-policy-test] switch requested from GUI.\n")

    def train_homing_policy(self):
        dialog_settings = self.homing_dialog.get_settings() if self.homing_dialog is not None else None
        settings = self._collect_homing_ui_settings(dialog_settings)
        if not settings.get("selected_datasets"):
            QMessageBox.warning(self, "Homing Training", "Select at least one collected Homing dataset.")
            return
        if self.homing_dialog is not None:
            self.homing_dialog.clear_log()
            self.homing_dialog.set_status("training")
        self._start_homing_worker("train", {
            **settings,
            "epochs": to_int(settings.get("epochs", 30), 30),
            "batch_size": to_int(settings.get("batch_size", 256), 256),
            "learning_rate": to_float(settings.get("learning_rate", 1e-3), 1e-3),
            "val_ratio": to_float(settings.get("val_ratio", 0.1), 0.1),
            "hidden_dim": to_int(settings.get("hidden_dim", 256), 256),
            "seed": to_int(settings.get("seed", 42), 42),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def rl_fine_tune_homing_policy(self):
        dialog_settings = self.homing_dialog.get_settings() if self.homing_dialog is not None else None
        settings = self._collect_homing_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "Homing Training", "Select a valid stand-drive ONNX policy.")
            return
        supervised_init = str(settings.get("ppo_supervised_init", "1")).strip().lower() not in ("0", "false", "no", "off")
        checkpoint_path = settings.get("checkpoint_path", "")
        if supervised_init and not os.path.isfile(checkpoint_path):
            QMessageBox.warning(self, "Homing Training", "Select a Homing checkpoint to initialize PPO fine-tuning.")
            return
        if supervised_init:
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_type = str(checkpoint.get("checkpoint_type", "supervised" if "model_state" in checkpoint else "unknown"))
                if checkpoint_type != "supervised" or "model_state" not in checkpoint:
                    QMessageBox.warning(self, "Homing Training", "RL Fine Tune must start from homing_policy_supervised.pt. Run Train Policy first or select the supervised checkpoint.")
                    return
            except Exception as exc:
                QMessageBox.warning(self, "Homing Training", f"Could not validate supervised checkpoint: {exc}")
                return
        if self.homing_dialog is not None:
            self.homing_dialog.clear_log()
            self.homing_dialog.set_status("ppo fine-tuning")
        self._start_homing_worker("rl_train", {
            **settings,
            "rollout_steps": to_int(settings.get("rollout_steps", 1000), 1000),
            "homing_trajectory_seconds": to_float(settings.get("homing_trajectory_seconds", 3.0), 3.0),
            "homing_stand_warmup_steps": to_int(settings.get("homing_stand_warmup_steps", 200), 200),
            "ppo_total_steps": to_int(settings.get("ppo_total_steps", 20000), 20000),
            "ppo_num_envs": to_int(settings.get("ppo_num_envs", 4), 4),
            "ppo_rollout_steps": to_int(settings.get("ppo_rollout_steps", 256), 256),
            "ppo_epochs": to_int(settings.get("ppo_epochs", 4), 4),
            "ppo_learning_rate": to_float(settings.get("ppo_learning_rate", 3e-4), 3e-4),
            "ppo_domain_randomize": to_float(settings.get("ppo_domain_randomize", 0.3), 0.3),
            "ppo_supervised_init": bool(supervised_init),
            "ppo_use_trajectory_reward": str(settings.get("ppo_use_trajectory_reward", "1")).strip().lower() not in ("0", "false", "no", "off"),
            "ppo_mask_wheel_actions": str(settings.get("ppo_mask_wheel_actions", "1")).strip().lower() not in ("0", "false", "no", "off"),
            "ppo_strategy_preset": str(settings.get("ppo_strategy_preset", "light")).strip(),
            "reward_track": to_float(settings.get("reward_track", 6.0), 6.0),
            "reward_base_acc": to_float(settings.get("reward_base_acc", 0.002), 0.002),
            "reward_upright": to_float(settings.get("reward_upright", 2.0), 2.0),
            "reward_action_rate": to_float(settings.get("reward_action_rate", 0.04), 0.04),
            "reward_contact": to_float(settings.get("reward_contact", 0.0005), 0.0005),
            "hidden_dim": to_int(settings.get("hidden_dim", 256), 256),
            "seed": to_int(settings.get("seed", 42), 42),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def rl_fine_tune_ctbc_policy(self):
        dialog_settings = self.ctbc_dialog.get_settings() if self.ctbc_dialog is not None else None
        settings = self._collect_ctbc_ui_settings(dialog_settings)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "CTBC Stair Reflex", "Select a valid base ONNX policy.")
            return
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.clear_log()
            self.ctbc_dialog.set_status("tuning stair controller")
        self._start_ctbc_worker("tune_controller", {
            **settings,
            "rollout_steps": to_int(settings.get("rollout_steps", 1000), 1000),
            "ppo_total_steps": to_int(settings.get("ppo_total_steps", 1000000), 1000000),
            "ppo_num_envs": to_int(settings.get("ppo_num_envs", 32), 32),
            "ppo_rollout_steps": to_int(settings.get("ppo_rollout_steps", 512), 512),
            "ppo_epochs": to_int(settings.get("ppo_epochs", 4), 4),
            "ppo_learning_rate": to_float(settings.get("ppo_learning_rate", 5e-5), 5e-5),
            "ppo_domain_randomize": to_float(settings.get("ppo_domain_randomize", 0.05), 0.05),
            "ppo_supervised_init": False,
            "ppo_use_trajectory_reward": True,
            "ppo_mask_wheel_actions": False,
            "reward_track": to_float(settings.get("reward_track", 1.2), 1.2),
            "reward_upright": to_float(settings.get("reward_upright", 2.0), 2.0),
            "reward_action_rate": to_float(settings.get("reward_action_rate", 0.04), 0.04),
            "ctbc_contact_threshold": to_float(settings.get("ctbc_contact_threshold", 30.0), 30.0),
            "ctbc_contact_window": to_int(settings.get("ctbc_contact_window", 3), 3),
            "ctbc_lift_amplitude": to_float(settings.get("ctbc_lift_amplitude", 0.90), 0.90),
            "ctbc_lift_period": to_float(settings.get("ctbc_lift_period", 0.75), 0.75),
            "ctbc_anneal_ratio": to_float(settings.get("ctbc_anneal_ratio", 0.70), 0.70),
            "ctbc_episode_steps": to_int(settings.get("ctbc_episode_steps", 1024), 1024),
            "ctbc_residual_limit": to_float(settings.get("ctbc_residual_limit", 4.0), 4.0),
            "ctbc_gate_height_threshold": to_float(settings.get("ctbc_gate_height_threshold", 0.06), 0.06),
            "ctbc_gate_height_softness": to_float(settings.get("ctbc_gate_height_softness", 0.025), 0.025),
            "ctbc_gate_rise": to_float(settings.get("ctbc_gate_rise", 0.35), 0.35),
            "ctbc_gate_fall": to_float(settings.get("ctbc_gate_fall", 0.08), 0.08),
            "ctbc_gate_lift_threshold": to_float(settings.get("ctbc_gate_lift_threshold", 0.25), 0.25),
            "ctbc_gate_reward_threshold": to_float(settings.get("ctbc_gate_reward_threshold", 0.35), 0.35),
            "ctbc_assist_trigger_gate": to_float(settings.get("ctbc_assist_trigger_gate", 0.12), 0.12),
            "ctbc_assist_gate_floor": to_float(settings.get("ctbc_assist_gate_floor", 0.85), 0.85),
            "ctbc_assist_min": to_float(settings.get("ctbc_assist_min", 0.0), 0.0),
            "ctbc_gate_residual_runtime": str(settings.get("ctbc_gate_residual_runtime", "0")),
            "ctbc_anneal_bc_with_assist": str(settings.get("ctbc_anneal_bc_with_assist", "1")),
            "ctbc_distill_primitive": str(settings.get("ctbc_distill_primitive", "1")),
            "ctbc_bc_weight_min": to_float(settings.get("ctbc_bc_weight_min", 0.15), 0.15),
            "ctbc_reflex_only": str(settings.get("ctbc_reflex_only", "1")),
            "ctbc_controller_candidates": to_int(settings.get("ctbc_controller_candidates", 64), 64),
            "ctbc_reflex_samples": to_int(settings.get("ctbc_reflex_samples", 8192), 8192),
            "ctbc_reflex_epochs": to_int(settings.get("ctbc_reflex_epochs", 12), 12),
            "ctbc_reflex_batch": to_int(settings.get("ctbc_reflex_batch", 256), 256),
            "ctbc_reflex_lr": to_float(settings.get("ctbc_reflex_lr", 3e-4), 3e-4),
            "ctbc_reflex_flat_ratio": to_float(settings.get("ctbc_reflex_flat_ratio", 0.35), 0.35),
            "ctbc_reflex_gain": to_float(settings.get("ctbc_reflex_gain", 1.0), 1.0),
            "ctbc_reflex_segment_steps": to_int(settings.get("ctbc_reflex_segment_steps", 128), 128),
            "ctbc_fast_teacher_steps": to_int(settings.get("ctbc_fast_teacher_steps", 4096), 4096),
            "ctbc_fast_teacher_epochs": to_int(settings.get("ctbc_fast_teacher_epochs", 6), 6),
            "ctbc_fast_teacher_batch": to_int(settings.get("ctbc_fast_teacher_batch", 256), 256),
            "ctbc_fast_teacher_lr": to_float(settings.get("ctbc_fast_teacher_lr", 2e-4), 2e-4),
            "ctbc_fast_teacher_gain": to_float(settings.get("ctbc_fast_teacher_gain", 1.0), 1.0),
            "ctbc_fast_teacher_stair_height": to_float(settings.get("ctbc_fast_teacher_stair_height", 0.12), 0.12),
            "ctbc_safe_tilt": to_float(settings.get("ctbc_safe_tilt", 0.22), 0.22),
            "ctbc_emergency_tilt": to_float(settings.get("ctbc_emergency_tilt", 0.34), 0.34),
            "ctbc_terminate_tilt": to_float(settings.get("ctbc_terminate_tilt", 0.42), 0.42),
            "ctbc_tilt_guard_penalty": to_float(settings.get("ctbc_tilt_guard_penalty", 8.0), 8.0),
            "ctbc_bad_contact_threshold": to_float(settings.get("ctbc_bad_contact_threshold", 1.0), 1.0),
            "ctbc_bad_contact_penalty": to_float(settings.get("ctbc_bad_contact_penalty", 20.0), 20.0),
            "ctbc_lift_cooldown": to_float(settings.get("ctbc_lift_cooldown", 0.35), 0.35),
            "ctbc_contact_baseline_alpha": to_float(settings.get("ctbc_contact_baseline_alpha", 0.02), 0.02),
            "ctbc_contact_spike_threshold": to_float(settings.get("ctbc_contact_spike_threshold", 80.0), 80.0),
            "ctbc_force_alternating_lift": str(settings.get("ctbc_force_alternating_lift", "1")),
            "ctbc_curriculum_enabled": str(settings.get("ctbc_curriculum_enabled", "1")),
            "ctbc_stair_height_min": to_float(settings.get("ctbc_stair_height_min", 0.025), 0.025),
            "ctbc_stair_height_max": to_float(settings.get("ctbc_stair_height_max", 0.20), 0.20),
            "ctbc_curriculum_ratio": to_float(settings.get("ctbc_curriculum_ratio", 0.60), 0.60),
            "ctbc_select_after_ratio": to_float(settings.get("ctbc_select_after_ratio", 0.70), 0.70),
            "ctbc_shoulder_gain": to_float(settings.get("ctbc_shoulder_gain", 0.50), 0.50),
            "ctbc_leg_gain": to_float(settings.get("ctbc_leg_gain", 0.0), 0.0),
            "ctbc_leg_push_gain": to_float(settings.get("ctbc_leg_push_gain", 1.75), 1.75),
            "ctbc_hip_gain": to_float(settings.get("ctbc_hip_gain", 0.0), 0.0),
            "ctbc_stance_gain": to_float(settings.get("ctbc_stance_gain", 0.30), 0.30),
            "ctbc_wheel_push_gain": to_float(settings.get("ctbc_wheel_push_gain", 0.0), 0.0),
            "ctbc_ff_clip": to_float(settings.get("ctbc_ff_clip", 4.0), 4.0),
            "ctbc_action_clip": to_float(settings.get("ctbc_action_clip", 4.0), 4.0),
            "ctbc_compensate_action_scale": str(settings.get("ctbc_compensate_action_scale", "1")),
            "ctbc_clearance_target": to_float(settings.get("ctbc_clearance_target", 0.14), 0.14),
            "ctbc_base_height_target": to_float(settings.get("ctbc_base_height_target", 0.14), 0.14),
            "ctbc_clearance_stair_ratio": to_float(settings.get("ctbc_clearance_stair_ratio", 0.90), 0.90),
            "ctbc_climb_stair_ratio": to_float(settings.get("ctbc_climb_stair_ratio", 0.75), 0.75),
            "ctbc_reward_lift": to_float(settings.get("ctbc_reward_lift", 2.0), 2.0),
            "ctbc_reward_clearance": to_float(settings.get("ctbc_reward_clearance", 1.0), 1.0),
            "ctbc_reward_wheel_clearance": to_float(settings.get("ctbc_reward_wheel_clearance", 4.0), 4.0),
            "ctbc_reward_base_height": to_float(settings.get("ctbc_reward_base_height", 4.0), 4.0),
            "ctbc_reward_stair_success": to_float(settings.get("ctbc_reward_stair_success", 5.0), 5.0),
            "ctbc_hard_stair_threshold": to_float(settings.get("ctbc_hard_stair_threshold", 0.14), 0.14),
            "ctbc_hard_stair_fail_penalty": to_float(settings.get("ctbc_hard_stair_fail_penalty", 1.5), 1.5),
            "ctbc_reward_forward_progress": to_float(settings.get("ctbc_reward_forward_progress", 35.0), 35.0),
            "ctbc_min_forward_progress": to_float(settings.get("ctbc_min_forward_progress", 0.010), 0.010),
            "ctbc_reward_stair_forward": to_float(settings.get("ctbc_reward_stair_forward", 2.0), 2.0),
            "ctbc_reward_stair_motion": to_float(settings.get("ctbc_reward_stair_motion", 4.0), 4.0),
            "ctbc_no_progress_penalty": to_float(settings.get("ctbc_no_progress_penalty", 1.0), 1.0),
            "ctbc_reward_height_progress": to_float(settings.get("ctbc_reward_height_progress", 30.0), 30.0),
            "ctbc_reward_balance_on_stair": to_float(settings.get("ctbc_reward_balance_on_stair", 0.7), 0.7),
            "ctbc_min_climb_height": to_float(settings.get("ctbc_min_climb_height", 0.015), 0.015),
            "ctbc_no_climb_penalty": to_float(settings.get("ctbc_no_climb_penalty", 0.12), 0.12),
            "ctbc_base_imitation": to_float(settings.get("ctbc_base_imitation", 0.5), 0.5),
            "ctbc_non_wheel_contact_penalty": to_float(settings.get("ctbc_non_wheel_contact_penalty", 4.0), 4.0),
            "ctbc_command_x_min": to_float(settings.get("ctbc_command_x_min", 0.35), 0.35),
            "ctbc_command_x_max": to_float(settings.get("ctbc_command_x_max", 0.70), 0.70),
            "ctbc_command_y_abs": to_float(settings.get("ctbc_command_y_abs", 0.03), 0.03),
            "ctbc_command_yaw_abs": to_float(settings.get("ctbc_command_yaw_abs", 0.05), 0.05),
            "hidden_dim": to_int(settings.get("hidden_dim", 256), 256),
            "seed": to_int(settings.get("seed", 42), 42),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def test_ctbc_export_policy(self):
        settings = self._collect_ctbc_ui_settings(self.ctbc_dialog.get_settings() if self.ctbc_dialog is not None else None)
        if not os.path.isfile(settings.get("policy_path", "")):
            QMessageBox.warning(self, "CTBC Stair Reflex", "Select a valid base ONNX policy.")
            return
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_status("testing stair controller")
            self.ctbc_dialog.append_log("[ctbc-controller-test] launching render test with base ONNX + deterministic controller.\n")
        self._start_homing_command_timer()
        self._start_ctbc_worker("test_controller", {
            **settings,
            "test_steps": to_int(settings.get("ctbc_episode_steps", 1000), 1000),
            "ctbc_residual_limit": to_float(settings.get("ctbc_residual_limit", 4.0), 4.0),
            "ctbc_gate_height_threshold": to_float(settings.get("ctbc_gate_height_threshold", 0.06), 0.06),
            "ctbc_gate_height_softness": to_float(settings.get("ctbc_gate_height_softness", 0.025), 0.025),
            "ctbc_gate_rise": to_float(settings.get("ctbc_gate_rise", 0.35), 0.35),
            "ctbc_gate_fall": to_float(settings.get("ctbc_gate_fall", 0.08), 0.08),
            "ctbc_gate_lift_threshold": to_float(settings.get("ctbc_gate_lift_threshold", 0.25), 0.25),
            "ctbc_gate_reward_threshold": to_float(settings.get("ctbc_gate_reward_threshold", 0.35), 0.35),
            "ctbc_assist_trigger_gate": to_float(settings.get("ctbc_assist_trigger_gate", 0.12), 0.12),
            "ctbc_assist_gate_floor": to_float(settings.get("ctbc_assist_gate_floor", 0.85), 0.85),
            "ctbc_assist_min": to_float(settings.get("ctbc_assist_min", 0.0), 0.0),
            "ctbc_safe_tilt": to_float(settings.get("ctbc_safe_tilt", 0.22), 0.22),
            "ctbc_emergency_tilt": to_float(settings.get("ctbc_emergency_tilt", 0.34), 0.34),
            "ctbc_terminate_tilt": to_float(settings.get("ctbc_terminate_tilt", 0.42), 0.42),
            "ctbc_tilt_guard_penalty": to_float(settings.get("ctbc_tilt_guard_penalty", 8.0), 8.0),
            "ctbc_lift_cooldown": to_float(settings.get("ctbc_lift_cooldown", 0.35), 0.35),
            "ctbc_contact_baseline_alpha": to_float(settings.get("ctbc_contact_baseline_alpha", 0.02), 0.02),
            "ctbc_contact_spike_threshold": to_float(settings.get("ctbc_contact_spike_threshold", 80.0), 80.0),
            "ctbc_shoulder_gain": to_float(settings.get("ctbc_shoulder_gain", 0.50), 0.50),
            "ctbc_leg_gain": to_float(settings.get("ctbc_leg_gain", 0.0), 0.0),
            "ctbc_leg_push_gain": to_float(settings.get("ctbc_leg_push_gain", 1.75), 1.75),
            "ctbc_hip_gain": to_float(settings.get("ctbc_hip_gain", 0.0), 0.0),
            "ctbc_stance_gain": to_float(settings.get("ctbc_stance_gain", 0.30), 0.30),
            "ctbc_ff_clip": to_float(settings.get("ctbc_ff_clip", 4.0), 4.0),
            "ctbc_action_clip": to_float(settings.get("ctbc_action_clip", 4.0), 4.0),
            "ctbc_compensate_action_scale": str(settings.get("ctbc_compensate_action_scale", "1")),
            "ctbc_clearance_target": to_float(settings.get("ctbc_clearance_target", 0.08), 0.08),
            "ctbc_base_height_target": to_float(settings.get("ctbc_base_height_target", 0.06), 0.06),
            "command_values": list(self.current_command_values),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def test_ctbc_primitive(self):
        settings = self._collect_ctbc_ui_settings(self.ctbc_dialog.get_settings() if self.ctbc_dialog is not None else None)
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_status("testing ctbc primitive")
            self.ctbc_dialog.append_log("[ctbc-primitive-test] launching render test with gravity off and frozen base.\n")
        self._start_ctbc_worker("test_primitive", {
            **settings,
            "test_steps": to_int(settings.get("ctbc_episode_steps", 1200), 1200),
            "ctbc_lift_amplitude": to_float(settings.get("ctbc_lift_amplitude", 0.90), 0.90),
            "ctbc_lift_period": to_float(settings.get("ctbc_lift_period", 0.75), 0.75),
            "ctbc_lift_cooldown": to_float(settings.get("ctbc_lift_cooldown", 0.35), 0.35),
            "ctbc_shoulder_gain": to_float(settings.get("ctbc_shoulder_gain", 0.50), 0.50),
            "ctbc_leg_gain": to_float(settings.get("ctbc_leg_gain", 0.0), 0.0),
            "ctbc_leg_push_gain": to_float(settings.get("ctbc_leg_push_gain", 1.75), 1.75),
            "ctbc_hip_gain": to_float(settings.get("ctbc_hip_gain", 0.0), 0.0),
            "ctbc_stance_gain": to_float(settings.get("ctbc_stance_gain", 0.30), 0.30),
            "ctbc_ff_clip": to_float(settings.get("ctbc_ff_clip", 4.0), 4.0),
            "ctbc_action_clip": to_float(settings.get("ctbc_action_clip", 4.0), 4.0),
            "ctbc_compensate_action_scale": str(settings.get("ctbc_compensate_action_scale", "1")),
            "env_overrides": self._homing_env_overrides(settings.get("env_id")),
        })

    def export_homing_onnx(self):
        settings = self._collect_homing_ui_settings(self.homing_dialog.get_settings() if self.homing_dialog is not None else None)
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        default_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "homing", "latest")
        checkpoint_path = settings.get("checkpoint_path") or os.path.join(default_dir, "homing_policy.pt")
        output_path = settings.get("output_path", "")
        if not output_path:
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Homing Policy ONNX",
                os.path.join(default_dir, "homing_policy.onnx"),
                "ONNX Files (*.onnx)"
            )
            if not output_path:
                return
            settings["output_path"] = output_path
        if self.homing_dialog is not None:
            self.homing_dialog.set_status("exporting")
        self._start_homing_worker("export", {
            **settings,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "env_overrides": self._homing_env_overrides(env_id),
        })

    def export_ctbc_onnx(self):
        settings = self._collect_ctbc_ui_settings(self.ctbc_dialog.get_settings() if self.ctbc_dialog is not None else None)
        env_id = settings.get("env_id", self.env_id_cb.currentText())
        default_dir = os.path.join(self._repo_root(), "envs", env_id, "weights", "ctbc", "latest")
        checkpoint_path = settings.get("checkpoint_path") or os.path.join(default_dir, "ctbc_policy_ppo.pt")
        output_path = settings.get("output_path") or os.path.join(default_dir, "ctbc_policy.onnx")
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_status("exporting")
        self._start_ctbc_worker("export", {
            **settings,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "env_overrides": self._homing_env_overrides(env_id),
        })

    def on_homing_log(self, message):
        if self.homing_dialog is not None:
            self.homing_dialog.append_log(message)

    def stop_homing_job(self):
        self._stop_homing_command_timer()
        if self.homing_worker is not None:
            self.homing_worker.request_stop()
        if self.homing_dialog is not None:
            self.homing_dialog.append_log("[homing] stop requested by user.\n")
            self.homing_dialog.set_status("stopping")

    def on_ctbc_log(self, message):
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.append_log(message)

    def stop_ctbc_job(self):
        self._stop_homing_command_timer()
        if self.ctbc_worker is not None:
            self.ctbc_worker.request_stop()
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.append_log("[ctbc] stop requested by user.\n")
            self.ctbc_dialog.set_status("stopping")

    def on_homing_finished(self, summary):
        self._stop_homing_command_timer()
        self._homing_last_summary = dict(summary or {})
        env_id = self._homing_last_summary.get("env_id", self.homing_settings.get("env_id", self.env_id_cb.currentText()))
        self._homing_last_summary_by_env[env_id] = dict(self._homing_last_summary)
        if self.homing_dialog is not None:
            self.homing_dialog.set_running(False)
            self.homing_dialog.set_status("done")
        if self._homing_last_summary.get("checkpoint_path"):
            self.homing_settings["checkpoint_path"] = str(self._homing_last_summary.get("checkpoint_path"))
            self.homing_settings_by_env[env_id] = dict(self.homing_settings)
        self._refresh_homing_dialog()
        if self._homing_last_summary.get("stopped", False):
            QMessageBox.information(self, "Homing Training", "Homing job stopped.")
        else:
            QMessageBox.information(self, "Homing Training", "Homing job finished.")

    def on_ctbc_finished(self, summary):
        self._stop_homing_command_timer()
        self._homing_last_summary = dict(summary or {})
        env_id = self._homing_last_summary.get("env_id", self.ctbc_settings.get("env_id", self.env_id_cb.currentText()))
        self._homing_last_summary_by_env[env_id] = dict(self._homing_last_summary)
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_running(False)
            self.ctbc_dialog.set_status("done")
        if self._homing_last_summary.get("checkpoint_path"):
            self.ctbc_settings["checkpoint_path"] = str(self._homing_last_summary.get("checkpoint_path"))
        if self._homing_last_summary.get("onnx_path"):
            self.ctbc_settings["output_path"] = str(self._homing_last_summary.get("onnx_path"))
        selected = self._homing_last_summary.get("selected_metrics", {})
        if isinstance(selected, dict) and isinstance(selected.get("params", None), dict):
            for key, value in selected["params"].items():
                self.ctbc_settings[str(key)] = str(value)
        self.ctbc_settings_by_env[env_id] = dict(self.ctbc_settings)
        self._refresh_ctbc_dialog()
        if self._homing_last_summary.get("stopped", False):
            QMessageBox.information(self, "CTBC Stair Reflex", "CTBC job stopped.")
        else:
            QMessageBox.information(self, "CTBC Stair Reflex", "CTBC job finished.")

    def on_homing_error(self, error_msg):
        self._stop_homing_command_timer()
        if self.homing_dialog is not None:
            self.homing_dialog.set_running(False)
            self.homing_dialog.append_log(f"[homing] ERROR: {error_msg}\n")
            self.homing_dialog.set_status("error")
        QMessageBox.critical(self, "Homing Training", error_msg)

    def on_ctbc_error(self, error_msg):
        self._stop_homing_command_timer()
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_running(False)
            self.ctbc_dialog.append_log(f"[ctbc] ERROR: {error_msg}\n")
            self.ctbc_dialog.set_status("error")
        QMessageBox.critical(self, "CTBC Stair Reflex", error_msg)

    def _on_homing_thread_finished(self):
        self._stop_homing_command_timer()
        self.homing_thread = None
        self.homing_worker = None
        self.homing_worker_mode = None
        if self.homing_dialog is not None:
            self.homing_dialog.set_running(False)

    def _on_ctbc_thread_finished(self):
        self._stop_homing_command_timer()
        self.ctbc_thread = None
        self.ctbc_worker = None
        self.ctbc_worker_mode = None
        if self.ctbc_dialog is not None:
            self.ctbc_dialog.set_running(False)

    def open_hardware_settings(self):
        env_id = self.env_id_cb.currentText()
        self._ensure_hardware_defaults()
        dialog = HardwareSettingsDialog((self.hardware_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.hardware_settings = dialog.get_settings()
            # Save back to per-env cache so it persists after env switches
            self.hardware_settings_by_env[env_id] = (self.hardware_settings).copy()

    def open_actuator_settings(self):
        env_id = self.env_id_cb.currentText()
        self._ensure_actuator_defaults()
        dialog = ActuatorSettingsDialog((self.actuator_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.actuator_settings = dialog.get_settings()
            self.actuator_settings_by_env[env_id] = (self.actuator_settings).copy()

    def open_action_scale_settings(self):
        env_id = self.env_id_cb.currentText()
        self._ensure_action_scale_defaults()
        dialog = ActionScaleSettingsDialog(list(self.action_scales), [dict(item) for item in self.action_clippings], self)
        if dialog.exec_() == QDialog.Accepted:
            self.action_scales, self.action_clippings = dialog.get_settings()
            self.action_scales_by_env[env_id] = list(self.action_scales)
            self.action_clippings_by_env[env_id] = [dict(item) for item in self.action_clippings]

    def open_observation_settings(self):
        # Open the dialog with the latest settings for the current env
        env_id = self.env_id_cb.currentText()
        self._ensure_observation_defaults()  # Sync cache
        dialog = ObservationSettingsDialog((self.observation_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.observation_settings = dialog.get_settings()
            # Save current env settings back into the cache (so they restore next time)
            self.obs_settings_by_env[env_id] = (self.observation_settings).copy()
            # Mark that user manually changed settings (for reference)
            self.observation_overridden_by_user = True

    def open_initial_pose_settings(self):
        env_id = self.env_id_cb.currentText()
        self._ensure_initial_pose_defaults()
        dialog = InitialPoseSettingsDialog((self.initial_pose_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.initial_pose_settings = dialog.get_settings()
            self.initial_pose_settings_by_env[env_id] = {
                "base_z": str((self.initial_pose_settings).get("base_z", "0.3")),
                "joints": dict((self.initial_pose_settings).get("joints", {}))
            }

    def open_final_pose_settings(self):
        env_id = self.homing_dialog.get_settings().get("env_id", self.env_id_cb.currentText()) if self.homing_dialog is not None else self.env_id_cb.currentText()
        self._ensure_final_pose_defaults_for_env(env_id)
        dialog = FinalPoseSettingsDialog(dict(self.final_pose_settings_by_env.get(env_id, {})), self)
        if dialog.exec_() == QDialog.Accepted:
            self.final_pose_settings = dialog.get_settings()
            self.final_pose_settings_by_env[env_id] = {
                "joints": dict((self.final_pose_settings).get("joints", {})),
                "velocities": dict((self.final_pose_settings).get("velocities", {})),
                "same": bool((self.final_pose_settings).get("same", True)),
                "priorities": dict((self.final_pose_settings).get("priorities", {})),
            }
            self._ensure_homing_defaults(env_id)
            self.homing_settings["final_pos"] = self._final_pose_csv(env_id, "joints")
            self.homing_settings["final_vel"] = self._final_pose_csv(env_id, "velocities")
            self.homing_settings["final_pose_same"] = "1" if self.final_pose_settings_by_env[env_id].get("same", True) else "0"
            self.homing_settings["final_pose_priorities"] = self._final_pose_csv(env_id, "priorities")
            self.homing_settings_by_env[env_id] = dict(self.homing_settings)
            self._refresh_homing_dialog()

    def open_homing_command_range_settings(self):
        env_id = self.homing_dialog.get_settings().get("env_id", self.env_id_cb.currentText()) if self.homing_dialog is not None else self.env_id_cb.currentText()
        self._ensure_homing_command_ranges_for_env(env_id)
        dialog = CommandRangeSettingsDialog(dict(self.homing_command_ranges_by_env.get(env_id, {})), self)
        if dialog.exec_() == QDialog.Accepted:
            self.homing_command_ranges = dialog.get_settings()
            self.homing_command_ranges_by_env[env_id] = {
                "mins": list((self.homing_command_ranges).get("mins", [])),
                "maxs": list((self.homing_command_ranges).get("maxs", [])),
            }
            self._ensure_homing_command_ranges_for_env(env_id)
            self._ensure_homing_defaults(env_id)
            self.homing_settings["command_mins"] = self._homing_command_range_csv(env_id, "mins")
            self.homing_settings["command_maxs"] = self._homing_command_range_csv(env_id, "maxs")
            mins = self.homing_command_ranges_by_env[env_id]["mins"]
            maxs = self.homing_command_ranges_by_env[env_id]["maxs"]
            if mins:
                self.homing_settings["command_min"] = str(mins[0])
            if maxs:
                self.homing_settings["command_max"] = str(maxs[0])
            self.homing_settings_by_env[env_id] = dict(self.homing_settings)
            self._refresh_homing_dialog()

    def open_ctbc_command_range_settings(self):
        env_id = self.ctbc_dialog.get_settings().get("env_id", self.env_id_cb.currentText()) if self.ctbc_dialog is not None else self.env_id_cb.currentText()
        self._ensure_homing_command_ranges_for_env(env_id)
        dialog = CommandRangeSettingsDialog(dict(self.homing_command_ranges_by_env.get(env_id, {})), self)
        if dialog.exec_() == QDialog.Accepted:
            self.homing_command_ranges = dialog.get_settings()
            self.homing_command_ranges_by_env[env_id] = {
                "mins": list((self.homing_command_ranges).get("mins", [])),
                "maxs": list((self.homing_command_ranges).get("maxs", [])),
            }
            self._ensure_homing_command_ranges_for_env(env_id)
            self._ensure_ctbc_defaults(env_id)
            self.ctbc_settings["command_mins"] = self._homing_command_range_csv(env_id, "mins")
            self.ctbc_settings["command_maxs"] = self._homing_command_range_csv(env_id, "maxs")
            mins = self.homing_command_ranges_by_env[env_id]["mins"]
            maxs = self.homing_command_ranges_by_env[env_id]["maxs"]
            if mins:
                self.ctbc_settings["command_min"] = str(mins[0])
            if maxs:
                self.ctbc_settings["command_max"] = str(maxs[0])
            self.ctbc_settings_by_env[env_id] = dict(self.ctbc_settings)
            self._refresh_ctbc_dialog()

    def open_depth_randomization_settings(self):
        env_id = self.env_id_cb.currentText()
        self._ensure_depth_randomization_defaults()
        dialog = DepthRandomizationSettingsDialog((self.depth_randomization_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.depth_randomization_settings = dialog.get_settings()
            self.depth_randomization_settings_by_env[env_id] = dict(self.depth_randomization_settings)

    # ---------------- Run / Gather Config ----------------

    def start_test(self):
        # Ensure latest settings for the current env
        self._last_run_had_error = False
        self._ensure_actuator_defaults()
        self._ensure_action_scale_defaults()
        self._ensure_observation_defaults()
        self._ensure_hardware_defaults()
        self._ensure_initial_pose_defaults()
        self._ensure_monitor_defaults()
        self._ensure_fine_tune_defaults()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Test running...")
        self.log_output.clear()
        self._log_buffer = ""
        self._pending_log_chunks.clear()
        self._log_color_index = 0
        self._joint_color_map.clear()
        self._redirect_log_streams()
        self._update_status_label()
        self.position_command_cb.setEnabled(False)
        config = self._gather_config()
        if config is None:
            self._restore_log_streams()
            return
        policy_file_path = self.policy_file_le.text().strip()
        if not policy_file_path or not os.path.isfile(policy_file_path):
            self._restore_log_streams()
            QMessageBox.critical(self, "Error", "Please select a valid ONNX file.")
            self.position_command_cb.setEnabled(True)
            self._reset_ui_after_test()
            return
        encoder_file_path = self.encoder_file_le.text().strip()
        self.tester = Tester()
        self.tester.load_config(config)
        self.tester.load_policy(policy_file_path)
        self.tester.overlayUpdated.connect(self._update_monitor_overlay)
        self.tester.depthUpdated.connect(self._update_depth_overlay)
        self.tester.alphaUpdated.connect(self._update_alpha_overlay)
        if self.policy_type_cb.currentText().strip().lower() == "encoder+mlp":
            if not encoder_file_path or not os.path.isfile(encoder_file_path):
                self._restore_log_streams()
                QMessageBox.critical(self, "Error", "Please select a valid Encoder ONNX file.")
                self.position_command_cb.setEnabled(True)
                self._reset_ui_after_test()
                return
            self.tester.load_encoder(encoder_file_path)
        self._apply_fine_tune_settings_to_tester()
        self.tester.set_monitor_joints(self.monitor_settings.get("selected_joints", []))
        self._init_default_command_values()
        for i, value in enumerate(self.current_command_values):
            self.tester.update_command(i, value)
        self.tester.stepFinished.connect(self.send_current_command)
        self.thread = QThread()
        self.worker = TesterWorker(self.tester)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_test_finished)
        self.worker.error.connect(self.on_test_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _gather_config(self):
        try:
            self._ensure_actuator_defaults()
            self._ensure_action_scale_defaults()
            self._ensure_hardware_defaults()
            self._ensure_initial_pose_defaults()
            self._ensure_monitor_defaults()
            self._ensure_fine_tune_defaults()
            self._ensure_depth_randomization_defaults()
            # hardware: convert numeric strings to float where applicable
            hardware_numeric = {
                k: self._normalize_hardware_value(v)
                for k, v in self.hardware_settings.items()
            }
            actuator = (self.actuator_settings).copy()
            action_scales = [to_float(v, 1.0) for v in self.action_scales]
            action_clippings = []
            for item in self.action_clippings:
                item = item if isinstance(item, dict) else {}
                min_value = to_float(item.get("min", -1.0), -1.0)
                max_value = to_float(item.get("max", 1.0), 1.0)
                if min_value > max_value:
                    min_value, max_value = max_value, min_value
                action_clippings.append({
                    "enabled": bool(item.get("enabled", False)),
                    "min": min_value,
                    "max": max_value,
                })
            initial_positions = {
                "base_z": to_float(self.initial_pose_settings.get("base_z", self._make_initial_pose_defaults(self.env_id_cb.currentText()).get("base_z", 0.3)), 0.3),
                "joints": {
                    joint_name: to_float(value, 0.0)
                    for joint_name, value in (self.initial_pose_settings.get("joints", {})).items()
                }
            }
            hm_x_forward = to_float(self.hm_x_fwd_le.text(), 0.5)
            hm_x_backward = to_float(self.hm_x_bwd_le.text(), 0.5)
            hm_y_left = to_float(self.hm_y_left_le.text(), 0.3)
            hm_y_right = to_float(self.hm_y_right_le.text(), 0.3)
            hm_resolution = to_float(self.hm_resolution_le.text(), 0.1)
            depth_scale = max(1, to_int(self.depth_scale_le.text(), 8))
            inference_onnx_path = str(self.dataset_height_map_settings.get("inference_onnx_path", "")).strip()
            inference_visualize = bool(self.hm_inference_cb.isChecked())
            if inference_visualize and (not inference_onnx_path or not os.path.isfile(inference_onnx_path)):
                raise RuntimeError("Select a valid height-map inference ONNX file before enabling inference visualization.")
            hm_size_x = hm_x_forward + hm_x_backward
            hm_size_y = hm_y_left + hm_y_right
            hm_res_x, hm_res_y = self._compute_height_map_grid(
                hm_x_forward,
                hm_x_backward,
                hm_y_left,
                hm_y_right,
                hm_resolution,
            )
            dataset_height_map = {
                "enabled": bool(self.depth_dataset_save_cb.isChecked()) or bool(self.hm_visualize_cb.isChecked()) or inference_visualize,
                "visualize": bool(self.hm_visualize_cb.isChecked()),
                "inference_visualize": inference_visualize,
                "inference_onnx_path": inference_onnx_path,
                "frame_body": self._default_height_map_frame_body(self.env_id_cb.currentText()),
                "x_forward": hm_x_forward,
                "x_backward": hm_x_backward,
                "y_left": hm_y_left,
                "y_right": hm_y_right,
                "size_x": hm_size_x,
                "size_y": hm_size_y,
                "resolution": hm_resolution,
                "res_x": hm_res_x,
                "res_y": hm_res_y,
            }
            self.dataset_height_map_settings = {
                "x_forward": str(dataset_height_map["x_forward"]),
                "x_backward": str(dataset_height_map["x_backward"]),
                "y_left": str(dataset_height_map["y_left"]),
                "y_right": str(dataset_height_map["y_right"]),
                "resolution": str(dataset_height_map["resolution"]),
                "visualize": dataset_height_map["visualize"],
                "inference_visualize": dataset_height_map["inference_visualize"],
                "inference_onnx_path": dataset_height_map["inference_onnx_path"],
                "frame_body": dataset_height_map["frame_body"],
                "depth_scale": str(depth_scale),
            }
            self.dataset_height_map_settings_by_env[self.env_id_cb.currentText()] = dict(self.dataset_height_map_settings)
            depth_randomization_cfg = {
                "enabled": bool(self.depth_randomization_settings.get("enabled", False)),
                "camera_xyz_shift_m": to_float(self.depth_randomization_settings.get("camera_xyz_shift_m", 0.01), 0.01),
                "camera_pitch_shift_deg": to_float(self.depth_randomization_settings.get("camera_pitch_shift_deg", 1.0), 1.0),
                "camera_fov_shift_deg": to_float(self.depth_randomization_settings.get("camera_fov_shift_deg", 1.0), 1.0),
                "gaussian_prob": to_float(self.depth_randomization_settings.get("gaussian_prob", 0.3), 0.3),
                "gaussian_stddev": to_float(self.depth_randomization_settings.get("gaussian_stddev", 0.01), 0.01),
                "rotation_prob": to_float(self.depth_randomization_settings.get("rotation_prob", 0.3), 0.3),
                "rotation_deg": to_float(self.depth_randomization_settings.get("rotation_deg", 2.0), 2.0),
                "edge_noise_prob": to_float(self.depth_randomization_settings.get("edge_noise_prob", 0.3), 0.3),
                "edge_noise_ratio": to_float(self.depth_randomization_settings.get("edge_noise_ratio", 0.03), 0.03),
                "small_object_prob": to_float(self.depth_randomization_settings.get("small_object_prob", 0.3), 0.3),
                "small_object_ratio": to_float(self.depth_randomization_settings.get("small_object_ratio", 0.02), 0.02),
                "small_object_count": to_int(self.depth_randomization_settings.get("small_object_count", 6), 6),
                "spot_noise_prob": to_float(self.depth_randomization_settings.get("spot_noise_prob", 0.3), 0.3),
                "spot_noise_ratio": to_float(self.depth_randomization_settings.get("spot_noise_ratio", 0.03), 0.03),
            }
            self.depth_randomization_settings = {
                "enabled": depth_randomization_cfg["enabled"],
                "camera_xyz_shift_m": str(depth_randomization_cfg["camera_xyz_shift_m"]),
                "camera_pitch_shift_deg": str(depth_randomization_cfg["camera_pitch_shift_deg"]),
                "camera_fov_shift_deg": str(depth_randomization_cfg["camera_fov_shift_deg"]),
                "gaussian_prob": str(depth_randomization_cfg["gaussian_prob"]),
                "gaussian_stddev": str(depth_randomization_cfg["gaussian_stddev"]),
                "rotation_prob": str(depth_randomization_cfg["rotation_prob"]),
                "rotation_deg": str(depth_randomization_cfg["rotation_deg"]),
                "edge_noise_prob": str(depth_randomization_cfg["edge_noise_prob"]),
                "edge_noise_ratio": str(depth_randomization_cfg["edge_noise_ratio"]),
                "small_object_prob": str(depth_randomization_cfg["small_object_prob"]),
                "small_object_ratio": str(depth_randomization_cfg["small_object_ratio"]),
                "small_object_count": str(depth_randomization_cfg["small_object_count"]),
                "spot_noise_prob": str(depth_randomization_cfg["spot_noise_prob"]),
                "spot_noise_ratio": str(depth_randomization_cfg["spot_noise_ratio"]),
            }
            self.depth_randomization_settings_by_env[self.env_id_cb.currentText()] = dict(self.depth_randomization_settings)

            # settings: copy latest settings for the current env
            env_id = self.env_id_cb.currentText()
            self._ensure_observation_defaults()
            settings_cfg = (self.observation_settings).copy()

            # height_map patching with env YAML defaults
            env_cfg = self.env_config.get(env_id, {}) or {}
            env_settings_cfg = env_cfg.get("settings", env_cfg) if isinstance(env_cfg, dict) else {}
            yaml_hm = env_settings_cfg.get("height_map", {}) if isinstance(env_settings_cfg.get("height_map", {}), dict) else {}
            yaml_size_x = to_float(yaml_hm.get("size_x", 1.0), 1.0)
            yaml_size_y = to_float(yaml_hm.get("size_y", 0.6), 0.6)
            yaml_hm_defaults = {
                "x_forward": to_float(yaml_hm.get("x_forward", yaml_size_x / 2.0), yaml_size_x / 2.0),
                "x_backward": to_float(yaml_hm.get("x_backward", yaml_size_x / 2.0), yaml_size_x / 2.0),
                "y_left": to_float(yaml_hm.get("y_left", yaml_size_y / 2.0), yaml_size_y / 2.0),
                "y_right": to_float(yaml_hm.get("y_right", yaml_size_y / 2.0), yaml_size_y / 2.0),
                "size_x": yaml_size_x,
                "size_y": yaml_size_y,
                "res_x": to_int(yaml_hm.get("res_x", 15)),
                "res_y": to_int(yaml_hm.get("res_y", 9)),
                "resolution": to_float(yaml_hm.get("resolution", 0.1), 0.1),
                "target_height": to_float(yaml_hm.get("target_height", 0.5), 0.5),
                "clipping_min": to_float(yaml_hm.get("clipping_min", 0.0), 0.0),
                "clipping_max": to_float(yaml_hm.get("clipping_max", 0.33), 0.33),
                "point_stride": to_int(yaml_hm.get("point_stride", 16), 16),
                "max_range": to_float(yaml_hm.get("max_range", 2.5), 2.5),
                "camera_update_freq": to_float(yaml_hm.get("camera_update_freq", 10.0), 10.0),
                "debug_print": bool(yaml_hm.get("debug_print", False)),
            }

            hm_val = settings_cfg.get("height_map", None)
            if isinstance(hm_val, dict):
                hm_val.setdefault("x_forward", yaml_hm_defaults["x_forward"])
                hm_val.setdefault("x_backward", yaml_hm_defaults["x_backward"])
                hm_val.setdefault("y_left", yaml_hm_defaults["y_left"])
                hm_val.setdefault("y_right", yaml_hm_defaults["y_right"])
                hm_val.setdefault("size_x", yaml_hm_defaults["size_x"])
                hm_val.setdefault("size_y", yaml_hm_defaults["size_y"])
                hm_val.setdefault("res_x", yaml_hm_defaults["res_x"])
                hm_val.setdefault("res_y", yaml_hm_defaults["res_y"])
                hm_val.setdefault("resolution", yaml_hm_defaults["resolution"])
                hm_val.setdefault("freq", 50)
                hm_val.setdefault("scale", 1.0)
                hm_val.setdefault("target_height", yaml_hm_defaults["target_height"])
                hm_val.setdefault("clipping_min", yaml_hm_defaults["clipping_min"])
                hm_val.setdefault("clipping_max", yaml_hm_defaults["clipping_max"])
                hm_val.setdefault("point_stride", yaml_hm_defaults["point_stride"])
                hm_val.setdefault("max_range", yaml_hm_defaults["max_range"])
                hm_val.setdefault("camera_update_freq", yaml_hm_defaults["camera_update_freq"])
                hm_val.setdefault("debug_print", yaml_hm_defaults["debug_print"])
                settings_cfg["height_map"] = hm_val
            elif hm_val is None:
                settings_cfg["height_map"] = None
            else:
                settings_cfg["height_map"] = None

            fine_tune_cfg = self._collect_fine_tune_ui_settings()

            config = {
                "env": {
                    "id": env_id,
                    "terrain": self.terrain_id_cb.currentText(),
                    "max_duration": float(self.max_duration_le.text().strip()),
                    "position_command": self.position_command_cb.isChecked()
                },
                "settings": settings_cfg,
                "observation": settings_cfg,  # backward-compatibility alias
                "policy": {
                    "policy_type": self.policy_type_cb.currentText(),
                    "h_in_dim": int(self.h_in_dim_le.text().strip()),
                    "c_in_dim": int(self.c_in_dim_le.text().strip()),
                    "onnx_file": os.path.basename(self.policy_file_le.text())
                },
                "random": {
                    "precision": self.precision_cb.currentText(),
                    "sensor_noise": self.sensor_noise_cb.currentText(),
                    "init_noise": self.init_noise_slider.value() / 100.0,
                    "sliding_friction": self.sliding_friction_slider.value() / 100.0,
                    "torsional_friction": self.torsional_friction_slider.value() / 100.0,
                    "rolling_friction": self.rolling_friction_slider.value() / 100.0,
                    "friction_loss": self.friction_loss_slider.value() / 100.0,
                    "action_delay_prob": self.action_delay_prob_slider.value() / 100.0,
                    "mass_noise": self.mass_noise_slider.value() / 100.0,
                    "load": self.load_slider.value() / 10.0
                },
                "action_scales": action_scales,
                "action_clippings": action_clippings,
                "actuator": actuator,
                "hardware": hardware_numeric,
                "initial_positions": initial_positions,
                "joint_offsets": initial_positions,
                "monitoring": {
                    "selected_joints": list(self.monitor_settings.get("selected_joints", [])),
                    "depth_enabled": bool(self.depth_window_toggle_cb.isChecked()) or inference_visualize,
                    "dataset_enabled": bool(self.depth_dataset_save_cb.isChecked()),
                    "depth_scale": depth_scale,
                    "depth_randomization": depth_randomization_cfg,
                    "height_map": dataset_height_map,
                    "moe_alpha_enabled": bool(getattr(self, "moe_alpha_vis_cb", None) and self.moe_alpha_vis_cb.isChecked()),
                    "moe_alpha_onnx_path": self._moe_alpha_onnx_path(self.env_id_cb.currentText()),
                },
                "fine_tune": {
                    "enabled": bool(fine_tune_cfg.get("enabled", False)),
                    "ridge_lambda": to_float(fine_tune_cfg.get("ridge_lambda", 1e-4), 1e-4),
                    "max_samples": to_int(fine_tune_cfg.get("max_samples", 5000), 5000),
                }
            }

            # random_table (only if present)
            cur_file_path = os.path.abspath(__file__)
            random_path = os.path.join(os.path.dirname(cur_file_path), "../config/random_table.yaml")
            random_path = os.path.abspath(random_path)
            if os.path.isfile(random_path):
                with open(random_path) as f:
                    random_config = yaml.full_load(f)
                if isinstance(random_config, dict) and "random_table" in random_config:
                    config["random_table"] = random_config["random_table"]
            return config
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parameter setting error: {e}")
            self._reset_ui_after_test()
            return None

    def _reset_ui_after_test(self):
        self._restore_log_streams()
        self.mujoco_overlay.clear_overlay()
        self.alpha_overlay.clear_overlay()
        self.depth_image_widget.clear_frame()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Waiting ...")
        self._update_fine_tune_status_label()
        self._update_vision_train_status_label()

    def reset_command_buttons(self):
        for key in list(self.active_keys.keys()):
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            if not self._is_discrete_command(cmd_index):
                default_value = self._get_default_command_value(cmd_index)
                self._update_command_button(cmd_index, default_value)
            self.active_keys.pop(key)

    def on_test_finished(self):
        self.reset_command_buttons()
        the_text = "Test complete"
        self.status_label.setText(the_text)
        self._show_monitor_plot_if_enabled()
        self._reset_ui_after_test()
        self.position_command_cb.setEnabled(True)
        if not self._last_run_had_error:
            reply = QMessageBox.question(
                self,
                "Check Report",
                "Test has finished. Would you like to view the report?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                policy_file_path = self.policy_file_le.text().strip()
                report_path = os.path.join(os.path.dirname(policy_file_path), "report.pdf")
                if os.path.isfile(report_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
                else:
                    QMessageBox.warning(self, "Warning", "Report file (report.pdf) does not exist.")
        
    def on_test_error(self, error_msg):
        self._last_run_had_error = True
        self._show_monitor_plot_if_enabled()
        QMessageBox.critical(self, "Test Error", error_msg)
        self.status_label.setText("Error occurred")
        self._reset_ui_after_test()

    def stop_test(self):
        if self.tester:
            try:
                self.tester.stop()
                self.status_label.setText("Test stop requested")
                self.stop_button.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Test stop error: {e}")

    def closeEvent(self, event):
        self.depth_image_widget.clear_frame()
        self._restore_log_streams()
        self.mujoco_overlay.clear_overlay()
        self.alpha_overlay.clear_overlay()
        if self.vision_train_dialog is not None and self.vision_train_dialog.isVisible():
            self.vision_train_dialog.close()
        if self.vision_train_thread is not None and self.vision_train_thread.isRunning():
            self.vision_train_thread.quit()
            self.vision_train_thread.wait(1000)
        if self.moe_dialog is not None and self.moe_dialog.isVisible():
            self.moe_dialog.close()
        if self.moe_manual_dialog is not None and self.moe_manual_dialog.isVisible():
            self.moe_manual_dialog.close()
        if self.moe_thread is not None and self.moe_thread.isRunning():
            self.moe_thread.quit()
            self.moe_thread.wait(1000)
        if self.homing_dialog is not None and self.homing_dialog.isVisible():
            self.homing_dialog.close()
        if self.ctbc_dialog is not None and self.ctbc_dialog.isVisible():
            self.ctbc_dialog.close()
        if self.homing_thread is not None and self.homing_thread.isRunning():
            self.homing_thread.quit()
            self.homing_thread.wait(1000)
        if self.ctbc_thread is not None and self.ctbc_thread.isRunning():
            self.ctbc_thread.quit()
            self.ctbc_thread.wait(1000)
        super().closeEvent(event)
