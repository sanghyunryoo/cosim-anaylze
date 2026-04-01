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
from ui.custom_widgets import MujocoOverlayWidget, NoWheelComboBox, NoWheelSlider, NonClickableButton
from ui.dialogs.action_scale_settings import ActionScaleSettingsDialog
from ui.dialogs.actuator_settings import ActuatorSettingsDialog
from ui.dialogs.hardware_settings import HardwareSettingsDialog
from ui.dialogs.observation_settings import ObservationSettingsDialog
from ui.dialogs.initial_pose_settings import InitialPoseSettingsDialog
from ui.dialogs.fine_tune_bias_editor import FineTuneBiasEditorDialog
from ui.workers import TesterWorker
from PyQt5.QtWidgets import QSizePolicy
from envs.initial_pose import get_default_initial_joint_map, get_initial_pose_joint_names


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

        self.obs_types = ["dof_pos", "dof_vel", "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel", "projected_gravity", "height_map", "last_action"]

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
        self.thread = None
        self.worker = None
        self.tester = None
        self.current_command_values = [0.0] * 6
        self.command_sensitivity_le_list = []
        self.max_command_value_le_list = []
        self.command_initial_value_le_list = []
        self.command_timer = None
        self.actuator_settings = {}
        self.actuator_settings_by_env = {}
        self.action_scales = []
        self.action_scales_by_env = {}
        self.hardware_settings = {}
        self.hardware_settings_by_env = {}
        self.initial_pose_settings = {}
        self.initial_pose_settings_by_env = {}
        self.monitor_settings = {}
        self.monitor_settings_by_env = {}
        self.monitor_joint_checkboxes = {}
        self.fine_tune_settings = {}
        self.fine_tune_settings_by_env = {}
        self.fine_tune_bias_dialog = None
        self.mujoco_overlay = MujocoOverlayWidget()
        self.mujoco_overlay.closed.connect(self._on_monitor_overlay_closed)
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
            "height_map": {"size_x": 1.0, "size_y": 0.6, "res_x": 15, "res_y": 9, "freq": 50, "scale": 1.0},
            "dof_pos": None,
            "dof_vel": None,
            "lin_vel_x": None,
            "lin_vel_y": None,
            "lin_vel_z": None,
            "ang_vel": None,
            "projected_gravity": None,
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

        height_in_order = ("height_map" in stacked_list) or ("height_map" in non_stacked_list)
        if height_in_order:
            height_map_yaml = settings_cfg.get("height_map", {}) if isinstance(settings_cfg.get("height_map", {}), dict) else {}
            height_map_val = {
                "size_x": to_float(height_map_yaml.get("size_x", 1.0)),
                "size_y": to_float(height_map_yaml.get("size_y", 0.6)),
                "res_x": to_int(height_map_yaml.get("res_x", 15)),
                "res_y": to_int(height_map_yaml.get("res_y", 9)),
                "freq": 50,
                "scale": 1.0,
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

    def _ensure_action_scale_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.action_scales_by_env:
            self.action_scales_by_env[env_id] = self._make_action_scale_defaults(env_id)
        self.action_scales = list(self.action_scales_by_env[env_id])

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
        if env_id not in self.hardware_settings_by_env:
            self.hardware_settings_by_env[env_id] = self._make_hardware_defaults(env_id)
        self.hardware_settings = (self.hardware_settings_by_env[env_id]).copy()

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
        raw = env_cfg.get("initial_positions", {}) or {}
        joint_defaults = get_default_initial_joint_map(env_id)
        joints_raw = raw.get("joints", raw) if isinstance(raw, dict) else {}
        if isinstance(joints_raw, dict):
            for joint_name in joint_defaults:
                if joint_name in joints_raw:
                    joint_defaults[joint_name] = str(joints_raw[joint_name])
                else:
                    joint_defaults[joint_name] = str(joint_defaults[joint_name])
        else:
            for joint_name in joint_defaults:
                joint_defaults[joint_name] = str(joint_defaults[joint_name])
        return {"joints": joint_defaults}

    def _ensure_initial_pose_defaults(self):
        env_id = self.env_id_cb.currentText()
        if env_id not in self.initial_pose_settings_by_env:
            self.initial_pose_settings_by_env[env_id] = self._make_initial_pose_defaults(env_id)
        self.initial_pose_settings = {
            "joints": dict((self.initial_pose_settings_by_env[env_id]).get("joints", {}))
        }

    def _make_monitor_defaults(self, env_id: str):
        joint_names = list(get_initial_pose_joint_names(env_id))
        default_selected = joint_names[: min(4, len(joint_names))]
        return {
            "available_joints": joint_names,
            "selected_joints": list(default_selected),
        }

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

        if new_env_id in self.actuator_settings_by_env:
            self.actuator_settings = (self.actuator_settings_by_env[new_env_id]).copy()
        else:
            self.actuator_settings = self._make_actuator_defaults(new_env_id)
            self.actuator_settings_by_env[new_env_id] = (self.actuator_settings).copy()

        if new_env_id in self.hardware_settings_by_env:
            self.hardware_settings = (self.hardware_settings_by_env[new_env_id]).copy()
        else:
            self.hardware_settings = self._make_hardware_defaults(new_env_id)
            self.hardware_settings_by_env[new_env_id] = (self.hardware_settings).copy()

        if new_env_id in self.initial_pose_settings_by_env:
            self.initial_pose_settings = {
                "joints": dict((self.initial_pose_settings_by_env[new_env_id]).get("joints", {}))
            }
        else:
            self.initial_pose_settings = self._make_initial_pose_defaults(new_env_id)
            self.initial_pose_settings_by_env[new_env_id] = {
                "joints": dict((self.initial_pose_settings).get("joints", {}))
            }

        if new_env_id in self.monitor_settings_by_env:
            self.monitor_settings = dict(self.monitor_settings_by_env[new_env_id])
        else:
            self.monitor_settings = self._make_monitor_defaults(new_env_id)
            self.monitor_settings_by_env[new_env_id] = dict(self.monitor_settings)

        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}

        # UI upper bounds (example retained)
        command_0_max = "1.5"
        command_2_max = "1.5"
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

        self._refresh_monitor_joint_checkboxes()
        self._sync_fine_tune_controls_from_cache()

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
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping and key not in self.active_keys:
            btn, cmd_index, direction = self.key_mapping[key]
            btn.setChecked(True)
            self.active_keys[key] = {"cmd_index": cmd_index, "direction": direction}

    def handle_key_release(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping:
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            if key in self.active_keys:
                self.active_keys.pop(key)
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

    def send_current_command(self):
        # Apply key-driven deltas within bounds, update tester and status
        for key_info in self.active_keys.values():
            cmd_index = key_info["cmd_index"]
            direction = key_info["direction"]
            step = self._parse_float(self.command_sensitivity_le_list[cmd_index].text(), 0.1)
            max_command_value = self._parse_float(self.max_command_value_le_list[cmd_index].text(), 2.0)
            current_value = self.current_command_values[cmd_index]
            new_value = current_value + direction * step
            if direction > 0:
                new_value = min(new_value, max_command_value)
            else:
                new_value = max(new_value, -max_command_value)
            self.current_command_values[cmd_index] = new_value
            self._update_command_button(cmd_index, new_value)
        if self.tester:
            for i, value in enumerate(self.current_command_values):
                self.tester.update_command(i, value)
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
        top_h_layout.addLayout(log_v_layout, 2)
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
        self.sensor_noise_cb.setCurrentText("low")
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
        form_layout.addRow("Sliding Friction:", create_slider_row(self.sliding_friction_slider, 0, 100, 80, 100, 2))
        self.torsional_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Torsional Friction:", create_slider_row(self.torsional_friction_slider, 0, 10, 2, 100, 2))
        self.rolling_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Rolling Friction:", create_slider_row(self.rolling_friction_slider, 0, 10, 1, 100, 2))
        self.friction_loss_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Friction Loss:", create_slider_row(self.friction_loss_slider, 0, 100, 10, 100, 2))
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

    def _create_command_settings_group(self, parent_layout):
        command_group = QGroupBox("Command Settings")
        command_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        grid_layout = QGridLayout(command_group)
        grid_layout.addWidget(QLabel("Index"), 0, 0)
        grid_layout.addWidget(QLabel("Sensitivity"), 0, 1)
        grid_layout.addWidget(QLabel("Max Value"), 0, 2)
        grid_layout.addWidget(QLabel("Initial Value"), 0, 3)

        # command[3] initial value is taken from the env's 'command' section
        settings = self.env_config.get(self.env_id_cb.currentText(), {}) or {}
        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}
        cmd3_init = str(to_float(cmd_cfg.get("command_3_initial", 0.0), 0.0))

        for i in range(6):  # indices 0~5
            label = QLabel(f"command[{i}]")
            sensitivity_le = QLineEdit("0.02")
            max_value_le = QLineEdit("1.5" if i in [0, 1, 2] else "1")
            init_value_widget = QLineEdit(cmd3_init) if i == 3 else QLabel("0.0")
            grid_layout.addWidget(label, i + 1, 0)
            grid_layout.addWidget(sensitivity_le, i + 1, 1)
            grid_layout.addWidget(max_value_le, i + 1, 2)
            grid_layout.addWidget(init_value_widget, i + 1, 3)
            self.command_sensitivity_le_list.append(sensitivity_le)
            self.max_command_value_le_list.append(max_value_le)
            self.command_initial_value_le_list.append(init_value_widget)
        self.position_command_cb = QCheckBox("Position Command")
        self.position_command_cb.setChecked(False)
        row_position = 6 + 1
        grid_layout.addWidget(self.position_command_cb, row_position, 0, 1, 4, Qt.AlignLeft)
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
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.setMinimumHeight(120)
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
        dialog = ActionScaleSettingsDialog(list(self.action_scales), self)
        if dialog.exec_() == QDialog.Accepted:
            self.action_scales = dialog.get_settings()
            self.action_scales_by_env[env_id] = list(self.action_scales)

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
                "joints": dict((self.initial_pose_settings).get("joints", {}))
            }

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
            # hardware: convert numeric strings to float where applicable
            hardware_numeric = {k: to_float(v, v) for k, v in self.hardware_settings.items()}
            actuator = (self.actuator_settings).copy()
            action_scales = [to_float(v, 1.0) for v in self.action_scales]
            initial_positions = {
                "joints": {
                    joint_name: to_float(value, 0.0)
                    for joint_name, value in (self.initial_pose_settings.get("joints", {})).items()
                }
            }

            # settings: copy latest settings for the current env
            env_id = self.env_id_cb.currentText()
            self._ensure_observation_defaults()
            settings_cfg = (self.observation_settings).copy()

            # height_map patching with env YAML defaults
            env_cfg = self.env_config.get(env_id, {}) or {}
            env_settings_cfg = env_cfg.get("settings", env_cfg) if isinstance(env_cfg, dict) else {}
            yaml_hm = env_settings_cfg.get("height_map", {}) if isinstance(env_settings_cfg.get("height_map", {}), dict) else {}
            yaml_hm_defaults = {
                "size_x": to_float(yaml_hm.get("size_x", 1.0)),
                "size_y": to_float(yaml_hm.get("size_y", 0.6)),
                "res_x": to_int(yaml_hm.get("res_x", 15)),
                "res_y": to_int(yaml_hm.get("res_y", 9)),
            }

            hm_val = settings_cfg.get("height_map", None)
            if isinstance(hm_val, dict):
                hm_val.setdefault("size_x", yaml_hm_defaults["size_x"])
                hm_val.setdefault("size_y", yaml_hm_defaults["size_y"])
                hm_val.setdefault("res_x", yaml_hm_defaults["res_x"])
                hm_val.setdefault("res_y", yaml_hm_defaults["res_y"])
                hm_val.setdefault("freq", 50)
                hm_val.setdefault("scale", 1.0)
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
                "actuator": actuator,
                "hardware": hardware_numeric,
                "initial_positions": initial_positions,
                "monitoring": {
                    "selected_joints": list(self.monitor_settings.get("selected_joints", [])),
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
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Waiting ...")
        self._update_fine_tune_status_label()

    def reset_command_buttons(self):
        for key in list(self.active_keys.keys()):
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
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
        self._restore_log_streams()
        self.mujoco_overlay.clear_overlay()
        super().closeEvent(event)
