# ui/dialogs/observation_settings.py
# -*- coding: utf-8 -*-
"""
Observation Settings dialog.

Design goals
------------
1) Saved-first priority:
   - Whatever the user saved with "OK" previously (passed in via `settings` from MainWindow)
     MUST be reflected in the UI and used as defaults, regardless of YAML/env defaults.
   - This dialog reads YAML/env defaults only as a fallback when a field isn't present in
     saved settings.

2) Robustness:
   - All text inputs are validated (int/float) with sensible fallbacks.
   - Height automatically adapts to content (dynamic rows + scroll area).

3) Separation of concerns:
   - The dialog is purely a view/editor of a settings dict. The caller (MainWindow)
     is responsible for caching per-env settings and passing them back next time.
     The dialog itself always prefers the provided `settings` over env defaults,
     so whenever the caller passes previously saved values, the UI will display them.

What gets returned by get_settings()
-----------------------------------
{
  "stacked_obs_order": [...],
  "non_stacked_obs_order": [...],
  "stack_size": int,
  "command_dim": int,
  "command_scales": { "0": float, "1": float, ... },
  "height_map": {
      "size_x": float, "size_y": float, "res_x": int, "res_y": int,
      "freq": int, "scale": float, "target_height": float,
      "clipping_min": float, "clipping_max": float
  } or None,
  # Per-observation entries (for each obs in self.obs_types). If present in any order,
  # value is {"freq": int, "scale": float}, otherwise None.
  "dof_pos": {...} or None,
  "dof_vel": {...} or None,
  ...
}

"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QGroupBox, QGridLayout,
    QScrollArea, QLineEdit, QWidget, QDialogButtonBox, QSizePolicy, QApplication, QStyle
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QDoubleValidator, QIntValidator

# Project helpers
from ui.utils import to_float, to_int, normalize_numkey_float_values
from ui.custom_widgets import NoWheelComboBox


class ObservationSettingsDialog(QDialog):
    def __init__(self, settings, parent):
        """
        Parameters
        ----------
        settings : dict
            The current (saved) observation settings for the active environment.
            These take absolute precedence over YAML/env defaults in this dialog.
        parent : MainWindow
            Used to read env_id and env_config for fallback defaults and scales.
        """
        super().__init__(parent)
        self.setWindowTitle("Observation Settings")

        # Keep obs_types consistent with MainWindow
        self.obs_types = [
            "dof_pos", "dof_vel", "ang_vel",
            "lin_vel_x", "lin_vel_y", "lin_vel_z",
            "projected_gravity",
            "lower_ang_vel", "upper_ang_vel",
            "lower_projected_gravity", "upper_projected_gravity",
            "lower_imu_ang_vel", "upper_imu_ang_vel",
            "lower_imu_projected_gravity", "upper_imu_projected_gravity",
            "height_map", "masked_height_map", "camera_height_map",
            "last_action",
        ]

        # Saved settings (highest priority source)
        self.settings = settings if isinstance(settings, dict) else {}

        # Parent ref (provides env_id, env_config)
        self.parent_widget = parent

        # Dynamic row stores
        # Each element is a dict: {"layout": QHBoxLayout, "combo": QComboBox, "freq": QComboBox, "scale": QComboBox}
        self.stacked_rows = []
        self.non_rows = []

        # Command scale widgets
        self.cmd_scale_cbs = []
        self.command_scales_grid = None

        # Build UI and then load saved settings into it
        self._setup_ui()
        self._load_existing_settings()  # <- saved-first priority

    # ---------- Small helpers ----------

    def _scale_options(self):
        """Common numeric options for scale selection combos."""
        return ["0", "0.01", "0.05", "0.1", "0.15", "0.25", "0.5", "0.75", "1.0", "2.0", "2.5", "5"]

    # ---------- UI construction ----------

    def _setup_ui(self):
        # ===== Outer layout =====
        self._outer_layout = QVBoxLayout(self)

        # ===== Scroll area (holds the main content) =====
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._outer_layout.addWidget(self._scroll)

        # ===== Inner widget/layout (everything scrolls inside here) =====
        self._inner_widget = QWidget()
        self._inner_layout = QVBoxLayout(self._inner_widget)

        # ---- small hint label
        obs_label = QLabel("State Order: [Stacked]  →  [Non-Stacked]  →  [Command]")
        obs_label.setAlignment(Qt.AlignCenter)
        obs_label.setStyleSheet(
            "QLabel { font-size: 7pt; color: #222; letter-spacing: 1px; }"
        )
        obs_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._inner_layout.addWidget(obs_label)

        # ---- read current environment config (fallback defaults only)
        env_id = self.parent_widget.env_id_cb.currentText()
        env_cfg = self.parent_widget.env_config.get(env_id, {}) or {}
        cmd_cfg_raw = env_cfg.get("command", {}) if isinstance(env_cfg.get("command", {}), dict) else {}
        command_scales_from_cfg = normalize_numkey_float_values(env_cfg.get("command_scales", {}))
        default_stack_size = to_int(env_cfg.get("stack_size", self.settings.get("stack_size", 3)), 3)

        # Saved-first for command_dim
        cmd_dim_val = to_int(self.settings.get("command_dim", cmd_cfg_raw.get("command_dim", 6)), 6)

        # ---------------- Stacked Observation ----------------
        stacked_group = QGroupBox("Stacked Observation")
        stacked_v = QVBoxLayout()

        # Stack size
        stack_size_h = QHBoxLayout()
        stack_size_h.addWidget(QLabel("Stack Size:"))
        self.stack_size_cb = NoWheelComboBox()
        self.stack_size_cb.addItems([str(i) for i in range(1, 16)])
        self.stack_size_cb.setCurrentText(str(self.settings.get("stack_size", default_stack_size)))
        stack_size_h.addWidget(self.stack_size_cb)
        stacked_v.addLayout(stack_size_h)

        # Rows container + add button
        self.stacked_container = QVBoxLayout()
        stacked_v.addLayout(self.stacked_container)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self.add_stacked())
        stacked_v.addWidget(add_btn)
        stacked_group.setLayout(stacked_v)

        # ---------------- Non-Stacked Observation ----------------
        non_group = QGroupBox("Non-Stacked Observation")
        non_v = QVBoxLayout()
        self.non_container = QVBoxLayout()
        non_v.addLayout(self.non_container)
        add_non = QPushButton("Add")
        add_non.clicked.connect(lambda: self.add_non())
        non_v.addWidget(add_non)
        non_group.setLayout(non_v)

        # Two groups side-by-side
        areas_layout = QHBoxLayout()
        areas_layout.addWidget(stacked_group)
        areas_layout.addWidget(non_group)
        self._inner_layout.addLayout(areas_layout)

        # ---------------- Height Map detail (size/res) ----------------
        height_group = QGroupBox("Height Map")
        height_layout = QFormLayout()
        size_res_layout = QHBoxLayout()

        hm_yaml = env_cfg.get("height_map", {}) if isinstance(env_cfg.get("height_map", {}), dict) else {}
        hm_settings = self.settings.get("height_map", {}) if isinstance(self.settings.get("height_map", {}), dict) else {}
        hm_default = {
            "size_x": to_float(hm_settings.get("size_x", hm_yaml.get("size_x", 1.0))),
            "size_y": to_float(hm_settings.get("size_y", hm_yaml.get("size_y", 0.6))),
            "res_x": to_int(hm_settings.get("res_x", hm_yaml.get("res_x", 15))),
            "res_y": to_int(hm_settings.get("res_y", hm_yaml.get("res_y", 9))),
            "target_height": to_float(hm_settings.get("target_height", hm_yaml.get("target_height", 0.5))),
            "clipping_min": to_float(hm_settings.get("clipping_min", hm_yaml.get("clipping_min", 0.0))),
            "clipping_max": to_float(hm_settings.get("clipping_max", hm_yaml.get("clipping_max", 0.33))),
            "point_stride": to_int(hm_settings.get("point_stride", hm_yaml.get("point_stride", 16))),
            "max_range": to_float(hm_settings.get("max_range", hm_yaml.get("max_range", 2.5))),
            "camera_update_freq": to_float(
                hm_settings.get("camera_update_freq", hm_yaml.get("camera_update_freq", 10.0))
            ),
            "debug_print": bool(hm_settings.get("debug_print", hm_yaml.get("debug_print", False))),
        }
        self.height_map_point_stride_default = hm_default["point_stride"]
        self.height_map_max_range_default = hm_default["max_range"]
        self.height_map_camera_update_freq_default = hm_default["camera_update_freq"]
        self.height_map_debug_print_default = hm_default["debug_print"]

        # Validators
        double_validator = QDoubleValidator(0.000001, 1e6, 4)
        double_validator.setNotation(QDoubleValidator.StandardNotation)
        target_height_validator = QDoubleValidator(-1e6, 1e6, 4)
        target_height_validator.setNotation(QDoubleValidator.StandardNotation)
        int_validator = QIntValidator(1, 10000)

        # Size X
        self.height_size_x_le = QLineEdit(str(hm_default["size_x"]))
        self.height_size_x_le.setFixedWidth(60)
        self.height_size_x_le.setPlaceholderText("x (m)")
        self.height_size_x_le.setValidator(double_validator)
        size_res_layout.addWidget(self.height_size_x_le)

        # ×
        size_res_layout.addWidget(QLabel("×"))

        # Size Y
        self.height_size_y_le = QLineEdit(str(hm_default["size_y"]))
        self.height_size_y_le.setFixedWidth(60)
        self.height_size_y_le.setPlaceholderText("y (m)")
        self.height_size_y_le.setValidator(double_validator)
        size_res_layout.addWidget(self.height_size_y_le)

        # Resolution label
        size_res_layout.addWidget(QLabel("  Resolution:"))

        # Res X
        self.height_res_x_le = QLineEdit(str(hm_default["res_x"]))
        self.height_res_x_le.setFixedWidth(60)
        self.height_res_x_le.setPlaceholderText("res_x")
        self.height_res_x_le.setValidator(int_validator)
        size_res_layout.addWidget(self.height_res_x_le)

        # ×
        size_res_layout.addWidget(QLabel("×"))

        # Res Y
        self.height_res_y_le = QLineEdit(str(hm_default["res_y"]))
        self.height_res_y_le.setFixedWidth(60)
        self.height_res_y_le.setPlaceholderText("res_y")
        self.height_res_y_le.setValidator(int_validator)
        size_res_layout.addWidget(self.height_res_y_le)

        height_layout.addRow("Size (m):", size_res_layout)

        target_layout = QHBoxLayout()
        self.height_target_height_le = QLineEdit(str(hm_default["target_height"]))
        self.height_target_height_le.setFixedWidth(60)
        self.height_target_height_le.setPlaceholderText("m")
        self.height_target_height_le.setValidator(target_height_validator)
        target_layout.addWidget(self.height_target_height_le)
        target_layout.addStretch(1)
        height_layout.addRow("Target Height:", target_layout)

        clipping_layout = QHBoxLayout()
        self.height_clipping_min_le = QLineEdit(str(hm_default["clipping_min"]))
        self.height_clipping_min_le.setFixedWidth(60)
        self.height_clipping_min_le.setPlaceholderText("min")
        self.height_clipping_min_le.setValidator(target_height_validator)
        clipping_layout.addWidget(self.height_clipping_min_le)
        clipping_layout.addWidget(QLabel("to"))
        self.height_clipping_max_le = QLineEdit(str(hm_default["clipping_max"]))
        self.height_clipping_max_le.setFixedWidth(60)
        self.height_clipping_max_le.setPlaceholderText("max")
        self.height_clipping_max_le.setValidator(target_height_validator)
        clipping_layout.addWidget(self.height_clipping_max_le)
        clipping_layout.addStretch(1)
        height_layout.addRow("Clipping:", clipping_layout)

        height_group.setLayout(height_layout)
        self._inner_layout.addWidget(height_group)

        # ---------------- Command ----------------
        command_group = QGroupBox("Command")
        command_layout = QFormLayout()

        # command_dim (1~6)
        self.command_dim_cb = NoWheelComboBox()
        self.command_dim_cb.addItems([str(i) for i in range(1, 7)])
        self.command_dim_cb.setCurrentText(str(cmd_dim_val))
        command_layout.addRow("Command Dim:", self.command_dim_cb)

        # Per-index scales grid
        self.command_scales_group = QGroupBox("Command Scales")
        self.command_scales_grid = QGridLayout(self.command_scales_group)
        self.command_scales_grid.setColumnStretch(1, 1)
        self._rebuild_command_scales(command_scales_from_cfg, int(self.command_dim_cb.currentText()))
        self.command_dim_cb.currentTextChanged.connect(
            lambda _:
                (self._rebuild_command_scales(command_scales_from_cfg, int(self.command_dim_cb.currentText())),
                 QTimer.singleShot(0, self._recalculate_height))
        )
        command_layout.addRow(self.command_scales_group)
        command_group.setLayout(command_layout)
        self._inner_layout.addWidget(command_group)

        # Put the inner widget into the scroll area
        self._scroll.setWidget(self._inner_widget)

        # ---------------- Dialog buttons (fixed at bottom) ----------------
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)   # Caller will read get_settings() on Accepted
        self._buttons.rejected.connect(self.reject)
        self._outer_layout.addWidget(self._buttons)

        # Set width and auto-resize height after UI build
        self.setMaximumWidth(900)
        QTimer.singleShot(0, self._recalculate_height)

    # ---------- Sizing logic ----------

    def _recalculate_height(self):
        """Measure content height and resize dialog accordingly (with a safe clamp)."""
        try:
            content_h = self._inner_widget.sizeHint().height()
            chrome_h = self._buttons.sizeHint().height()
            try:
                frame_h = self.style().pixelMetric(QStyle.PM_TitleBarHeight) \
                          + 2 * self.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
            except Exception:
                frame_h = 40

            total_h = int(content_h + chrome_h + frame_h)
            scr = (self.screen() if hasattr(self, "screen") else None) or QApplication.primaryScreen()
            avail_h = scr.availableGeometry().height() if scr else 900
            max_h = int(avail_h * 0.90)

            target_h = max(400, min(max_h, total_h))
            self.resize(900, target_h)
        except Exception:
            # Fallback to a safe default if measurement fails
            scr = (self.screen() if hasattr(self, "screen") else None) or QApplication.primaryScreen()
            avail_h = scr.availableGeometry().height() if scr else 900
            self.resize(900, int(avail_h * 0.90))

    # ---------- Command scale helpers ----------

    def _rebuild_command_scales(self, command_scales_from_cfg: dict, cmd_dim: int):
        """
        Rebuild the (index, scale) grid based on current command_dim, preferring saved values.

        Priority: self.settings["command_scales"]  >  env_cfg["command_scales"]  >  1.0
        """
        # Clear current grid content
        while self.command_scales_grid.count():
            item = self.command_scales_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.cmd_scale_cbs.clear()

        # Header
        self.command_scales_grid.addWidget(QLabel("Index"), 0, 0)
        self.command_scales_grid.addWidget(QLabel("Scale"), 0, 1)

        # Saved-first prior
        prior = normalize_numkey_float_values(self.settings.get("command_scales", {}))
        options = self._scale_options()

        for i in range(cmd_dim):
            idx_str = str(i)
            default_val = to_float(prior.get(idx_str, command_scales_from_cfg.get(idx_str, 1.0)), 1.0)

            row = i + 1
            self.command_scales_grid.addWidget(QLabel(f"{i}"), row, 0)

            cb = NoWheelComboBox()
            cb.addItems(options)
            # Use exact match if available, otherwise fall back to "1.0"
            cb.setCurrentText(str(default_val) if str(default_val) in options else "1.0")
            self.command_scales_grid.addWidget(cb, row, 1)
            self.cmd_scale_cbs.append(cb)

    # ---------- Dynamic row management ----------

    def add_stacked(self, selected: str = "", freq: int = 50, scale: float = 1.0):
        """Add one Stacked row: [obs_type][Freq][Scale][Delete]."""
        h = QHBoxLayout()

        # obs type
        combo = NoWheelComboBox()
        combo.addItems(self.obs_types)
        if selected and combo.findText(selected) < 0:
            combo.addItem(selected)
        if selected:
            combo.setCurrentText(selected)
        h.addWidget(combo)

        # freq
        h.addWidget(QLabel("Freq:"))
        freq_cb = NoWheelComboBox()
        freq_cb.addItems(["10", "25", "50"])
        freq_cb.setCurrentText(str(freq))
        h.addWidget(freq_cb)

        # scale (prefer saved or default scale for that obs)
        h.addWidget(QLabel("Scale:"))
        default_scale = self.get_default_scale(selected or combo.currentText())
        scale_cb = NoWheelComboBox()
        scale_cb.addItems(self._scale_options())
        items = [scale_cb.itemText(j) for j in range(scale_cb.count())]
        scale_cb.setCurrentText(str(scale) if str(scale) in items else str(default_scale))
        h.addWidget(scale_cb)

        # delete
        remove = QPushButton("Delete")
        remove.clicked.connect(lambda: self.remove_layout(h, self.stacked_container, self.stacked_rows))
        h.addWidget(remove)

        self.stacked_container.addLayout(h)
        self.stacked_rows.append({"layout": h, "combo": combo, "freq": freq_cb, "scale": scale_cb})

        QTimer.singleShot(0, self._recalculate_height)

    def add_non(self, selected: str = "", freq: int = 50, scale: float = 1.0):
        """Add one Non-Stacked row: [obs_type][Freq][Scale][Delete]."""
        h = QHBoxLayout()

        combo = NoWheelComboBox()
        combo.addItems(self.obs_types)
        if selected and combo.findText(selected) < 0:
            combo.addItem(selected)
        if selected:
            combo.setCurrentText(selected)
        h.addWidget(combo)

        h.addWidget(QLabel("Freq:"))
        freq_cb = NoWheelComboBox()
        freq_cb.addItems(["10", "25", "50"])
        freq_cb.setCurrentText(str(freq))
        h.addWidget(freq_cb)

        h.addWidget(QLabel("Scale:"))
        default_scale = self.get_default_scale(selected or combo.currentText())
        scale_cb = NoWheelComboBox()
        scale_cb.addItems(self._scale_options())
        items = [scale_cb.itemText(j) for j in range(scale_cb.count())]
        scale_cb.setCurrentText(str(scale) if str(scale) in items else str(default_scale))
        h.addWidget(scale_cb)

        remove = QPushButton("Delete")
        remove.clicked.connect(lambda: self.remove_layout(h, self.non_container, self.non_rows))
        h.addWidget(remove)

        self.non_container.addLayout(h)
        self.non_rows.append({"layout": h, "combo": combo, "freq": freq_cb, "scale": scale_cb})

        QTimer.singleShot(0, self._recalculate_height)

    def get_default_scale(self, obs_type: str) -> float:
        """Read default scale for a given obs_type from the current env's obs_scales."""
        env_id = self.parent_widget.env_id_cb.currentText()
        env_cfg = self.parent_widget.env_config.get(env_id, {}) or {}
        obs_scales = env_cfg.get("obs_scales", {}) or {}
        return to_float(obs_scales.get(obs_type, 1.0), 1.0)

    def remove_layout(self, layout, container, row_store):
        """Remove a dynamic row and clean up its widgets."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        container.removeItem(layout)
        layout.deleteLater()
        for i, row in enumerate(row_store):
            if row["layout"] is layout:
                row_store.pop(i)
                break
        QTimer.singleShot(0, self._recalculate_height)

    # ---------- Data extraction ----------

    def _extract_height_map_freq_scale_from_rows(self):
        """
        Find the first 'height_map' row across stacked/non-stacked and
        return (True, freq, scale). If none found, return (False, None, None).
        """
        for rows in (self.stacked_rows, self.non_rows):
            for row in rows:
                if row["combo"].currentText() in ("height_map", "masked_height_map", "camera_height_map"):
                    freq_val = to_int(row["freq"].currentText(), 50)
                    scale_val = to_float(row["scale"].currentText(), 1.0)
                    return True, freq_val, scale_val
        return False, None, None

    def get_settings(self) -> dict:
        """
        Aggregate all UI selections into a settings dictionary
        that matches what MainWindow expects and caches per-env.
        """
        stacked_order = []
        non_order = []
        obs_dict = {}

        # stacked rows
        for row in self.stacked_rows:
            obs_type = row["combo"].currentText()
            if obs_type:
                stacked_order.append(obs_type)
                obs_dict[obs_type] = {
                    "freq": int(row["freq"].currentText()),
                    "scale": float(row["scale"].currentText())
                }

        # non-stacked rows
        for row in self.non_rows:
            obs_type = row["combo"].currentText()
            if obs_type:
                non_order.append(obs_type)
                obs_dict[obs_type] = {
                    "freq": int(row["freq"].currentText()),
                    "scale": float(row["scale"].currentText())
                }

        # ensure all obs_types exist as keys (None if not selected)
        for obs_type in self.obs_types:
            if obs_type not in obs_dict:
                obs_dict[obs_type] = None

        # command_scales
        command_scales = {str(i): float(cb.currentText()) for i, cb in enumerate(self.cmd_scale_cbs)}

        # height map detail
        sx = to_float(self.height_size_x_le.text(), 1.0)
        sy = to_float(self.height_size_y_le.text(), 0.6)
        rx = to_int(self.height_res_x_le.text(), 15)
        ry = to_int(self.height_res_y_le.text(), 9)
        target_height = to_float(self.height_target_height_le.text(), 0.5)
        clipping_min = to_float(self.height_clipping_min_le.text(), 0.0)
        clipping_max = to_float(self.height_clipping_max_le.text(), 0.33)
        if clipping_min > clipping_max:
            clipping_min, clipping_max = clipping_max, clipping_min

        hm_selected, hm_freq, hm_scale = self._extract_height_map_freq_scale_from_rows()
        height_map = {
            "size_x": sx, "size_y": sy, "res_x": rx, "res_y": ry,
            "freq": hm_freq, "scale": hm_scale, "target_height": target_height,
            "clipping_min": clipping_min, "clipping_max": clipping_max,
            "point_stride": self.height_map_point_stride_default,
            "max_range": self.height_map_max_range_default,
            "camera_update_freq": self.height_map_camera_update_freq_default,
            "debug_print": self.height_map_debug_print_default,
        } if hm_selected else None

        # Avoid duplicate/ambiguous "height_map" entry (details live in height_map above)
        obs_dict.pop("height_map", None)

        stack_size = int(self.stack_size_cb.currentText())
        return {
            "stacked_obs_order": stacked_order,
            "non_stacked_obs_order": non_order,
            "stack_size": stack_size,
            "command_dim": int(self.command_dim_cb.currentText()),
            "command_scales": command_scales,
            "height_map": height_map,
            **obs_dict
        }

    # ---------- Saved-first population ----------

    def _load_existing_settings(self):
        """
        Apply saved settings into the UI (highest priority), falling back to env defaults
        only if a specific field wasn't saved yet.
        """
        # stack size (saved-first)
        env_id = self.parent_widget.env_id_cb.currentText()
        default_stack_size = to_int(
            self.parent_widget.env_config.get(env_id, {}).get("stack_size", 3), 3
        )
        self.stack_size_cb.setCurrentText(str(self.settings.get("stack_size", default_stack_size)))

        # stacked rows
        for obs in self.settings.get("stacked_obs_order", []):
            obs_cfg = self.settings.get(obs)
            freq = 50
            scale = self.get_default_scale(obs)
            if isinstance(obs_cfg, dict):
                freq = to_int(obs_cfg.get("freq", freq), freq)
                scale = to_float(obs_cfg.get("scale", scale), scale)
            self.add_stacked(obs, freq, scale)

        # non-stacked rows
        for obs in self.settings.get("non_stacked_obs_order", []):
            obs_cfg = self.settings.get(obs)
            freq = 50
            scale = self.get_default_scale(obs)
            if isinstance(obs_cfg, dict):
                freq = to_int(obs_cfg.get("freq", freq), freq)
                scale = to_float(obs_cfg.get("scale", scale), scale)
            self.add_non(obs, freq, scale)

        # command_dim (the scales grid itself prefers saved command_scales)
        default_cmd_dim = to_int(
            self.parent_widget.env_config.get(env_id, {}).get("command", {}).get("command_dim", 6), 6
        )
        self.command_dim_cb.setCurrentText(str(self.settings.get("command_dim", default_cmd_dim)))
