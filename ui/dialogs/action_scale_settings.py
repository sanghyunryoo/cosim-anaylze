from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QDialogButtonBox,
    QScrollArea,
    QWidget,
    QGridLayout,
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt


class ActionScaleSettingsDialog(QDialog):
    def __init__(self, action_scales, action_clippings, parent):
        super().__init__(parent)
        self.action_scales = list(action_scales) if isinstance(action_scales, (list, tuple)) else []
        self.action_clippings = list(action_clippings) if isinstance(action_clippings, (list, tuple)) else []
        self.setWindowTitle("Action Scale Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.addWidget(QLabel("Joint"), 0, 0)
        grid.addWidget(QLabel("Scale"), 0, 1)
        grid.addWidget(QLabel("Clip"), 0, 2)
        grid.addWidget(QLabel("Min"), 0, 3)
        grid.addWidget(QLabel("Max"), 0, 4)
        self.scale_fields = []
        self.clip_enabled_fields = []
        self.clip_min_fields = []
        self.clip_max_fields = []
        for i, value in enumerate(self.action_scales):
            clip = self.action_clippings[i] if i < len(self.action_clippings) else {}
            if not isinstance(clip, dict):
                clip = {}

            scale_le = QLineEdit(str(value))
            scale_le.setValidator(QDoubleValidator())

            enabled_cb = QCheckBox()
            enabled_cb.setChecked(bool(clip.get("enabled", False)))

            min_le = QLineEdit(str(clip.get("min", -1.0)))
            min_le.setValidator(QDoubleValidator())
            max_le = QLineEdit(str(clip.get("max", 1.0)))
            max_le.setValidator(QDoubleValidator())

            grid.addWidget(QLabel(f"action[{i}]"), i + 1, 0)
            grid.addWidget(scale_le, i + 1, 1)
            grid.addWidget(enabled_cb, i + 1, 2)
            grid.addWidget(min_le, i + 1, 3)
            grid.addWidget(max_le, i + 1, 4)

            self.scale_fields.append(scale_le)
            self.clip_enabled_fields.append(enabled_cb)
            self.clip_min_fields.append(min_le)
            self.clip_max_fields.append(max_le)

        scroll.setWidget(inner)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        self.setMaximumHeight(600)

    def get_settings(self):
        scales = []
        clippings = []
        for i, le in enumerate(self.scale_fields):
            txt = le.text().strip()
            scales.append(float(txt) if txt else 0.0)

            min_txt = self.clip_min_fields[i].text().strip()
            max_txt = self.clip_max_fields[i].text().strip()
            min_value = float(min_txt) if min_txt else 0.0
            max_value = float(max_txt) if max_txt else 0.0
            if min_value > max_value:
                min_value, max_value = max_value, min_value
            clippings.append({
                "enabled": self.clip_enabled_fields[i].isChecked(),
                "min": min_value,
                "max": max_value,
            })
        return scales, clippings
