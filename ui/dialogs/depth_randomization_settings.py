from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class DepthRandomizationSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = dict(settings or {})
        self.setWindowTitle("Depth Randomization Settings")
        self._setup_ui()

    def _add_numeric_row(self, layout, label_text, key, validator, suffix_text=None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        line_edit = QLineEdit(str(self.settings.get(key, "")))
        line_edit.setValidator(validator)
        line_edit.setFixedWidth(84)
        row_layout.addWidget(line_edit)
        if suffix_text:
            suffix = QLabel(suffix_text)
            suffix.setStyleSheet("color: #64748B;")
            row_layout.addWidget(suffix)
        row_layout.addStretch(1)
        layout.addRow(label_text, row)
        return line_edit

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        description = QLabel(
            "Configure paper-style depth randomization for dataset generation. "
            "These settings are stored per environment and included in the run config."
        )
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.enable_cb = QCheckBox("Enable depth randomization settings")
        self.enable_cb.setChecked(bool(self.settings.get("enabled", False)))
        main_layout.addWidget(self.enable_cb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(10)

        self.fields = {}

        pose_group = QGroupBox("Camera Pose / FOV")
        pose_layout = QFormLayout(pose_group)
        pose_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["camera_xyz_shift_m"] = self._add_numeric_row(
            pose_layout, "XYZ shift:", "camera_xyz_shift_m", QDoubleValidator(0.0, 1.0, 6), "m"
        )
        self.fields["camera_pitch_shift_deg"] = self._add_numeric_row(
            pose_layout, "Pitch shift:", "camera_pitch_shift_deg", QDoubleValidator(0.0, 180.0, 6), "deg"
        )
        self.fields["camera_fov_shift_deg"] = self._add_numeric_row(
            pose_layout, "FOV shift:", "camera_fov_shift_deg", QDoubleValidator(0.0, 180.0, 6), "deg"
        )
        inner_layout.addWidget(pose_group)

        gaussian_group = QGroupBox("Gaussian Noise")
        gaussian_layout = QFormLayout(gaussian_group)
        gaussian_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["gaussian_prob"] = self._add_numeric_row(
            gaussian_layout, "Probability:", "gaussian_prob", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["gaussian_stddev"] = self._add_numeric_row(
            gaussian_layout, "Stddev:", "gaussian_stddev", QDoubleValidator(0.0, 1.0, 6)
        )
        inner_layout.addWidget(gaussian_group)

        rotation_group = QGroupBox("Image Rotation")
        rotation_layout = QFormLayout(rotation_group)
        rotation_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["rotation_prob"] = self._add_numeric_row(
            rotation_layout, "Probability:", "rotation_prob", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["rotation_deg"] = self._add_numeric_row(
            rotation_layout, "Max rotation:", "rotation_deg", QDoubleValidator(0.0, 180.0, 6), "deg"
        )
        inner_layout.addWidget(rotation_group)

        edge_group = QGroupBox("Edge Noise")
        edge_layout = QFormLayout(edge_group)
        edge_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["edge_noise_prob"] = self._add_numeric_row(
            edge_layout, "Probability:", "edge_noise_prob", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["edge_noise_ratio"] = self._add_numeric_row(
            edge_layout, "Edge ratio:", "edge_noise_ratio", QDoubleValidator(0.0, 1.0, 6)
        )
        inner_layout.addWidget(edge_group)

        object_group = QGroupBox("Small Objects")
        object_layout = QFormLayout(object_group)
        object_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["small_object_prob"] = self._add_numeric_row(
            object_layout, "Probability:", "small_object_prob", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["small_object_ratio"] = self._add_numeric_row(
            object_layout, "Area ratio:", "small_object_ratio", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["small_object_count"] = self._add_numeric_row(
            object_layout, "Max count:", "small_object_count", QIntValidator(0, 999)
        )
        inner_layout.addWidget(object_group)

        spot_group = QGroupBox("Spot Noise")
        spot_layout = QFormLayout(spot_group)
        spot_layout.setLabelAlignment(Qt.AlignRight)
        self.fields["spot_noise_prob"] = self._add_numeric_row(
            spot_layout, "Probability:", "spot_noise_prob", QDoubleValidator(0.0, 1.0, 6)
        )
        self.fields["spot_noise_ratio"] = self._add_numeric_row(
            spot_layout, "Coverage ratio:", "spot_noise_ratio", QDoubleValidator(0.0, 1.0, 6)
        )
        inner_layout.addWidget(spot_group)
        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.resize(520, 720)

    def get_settings(self):
        values = {"enabled": bool(self.enable_cb.isChecked())}
        for key, field in self.fields.items():
            values[key] = field.text().strip()
        return values
