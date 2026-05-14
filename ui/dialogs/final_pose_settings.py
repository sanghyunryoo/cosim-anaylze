from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt


class FinalPoseSettingsDialog(QDialog):
    def __init__(self, final_pose_settings, parent):
        super().__init__(parent)
        self.final_pose_settings = {
            "joints": dict((final_pose_settings or {}).get("joints", {})),
            "velocities": dict((final_pose_settings or {}).get("velocities", {})),
        }
        self.setWindowTitle("Final Pose Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        description = QLabel("Set the homing target joint positions and velocities.")
        description.setWordWrap(True)
        main_layout.addWidget(description)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.pos_fields = {}
        self.vel_fields = {}

        joints = self.final_pose_settings.get("joints", {})
        velocities = self.final_pose_settings.get("velocities", {})
        validator = QDoubleValidator()
        for joint_name, value in joints.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            pos_le = QLineEdit(str(value))
            vel_le = QLineEdit(str(velocities.get(joint_name, 0.0)))
            pos_le.setValidator(validator)
            vel_le.setValidator(validator)
            row_layout.addWidget(QLabel("pos"))
            row_layout.addWidget(pos_le)
            row_layout.addWidget(QLabel("vel"))
            row_layout.addWidget(vel_le)
            form_layout.addRow(QLabel(joint_name), row)
            self.pos_fields[joint_name] = pos_le
            self.vel_fields[joint_name] = vel_le

        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.resize(620, 680)

    def get_settings(self):
        return {
            "joints": {joint_name: field.text() for joint_name, field in self.pos_fields.items()},
            "velocities": {joint_name: field.text() for joint_name, field in self.vel_fields.items()},
        }
