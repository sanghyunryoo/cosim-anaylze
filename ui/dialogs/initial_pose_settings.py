from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QScrollArea, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt


class InitialPoseSettingsDialog(QDialog):
    def __init__(self, initial_pose_settings, parent):
        super().__init__(parent)
        self.initial_pose_settings = (initial_pose_settings or {}).copy()
        self.setWindowTitle("Initial Pose Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        description = QLabel("Set the initial joint positions applied before the test starts.")
        description.setWordWrap(True)
        main_layout.addWidget(description)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.fields = {}

        joints = self.initial_pose_settings.get("joints", {})
        for joint_name, value in joints.items():
            label = QLabel(joint_name)
            le = QLineEdit(str(value))
            le.setValidator(QDoubleValidator())
            form_layout.addRow(label, le)
            self.fields[joint_name] = le

        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.resize(520, 600)

    def get_settings(self):
        return {
            "joints": {joint_name: field.text() for joint_name, field in self.fields.items()}
        }
