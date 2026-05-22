from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
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
        same = (final_pose_settings or {}).get("same", True)
        if not isinstance(same, bool):
            same = str(same).strip().lower() not in ("0", "false", "no", "off")
        self.final_pose_settings = {
            "joints": dict((final_pose_settings or {}).get("joints", {})),
            "velocities": dict((final_pose_settings or {}).get("velocities", {})),
            "same": bool(same),
            "priorities": dict((final_pose_settings or {}).get("priorities", {})),
        }
        self._syncing_priority = False
        self.setWindowTitle("Final Pose Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        description = QLabel("Set the homing target joint positions, velocities, and movement order.")
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.same_cb = QCheckBox("Same")
        self.same_cb.setChecked(bool(self.final_pose_settings.get("same", True)))
        main_layout.addWidget(self.same_cb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.pos_fields = {}
        self.vel_fields = {}
        self.priority_fields = {}
        self.priority_groups = {}

        joints = self.final_pose_settings.get("joints", {})
        velocities = self.final_pose_settings.get("velocities", {})
        priorities = self.final_pose_settings.get("priorities", {})
        validator = QDoubleValidator()
        for index, (joint_name, value) in enumerate(joints.items(), start=1):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            pos_le = QLineEdit(str(value))
            vel_le = QLineEdit(str(velocities.get(joint_name, 0.0)))
            priority_le = QLineEdit(str(priorities.get(joint_name, index)))
            pos_le.setValidator(validator)
            vel_le.setValidator(validator)
            priority_le.setFixedWidth(52)
            row_layout.addWidget(QLabel("pos"))
            row_layout.addWidget(pos_le)
            row_layout.addWidget(QLabel("vel"))
            row_layout.addWidget(vel_le)
            row_layout.addWidget(QLabel("priority"))
            row_layout.addWidget(priority_le)
            form_layout.addRow(QLabel(joint_name), row)
            self.pos_fields[joint_name] = pos_le
            self.vel_fields[joint_name] = vel_le
            self.priority_fields[joint_name] = priority_le
            group_key = self._priority_group_key(joint_name)
            self.priority_groups.setdefault(group_key, []).append(joint_name)
            priority_le.textChanged.connect(
                lambda text, name=joint_name: self._sync_priority_group(name, text)
            )

        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)
        self.same_cb.toggled.connect(self._set_priority_enabled)
        self._set_priority_enabled(self.same_cb.isChecked())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.resize(620, 680)

    def get_settings(self):
        self._normalize_priority_groups()
        return {
            "joints": {joint_name: field.text() for joint_name, field in self.pos_fields.items()},
            "velocities": {joint_name: field.text() for joint_name, field in self.vel_fields.items()},
            "same": self.same_cb.isChecked(),
            "priorities": {joint_name: field.text() for joint_name, field in self.priority_fields.items()},
        }

    def _set_priority_enabled(self, same):
        for field in self.priority_fields.values():
            field.setEnabled(not bool(same))

    @staticmethod
    def _priority_group_key(joint_name):
        name = str(joint_name)
        for prefix in ("left_", "right_", "FL_", "FR_", "RL_", "RR_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _sync_priority_group(self, source_joint, text):
        if self._syncing_priority:
            return
        group_key = self._priority_group_key(source_joint)
        group = self.priority_groups.get(group_key, [])
        if len(group) <= 1:
            return
        self._syncing_priority = True
        try:
            for joint_name in group:
                if joint_name == source_joint:
                    continue
                field = self.priority_fields.get(joint_name)
                if field is not None and field.text() != text:
                    field.setText(text)
        finally:
            self._syncing_priority = False

    def _normalize_priority_groups(self):
        for group in self.priority_groups.values():
            if len(group) <= 1:
                continue
            value = ""
            for joint_name in group:
                value = self.priority_fields[joint_name].text()
                if value:
                    break
            for joint_name in group:
                if self.priority_fields[joint_name].text() != value:
                    self.priority_fields[joint_name].setText(value)
