from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ActuatorSettingsDialog(QDialog):
    def __init__(self, actuator_settings, parent):
        super().__init__(parent)
        defaults = {
            "hip_mode": "pd",
            "hip_net_path": "",
            "shoulder_mode": "actuator_net",
            "shoulder_net_path": "",
            "leg_mode": "actuator_net",
            "leg_net_path": "",
            "wheel_mode": "pd",
            "wheel_net_path": "",
        }
        incoming = actuator_settings if isinstance(actuator_settings, dict) else {}
        if "mode" in incoming:
            global_mode = str(incoming.get("mode", "pd")).strip()
            incoming.setdefault("hip_mode", global_mode)
            incoming.setdefault("shoulder_mode", global_mode)
            incoming.setdefault("leg_mode", global_mode)
            incoming.setdefault("wheel_mode", global_mode)
        self.actuator_settings = {**defaults, **incoming}
        self.setWindowTitle("Actuator Settings")
        self._setup_ui()
        self._load_existing_settings()
        self._apply_mode_enabled_states()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.hip_mode_cb = QComboBox()
        self.hip_mode_cb.addItems(["pd", "actuator_net"])
        self.hip_mode_cb.currentTextChanged.connect(lambda _: self._apply_mode_enabled_states())
        form_layout.addRow("Hip Mode:", self.hip_mode_cb)
        self.hip_net_path_le = QLineEdit()
        form_layout.addRow("Hip Net:", self._make_path_row(self.hip_net_path_le, self._browse_hip_net))

        self.shoulder_mode_cb = QComboBox()
        self.shoulder_mode_cb.addItems(["pd", "actuator_net"])
        self.shoulder_mode_cb.currentTextChanged.connect(lambda _: self._apply_mode_enabled_states())
        form_layout.addRow("Shoulder Mode:", self.shoulder_mode_cb)
        self.shoulder_net_path_le = QLineEdit()
        form_layout.addRow(
            "Shoulder Net:",
            self._make_path_row(self.shoulder_net_path_le, self._browse_shoulder_net),
        )

        self.leg_mode_cb = QComboBox()
        self.leg_mode_cb.addItems(["pd", "actuator_net"])
        self.leg_mode_cb.currentTextChanged.connect(lambda _: self._apply_mode_enabled_states())
        form_layout.addRow("Leg Mode:", self.leg_mode_cb)
        self.leg_net_path_le = QLineEdit()
        form_layout.addRow("Leg Net:", self._make_path_row(self.leg_net_path_le, self._browse_leg_net))

        self.wheel_mode_cb = QComboBox()
        self.wheel_mode_cb.addItems(["pd", "actuator_net"])
        self.wheel_mode_cb.currentTextChanged.connect(lambda _: self._apply_mode_enabled_states())
        form_layout.addRow("Wheel Mode:", self.wheel_mode_cb)
        self.wheel_net_path_le = QLineEdit()
        form_layout.addRow(
            "Wheel Net:",
            self._make_path_row(self.wheel_net_path_le, self._browse_wheel_net),
        )

        main_layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        
    def _make_path_row(self, line_edit, browse_handler):
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(browse_handler)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, 1)
        row.addWidget(browse_btn)
        widget = QWidget()
        widget.setLayout(row)
        return widget

    def _load_existing_settings(self):
        self.hip_mode_cb.setCurrentText(self._normalize_mode(self.actuator_settings.get("hip_mode", "pd")))
        self.shoulder_mode_cb.setCurrentText(self._normalize_mode(self.actuator_settings.get("shoulder_mode", "actuator_net")))
        self.leg_mode_cb.setCurrentText(self._normalize_mode(self.actuator_settings.get("leg_mode", "actuator_net")))
        self.wheel_mode_cb.setCurrentText(self._normalize_mode(self.actuator_settings.get("wheel_mode", "pd")))
        self.hip_net_path_le.setText(str(self.actuator_settings.get("hip_net_path", "")))
        self.shoulder_net_path_le.setText(str(self.actuator_settings.get("shoulder_net_path", "")))
        self.leg_net_path_le.setText(str(self.actuator_settings.get("leg_net_path", "")))
        self.wheel_net_path_le.setText(str(self.actuator_settings.get("wheel_net_path", "")))

    @staticmethod
    def _normalize_mode(mode_text):
        mode = str(mode_text).strip().lower()
        return "actuator_net" if mode == "actuator_net" else "pd"

    def _apply_mode_enabled_states(self):
        self.hip_net_path_le.setEnabled(self.hip_mode_cb.currentText() == "actuator_net")
        self.shoulder_net_path_le.setEnabled(self.shoulder_mode_cb.currentText() == "actuator_net")
        self.leg_net_path_le.setEnabled(self.leg_mode_cb.currentText() == "actuator_net")
        self.wheel_net_path_le.setEnabled(self.wheel_mode_cb.currentText() == "actuator_net")

    def _browse_hip_net(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Hip Net TorchScript File",
            "",
            "TorchScript Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.hip_net_path_le.setText(file_path)

    def _browse_shoulder_net(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Shoulder Net TorchScript File",
            "",
            "TorchScript Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.shoulder_net_path_le.setText(file_path)

    def _browse_leg_net(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Leg Net TorchScript File",
            "",
            "TorchScript Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.leg_net_path_le.setText(file_path)

    def _browse_wheel_net(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Wheel Net TorchScript File",
            "",
            "TorchScript Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.wheel_net_path_le.setText(file_path)

    def get_settings(self):
        return {
            "hip_mode": self.hip_mode_cb.currentText().strip(),
            "hip_net_path": self.hip_net_path_le.text().strip(),
            "shoulder_mode": self.shoulder_mode_cb.currentText().strip(),
            "shoulder_net_path": self.shoulder_net_path_le.text().strip(),
            "leg_mode": self.leg_mode_cb.currentText().strip(),
            "leg_net_path": self.leg_net_path_le.text().strip(),
            "wheel_mode": self.wheel_mode_cb.currentText().strip(),
            "wheel_net_path": self.wheel_net_path_le.text().strip(),
        }
