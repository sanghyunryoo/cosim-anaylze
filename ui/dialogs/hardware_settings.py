from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QDialogButtonBox,
    QScrollArea, QWidget, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator


def _is_float_like(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


class HardwareSettingsDialog(QDialog):
    def __init__(self, hardware_settings, parent):
        super().__init__(parent)
        self.hardware_settings = (hardware_settings or {}).copy()
        self.setWindowTitle("Hardware Settings")
        self._setup_ui()

    def _setup_ui(self):
        # Main vertical layout for the dialog
        main_layout = QVBoxLayout(self)

        # Create a scroll area to allow scrolling when height exceeds limit
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Allow resizing of the scroll area content
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scrollbar

        # Create an inner widget that will hold the form layout
        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.fields = {}

        # Build editable fields for each hardware setting
        for key, value in self.hardware_settings.items():
            label = QLabel(key)
            if key == "actuator_mode":
                cb = QComboBox()
                cb.addItems(["torque", "position"])
                current = str(value).strip().lower()
                cb.setCurrentText(current if current in ("torque", "position") else "torque")
                form_layout.addRow(label, cb)
                self.fields[key] = cb
            elif isinstance(value, bool):
                cb = QCheckBox()
                cb.setChecked(value)
                form_layout.addRow(label, cb)
                self.fields[key] = cb
            else:
                le = QLineEdit(str(value))
                if _is_float_like(value):
                    le.setValidator(QDoubleValidator())  # Allow only floating-point input
                form_layout.addRow(label, le)
                self.fields[key] = le

        # Set the inner widget inside the scroll area
        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        # OK / Cancel buttons at the bottom
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)   # Close dialog with "Accepted" state
        buttons.rejected.connect(self.reject)   # Close dialog with "Rejected" state
        main_layout.addWidget(buttons)

        # Set maximum height to 600px; scrolling will be enabled if exceeded
        self.setMaximumHeight(600)

    def get_settings(self):
        # Return current field values as a dictionary
        settings = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QCheckBox):
                settings[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                settings[key] = widget.currentText()
            else:
                settings[key] = widget.text()
        return settings
