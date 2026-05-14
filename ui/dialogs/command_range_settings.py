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


class CommandRangeSettingsDialog(QDialog):
    def __init__(self, command_ranges, parent=None):
        super().__init__(parent)
        self.command_ranges = {
            "mins": list((command_ranges or {}).get("mins", [])),
            "maxs": list((command_ranges or {}).get("maxs", [])),
        }
        self.setWindowTitle("Command Range Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        description = QLabel("Set min/max sampling range for each command dimension.")
        description.setWordWrap(True)
        main_layout.addWidget(description)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.min_fields = []
        self.max_fields = []
        validator = QDoubleValidator()
        dim = max(len(self.command_ranges["mins"]), len(self.command_ranges["maxs"]))
        for i in range(dim):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            min_le = QLineEdit(str(self.command_ranges["mins"][i] if i < len(self.command_ranges["mins"]) else -1.0))
            max_le = QLineEdit(str(self.command_ranges["maxs"][i] if i < len(self.command_ranges["maxs"]) else 1.0))
            min_le.setValidator(validator)
            max_le.setValidator(validator)
            row_layout.addWidget(QLabel("min"))
            row_layout.addWidget(min_le)
            row_layout.addWidget(QLabel("max"))
            row_layout.addWidget(max_le)
            form_layout.addRow(QLabel(f"command_{i}"), row)
            self.min_fields.append(min_le)
            self.max_fields.append(max_le)

        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.resize(520, 520)

    def get_settings(self):
        return {
            "mins": [field.text() for field in self.min_fields],
            "maxs": [field.text() for field in self.max_fields],
        }
