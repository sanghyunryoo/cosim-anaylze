from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QScrollArea,
    QWidget,
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt


class ActionScaleSettingsDialog(QDialog):
    def __init__(self, action_scales, parent):
        super().__init__(parent)
        self.action_scales = list(action_scales) if isinstance(action_scales, (list, tuple)) else []
        self.setWindowTitle("Action Scale Settings")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        form_layout = QFormLayout(inner)
        self.fields = []
        for i, value in enumerate(self.action_scales):
            le = QLineEdit(str(value))
            le.setValidator(QDoubleValidator())
            form_layout.addRow(QLabel(f"action_scale[{i}]"), le)
            self.fields.append(le)

        scroll.setWidget(inner)
        main_layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        self.setMaximumHeight(600)

    def get_settings(self):
        out = []
        for le in self.fields:
            txt = le.text().strip()
            out.append(float(txt) if txt else 0.0)
        return out
