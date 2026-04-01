from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QDoubleValidator


class FineTuneBiasEditorDialog(QDialog):
    biasChanged = pyqtSignal(list)

    def __init__(self, action_dim, values=None, parent=None):
        super().__init__(parent)
        self.action_dim = int(action_dim)
        self.setModal(False)
        self.setWindowTitle("Fine-tune Action Bias")
        self._fields = []

        layout = QVBoxLayout(self)
        form = QFormLayout()
        values = list(values or [])
        for idx in range(self.action_dim):
            line_edit = QLineEdit(str(values[idx]) if idx < len(values) else "0.0")
            validator = QDoubleValidator(self)
            validator.setDecimals(6)
            line_edit.setValidator(validator)
            line_edit.editingFinished.connect(self._emit_bias)
            self._fields.append(line_edit)
            form.addRow(f"action[{idx}]", line_edit)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        zero_btn = QPushButton("Zero All")
        zero_btn.clicked.connect(self._zero_all)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._emit_bias)
        button_row.addWidget(zero_btn)
        button_row.addWidget(apply_btn)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        layout.addWidget(buttons)

    def get_bias(self):
        values = []
        for field in self._fields:
            try:
                values.append(float(field.text()) if field.text().strip() else 0.0)
            except Exception:
                values.append(0.0)
        return values

    def set_bias(self, values):
        values = list(values or [])
        for idx, field in enumerate(self._fields):
            field.setText(str(values[idx]) if idx < len(values) else "0.0")

    def _zero_all(self):
        for field in self._fields:
            field.setText("0.0")
        self._emit_bias()

    def _emit_bias(self):
        self.biasChanged.emit(self.get_bias())
