import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MoEManualDialog(QDialog):
    exportRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Manual MoE Export")
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        formula_label = QLabel(
            "Formula: alpha = last obs value, clipped to 0..1; physical = (1-alpha)*A(obs[:-1])*Scale A + alpha*B(obs[:-1])*Scale B; action = physical / Output scale"
        )
        formula_label.setWordWrap(True)
        formula_label.setStyleSheet("font-weight: 600; color: #1f2937;")
        layout.addWidget(formula_label)

        form = QFormLayout()
        layout.addLayout(form)

        self.policy_a_le = QLineEdit()
        self.policy_b_le = QLineEdit()
        self.policy_a_scales_le = QLineEdit()
        self.policy_b_scales_le = QLineEdit()
        self.output_scales_le = QLineEdit()
        self.output_le = QLineEdit()

        form.addRow("Policy A:", self._file_row(self.policy_a_le, save=False))
        form.addRow("Policy B:", self._file_row(self.policy_b_le, save=False))
        form.addRow("Scale A:", self.policy_a_scales_le)
        form.addRow("Scale B:", self.policy_b_scales_le)
        form.addRow("Output scale:", self.output_scales_le)
        form.addRow("Output:", self._file_row(self.output_le, save=True))

        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)
        form.addRow("Status:", self.status_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        layout.addWidget(self.log_output, 1)

        action_row = QHBoxLayout()
        self.export_btn = QPushButton("Export ONNX")
        self.close_btns = QDialogButtonBox(QDialogButtonBox.Close)
        action_row.addWidget(self.export_btn)
        action_row.addStretch()
        action_row.addWidget(self.close_btns)
        layout.addLayout(action_row)

        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.close_btns.rejected.connect(self.close)
        self.close_btns.accepted.connect(self.close)

    def _file_row(self, line_edit, save=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse(line_edit, save))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return row

    def _browse(self, line_edit, save):
        current = line_edit.text().strip()
        start_dir = os.path.dirname(current) if current else os.getcwd()
        if save:
            path, _ = QFileDialog.getSaveFileName(self, "Export Manual MoE ONNX", start_dir, "ONNX Files (*.onnx)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Policy", start_dir, "ONNX Files (*.onnx)")
        if path:
            line_edit.setText(path)

    def get_settings(self):
        return {
            "policy_a_path": self.policy_a_le.text().strip(),
            "policy_b_path": self.policy_b_le.text().strip(),
            "policy_a_action_scales": self.policy_a_scales_le.text().strip(),
            "policy_b_action_scales": self.policy_b_scales_le.text().strip(),
            "output_action_scales": self.output_scales_le.text().strip(),
            "output_path": self.output_le.text().strip(),
        }

    def set_settings(self, settings):
        settings = dict(settings or {})
        self.policy_a_le.setText(str(settings.get("policy_a_path", "")))
        self.policy_b_le.setText(str(settings.get("policy_b_path", "")))
        self.policy_a_scales_le.setText(str(settings.get("policy_a_action_scales", "")))
        self.policy_b_scales_le.setText(str(settings.get("policy_b_action_scales", "")))
        self.output_scales_le.setText(str(settings.get("output_action_scales", "")))
        self.output_le.setText(str(settings.get("output_path", "")))

    def set_status(self, text):
        self.status_label.setText(str(text))

    def set_running(self, running):
        self.export_btn.setEnabled(not bool(running))

    def append_log(self, message):
        if message:
            self.log_output.append(str(message).rstrip("\n"))
