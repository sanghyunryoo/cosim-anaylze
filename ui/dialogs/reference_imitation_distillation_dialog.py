import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ReferenceImitationDistillationDialog(QDialog):
    """One-step UI for reference-teacher trajectory ONNX distillation."""

    trainRequested = pyqtSignal()
    stopRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.setModal(False)
        self.setWindowTitle("Reference → Locomotion Policy Export")
        self.resize(760, 590)

        layout = QVBoxLayout(self)
        description = QLabel(
            "The selected 276-D Isaac reference teacher is distilled in MuJoCo into a standalone trajectory ONNX. "
            "In Observation Settings, add Reference Progress once under Non-Stacked Observation and set Command Dim to 0. "
            "During testing, the selected clip supplies that normalized progress; the student is fixed at 90-D locomotion + 1-D phase = 91-D."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-weight: 600; color: #1f2937;")
        layout.addWidget(description)

        contract = QLabel(
            "Student phase input: [reference_progress] last in Non-Stacked (fixed 50 Hz, scale = 1); command is excluded."
        )
        contract.setWordWrap(True)
        contract.setStyleSheet("color: #2563EB;")
        layout.addWidget(contract)

        setup_group = QGroupBox("Export Setup")
        setup = QFormLayout(setup_group)
        setup.setLabelAlignment(Qt.AlignRight)
        self.motion_le = QLineEdit()
        self.motion_le.setReadOnly(True)
        self.teacher_le = QLineEdit()
        self.output_le = QLineEdit()
        # These defaults are the first configuration that passed the full
        # nominal + five reset-perturbation ONNX gate for Humanoid Light.
        self.samples_le = QLineEdit("6000")
        self.rounds_le = QLineEdit("6")
        self.epochs_le = QLineEdit("25")
        self.batch_le = QLineEdit("512")
        self.lr_le = QLineEdit("5e-4")
        self.hidden_le = QLineEdit("512")
        self.seed_le = QLineEdit("42")
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        setup.addRow("Reference motion:", self.motion_le)
        setup.addRow("Teacher ONNX (276 → 26):", self._file_row(self.teacher_le, save=False, pattern="ONNX Files (*.onnx)"))
        setup.addRow("Export student ONNX (91 → 26):", self._file_row(self.output_le, save=True, pattern="ONNX Files (*.onnx)"))
        setup.addRow("Samples / DAgger round:", self.samples_le)
        setup.addRow("DAgger rounds:", self.rounds_le)
        setup.addRow("Epochs / round:", self.epochs_le)
        setup.addRow("Batch size:", self.batch_le)
        setup.addRow("Learning rate:", self.lr_le)
        setup.addRow("Hidden width:", self.hidden_le)
        setup.addRow("Seed:", self.seed_le)
        setup.addRow("Status:", self.status_label)
        layout.addWidget(setup_group)

        log_group = QGroupBox("Distillation Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.document().setMaximumBlockCount(5000)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setStyleHint(QFont.Monospace)
        self.log_output.setFont(fixed_font)
        self.log_output.setStyleSheet("QTextEdit { background: #000; color: #f5f5f5; border: 1px solid #333; }")
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group, 1)

        actions = QHBoxLayout()
        self.train_btn = QPushButton("Train & Export Locomotion ONNX")
        self.stop_btn = QPushButton("Stop After Current Step")
        self.stop_btn.setEnabled(False)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        actions.addWidget(self.train_btn)
        actions.addWidget(self.stop_btn)
        actions.addStretch()
        actions.addWidget(close)
        layout.addLayout(actions)
        self.train_btn.clicked.connect(self.trainRequested.emit)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        close.rejected.connect(self.close)
        close.accepted.connect(self.close)

    def _file_row(self, line_edit, save: bool, pattern: str):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse(line_edit, save, pattern))
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse)
        return row

    def _browse(self, line_edit, save: bool, pattern: str):
        current = line_edit.text().strip()
        start_dir = os.path.dirname(current) if current else os.getcwd()
        if save:
            path, _ = QFileDialog.getSaveFileName(self, "Export Locomotion ONNX", start_dir, pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Reference Teacher ONNX", start_dir, pattern)
        if path:
            line_edit.setText(path)

    def set_context(self, motion_path: str, teacher_path: str, output_path: str):
        self.motion_le.setText(str(motion_path))
        if not self.teacher_le.text().strip() or not self._running:
            self.teacher_le.setText(str(teacher_path))
        if not self.output_le.text().strip() or not self._running:
            self.output_le.setText(str(output_path))

    def settings(self) -> dict:
        return {
            "teacher_policy_path": self.teacher_le.text().strip(),
            "output_path": self.output_le.text().strip(),
            "samples_per_round": self.samples_le.text().strip(),
            "dagger_rounds": self.rounds_le.text().strip(),
            "epochs_per_round": self.epochs_le.text().strip(),
            "batch_size": self.batch_le.text().strip(),
            "learning_rate": self.lr_le.text().strip(),
            "hidden_dim": self.hidden_le.text().strip(),
            "seed": self.seed_le.text().strip(),
        }

    def set_running(self, running: bool):
        self._running = bool(running)
        for widget in (
            self.teacher_le, self.output_le, self.samples_le, self.rounds_le,
            self.epochs_le, self.batch_le, self.lr_le, self.hidden_le, self.seed_le,
        ):
            widget.setEnabled(not self._running)
        self.train_btn.setEnabled(not self._running)
        self.stop_btn.setEnabled(self._running)
        if self._running:
            self.status_label.setText("Training in MuJoCo…")

    def set_status(self, status: str):
        self.status_label.setText(str(status))

    def append_log(self, text: str):
        text = str(text).rstrip("\n")
        if not text:
            return
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + "\n")
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()
