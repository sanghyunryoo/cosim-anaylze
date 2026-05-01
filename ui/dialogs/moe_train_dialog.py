import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MoETrainDialog(QDialog):
    collectRequested = pyqtSignal()
    trainRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    stopRequested = pyqtSignal()

    def __init__(self, env_ids, terrain_ids, parent=None):
        super().__init__(parent)
        self._running = False
        self._available_by_path = {}

        self.setModal(False)
        self.setWindowTitle("MoE Training")
        self.resize(1040, 760)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        formula_label = QLabel(
            "Formula: action = (1 - alpha) * Policy A(obs) + alpha * Policy B(obs)    |    alpha = Gate(obs)"
        )
        formula_label.setWordWrap(True)
        formula_label.setStyleSheet("font-weight: 600; color: #1f2937;")
        layout.addWidget(formula_label)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        layout.addLayout(top_row)

        setup_group = QGroupBox("Collection Setup")
        setup_layout = QFormLayout(setup_group)
        setup_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(setup_group, 2)

        self.env_cb = QComboBox()
        self.env_cb.addItems(list(env_ids or []))
        setup_layout.addRow("Robot:", self.env_cb)

        self.policy_a_le = QLineEdit()
        self.policy_b_le = QLineEdit()
        setup_layout.addRow("Policy A:", self._file_row(self.policy_a_le))
        setup_layout.addRow("Policy B:", self._file_row(self.policy_b_le))

        self.samples_le = QLineEdit("200000")
        self.rollout_steps_le = QLineEdit("1000")
        self.boundary_le = QLineEdit("8.0")
        self.command_min_le = QLineEdit("-1.0")
        self.command_max_le = QLineEdit("1.0")
        self.seed_le = QLineEdit("42")
        self.lambda_smooth_le = QLineEdit("0")
        self.cmd_alpha_penalty_le = QLineEdit("0")
        self.cmd_label_threshold_le = QLineEdit("0.2")
        self.cmd_label_alpha_le = QLineEdit("0")
        setup_layout.addRow("Samples:", self.samples_le)
        setup_layout.addRow("Steps/reset:", self.rollout_steps_le)
        setup_layout.addRow("Boundary m:", self.boundary_le)
        setup_layout.addRow("Cmd min:", self.command_min_le)
        setup_layout.addRow("Cmd max:", self.command_max_le)
        setup_layout.addRow("Seed:", self.seed_le)
        setup_layout.addRow("Smoothness:", self.lambda_smooth_le)
        setup_layout.addRow("Cmd Alpha Penalty:", self.cmd_alpha_penalty_le)
        setup_layout.addRow("Cmd Label Thresh:", self.cmd_label_threshold_le)
        setup_layout.addRow("Cmd Flat Alpha:", self.cmd_label_alpha_le)
        penalty_hint = QLabel("Flat labels only: keeps alpha low when command[1]/command[2] are nonzero.")
        penalty_hint.setWordWrap(True)
        penalty_hint.setStyleSheet("color: #64748B;")
        setup_layout.addRow("", penalty_hint)
        label_hint = QLabel("If flat and |cmd[1]| or |cmd[2]| exceeds threshold, alpha_label is capped to Cmd Flat Alpha.")
        label_hint.setWordWrap(True)
        label_hint.setStyleSheet("color: #64748B;")
        setup_layout.addRow("", label_hint)

        terrain_group = QGroupBox("Terrains")
        terrain_layout = QVBoxLayout(terrain_group)
        self.terrain_list = QListWidget()
        self.terrain_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for terrain in terrain_ids:
            item = QListWidgetItem(terrain)
            item.setSelected(terrain in {"flat", "rocky_easy", "rocky_hard", "stairs_up_easy", "stairs_up_normal", "stairs_up_hard"})
            self.terrain_list.addItem(item)
        terrain_layout.addWidget(self.terrain_list)
        top_row.addWidget(terrain_group, 1)

        data_group = QGroupBox("Datasets / Training")
        data_layout = QFormLayout(data_group)
        data_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(data_group, 2)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        data_layout.addRow("Datasets:", self.dataset_list)

        self.epochs_le = QLineEdit("30")
        self.batch_size_le = QLineEdit("256")
        self.lr_le = QLineEdit("1e-3")
        self.val_ratio_le = QLineEdit("0.1")
        data_layout.addRow("Epochs:", self.epochs_le)
        data_layout.addRow("Batch size:", self.batch_size_le)
        data_layout.addRow("Learning rate:", self.lr_le)
        data_layout.addRow("Val ratio:", self.val_ratio_le)

        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)
        data_layout.addRow("Status:", self.status_label)

        log_group = QGroupBox("MoE Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.document().setMaximumBlockCount(10000)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setStyleHint(QFont.Monospace)
        self.log_output.setFont(fixed_font)
        self.log_output.setStyleSheet("QTextEdit { background-color: #000000; color: #f5f5f5; border: 1px solid #333333; }")
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group, 1)

        action_row = QHBoxLayout()
        self.collect_btn = QPushButton("Collect Data")
        self.train_btn = QPushButton("Train Gate")
        self.export_btn = QPushButton("Export ONNX")
        self.refresh_btn = QPushButton("Refresh")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.close_btns = QDialogButtonBox(QDialogButtonBox.Close)
        action_row.addWidget(self.collect_btn)
        action_row.addWidget(self.train_btn)
        action_row.addWidget(self.export_btn)
        action_row.addWidget(self.refresh_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addStretch()
        action_row.addWidget(self.close_btns)
        layout.addLayout(action_row)

        self.collect_btn.clicked.connect(self.collectRequested.emit)
        self.train_btn.clicked.connect(self.trainRequested.emit)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.close_btns.rejected.connect(self.close)
        self.close_btns.accepted.connect(self.close)

    def _file_row(self, line_edit):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_policy(line_edit))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return row

    def _browse_policy(self, line_edit):
        start_dir = os.path.dirname(line_edit.text().strip()) if line_edit.text().strip() else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Policy", start_dir, "ONNX Files (*.onnx)")
        if path:
            line_edit.setText(path)

    def get_selected_terrains(self):
        return [item.text() for item in self.terrain_list.selectedItems()]

    def set_available_datasets(self, datasets, selected_paths=None):
        self._available_by_path = {entry["path"]: dict(entry) for entry in datasets}
        selected_paths = set(selected_paths or [])
        self.dataset_list.clear()
        for entry in datasets:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.UserRole, entry["path"])
            item.setSelected(entry["path"] in selected_paths)
            self.dataset_list.addItem(item)

    def get_selected_dataset_paths(self):
        return [item.data(Qt.UserRole) for item in self.dataset_list.selectedItems()]

    def get_settings(self):
        return {
            "env_id": self.env_cb.currentText(),
            "policy_a_path": self.policy_a_le.text().strip(),
            "policy_b_path": self.policy_b_le.text().strip(),
            "terrains": self.get_selected_terrains(),
            "samples": self.samples_le.text().strip(),
            "rollout_steps": self.rollout_steps_le.text().strip(),
            "boundary_m": self.boundary_le.text().strip(),
            "command_min": self.command_min_le.text().strip(),
            "command_max": self.command_max_le.text().strip(),
            "seed": self.seed_le.text().strip(),
            "epochs": self.epochs_le.text().strip(),
            "batch_size": self.batch_size_le.text().strip(),
            "learning_rate": self.lr_le.text().strip(),
            "lambda_smooth": self.lambda_smooth_le.text().strip(),
            "cmd_alpha_penalty": self.cmd_alpha_penalty_le.text().strip(),
            "cmd_label_threshold": self.cmd_label_threshold_le.text().strip(),
            "cmd_label_alpha": self.cmd_label_alpha_le.text().strip(),
            "val_ratio": self.val_ratio_le.text().strip(),
            "selected_datasets": self.get_selected_dataset_paths(),
        }

    def set_settings(self, settings):
        settings = dict(settings or {})
        if settings.get("env_id"):
            self.env_cb.setCurrentText(str(settings.get("env_id")))
        self.policy_a_le.setText(str(settings.get("policy_a_path", "")))
        self.policy_b_le.setText(str(settings.get("policy_b_path", "")))
        self.samples_le.setText(str(settings.get("samples", "200000")))
        self.rollout_steps_le.setText(str(settings.get("rollout_steps", "1000")))
        self.boundary_le.setText(str(settings.get("boundary_m", "8.0")))
        self.command_min_le.setText(str(settings.get("command_min", "-1.0")))
        self.command_max_le.setText(str(settings.get("command_max", "1.0")))
        self.seed_le.setText(str(settings.get("seed", "42")))
        self.epochs_le.setText(str(settings.get("epochs", "30")))
        self.batch_size_le.setText(str(settings.get("batch_size", "256")))
        self.lr_le.setText(str(settings.get("learning_rate", "1e-3")))
        self.lambda_smooth_le.setText(str(settings.get("lambda_smooth", "0")))
        self.cmd_alpha_penalty_le.setText(str(settings.get("cmd_alpha_penalty", "0")))
        self.cmd_label_threshold_le.setText(str(settings.get("cmd_label_threshold", "0.2")))
        self.cmd_label_alpha_le.setText(str(settings.get("cmd_label_alpha", "0")))
        self.val_ratio_le.setText(str(settings.get("val_ratio", "0.1")))
        terrain_set = set(settings.get("terrains", []))
        if terrain_set:
            for i in range(self.terrain_list.count()):
                item = self.terrain_list.item(i)
                item.setSelected(item.text() in terrain_set)

    def set_status(self, text):
        self.status_label.setText(str(text))

    def clear_log(self):
        self.log_output.clear()

    def append_log(self, message):
        if not message:
            return
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.insertPlainText(str(message))
        self.log_output.ensureCursorVisible()

    def set_running(self, running):
        self._running = bool(running)
        self.stop_btn.setEnabled(self._running)
        for widget in (
            self.collect_btn, self.train_btn, self.export_btn, self.refresh_btn,
            self.env_cb, self.policy_a_le, self.policy_b_le, self.terrain_list,
            self.dataset_list, self.close_btns,
        ):
            widget.setEnabled(not self._running)

    def closeEvent(self, event):
        if self._running:
            QMessageBox.warning(self, "MoE Training", "MoE job is running. Wait for it to finish before closing this window.")
            event.ignore()
            return
        super().closeEvent(event)
