import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
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
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class HomingTrainDialog(QDialog):
    collectRequested = pyqtSignal()
    trainRequested = pyqtSignal()
    rlTrainRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    finalPoseRequested = pyqtSignal()
    testTeacherRequested = pyqtSignal()
    testPolicyRequested = pyqtSignal()
    switchPolicyRequested = pyqtSignal()
    commandRangeRequested = pyqtSignal()
    envChanged = pyqtSignal(str, str)

    def __init__(self, env_ids, terrain_ids, parent=None):
        super().__init__(parent)
        self._running = False
        self.setModal(False)
        self.setWindowTitle("Homing Policy Training")
        self.resize(1080, 760)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        formula_label = QLabel(
            "Dataset policy: stand-drive rollout observations + homing teacher labels. Exported ONNX input matches the stand-drive obs shape."
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
        self._current_env = self.env_cb.currentText()
        self.env_cb.currentTextChanged.connect(self._on_env_changed)
        self.policy_le = QLineEdit()
        self.samples_le = QLineEdit("50000")
        self.rollout_steps_le = QLineEdit("1000")
        self.trajectory_seconds_le = QLineEdit("3.0")
        self.stand_warmup_steps_le = QLineEdit("200")
        self.balance_blend_le = QLineEdit("0.0")
        self.command_range_btn = QPushButton("Command Range Settings")
        self.command_range_btn.clicked.connect(self.commandRangeRequested.emit)
        self.command_range_summary_label = QLabel("")
        self.command_range_summary_label.setWordWrap(True)
        self.command_range_summary_label.setStyleSheet("color: #64748B;")
        self.seed_le = QLineEdit("42")
        self.final_pose_btn = QPushButton("Final Pose Settings")
        self.final_pose_btn.clicked.connect(self.finalPoseRequested.emit)
        self.final_pose_summary_label = QLabel("")
        self.final_pose_summary_label.setWordWrap(True)
        self.final_pose_summary_label.setStyleSheet("color: #64748B;")

        setup_layout.addRow("Robot:", self.env_cb)
        setup_layout.addRow("Stand Policy:", self._file_row(self.policy_le, save=False))
        setup_layout.addRow("Target:", self.final_pose_btn)
        setup_layout.addRow("", self.final_pose_summary_label)
        setup_layout.addRow("Samples:", self.samples_le)
        setup_layout.addRow("Steps/reset:", self.rollout_steps_le)
        setup_layout.addRow("Trajectory sec:", self.trajectory_seconds_le)
        setup_layout.addRow("Stand warmup:", self.stand_warmup_steps_le)
        setup_layout.addRow("Balance blend:", self.balance_blend_le)
        setup_layout.addRow("Command:", self.command_range_btn)
        setup_layout.addRow("", self.command_range_summary_label)
        setup_layout.addRow("Seed:", self.seed_le)

        train_group = QGroupBox("Datasets / Training")
        train_layout = QFormLayout(train_group)
        train_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(train_group, 2)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.epochs_le = QLineEdit("30")
        self.batch_size_le = QLineEdit("256")
        self.lr_le = QLineEdit("1e-3")
        self.val_ratio_le = QLineEdit("0.1")
        self.hidden_dim_le = QLineEdit("256")
        self.ppo_steps_le = QLineEdit("20000")
        self.ppo_envs_le = QLineEdit("4")
        self.ppo_rollout_le = QLineEdit("256")
        self.ppo_epochs_le = QLineEdit("4")
        self.ppo_lr_le = QLineEdit("3e-4")
        self.ppo_randomize_le = QLineEdit("0.3")
        self.reward_track_le = QLineEdit("6.0")
        self.reward_balance_le = QLineEdit("0.002")
        self.reward_upright_le = QLineEdit("2.0")
        self.reward_smooth_le = QLineEdit("0.04")
        self.reward_contact_le = QLineEdit("0.0005")
        self.checkpoint_le = QLineEdit()
        self.output_le = QLineEdit()
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)

        train_layout.addRow("Datasets:", self.dataset_list)
        train_layout.addRow("Epochs:", self.epochs_le)
        train_layout.addRow("Batch size:", self.batch_size_le)
        train_layout.addRow("Learning rate:", self.lr_le)
        train_layout.addRow("Val ratio:", self.val_ratio_le)
        train_layout.addRow("Hidden dim:", self.hidden_dim_le)
        train_layout.addRow("PPO steps:", self.ppo_steps_le)
        train_layout.addRow("PPO envs:", self.ppo_envs_le)
        train_layout.addRow("PPO rollout:", self.ppo_rollout_le)
        train_layout.addRow("PPO epochs:", self.ppo_epochs_le)
        train_layout.addRow("PPO lr:", self.ppo_lr_le)
        train_layout.addRow("Domain rand:", self.ppo_randomize_le)
        train_layout.addRow("R track:", self.reward_track_le)
        train_layout.addRow("R base accel:", self.reward_balance_le)
        train_layout.addRow("R upright:", self.reward_upright_le)
        train_layout.addRow("R smooth:", self.reward_smooth_le)
        train_layout.addRow("R contact:", self.reward_contact_le)
        train_layout.addRow("Checkpoint:", self._file_row(self.checkpoint_le, save=False, title="Select Homing Checkpoint", pattern="PyTorch (*.pt)"))
        train_layout.addRow("Export ONNX:", self._file_row(self.output_le, save=True, title="Export Homing ONNX", pattern="ONNX Files (*.onnx)"))
        train_layout.addRow("Status:", self.status_label)

        rl_group = QGroupBox("RL Strategy")
        rl_layout = QFormLayout(rl_group)
        rl_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(rl_group, 1)

        self.rl_supervised_init_cb = QCheckBox("Use Homing checkpoint")
        self.rl_supervised_init_cb.setChecked(True)
        self.rl_trajectory_reward_cb = QCheckBox("Use trajectory reward")
        self.rl_trajectory_reward_cb.setChecked(True)
        self.rl_wheel_mask_cb = QCheckBox("Mask wheel actions")
        self.rl_wheel_mask_cb.setChecked(True)
        self.rl_preset_cb = QComboBox()
        self.rl_preset_cb.addItem("Light: checkpoint + trajectory", "light")
        self.rl_preset_cb.addItem("Pro: stand-drive pure RL", "pro")
        self.rl_preset_cb.currentIndexChanged.connect(self._apply_rl_preset)
        rl_layout.addRow("Preset:", self.rl_preset_cb)
        rl_layout.addRow("Init:", self.rl_supervised_init_cb)
        rl_layout.addRow("Reward:", self.rl_trajectory_reward_cb)
        rl_layout.addRow("Safety:", self.rl_wheel_mask_cb)

        log_group = QGroupBox("Homing Log")
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
        self.train_btn = QPushButton("Train Policy")
        self.rl_train_btn = QPushButton("RL Fine Tune")
        self.test_teacher_btn = QPushButton("Test Teacher")
        self.test_policy_btn = QPushButton("Test Export")
        self.switch_policy_btn = QPushButton("Switch Policy")
        self.export_btn = QPushButton("Export ONNX")
        self.refresh_btn = QPushButton("Refresh")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.switch_policy_btn.setEnabled(False)
        self.close_btns = QDialogButtonBox(QDialogButtonBox.Close)
        action_row.addWidget(self.collect_btn)
        action_row.addWidget(self.train_btn)
        action_row.addWidget(self.rl_train_btn)
        action_row.addWidget(self.test_teacher_btn)
        action_row.addWidget(self.test_policy_btn)
        action_row.addWidget(self.switch_policy_btn)
        action_row.addWidget(self.export_btn)
        action_row.addWidget(self.refresh_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addStretch()
        action_row.addWidget(self.close_btns)
        layout.addLayout(action_row)

        self.collect_btn.clicked.connect(self.collectRequested.emit)
        self.train_btn.clicked.connect(self.trainRequested.emit)
        self.rl_train_btn.clicked.connect(self.rlTrainRequested.emit)
        self.test_teacher_btn.clicked.connect(self.testTeacherRequested.emit)
        self.test_policy_btn.clicked.connect(self.testPolicyRequested.emit)
        self.switch_policy_btn.clicked.connect(self.switchPolicyRequested.emit)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.close_btns.rejected.connect(self.close)
        self.close_btns.accepted.connect(self.close)

    def _file_row(self, line_edit, save=False, title=None, pattern=None):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse(line_edit, save, title, pattern))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return row

    def _browse(self, line_edit, save, title=None, pattern=None):
        current = line_edit.text().strip()
        start_dir = os.path.dirname(current) if current else os.getcwd()
        title = title or ("Export File" if save else "Select File")
        pattern = pattern or "All Files (*)"
        if save:
            path, _ = QFileDialog.getSaveFileName(self, title, start_dir, pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, start_dir, pattern)
        if path:
            line_edit.setText(path)

    def set_available_datasets(self, datasets, selected_paths=None):
        selected_paths = set(selected_paths or [])
        self.dataset_list.clear()
        for entry in datasets:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.UserRole, entry["path"])
            item.setSelected(entry["path"] in selected_paths)
            self.dataset_list.addItem(item)

    def get_selected_dataset_paths(self):
        return [item.data(Qt.UserRole) for item in self.dataset_list.selectedItems()]

    def _on_env_changed(self, env_id):
        previous = self._current_env
        current = str(env_id)
        self._current_env = current
        if previous != current:
            self.envChanged.emit(previous, current)

    def keyPressEvent(self, event):
        parent = self.parent()
        if parent is not None and hasattr(parent, "handle_key_press"):
            parent.handle_key_press(event)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        parent = self.parent()
        if parent is not None and hasattr(parent, "handle_key_release"):
            parent.handle_key_release(event)
            return
        super().keyReleaseEvent(event)

    def get_settings(self):
        return {
            "env_id": self.env_cb.currentText(),
            "policy_path": self.policy_le.text().strip(),
            "terrains": ["flat"],
            "samples": self.samples_le.text().strip(),
            "rollout_steps": self.rollout_steps_le.text().strip(),
            "homing_trajectory_seconds": self.trajectory_seconds_le.text().strip(),
            "homing_stand_warmup_steps": self.stand_warmup_steps_le.text().strip(),
            "homing_balance_blend": self.balance_blend_le.text().strip(),
            "command_min": str(getattr(self, "_command_min", "-1.0")),
            "command_max": str(getattr(self, "_command_max", "1.0")),
            "command_mins": str(getattr(self, "_command_mins", "")),
            "command_maxs": str(getattr(self, "_command_maxs", "")),
            "seed": self.seed_le.text().strip(),
            "epochs": self.epochs_le.text().strip(),
            "batch_size": self.batch_size_le.text().strip(),
            "learning_rate": self.lr_le.text().strip(),
            "val_ratio": self.val_ratio_le.text().strip(),
            "hidden_dim": self.hidden_dim_le.text().strip(),
            "ppo_total_steps": self.ppo_steps_le.text().strip(),
            "ppo_num_envs": self.ppo_envs_le.text().strip(),
            "ppo_rollout_steps": self.ppo_rollout_le.text().strip(),
            "ppo_epochs": self.ppo_epochs_le.text().strip(),
            "ppo_learning_rate": self.ppo_lr_le.text().strip(),
            "ppo_domain_randomize": self.ppo_randomize_le.text().strip(),
            "reward_track": self.reward_track_le.text().strip(),
            "reward_base_acc": self.reward_balance_le.text().strip(),
            "reward_upright": self.reward_upright_le.text().strip(),
            "reward_action_rate": self.reward_smooth_le.text().strip(),
            "reward_contact": self.reward_contact_le.text().strip(),
            "ppo_supervised_init": "1" if self.rl_supervised_init_cb.isChecked() else "0",
            "ppo_use_trajectory_reward": "1" if self.rl_trajectory_reward_cb.isChecked() else "0",
            "ppo_mask_wheel_actions": "1" if self.rl_wheel_mask_cb.isChecked() else "0",
            "ppo_strategy_preset": self.rl_preset_cb.currentData() or "light",
            "selected_datasets": self.get_selected_dataset_paths(),
            "checkpoint_path": self.checkpoint_le.text().strip(),
            "output_path": self.output_le.text().strip(),
        }

    def set_settings(self, settings):
        settings = dict(settings or {})
        if settings.get("env_id"):
            self.env_cb.blockSignals(True)
            self.env_cb.setCurrentText(str(settings.get("env_id")))
            self.env_cb.blockSignals(False)
            self._current_env = self.env_cb.currentText()
        self.policy_le.setText(str(settings.get("policy_path", "")))
        self.samples_le.setText(str(settings.get("samples", "50000")))
        self.rollout_steps_le.setText(str(settings.get("rollout_steps", "1000")))
        self.trajectory_seconds_le.setText(str(settings.get("homing_trajectory_seconds", "3.0")))
        self.stand_warmup_steps_le.setText(str(settings.get("homing_stand_warmup_steps", "200")))
        self.balance_blend_le.setText(str(settings.get("homing_balance_blend", "0.0")))
        self._command_min = str(settings.get("command_min", "-1.0"))
        self._command_max = str(settings.get("command_max", "1.0"))
        self._command_mins = str(settings.get("command_mins", ""))
        self._command_maxs = str(settings.get("command_maxs", ""))
        cmd_dim = len(self._command_mins.split(",")) if self._command_mins else 0
        self.command_range_summary_label.setText(f"command dims: {cmd_dim} | per-command min/max")
        self.seed_le.setText(str(settings.get("seed", "42")))
        self.epochs_le.setText(str(settings.get("epochs", "30")))
        self.batch_size_le.setText(str(settings.get("batch_size", "256")))
        self.lr_le.setText(str(settings.get("learning_rate", "1e-3")))
        self.val_ratio_le.setText(str(settings.get("val_ratio", "0.1")))
        self.hidden_dim_le.setText(str(settings.get("hidden_dim", "256")))
        self.ppo_steps_le.setText(str(settings.get("ppo_total_steps", "20000")))
        self.ppo_envs_le.setText(str(settings.get("ppo_num_envs", "4")))
        self.ppo_rollout_le.setText(str(settings.get("ppo_rollout_steps", "256")))
        self.ppo_epochs_le.setText(str(settings.get("ppo_epochs", "4")))
        self.ppo_lr_le.setText(str(settings.get("ppo_learning_rate", "3e-4")))
        self.ppo_randomize_le.setText(str(settings.get("ppo_domain_randomize", "0.3")))
        self.reward_track_le.setText(str(settings.get("reward_track", "6.0")))
        self.reward_balance_le.setText(str(settings.get("reward_base_acc", "0.002")))
        self.reward_upright_le.setText(str(settings.get("reward_upright", "2.0")))
        self.reward_smooth_le.setText(str(settings.get("reward_action_rate", "0.04")))
        self.reward_contact_le.setText(str(settings.get("reward_contact", "0.0005")))
        self.rl_supervised_init_cb.setChecked(str(settings.get("ppo_supervised_init", "1")).strip().lower() not in ("0", "false", "no", "off"))
        self.rl_trajectory_reward_cb.setChecked(str(settings.get("ppo_use_trajectory_reward", "1")).strip().lower() not in ("0", "false", "no", "off"))
        self.rl_wheel_mask_cb.setChecked(str(settings.get("ppo_mask_wheel_actions", "1")).strip().lower() not in ("0", "false", "no", "off"))
        preset = str(settings.get("ppo_strategy_preset", "light"))
        preset_index = self.rl_preset_cb.findData(preset)
        if preset_index >= 0:
            self.rl_preset_cb.blockSignals(True)
            self.rl_preset_cb.setCurrentIndex(preset_index)
            self.rl_preset_cb.blockSignals(False)
        self.checkpoint_le.setText(str(settings.get("checkpoint_path", "")))
        self.output_le.setText(str(settings.get("output_path", "")))
        joint_count = len(str(settings.get("final_pos", "")).split(",")) if settings.get("final_pos", "") else 0
        same = str(settings.get("final_pose_same", "1")).strip().lower() not in ("0", "false", "no", "off")
        mode = "same" if same else "priority"
        self.final_pose_summary_label.setText(f"flat terrain only | target joints: {joint_count} | move: {mode}")

    def _apply_rl_preset(self):
        preset = self.rl_preset_cb.currentData()
        if preset == "pro":
            self.rl_supervised_init_cb.setChecked(False)
            self.rl_trajectory_reward_cb.setChecked(False)
            self.rl_wheel_mask_cb.setChecked(True)
        else:
            self.rl_supervised_init_cb.setChecked(True)
            self.rl_trajectory_reward_cb.setChecked(True)
            self.rl_wheel_mask_cb.setChecked(True)

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
            self.collect_btn,
            self.train_btn,
            self.rl_train_btn,
            self.test_teacher_btn,
            self.test_policy_btn,
            self.command_range_btn,
            self.rl_preset_cb,
            self.rl_supervised_init_cb,
            self.rl_trajectory_reward_cb,
            self.rl_wheel_mask_cb,
            self.export_btn,
            self.refresh_btn,
            self.close_btns,
        ):
            widget.setEnabled(not self._running)
        self.switch_policy_btn.setEnabled(self._running)
