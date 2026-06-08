import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QComboBox,
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


class CtbcTrainDialog(QDialog):
    rlTrainRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    testPolicyRequested = pyqtSignal()
    testPrimitiveRequested = pyqtSignal()
    commandRangeRequested = pyqtSignal()
    envChanged = pyqtSignal(str, str)

    def __init__(self, env_ids, terrain_ids, parent=None):
        super().__init__(parent)
        self._running = False
        self._current_env = ""
        self.setModal(False)
        self.setWindowTitle("CTBC Stair Reflex Tuner")
        self.resize(900, 720)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        layout.addLayout(top_row)

        setup_group = QGroupBox("Policy / Scene")
        setup_layout = QFormLayout(setup_group)
        setup_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(setup_group, 2)

        self.env_cb = QComboBox()
        self.env_cb.addItems(list(env_ids or []))
        self._current_env = self.env_cb.currentText()
        self.env_cb.currentTextChanged.connect(self._on_env_changed)
        self.policy_le = QLineEdit()
        self.policy_le.setPlaceholderText("Base ONNX policy")
        self.output_le = QLineEdit()
        self.output_le.setPlaceholderText("Unused for controller tuning")
        self.checkpoint_le = QLineEdit()
        self.checkpoint_le.setPlaceholderText("Auto generated checkpoint")
        self.ctbc_terrain_cb = QComboBox()
        self.ctbc_terrain_cb.addItems([str(item) for item in (terrain_ids or [])])
        if self.ctbc_terrain_cb.findText("stairs_up_easy") >= 0:
            self.ctbc_terrain_cb.setCurrentText("stairs_up_easy")

        setup_layout.addRow("Robot:", self.env_cb)
        setup_layout.addRow("Base ONNX:", self._file_row(self.policy_le, save=False, title="Select Base ONNX", pattern="ONNX Files (*.onnx)"))
        setup_layout.addRow("Terrain:", self.ctbc_terrain_cb)

        reflex_group = QGroupBox("Stair Detection / Reflex Fit")
        reflex_layout = QFormLayout(reflex_group)
        reflex_layout.setLabelAlignment(Qt.AlignRight)
        top_row.addWidget(reflex_group, 1)

        self.ctbc_reflex_samples_le = QLineEdit("8192")
        self.ctbc_controller_candidates_le = QLineEdit("64")
        self.ctbc_reflex_epochs_le = QLineEdit("12")
        self.ctbc_reflex_lr_le = QLineEdit("3e-4")
        self.ctbc_reflex_flat_ratio_le = QLineEdit("0.35")
        self.ctbc_reflex_gain_le = QLineEdit("1.0")
        self.ctbc_reflex_segment_steps_le = QLineEdit("128")
        self.ctbc_stair_min_le = QLineEdit("0.025")
        self.ctbc_stair_max_le = QLineEdit("0.20")
        self.ctbc_episode_steps_le = QLineEdit("1200")

        reflex_layout.addRow("Candidates:", self.ctbc_controller_candidates_le)
        reflex_layout.addRow("Legacy samples:", self.ctbc_reflex_samples_le)
        reflex_layout.addRow("Epochs:", self.ctbc_reflex_epochs_le)
        reflex_layout.addRow("Learning rate:", self.ctbc_reflex_lr_le)
        reflex_layout.addRow("Flat ratio:", self.ctbc_reflex_flat_ratio_le)
        reflex_layout.addRow("Reflex gain:", self.ctbc_reflex_gain_le)
        reflex_layout.addRow("Segment steps:", self.ctbc_reflex_segment_steps_le)
        reflex_layout.addRow("Stair min m:", self.ctbc_stair_min_le)
        reflex_layout.addRow("Stair max m:", self.ctbc_stair_max_le)
        reflex_layout.addRow("Test steps:", self.ctbc_episode_steps_le)

        primitive_group = QGroupBox("Primitive Correction")
        primitive_layout = QFormLayout(primitive_group)
        primitive_layout.setLabelAlignment(Qt.AlignRight)
        layout.addWidget(primitive_group)

        self.ctbc_amplitude_le = QLineEdit("0.90")
        self.ctbc_period_le = QLineEdit("0.75")
        self.ctbc_cooldown_le = QLineEdit("0.35")
        self.ctbc_shoulder_gain_le = QLineEdit("0.50")
        self.ctbc_leg_gain_le = QLineEdit("0.0")
        self.ctbc_leg_push_gain_le = QLineEdit("1.75")
        self.ctbc_hip_gain_le = QLineEdit("0.0")
        self.ctbc_stance_gain_le = QLineEdit("0.30")
        self.ctbc_wheel_push_gain_le = QLineEdit("0.0")
        self.ctbc_ff_clip_le = QLineEdit("4.0")

        self.ctbc_reflex_flat_ratio_le.setToolTip("Fraction of flat samples with zero residual target. Higher preserves flat behavior more.")
        self.ctbc_reflex_gain_le.setToolTip("Scale applied to primitive residual targets during supervised reflex fitting.")
        self.ctbc_amplitude_le.setToolTip("Overall primitive scale.")
        self.ctbc_period_le.setToolTip("One lift cycle duration in seconds.")
        self.ctbc_cooldown_le.setToolTip("Pause between left/right lift cycles.")
        self.ctbc_shoulder_gain_le.setToolTip("Active-side shoulder swing.")
        self.ctbc_leg_gain_le.setToolTip("Active-side leg retract/lift gain.")
        self.ctbc_leg_push_gain_le.setToolTip("Active-side late push/extension gain.")
        self.ctbc_hip_gain_le.setToolTip("Active-side hip motion.")
        self.ctbc_stance_gain_le.setToolTip("Opposite-side support motion.")
        self.ctbc_wheel_push_gain_le.setToolTip("Extra wheel command during stair controller phase.")
        self.ctbc_ff_clip_le.setToolTip("Primitive normalized-action clamp.")

        primitive_layout.addRow("Amplitude:", self.ctbc_amplitude_le)
        primitive_layout.addRow("Period sec:", self.ctbc_period_le)
        primitive_layout.addRow("Cooldown sec:", self.ctbc_cooldown_le)
        primitive_layout.addRow("Shoulder gain:", self.ctbc_shoulder_gain_le)
        primitive_layout.addRow("Leg lift/retract:", self.ctbc_leg_gain_le)
        primitive_layout.addRow("Leg push:", self.ctbc_leg_push_gain_le)
        primitive_layout.addRow("Hip gain:", self.ctbc_hip_gain_le)
        primitive_layout.addRow("Stance gain:", self.ctbc_stance_gain_le)
        primitive_layout.addRow("Wheel push:", self.ctbc_wheel_push_gain_le)
        primitive_layout.addRow("Action clip:", self.ctbc_ff_clip_le)

        hint = QLabel("Tune Controller uses MuJoCo privileged stair state to search deterministic primitive parameters.")
        hint.setStyleSheet("color: #64748B;")
        layout.addWidget(hint)

        log_group = QGroupBox("Log")
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
        self.rl_train_btn = QPushButton("Tune Controller")
        self.test_primitive_btn = QPushButton("Test Primitive")
        self.test_policy_btn = QPushButton("Test Controller")
        self.refresh_btn = QPushButton("Refresh")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.status_label = QLabel("idle")
        self.close_btns = QDialogButtonBox(QDialogButtonBox.Close)
        action_row.addWidget(self.rl_train_btn)
        action_row.addWidget(self.test_primitive_btn)
        action_row.addWidget(self.test_policy_btn)
        action_row.addWidget(self.refresh_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.close_btns)
        layout.addLayout(action_row)

        self.rl_train_btn.clicked.connect(self.rlTrainRequested.emit)
        self.test_primitive_btn.clicked.connect(self.testPrimitiveRequested.emit)
        self.test_policy_btn.clicked.connect(self.testPolicyRequested.emit)
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

    def _browse(self, line_edit, save=False, title=None, pattern=None):
        current = line_edit.text().strip()
        start_dir = os.path.dirname(current) if current else os.getcwd()
        if save:
            path, _ = QFileDialog.getSaveFileName(self, title or "Export File", start_dir, pattern or "All Files (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, title or "Select File", start_dir, pattern or "All Files (*)")
        if path:
            line_edit.setText(path)

    def _on_env_changed(self, env_id):
        previous = self._current_env
        current = str(env_id)
        self._current_env = current
        if previous != current:
            self.envChanged.emit(previous, current)

    def get_settings(self):
        return {
            "env_id": self.env_cb.currentText(),
            "policy_path": self.policy_le.text().strip(),
            "output_path": self.output_le.text().strip(),
            "checkpoint_path": self.checkpoint_le.text().strip(),
            "ctbc_terrain": self.ctbc_terrain_cb.currentText(),
            "ctbc_reflex_only": "1",
            "ctbc_controller_candidates": self.ctbc_controller_candidates_le.text().strip(),
            "ctbc_reflex_samples": self.ctbc_reflex_samples_le.text().strip(),
            "ctbc_reflex_epochs": self.ctbc_reflex_epochs_le.text().strip(),
            "ctbc_reflex_lr": self.ctbc_reflex_lr_le.text().strip(),
            "ctbc_reflex_flat_ratio": self.ctbc_reflex_flat_ratio_le.text().strip(),
            "ctbc_reflex_gain": self.ctbc_reflex_gain_le.text().strip(),
            "ctbc_reflex_segment_steps": self.ctbc_reflex_segment_steps_le.text().strip(),
            "ctbc_reflex_batch": "256",
            "ctbc_stair_height_min": self.ctbc_stair_min_le.text().strip(),
            "ctbc_stair_height_max": self.ctbc_stair_max_le.text().strip(),
            "ctbc_episode_steps": self.ctbc_episode_steps_le.text().strip(),
            "ctbc_lift_amplitude": self.ctbc_amplitude_le.text().strip(),
            "ctbc_lift_period": self.ctbc_period_le.text().strip(),
            "ctbc_lift_cooldown": self.ctbc_cooldown_le.text().strip(),
            "ctbc_shoulder_gain": self.ctbc_shoulder_gain_le.text().strip(),
            "ctbc_leg_gain": self.ctbc_leg_gain_le.text().strip(),
            "ctbc_leg_push_gain": self.ctbc_leg_push_gain_le.text().strip(),
            "ctbc_hip_gain": self.ctbc_hip_gain_le.text().strip(),
            "ctbc_stance_gain": self.ctbc_stance_gain_le.text().strip(),
            "ctbc_wheel_push_gain": self.ctbc_wheel_push_gain_le.text().strip(),
            "ctbc_ff_clip": self.ctbc_ff_clip_le.text().strip(),
            "ctbc_action_clip": self.ctbc_ff_clip_le.text().strip(),
            "ctbc_compensate_action_scale": "1",
            "ppo_total_steps": "1",
            "ppo_num_envs": "1",
            "ppo_rollout_steps": "1",
            "ppo_epochs": "1",
            "ppo_learning_rate": "5e-5",
            "ppo_domain_randomize": "0.0",
            "hidden_dim": "256",
            "seed": "42",
        }

    def set_settings(self, settings):
        settings = dict(settings or {})
        if settings.get("env_id"):
            self.env_cb.blockSignals(True)
            self.env_cb.setCurrentText(str(settings.get("env_id")))
            self.env_cb.blockSignals(False)
            self._current_env = self.env_cb.currentText()
        self.policy_le.setText(str(settings.get("policy_path", "")))
        self.output_le.setText(str(settings.get("output_path", "")))
        self.checkpoint_le.setText(str(settings.get("checkpoint_path", "")))
        terrain = str(settings.get("ctbc_terrain", "stairs_up_easy"))
        if self.ctbc_terrain_cb.findText(terrain) >= 0:
            self.ctbc_terrain_cb.setCurrentText(terrain)
        self.ctbc_reflex_samples_le.setText(str(settings.get("ctbc_reflex_samples", "8192")))
        self.ctbc_controller_candidates_le.setText(str(settings.get("ctbc_controller_candidates", "64")))
        self.ctbc_reflex_epochs_le.setText(str(settings.get("ctbc_reflex_epochs", "12")))
        self.ctbc_reflex_lr_le.setText(str(settings.get("ctbc_reflex_lr", "3e-4")))
        self.ctbc_reflex_flat_ratio_le.setText(str(settings.get("ctbc_reflex_flat_ratio", "0.35")))
        self.ctbc_reflex_gain_le.setText(str(settings.get("ctbc_reflex_gain", "1.0")))
        self.ctbc_reflex_segment_steps_le.setText(str(settings.get("ctbc_reflex_segment_steps", "128")))
        self.ctbc_stair_min_le.setText(str(settings.get("ctbc_stair_height_min", "0.025")))
        self.ctbc_stair_max_le.setText(str(settings.get("ctbc_stair_height_max", "0.20")))
        self.ctbc_episode_steps_le.setText(str(settings.get("ctbc_episode_steps", "1200")))
        self.ctbc_amplitude_le.setText(str(settings.get("ctbc_lift_amplitude", "0.90")))
        self.ctbc_period_le.setText(str(settings.get("ctbc_lift_period", "0.75")))
        self.ctbc_cooldown_le.setText(str(settings.get("ctbc_lift_cooldown", "0.35")))
        self.ctbc_shoulder_gain_le.setText(str(settings.get("ctbc_shoulder_gain", "0.50")))
        self.ctbc_leg_gain_le.setText(str(settings.get("ctbc_leg_gain", "0.0")))
        self.ctbc_leg_push_gain_le.setText(str(settings.get("ctbc_leg_push_gain", "1.75")))
        self.ctbc_hip_gain_le.setText(str(settings.get("ctbc_hip_gain", "0.0")))
        self.ctbc_stance_gain_le.setText(str(settings.get("ctbc_stance_gain", "0.30")))
        self.ctbc_wheel_push_gain_le.setText(str(settings.get("ctbc_wheel_push_gain", "0.0")))
        self.ctbc_ff_clip_le.setText(str(settings.get("ctbc_ff_clip", settings.get("ctbc_action_clip", "4.0"))))

    def set_available_datasets(self, datasets, selected_paths=None):
        _ = datasets, selected_paths

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
            self.rl_train_btn,
            self.test_primitive_btn,
            self.test_policy_btn,
            self.refresh_btn,
            self.close_btns,
        ):
            widget.setEnabled(not self._running)
