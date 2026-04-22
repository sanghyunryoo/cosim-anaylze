from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
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


class VisionTrainDialog(QDialog):
    trainRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    refreshRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._available_by_path = {}

        self.setModal(False)
        self.setWindowTitle("Vision Train")
        self.resize(980, 720)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.env_label = QLabel("")
        self.env_label.setStyleSheet("font-weight: bold; color: #334155;")
        layout.addWidget(self.env_label)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        layout.addLayout(top_row, 1)

        dataset_group = QGroupBox("Training Datasets")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(8)
        top_row.addWidget(dataset_group, 3)

        dataset_hint = QLabel("Select one or more datasets on the left, add them to the training list, and repeat to include duplicates.")
        dataset_hint.setWordWrap(True)
        dataset_hint.setStyleSheet("color: #64748B;")
        dataset_layout.addWidget(dataset_hint)

        dataset_lists = QHBoxLayout()
        dataset_lists.setSpacing(8)
        dataset_layout.addLayout(dataset_lists, 1)

        available_col = QVBoxLayout()
        available_col.addWidget(QLabel("Available"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        available_col.addWidget(self.available_list, 1)
        dataset_lists.addLayout(available_col, 1)

        button_col = QVBoxLayout()
        button_col.addStretch()
        self.add_dataset_btn = QPushButton("Add ->")
        self.remove_dataset_btn = QPushButton("<- Remove")
        self.clear_dataset_btn = QPushButton("Clear")
        self.refresh_dataset_btn = QPushButton("Refresh")
        button_col.addWidget(self.add_dataset_btn)
        button_col.addWidget(self.remove_dataset_btn)
        button_col.addWidget(self.clear_dataset_btn)
        button_col.addWidget(self.refresh_dataset_btn)
        button_col.addStretch()
        dataset_lists.addLayout(button_col)

        selected_col = QVBoxLayout()
        selected_col.addWidget(QLabel("Selected For Training"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        selected_col.addWidget(self.selected_list, 1)
        dataset_lists.addLayout(selected_col, 1)

        param_group = QGroupBox("Training Parameters")
        param_layout = QFormLayout(param_group)
        param_layout.setLabelAlignment(Qt.AlignRight)
        param_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        param_layout.setSpacing(8)
        top_row.addWidget(param_group, 2)

        self.epochs_le = QLineEdit("10")
        self.batch_size_le = QLineEdit("64")
        self.learning_rate_le = QLineEdit("1e-3")
        self.latent_dim_le = QLineEdit("128")
        self.hidden_dim_le = QLineEdit("128")
        self.val_ratio_le = QLineEdit("0.1")
        self.seed_le = QLineEdit("42")
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)

        param_layout.addRow("Epochs:", self.epochs_le)
        param_layout.addRow("Batch size:", self.batch_size_le)
        param_layout.addRow("Learning rate:", self.learning_rate_le)
        param_layout.addRow("Latent dim:", self.latent_dim_le)
        param_layout.addRow("Hidden dim:", self.hidden_dim_le)
        param_layout.addRow("Val ratio:", self.val_ratio_le)
        param_layout.addRow("Seed:", self.seed_le)
        param_layout.addRow("Status:", self.status_label)

        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAcceptRichText(False)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.document().setMaximumBlockCount(5000)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setStyleHint(QFont.Monospace)
        self.log_output.setFont(fixed_font)
        self.log_output.setStyleSheet(
            "QTextEdit { background-color: #000000; color: #f5f5f5; border: 1px solid #333333; }"
        )
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group, 1)

        action_row = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.export_btn = QPushButton("Export ONNX")
        self.close_btns = QDialogButtonBox(QDialogButtonBox.Close)
        action_row.addWidget(self.train_btn)
        action_row.addWidget(self.export_btn)
        action_row.addStretch()
        action_row.addWidget(self.close_btns)
        layout.addLayout(action_row)

        self.add_dataset_btn.clicked.connect(self.add_selected_datasets)
        self.remove_dataset_btn.clicked.connect(self.remove_selected_datasets)
        self.clear_dataset_btn.clicked.connect(self.selected_list.clear)
        self.refresh_dataset_btn.clicked.connect(self.refreshRequested.emit)
        self.train_btn.clicked.connect(self.trainRequested.emit)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.close_btns.rejected.connect(self.close)
        self.close_btns.accepted.connect(self.close)

    def set_env_id(self, env_id: str):
        self.env_label.setText(f"Environment: {env_id}")

    def set_available_datasets(self, datasets, selected_paths=None):
        self._available_by_path = {entry["path"]: dict(entry) for entry in datasets}
        self.available_list.clear()
        for entry in datasets:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.UserRole, entry["path"])
            self.available_list.addItem(item)

        self.selected_list.clear()
        restored = list(selected_paths or [])
        if not restored and datasets:
            restored = [datasets[0]["path"]]
        for path in restored:
            self._append_selected_dataset(path)

    def add_selected_datasets(self):
        items = self.available_list.selectedItems()
        for item in items:
            self._append_selected_dataset(item.data(Qt.UserRole))

    def remove_selected_datasets(self):
        for item in self.selected_list.selectedItems():
            self.selected_list.takeItem(self.selected_list.row(item))

    def _append_selected_dataset(self, path: str):
        entry = self._available_by_path.get(path)
        label = entry["label"] if entry is not None else f"{path} | missing"
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, path)
        self.selected_list.addItem(item)

    def get_settings(self):
        return {
            "epochs": self.epochs_le.text().strip(),
            "batch_size": self.batch_size_le.text().strip(),
            "learning_rate": self.learning_rate_le.text().strip(),
            "latent_dim": self.latent_dim_le.text().strip(),
            "hidden_dim": self.hidden_dim_le.text().strip(),
            "val_ratio": self.val_ratio_le.text().strip(),
            "seed": self.seed_le.text().strip(),
            "selected_datasets": self.get_selected_dataset_paths(),
        }

    def get_selected_dataset_paths(self):
        return [
            self.selected_list.item(index).data(Qt.UserRole)
            for index in range(self.selected_list.count())
        ]

    def set_settings(self, settings: dict):
        settings = dict(settings or {})
        self.epochs_le.setText(str(settings.get("epochs", "10")))
        self.batch_size_le.setText(str(settings.get("batch_size", "64")))
        self.learning_rate_le.setText(str(settings.get("learning_rate", "1e-3")))
        self.latent_dim_le.setText(str(settings.get("latent_dim", "128")))
        self.hidden_dim_le.setText(str(settings.get("hidden_dim", "128")))
        self.val_ratio_le.setText(str(settings.get("val_ratio", "0.1")))
        self.seed_le.setText(str(settings.get("seed", "42")))

    def set_status(self, text: str):
        self.status_label.setText(text)

    def clear_log(self):
        self.log_output.clear()

    def append_log(self, message: str):
        if not message:
            return
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.insertPlainText(message)
        self.log_output.ensureCursorVisible()

    def set_running(self, running: bool):
        self._running = bool(running)
        self.train_btn.setEnabled(not self._running)
        self.export_btn.setEnabled(not self._running)
        self.add_dataset_btn.setEnabled(not self._running)
        self.remove_dataset_btn.setEnabled(not self._running)
        self.clear_dataset_btn.setEnabled(not self._running)
        self.refresh_dataset_btn.setEnabled(not self._running)
        self.available_list.setEnabled(not self._running)
        self.selected_list.setEnabled(not self._running)
        self.close_btns.setEnabled(not self._running)

    def closeEvent(self, event):
        if self._running:
            QMessageBox.warning(self, "Vision Train", "Training is running. Wait for it to finish before closing this window.")
            event.ignore()
            return
        super().closeEvent(event)
