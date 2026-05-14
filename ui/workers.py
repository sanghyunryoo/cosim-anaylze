from PyQt5.QtCore import QObject, pyqtSignal
from core.tester import Tester
from core.vision_heightmap_trainer import VisionHeightMapTrainer
from core.moe_trainer import MoETrainer
from core.homing_trainer import HomingTrainer


class TesterWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, tester: Tester):
        super().__init__()
        self.tester = tester
    def run(self):
        # Execute tester in a separate thread context
        try:
            self.tester.init_user_command()
            self.tester.test()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VisionTrainerWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, repo_root: str, env_id: str, dataset_paths, settings: dict):
        super().__init__()
        self.repo_root = repo_root
        self.env_id = env_id
        self.dataset_paths = list(dataset_paths or [])
        self.settings = dict(settings or {})
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def stop_requested(self):
        return bool(self._stop_requested)

    def run(self):
        try:
            trainer = VisionHeightMapTrainer(
                repo_root=self.repo_root,
                env_id=self.env_id,
                dataset_paths=self.dataset_paths,
                settings=self.settings,
                log_callback=self.log.emit,
                stop_callback=self.stop_requested,
            )
            summary = trainer.train()
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit(str(e))


class MoEWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, repo_root: str, settings: dict, mode: str):
        super().__init__()
        self.repo_root = repo_root
        self.settings = dict(settings or {})
        self.mode = str(mode)
        self._stop_requested = False
        self._switch_requested = False
        self._command_values = list(self.settings.get("command_values", []))

    def request_stop(self):
        self._stop_requested = True

    def request_policy_switch(self):
        self._switch_requested = True

    def switch_requested(self):
        return bool(self._switch_requested)

    def update_command_values(self, values):
        self._command_values = list(values or [])

    def command_values(self):
        return list(self._command_values)

    def stop_requested(self):
        return bool(self._stop_requested)

    def run(self):
        try:
            trainer = MoETrainer(
                repo_root=self.repo_root,
                settings=self.settings,
                log_callback=self.log.emit,
                stop_callback=self.stop_requested,
            )
            if self.mode == "collect":
                summary = trainer.collect()
            elif self.mode == "train":
                summary = trainer.train()
            elif self.mode == "export":
                checkpoint_path = self.settings.get("checkpoint_path", "")
                output_path = self.settings.get("output_path", "")
                exported = trainer.export_onnx_from_checkpoint(checkpoint_path, output_path)
                summary = {"onnx_path": exported}
            elif self.mode == "manual_export":
                exported = trainer.export_manual_moe_onnx(
                    self.settings.get("policy_a_path", ""),
                    self.settings.get("policy_b_path", ""),
                    0.0,
                    self.settings.get("output_path", ""),
                )
                summary = {
                    "onnx_path": exported,
                    "alpha_source": "final_command",
                }
            else:
                raise RuntimeError(f"Unknown MoE worker mode: {self.mode}")
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit(str(e))


class HomingWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, repo_root: str, settings: dict, mode: str):
        super().__init__()
        self.repo_root = repo_root
        self.settings = dict(settings or {})
        self.mode = str(mode)
        self._stop_requested = False
        self._switch_requested = False
        self._command_values = list(self.settings.get("command_values", []))

    def request_stop(self):
        self._stop_requested = True

    def request_policy_switch(self):
        self._switch_requested = True

    def switch_requested(self):
        return bool(self._switch_requested)

    def update_command_values(self, values):
        self._command_values = list(values or [])

    def command_values(self):
        return list(self._command_values)

    def stop_requested(self):
        return bool(self._stop_requested)

    def run(self):
        try:
            trainer = HomingTrainer(
                repo_root=self.repo_root,
                settings=self.settings,
                log_callback=self.log.emit,
                stop_callback=self.stop_requested,
                switch_callback=self.switch_requested,
                command_callback=self.command_values,
            )
            if self.mode == "collect":
                summary = trainer.collect()
            elif self.mode == "train":
                summary = trainer.train()
            elif self.mode == "rl_train":
                summary = trainer.fine_tune_rl()
            elif self.mode == "export":
                exported = trainer.export_onnx_from_checkpoint(
                    self.settings.get("checkpoint_path", ""),
                    self.settings.get("output_path", ""),
                )
                summary = {"onnx_path": exported}
            elif self.mode == "test_teacher":
                summary = trainer.test_teacher()
            elif self.mode == "test_policy":
                summary = trainer.test_export_policy()
            else:
                raise RuntimeError(f"Unknown Homing worker mode: {self.mode}")
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit(str(e))
