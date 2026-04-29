from PyQt5.QtCore import QObject, pyqtSignal
from core.tester import Tester
from core.vision_heightmap_trainer import VisionHeightMapTrainer
from core.moe_trainer import MoETrainer


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

    def request_stop(self):
        self._stop_requested = True

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
                    self.settings.get("manual_alpha", 0.0),
                    self.settings.get("output_path", ""),
                )
                summary = {
                    "onnx_path": exported,
                    "manual_alpha": float(self.settings.get("manual_alpha", 0.0)),
                }
            else:
                raise RuntimeError(f"Unknown MoE worker mode: {self.mode}")
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit(str(e))
