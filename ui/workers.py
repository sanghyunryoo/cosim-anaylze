from PyQt5.QtCore import QObject, pyqtSignal
from core.tester import Tester
from core.vision_heightmap_trainer import VisionHeightMapTrainer


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

    def run(self):
        try:
            trainer = VisionHeightMapTrainer(
                repo_root=self.repo_root,
                env_id=self.env_id,
                dataset_paths=self.dataset_paths,
                settings=self.settings,
                log_callback=self.log.emit,
            )
            summary = trainer.train()
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit(str(e))
