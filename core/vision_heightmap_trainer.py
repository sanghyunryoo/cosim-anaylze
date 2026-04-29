import json
import os
from dataclasses import dataclass

import numpy as np


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    random_split = None


class HeightMapSupervisionDataset(Dataset):
    def __init__(self, dataset_npz_paths):
        dataset_npz_paths = list(dataset_npz_paths or [])
        if not dataset_npz_paths:
            raise RuntimeError("At least one dataset.npz path is required.")

        depth_history_parts = []
        projected_gravity_parts = []
        height_map_parts = []
        self.dataset_paths = list(dataset_npz_paths)
        expected_depth_shape = None
        expected_state_shape = None
        expected_height_shape = None

        for dataset_npz_path in self.dataset_paths:
            with np.load(dataset_npz_path) as payload:
                depth_history = payload["depth_history"].astype(np.float32) / 255.0
                projected_gravity = payload["projected_gravity"].astype(np.float32)
                height_map = payload["height_map"].astype(np.float32)

            current_depth_shape = tuple(int(v) for v in depth_history.shape[1:])
            current_state_shape = tuple(int(v) for v in projected_gravity.shape[1:])
            current_height_shape = tuple(int(v) for v in height_map.shape[1:])
            if expected_depth_shape is None:
                expected_depth_shape = current_depth_shape
                expected_state_shape = current_state_shape
                expected_height_shape = current_height_shape
            elif (
                current_depth_shape != expected_depth_shape
                or current_state_shape != expected_state_shape
                or current_height_shape != expected_height_shape
            ):
                raise RuntimeError(
                    "Selected datasets must share the same depth/state/height-map shapes. "
                    f"Expected depth {expected_depth_shape}, state {expected_state_shape}, height_map {expected_height_shape}, "
                    f"but got depth {current_depth_shape}, state {current_state_shape}, height_map {current_height_shape} "
                    f"from {dataset_npz_path}."
                )

            depth_history_parts.append(depth_history)
            projected_gravity_parts.append(projected_gravity)
            height_map_parts.append(height_map)

        self.depth_history = np.concatenate(depth_history_parts, axis=0)
        self.projected_gravity = np.concatenate(projected_gravity_parts, axis=0)
        self.height_map = np.concatenate(height_map_parts, axis=0)

        if self.depth_history.ndim != 4:
            raise RuntimeError("depth_history must have shape [N, T, H, W].")
        if self.projected_gravity.ndim != 2:
            raise RuntimeError("projected_gravity must have shape [N, S].")
        if self.height_map.ndim != 3:
            raise RuntimeError("height_map must have shape [N, Hh, Wh].")

    def __len__(self):
        return int(self.depth_history.shape[0])

    def __getitem__(self, index):
        return {
            "depth_history": torch.from_numpy(self.depth_history[index]),
            "projected_gravity": torch.from_numpy(self.projected_gravity[index]),
            "height_map": torch.from_numpy(self.height_map[index]),
        }


class DepthFrameEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Stage1LSTMPredictor(nn.Module):
    def __init__(self, state_dim: int, height_map_shape, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.height_map_shape = tuple(int(v) for v in height_map_shape)
        self.frame_encoder = DepthFrameEncoder(latent_dim=latent_dim)
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.height_map_shape[0] * self.height_map_shape[1]),
        )

    def forward(self, depth_history, state_vec):
        batch_size, history_len, height, width = depth_history.shape
        x = depth_history.reshape(batch_size * history_len, 1, height, width)
        encoded = self.frame_encoder(x).reshape(batch_size, history_len, -1)
        _, (hidden, _) = self.lstm(encoded)
        last_hidden = hidden[-1]
        fused = torch.cat([last_hidden, state_vec], dim=-1)
        raw = self.head(fused).reshape(batch_size, 1, self.height_map_shape[0], self.height_map_shape[1])
        return raw


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Stage2UNetRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(32 + 16, 16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = nn.functional.interpolate(self.up2(b), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = nn.functional.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class VisionHeightMapPredictor(nn.Module):
    def __init__(self, state_dim: int, height_map_shape, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.stage1 = Stage1LSTMPredictor(
            state_dim=state_dim,
            height_map_shape=height_map_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.stage2 = Stage2UNetRefiner()

    def forward(self, depth_history, state_vec):
        raw = self.stage1(depth_history, state_vec)
        refined = self.stage2(raw)
        return refined, raw


class VisionHeightMapPredictorForExport(nn.Module):
    def __init__(self, predictor: VisionHeightMapPredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, depth_history, state_vec):
        refined, _ = self.predictor(depth_history, state_vec)
        return refined


@dataclass
class TrainingArtifacts:
    run_dir: str
    checkpoint_path: str
    onnx_path: str
    summary_path: str


class VisionHeightMapTrainer:
    def __init__(self, repo_root: str, env_id: str, dataset_paths=None, settings: dict = None, log_callback=None, stop_callback=None):
        self.repo_root = os.path.abspath(repo_root)
        self.env_id = str(env_id)
        self.dataset_paths = list(dataset_paths or [])
        self.settings = dict(settings or {})
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.device = "cpu"

    def _require_dependencies(self):
        if torch is None or nn is None or DataLoader is None:
            raise RuntimeError(
                "Vision predictor training requires the 'torch' package, and ONNX export requires 'onnx'. "
                "Install the packages from requirements.txt first."
            )

    def _dataset_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "dataset", "height_map_supervision")

    def _resolve_latest_dataset_npz(self):
        dataset_root = self._dataset_root()
        if not os.path.isdir(dataset_root):
            raise RuntimeError(f"No dataset directory found for env '{self.env_id}'.")
        run_dirs = [
            os.path.join(dataset_root, name)
            for name in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, name))
        ]
        run_dirs.sort(reverse=True)
        for run_dir in run_dirs:
            dataset_path = os.path.join(run_dir, "dataset.npz")
            if os.path.isfile(dataset_path):
                return dataset_path
        raise RuntimeError(f"No dataset.npz file found under {dataset_root}.")

    def _resolve_dataset_npz_paths(self):
        if self.dataset_paths:
            missing = [path for path in self.dataset_paths if not os.path.isfile(path)]
            if missing:
                missing_text = "\n".join(missing)
                raise RuntimeError(f"Selected dataset files were not found:\n{missing_text}")
            return list(self.dataset_paths)
        return [self._resolve_latest_dataset_npz()]

    def _log(self, message: str):
        text = str(message)
        if not text.endswith("\n"):
            text += "\n"
        if callable(self.log_callback):
            self.log_callback(text)
        print(text, end="")

    def _stop_requested(self):
        return bool(callable(self.stop_callback) and self.stop_callback())

    def _make_output_paths(self):
        output_root = os.path.join(self.repo_root, "envs", self.env_id, "weights", "vision_heightmap")
        os.makedirs(output_root, exist_ok=True)
        run_name = self.settings.get("run_name", "latest")
        run_dir = os.path.join(output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return TrainingArtifacts(
            run_dir=run_dir,
            checkpoint_path=os.path.join(run_dir, "vision_heightmap_predictor.pt"),
            onnx_path=os.path.join(run_dir, "vision_heightmap_predictor.onnx"),
            summary_path=os.path.join(run_dir, "train_summary.json"),
        )

    def train(self):
        self._require_dependencies()
        dataset_paths = self._resolve_dataset_npz_paths()
        self._log(f"[vision-train] loading {len(dataset_paths)} dataset(s)")
        for index, dataset_path in enumerate(dataset_paths, start=1):
            self._log(f"[vision-train] dataset[{index}] {dataset_path}")
        dataset = HeightMapSupervisionDataset(dataset_paths)
        sample_count = len(dataset)
        if sample_count < 8:
            raise RuntimeError("At least 8 dataset samples are required to train the vision predictor.")

        val_ratio = min(0.4, max(0.05, float(self.settings.get("val_ratio", 0.1))))
        val_count = max(1, int(sample_count * val_ratio))
        train_count = max(1, sample_count - val_count)
        if train_count <= 0:
            train_count = max(1, sample_count - 1)
            val_count = 1

        generator = torch.Generator().manual_seed(int(self.settings.get("seed", 42)))
        train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)

        batch_size = max(1, int(self.settings.get("batch_size", 64)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        state_dim = int(dataset.projected_gravity.shape[1])
        height_map_shape = tuple(int(v) for v in dataset.height_map.shape[1:])
        predictor = VisionHeightMapPredictor(
            state_dim=state_dim,
            height_map_shape=height_map_shape,
            latent_dim=max(16, int(self.settings.get("latent_dim", 128))),
            hidden_dim=max(16, int(self.settings.get("hidden_dim", 128))),
        )
        predictor.to(self.device)

        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=float(self.settings.get("learning_rate", 1e-3)),
            weight_decay=float(self.settings.get("weight_decay", 1e-6)),
        )
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        epochs = max(1, int(self.settings.get("epochs", 10)))

        best_val = None
        best_state = None
        history = []

        for epoch in range(epochs):
            if self._stop_requested():
                self._log("[vision-train] stop requested; ending after last completed epoch.")
                break
            predictor.train()
            train_loss_total = 0.0
            train_batches = 0
            for batch in train_loader:
                if self._stop_requested():
                    self._log("[vision-train] stop requested during training batch.")
                    break
                depth_history = batch["depth_history"].to(self.device)
                projected_gravity = batch["projected_gravity"].to(self.device)
                target = batch["height_map"].to(self.device).unsqueeze(1)

                refined, raw = predictor(depth_history, projected_gravity)
                loss_stage1 = mse_loss(raw, target)
                loss_stage2 = l1_loss(refined, target)
                loss = loss_stage1 + loss_stage2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_total += float(loss.item())
                train_batches += 1

            if self._stop_requested():
                break

            predictor.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if self._stop_requested():
                        self._log("[vision-train] stop requested during validation batch.")
                        break
                    depth_history = batch["depth_history"].to(self.device)
                    projected_gravity = batch["projected_gravity"].to(self.device)
                    target = batch["height_map"].to(self.device).unsqueeze(1)

                    refined, raw = predictor(depth_history, projected_gravity)
                    loss_stage1 = mse_loss(raw, target)
                    loss_stage2 = l1_loss(refined, target)
                    loss = loss_stage1 + loss_stage2
                    val_loss_total += float(loss.item())
                    val_batches += 1

            if self._stop_requested():
                break

            train_loss = train_loss_total / max(1, train_batches)
            val_loss = val_loss_total / max(1, val_batches)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            self._log(f"[vision-train] epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")

            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in predictor.state_dict().items()}

        if best_state is None:
            if self._stop_requested():
                return {
                    "env_id": self.env_id,
                    "dataset_path": dataset_paths[0],
                    "dataset_paths": dataset_paths,
                    "samples": sample_count,
                    "stopped": True,
                    "history": history,
                }
            raise RuntimeError("Vision predictor training did not produce a valid checkpoint.")

        predictor.load_state_dict(best_state)
        artifacts = self._make_output_paths()
        torch.save(
            {
                "state_dict": predictor.state_dict(),
                "state_dim": state_dim,
                "height_map_shape": list(height_map_shape),
                "latent_dim": max(16, int(self.settings.get("latent_dim", 128))),
                "hidden_dim": max(16, int(self.settings.get("hidden_dim", 128))),
                "dataset_path": dataset_paths[0],
                "dataset_paths": dataset_paths,
                "history": history,
            },
            artifacts.checkpoint_path,
        )
        summary = {
            "env_id": self.env_id,
            "dataset_path": dataset_paths[0],
            "dataset_paths": dataset_paths,
            "samples": sample_count,
            "train_samples": train_count,
            "val_samples": val_count,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(self.settings.get("learning_rate", 1e-3)),
            "best_val_loss": float(best_val),
            "checkpoint_path": artifacts.checkpoint_path,
            "onnx_path": artifacts.onnx_path,
            "history": history,
        }
        with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.export_onnx_from_checkpoint(artifacts.checkpoint_path, artifacts.onnx_path)
        return summary

    def export_onnx_from_checkpoint(self, checkpoint_path: str, output_path: str):
        self._require_dependencies()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        predictor = VisionHeightMapPredictor(
            state_dim=int(checkpoint["state_dim"]),
            height_map_shape=tuple(int(v) for v in checkpoint["height_map_shape"]),
            latent_dim=int(checkpoint.get("latent_dim", 128)),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        )
        predictor.load_state_dict(checkpoint["state_dict"])
        predictor.eval()
        export_model = VisionHeightMapPredictorForExport(predictor)

        dataset_paths = checkpoint.get("dataset_paths", [])
        dataset_path = dataset_paths[0] if dataset_paths else checkpoint.get("dataset_path", "")
        if not dataset_path or not os.path.isfile(dataset_path):
            dataset_path = self._resolve_dataset_npz_paths()[0]
        payload = np.load(dataset_path)
        depth_shape = tuple(int(v) for v in payload["depth_history"].shape[1:])
        state_dim = int(payload["projected_gravity"].shape[1])

        dummy_depth = torch.zeros((1, depth_shape[0], depth_shape[1], depth_shape[2]), dtype=torch.float32)
        dummy_state = torch.zeros((1, state_dim), dtype=torch.float32)
        torch.onnx.export(
            export_model,
            (dummy_depth, dummy_state),
            output_path,
            input_names=["depth_history", "projected_gravity"],
            output_names=["height_map"],
            dynamic_axes={
                "depth_history": {0: "batch"},
                "projected_gravity": {0: "batch"},
                "height_map": {0: "batch"},
            },
            opset_version=13,
        )
        return output_path
