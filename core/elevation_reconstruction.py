"""Collection and training utilities for elevation-map denoising."""

import json
import os
import copy
from datetime import datetime

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except Exception:
    torch = None


class ElevationRestorer(nn.Module if torch is not None else object):
    def __init__(self, map_shape, height_center=0.0, height_scale=1.0):
        super().__init__()
        self.map_shape = tuple(map_shape)
        self.register_buffer("height_center", torch.tensor(float(height_center)))
        self.register_buffer("height_scale", torch.tensor(float(height_scale)))
        # A spatial model is essential for sparse stair/beam cells: an MLP
        # tends to average those cells into the overwhelmingly flat background.
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, noisy_elevation):
        height, width = self.map_shape
        elevation = ((noisy_elevation - self.height_center) / self.height_scale).view(-1, 1, height, width)
        output = self.net(elevation).flatten(1)
        return self.height_center + self.height_scale * torch.tanh(output)


def save_dataset(output_root, noisy_maps, noisy_gravity, targets, metadata):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, "elevation_reconstruction", stamp)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "dataset.npz")
    np.savez_compressed(
        path,
        noisy_elevation=np.asarray(noisy_maps, dtype=np.float32),
        projected_gravity=np.asarray(noisy_gravity, dtype=np.float32),
        elevation_target=np.asarray(targets, dtype=np.float32),
    )
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def train(dataset_path, output_root, epochs=20, batch_size=512, learning_rate=1e-3, log=print):
    if torch is None:
        raise RuntimeError("Elevation reconstruction training requires torch.")
    with np.load(dataset_path) as payload:
        noisy = payload["noisy_elevation"].astype(np.float32)
        gravity = payload["projected_gravity"].astype(np.float32)
        target = payload["elevation_target"].astype(np.float32)
    metadata_path = os.path.join(os.path.dirname(dataset_path), "metadata.json")
    with open(metadata_path, encoding="utf-8") as handle:
        map_shape = tuple(json.load(handle).get("map_shape", (1, noisy.shape[1])))
    if len(noisy) < 8:
        raise RuntimeError("Collect at least 8 samples before training.")
    if noisy.ndim != 2 or gravity.shape != (len(noisy), 3) or target.shape != noisy.shape:
        raise RuntimeError("Invalid elevation reconstruction dataset shapes.")
    if len(map_shape) != 2 or int(np.prod(map_shape)) != noisy.shape[1]:
        raise RuntimeError(f"Dataset map_shape {map_shape} does not match {noisy.shape[1]} elevation values.")
    # Discard maps recorded after a fall/launch. Those maps are not useful for
    # an elevation restorer and otherwise create multi-metre regression targets.
    map_median = np.median(target, axis=1)
    map_range = np.ptp(target, axis=1)
    valid = (map_median >= 0.20) & (map_median <= 0.70) & (map_range <= 0.40)
    if int(valid.sum()) < 100:
        raise RuntimeError("Too few physically valid elevation maps after fall/outlier filtering.")
    if int(valid.sum()) != len(target):
        log(f"[elev-train] discarded {len(target) - int(valid.sum()):,}/{len(target):,} fall/outlier maps (median outside 0.20..0.70 m or range > 0.40 m)")
    noisy, gravity, target = noisy[valid], gravity[valid], target[valid]
    permutation = np.random.default_rng(20260828).permutation(len(target))
    validation_count = max(1, int(round(len(target) * 0.10)))
    validation_indices, train_indices = permutation[:validation_count], permutation[validation_count:]
    train_target = target[train_indices]
    lower, upper = float(train_target.min()), float(train_target.max())
    height_center, height_scale = (lower + upper) / 2.0, max((upper - lower) / 2.0 + 0.01, 0.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElevationRestorer(map_shape, height_center, height_scale).to(device)
    train_relief = np.ptp(target[train_indices], axis=1)
    sampler = WeightedRandomSampler(
        torch.from_numpy(1.0 + 8.0 * np.clip(train_relief / 0.20, 0.0, 1.0)).double(),
        num_samples=len(train_indices), replacement=True,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(noisy[train_indices]), torch.from_numpy(target[train_indices])),
        batch_size=max(1, int(batch_size)), sampler=sampler,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    # Height maps are mostly flat.  Plain MSE therefore learns the dominant
    # background height and erases the comparatively rare obstacle cells.
    # Weight cells by per-map relief and also match spatial gradients, so a
    # beam/stair edge matters even when it occupies only a few cells.
    def reconstruction_loss(prediction, target):
        relief = torch.abs(target - target.median(dim=1, keepdim=True).values)
        cell_weight = 1.0 + 12.0 * torch.clamp(relief / 0.05, 0.0, 1.0)
        cell_loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=0.02, reduction="none")
        cell_loss = (cell_weight * cell_loss).sum() / cell_weight.sum()
        pred_map = prediction.view(-1, 1, *map_shape)
        target_map = target.view(-1, 1, *map_shape)
        gradient_loss = (
            torch.nn.functional.smooth_l1_loss(pred_map[..., 1:, :] - pred_map[..., :-1, :], target_map[..., 1:, :] - target_map[..., :-1, :], beta=0.02)
            + torch.nn.functional.smooth_l1_loss(pred_map[..., :, 1:] - pred_map[..., :, :-1], target_map[..., :, 1:] - target_map[..., :, :-1], beta=0.02)
        ) / 2.0
        return cell_loss + gradient_loss, cell_loss, gradient_loss
    best_state = None; best_max_error = float("inf"); best_epoch = 0
    validation_map = torch.from_numpy(noisy[validation_indices]).to(device)
    validation_target = torch.from_numpy(target[validation_indices]).to(device)
    for epoch in range(max(1, int(epochs))):
        model.train(); total = cell_total = gradient_total = 0.0; count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss, cell_loss, gradient_loss = reconstruction_loss(model(x), y); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(x); cell_total += float(cell_loss.detach()) * len(x); gradient_total += float(gradient_loss.detach()) * len(x); count += len(x)
        model.eval()
        with torch.no_grad():
            validation_error = torch.abs(model(validation_map) - validation_target)
            validation_max = float(validation_error.max().cpu())
            validation_p99 = float(torch.quantile(validation_error, 0.99).cpu())
        if validation_max < best_max_error:
            best_max_error = validation_max; best_epoch = epoch + 1
            best_state = copy.deepcopy(model.cpu().state_dict()); model.to(device)
        log(f"[elev-train] epoch {epoch + 1}/{epochs} weighted_smooth_l1={cell_total / max(count, 1):.7f} gradient={gradient_total / max(count, 1):.7f} val_p99={validation_p99:.5f} val_max={validation_max:.5f}")
        if validation_max <= 0.05:
            log(f"[elev-train] strict validation passed at epoch {epoch + 1}: every held-out cell <= 0.05 m")
            break
    model.load_state_dict(best_state)
    run_dir = os.path.join(output_root, "elevation_reconstruction", "latest")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint = os.path.join(run_dir, "elevation_restorer.pt")
    onnx_path = os.path.join(run_dir, "elevation_restorer.onnx")
    torch.save({"state_dict": model.cpu().state_dict(), "map_dim": noisy.shape[1], "dataset_path": dataset_path,
                "map_shape": map_shape, "valid_samples": len(target), "validation_samples": validation_count, "best_epoch": best_epoch,
                "validation_max_abs_error": best_max_error, "height_center": height_center, "height_scale": height_scale}, checkpoint)
    model.eval()
    dummy_map = torch.zeros(1, noisy.shape[1])
    torch.onnx.export(model, dummy_map, onnx_path,
                      input_names=["noisy_elevation"], output_names=["elevation"],
                      opset_version=13, dynamo=False)
    # Fail the run if the exported artifact cannot be used by the deployment runtime.
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {"noisy_elevation": np.zeros((1, noisy.shape[1]), dtype=np.float32)},
    )[0]
    if tuple(output.shape) != (1, noisy.shape[1]):
        raise RuntimeError(f"Invalid exported ONNX output shape: {output.shape}")
    log(f"[elev-train] ONNX export verified: {onnx_path}")
    return {"checkpoint": checkpoint, "onnx": onnx_path, "samples": len(target), "validation_max_abs_error": best_max_error, "strict_validation_passed": best_max_error <= 0.05}
