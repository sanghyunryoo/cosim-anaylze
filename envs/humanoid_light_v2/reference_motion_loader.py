"""Isaac-compatible reference-motion sampling for Humanoid Light.

The reference PPO was trained in Isaac with a deliberately small, fixed
contract.  This module mirrors that contract in NumPy so the MuJoCo tester can
feed the exported ONNX exactly the same reference targets:

* retargeted ``base_frame_*``/``joint_angles`` clips are canonicalised using
  the same central finite differences as the Isaac loader;
* the retargeter joint order is reordered into the simulator policy order;
* control ticks select ``floor(t / motion.dt)`` -- there is no interpolation;
* future reference frames and the clamped sin/cos phase use Isaac's formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HUMANOID_LIGHT_REF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_yaw_joint",
    "torso_pitch_joint",
    "torso_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_joint",
    "head_joint",
)

DEFAULT_REFERENCE_MOTION = "82_82_08_poses_keypoints_settle5s_retargeted.npz"
FUTURE_FRAME_OFFSETS: tuple[int, ...] = (0, 1, 4)


def resolve_reference_motion(motion: str | Path | None, reference_dir: str | Path) -> Path:
    """Resolve a bundled filename/stem or an explicit ``.npz`` path."""

    directory = Path(reference_dir).expanduser().resolve()
    candidate = Path(motion or DEFAULT_REFERENCE_MOTION).expanduser()
    choices: list[Path] = []
    if candidate.is_file():
        choices.append(candidate)
    else:
        choices.append(directory / candidate.name)
        if candidate.suffix != ".npz":
            choices.append((directory / candidate.name).with_suffix(".npz"))

    for path in choices:
        if path.is_file():
            return path.resolve()

    available = ", ".join(path.name for path in sorted(directory.glob("*.npz")))
    raise FileNotFoundError(f"Reference motion not found: {motion!s}. Available bundled clips: {available}")


class HumanoidLightReferenceMotion:
    """A single retargeted clip in the requested simulator joint order."""

    def __init__(self, motion_path: str | Path, asset_joint_names: list[str] | tuple[str, ...]):
        self.motion_path = Path(motion_path).expanduser().resolve()
        if not self.motion_path.is_file():
            raise FileNotFoundError(f"Reference motion file not found: {self.motion_path}")

        self.asset_joint_names = tuple(asset_joint_names)
        if len(self.asset_joint_names) != len(HUMANOID_LIGHT_REF_JOINT_NAMES):
            raise ValueError("Humanoid Light reference motion requires exactly 26 controlled joints.")
        if set(self.asset_joint_names) != set(HUMANOID_LIGHT_REF_JOINT_NAMES):
            missing = sorted(set(HUMANOID_LIGHT_REF_JOINT_NAMES).difference(self.asset_joint_names))
            extra = sorted(set(self.asset_joint_names).difference(HUMANOID_LIGHT_REF_JOINT_NAMES))
            raise ValueError(f"Asset joint order does not match Humanoid Light reference schema. missing={missing}, extra={extra}")

        with np.load(self.motion_path, allow_pickle=False) as raw:
            keys = set(raw.files)
            required = {"base_frame_pos", "base_frame_wxyz", "joint_angles", "fps"}
            missing = sorted(required.difference(keys))
            if missing:
                raise KeyError(f"Reference motion missing required keys: {missing}")
            root_pos = np.asarray(raw["base_frame_pos"], dtype=np.float32)
            root_quat_wxyz = np.asarray(raw["base_frame_wxyz"], dtype=np.float32)
            dof_pos_source = np.asarray(raw["joint_angles"], dtype=np.float32)
            fps = float(np.asarray(raw["fps"]).item())

        if fps <= 0.0 or not np.isfinite(fps):
            raise ValueError(f"Reference motion fps must be a positive finite scalar, got {fps}")
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"base_frame_pos must be [T, 3], got {root_pos.shape}")
        if root_quat_wxyz.shape != (root_pos.shape[0], 4):
            raise ValueError(f"base_frame_wxyz must be [T, 4], got {root_quat_wxyz.shape}")
        if dof_pos_source.shape != (root_pos.shape[0], 26):
            raise ValueError(f"joint_angles must be [T, 26], got {dof_pos_source.shape}")
        if root_pos.shape[0] < 1:
            raise ValueError("Reference motion must contain at least one frame.")
        if not all(np.isfinite(value).all() for value in (root_pos, root_quat_wxyz, dof_pos_source)):
            raise ValueError("Reference motion contains non-finite values.")

        self.fps = fps
        self.dt = 1.0 / fps
        self.num_frames = int(root_pos.shape[0])
        self.duration_s = self.num_frames * self.dt
        self.root_pos = root_pos
        self.root_quat_wxyz = root_quat_wxyz
        self.source_to_asset_indices = np.asarray(
            [HUMANOID_LIGHT_REF_JOINT_NAMES.index(name) for name in self.asset_joint_names], dtype=np.int64
        )
        self.dof_pos = dof_pos_source[:, self.source_to_asset_indices]

        # Match HumanoidLightReferenceMotion._canonicalize_raw in Isaac exactly.
        if self.num_frames == 1:
            self.root_lin_vel = np.zeros_like(root_pos)
            self.dof_vel = np.zeros_like(self.dof_pos)
        else:
            self.root_lin_vel = np.gradient(root_pos, self.dt, axis=0).astype(np.float32)
            dof_vel_source = np.gradient(dof_pos_source, self.dt, axis=0).astype(np.float32)
            self.dof_vel = dof_vel_source[:, self.source_to_asset_indices]

    def frame_id(self, control_step: int, control_dt: float) -> int:
        """Return Isaac's clamped frame id for an environment control tick."""

        elapsed = float(control_step) * float(control_dt)
        return int(np.clip(np.floor(elapsed / self.dt), 0, self.num_frames - 1))

    def frame(self, frame_id: int) -> dict[str, np.ndarray]:
        frame_id = int(np.clip(frame_id, 0, self.num_frames - 1))
        return {
            "frame_id": frame_id,
            "root_pos": self.root_pos[frame_id],
            "root_quat_wxyz": self.root_quat_wxyz[frame_id],
            "root_lin_vel": self.root_lin_vel[frame_id],
            "dof_pos": self.dof_pos[frame_id],
            "dof_vel": self.dof_vel[frame_id],
        }

    def policy_targets(self, control_step: int, control_dt: float) -> np.ndarray:
        """Build Isaac's 180 target values plus two clamped phase values."""

        current_frame = self.frame_id(control_step, control_dt)
        targets: list[np.ndarray] = []
        for offset in FUTURE_FRAME_OFFSETS:
            ref = self.frame(current_frame + offset)
            # Isaac targets are physical-joint reference values.  The 90-D
            # proprioception is motor space; do not transform this term again.
            targets.append(
                np.concatenate(
                    (
                        ref["dof_pos"],
                        ref["dof_vel"],
                        ref["root_pos"][2:3],
                        ref["root_quat_wxyz"],
                        ref["root_lin_vel"],
                    ),
                    dtype=np.float32,
                )
            )
        phase = current_frame / max(self.num_frames - 1, 1)
        targets.append(np.asarray((np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)), dtype=np.float32))
        return np.concatenate(targets, dtype=np.float32)
