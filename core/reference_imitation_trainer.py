"""MuJoCo DAgger distillation for Humanoid Light reference imitation.

The Isaac ONNX remains the 276-D reference-motion teacher.  This trainer
learns a student using the user's normal locomotion observation contract plus
one explicit non-stacked normalized clip-progress observation.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from envs.build import build_env
from envs.humanoid_light_v2.reference_policy_manifest import (
    PHASE_LAYOUT,
    SCHEMA,
    manifest_path_for,
    sha256_file,
)

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


if nn is not None:
    class ReferenceTrajectoryStudentNet(nn.Module):
        """A trajectory-conditioned student with a learned action prior.

        The frozen action prior is populated solely from one offline teacher
        rollout.  It is *not* a teacher graph and no 182-D target is present
        in the exported ONNX.  The residual MLP sees the ordinary 90-D
        proprioception followed by normalized trajectory progress. It is trained
        with DAgger labels, so it can correct deviations from the nominal
        trajectory instead of merely replaying the NPZ pose.
        """

        def __init__(
            self,
            action_prior: np.ndarray,
            nominal_proprio: np.ndarray,
            input_dim: int,
            progress_index: int,
            action_dim: int = 26,
            hidden_dim: int = 256,
            fourier_frequencies: int = 12,
            residual_gate_scale: float = 0.05,
        ):
            super().__init__()
            prior = np.asarray(action_prior, dtype=np.float32)
            nominal = np.asarray(nominal_proprio, dtype=np.float32)
            if prior.ndim != 2 or prior.shape[1] != int(action_dim) or prior.shape[0] < 1:
                raise ValueError(f"Expected [num_control_steps, {action_dim}] action prior, got {prior.shape}.")
            if nominal.shape != (prior.shape[0], 90):
                raise ValueError(
                    "Expected nominal proprio table with shape "
                    f"({prior.shape[0]}, 90), got {nominal.shape}."
                )
            self.input_dim = int(input_dim)
            self.progress_index = int(progress_index)
            self.action_dim = int(action_dim)
            if not 0 <= self.progress_index < self.input_dim:
                raise ValueError(
                    f"Progress index must be in [0, {self.input_dim}), got {self.progress_index}."
                )
            self.num_control_steps = int(prior.shape[0])
            self.fourier_frequencies = max(0, int(fourier_frequencies))
            self.residual_gate_scale = max(float(residual_gate_scale), 1.0e-6)
            half_hidden = max(1, int(hidden_dim) // 2)
            lifted_dim = self.input_dim + 2 * self.fourier_frequencies
            self.feedback = nn.Sequential(
                nn.Linear(lifted_dim, int(hidden_dim)),
                nn.ELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ELU(),
                nn.Linear(int(hidden_dim), half_hidden),
                nn.ELU(),
                nn.Linear(half_hidden, self.action_dim),
            )
            # Before DAgger has demonstrated a needed correction, deployment
            # exactly follows the independently collected teacher-action
            # prior.  This makes stability testing meaningful from the first
            # epoch instead of relying on an arbitrarily initialized MLP.
            final_layer = self.feedback[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
            self.register_buffer("action_prior", torch.from_numpy(prior))
            self.register_buffer("nominal_proprio", torch.from_numpy(nominal))
            self.register_buffer(
                "fourier_frequency_values",
                torch.arange(1, self.fourier_frequencies + 1, dtype=torch.float32),
            )

        def forward(self, observation):
            progress = torch.clamp(observation[..., self.progress_index], 0.0, 1.0)
            index = torch.round(progress * float(max(self.num_control_steps - 1, 1))).to(torch.long)
            index = torch.clamp(index, 0, self.num_control_steps - 1)
            prior = self.action_prior.index_select(0, index.reshape(-1)).reshape(*index.shape, self.action_dim)
            nominal = self.nominal_proprio.index_select(0, index.reshape(-1)).reshape(*index.shape, 90)
            proprio_error = observation[..., :90] - nominal
            if self.fourier_frequencies <= 0:
                lifted = torch.cat((proprio_error, observation[..., 90:]), dim=-1)
            else:
                angle = 2.0 * torch.pi * progress.unsqueeze(-1) * self.fourier_frequency_values
                lifted = torch.cat((proprio_error, observation[..., 90:], torch.sin(angle), torch.cos(angle)), dim=-1)
            # This gate is structural, not a loss preference: when the state
            # equals the nominal MuJoCo trajectory, the correction is exactly
            # zero.  DAgger can consequently learn recovery actions without
            # corrupting the already-validated nominal solution.
            deviation = torch.linalg.vector_norm(proprio_error, ord=2, dim=-1, keepdim=True)
            gate = torch.clamp(deviation / self.residual_gate_scale, 0.0, 1.0)
            return prior + gate * self.feedback(lifted)
else:
    ReferenceTrajectoryStudentNet = None


class ReferenceImitationTrainer:
    """Collect on-policy DAgger labels in MuJoCo and export a student ONNX."""

    ACTION_DIM = 26
    TEACHER_OBS_DIM = 276

    def __init__(
        self,
        repo_root: str,
        settings: dict,
        log_callback: Callable[[str], None] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ):
        self.repo_root = str(repo_root)
        self.settings = dict(settings or {})
        self._log_callback = log_callback
        self._stop_callback = stop_callback

    def _log(self, message: str):
        if self._log_callback is not None:
            self._log_callback(str(message))
        else:
            print(message)

    def _stopped(self) -> bool:
        return bool(self._stop_callback is not None and self._stop_callback())

    @staticmethod
    def _require_dependencies():
        if torch is None or nn is None or ReferenceTrajectoryStudentNet is None:
            raise RuntimeError("Reference locomotion distillation requires the 'torch' package.")
        try:
            import onnx  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Reference locomotion ONNX export requires the 'onnx' package.") from exc

    @staticmethod
    def _onnx_shape(meta) -> int | None:
        shape = list(meta.shape)
        if not shape:
            return None
        return shape[-1] if isinstance(shape[-1], int) else None

    @staticmethod
    def _reference_duration_from_config(config: dict) -> float:
        """Read the NPZ duration for the teacher's one-clip action prior."""
        reference_cfg = config.get("reference_motion", {}) or {}
        motion_value = str(reference_cfg.get("motion", "")).strip()
        motion_path = Path(motion_value).expanduser()
        if not motion_path.is_file():
            motion_path = Path(str(reference_cfg.get("directory", ""))).expanduser() / motion_value
        try:
            with np.load(motion_path, allow_pickle=False) as motion:
                frames = int(np.asarray(motion["base_frame_pos"]).shape[0])
                fps = float(np.asarray(motion["fps"]).item())
            if frames < 1 or fps <= 0.0:
                raise ValueError("invalid frame/fps values")
            return frames / fps
        except Exception as exc:
            raise RuntimeError(f"Could not read duration from reference NPZ '{motion_path}': {exc}") from exc

    def _load_teacher(self, teacher_path: str):
        path = Path(teacher_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Reference teacher ONNX not found: {path}")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise RuntimeError("Reference teacher must be a single-input ONNX MLP.")
        input_dim = self._onnx_shape(inputs[0])
        output_dim = self._onnx_shape(outputs[0])
        if input_dim != self.TEACHER_OBS_DIM or output_dim != self.ACTION_DIM:
            raise RuntimeError(
                "Selected teacher does not match Humanoid Light reference contract: "
                f"expected [*, 276] -> [*, 26], got {list(inputs[0].shape)} -> {list(outputs[0].shape)}."
            )
        return path, session, inputs[0].name, [output.name for output in outputs]

    @staticmethod
    def _teacher_action(session, input_name: str, output_names: list[str], teacher_state: np.ndarray) -> np.ndarray:
        observation = np.asarray(teacher_state, dtype=np.float32).reshape(1, -1)
        output = session.run(output_names, {input_name: observation})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.shape != (ReferenceImitationTrainer.ACTION_DIM,):
            raise RuntimeError(f"Teacher ONNX returned {action.shape}; expected ({ReferenceImitationTrainer.ACTION_DIM},).")
        return action

    @staticmethod
    def _find_reference_wrapper(env):
        current = env
        while current is not None:
            if current.__class__.__name__ == "ReferenceMotionTargetWrapper":
                return current
            current = getattr(current, "env", None)
        raise RuntimeError("Reference teacher environment is missing ReferenceMotionTargetWrapper.")

    @staticmethod
    def _find_wrapper(env, class_name: str):
        current = env
        while current is not None:
            if current.__class__.__name__ == class_name:
                return current
            current = getattr(current, "env", None)
        return None

    @staticmethod
    def _observation_contract(settings: dict) -> dict:
        """The settings subset that determines a student input vector."""
        settings = settings or {}
        stacked = list(settings.get("stacked_obs_order", []) or [])
        non_stacked = list(settings.get("non_stacked_obs_order", []) or [])
        names = list(dict.fromkeys(stacked + non_stacked))
        per_observation = {}
        for name in names:
            value = settings.get(name) or {}
            per_observation[name] = {
                "freq": int(value.get("freq", 50)),
                "scale": float(value.get("scale", 1.0)),
            }
        return {
            "stack_size": int(settings.get("stack_size", 1)),
            "stacked_obs_order": stacked,
            "non_stacked_obs_order": non_stacked,
            "command_dim": int(settings.get("command_dim", 0)),
            "per_observation": per_observation,
        }

    @staticmethod
    def _reference_progress_frames(settings: dict) -> int:
        try:
            frames = int((settings or {}).get("reference_progress_frames", 0))
        except (TypeError, ValueError):
            frames = 0
        if frames < 1:
            raise RuntimeError(
                "Set Observation Settings → Reference Progress Frames to a positive 50 Hz frame count "
                "(for example, 779)."
            )
        return frames

    @classmethod
    def _student_layout(cls, teacher_env, student_config: dict) -> dict:
        """Map a 276-D teacher state into the configured student state.

        The teacher preserves its 94-D base contract.  The student takes its
        first 90 locomotion values and appends the one generated observation
        selected in Non-Stacked Observation.  Its final input is always 91-D.
        """
        settings = student_config.get("settings", student_config.get("observation", {})) or {}
        stacked = list(settings.get("stacked_obs_order", []) or [])
        non_stacked = list(settings.get("non_stacked_obs_order", []) or [])
        if "reference_progress" in stacked or non_stacked.count("reference_progress") != 1:
            raise RuntimeError(
                "Add 'reference_progress' exactly once to Observation Settings → Non-Stacked Observation."
            )
        if int(settings.get("command_dim", -1)) != 0:
            raise RuntimeError(
                "Set Observation Settings → Command Dim to 0 for the 91-D reference student."
            )
        phase_frame_count = cls._reference_progress_frames(settings)
        progress_cfg = settings.get("reference_progress") or {}
        if int(progress_cfg.get("freq", 0)) != 50 or not np.isclose(
            float(progress_cfg.get("scale", float("nan"))), 1.0
        ):
            raise RuntimeError("'reference_progress' must use freq=50 and scale=1.0.")
        reference_wrapper = cls._find_reference_wrapper(teacher_env)
        teacher_control_steps = max(
            1, int(np.floor(reference_wrapper.motion.duration_s / reference_wrapper.control_dt + 1.0e-6))
        )
        if phase_frame_count != teacher_control_steps:
            raise RuntimeError(
                "Reference Progress Frames must equal the selected teacher clip's 50 Hz control-frame count: "
                f"configured={phase_frame_count}, teacher={teacher_control_steps}."
            )
        command_wrapper = cls._find_wrapper(teacher_env, "CommandWrapper")
        state_builder = cls._find_wrapper(teacher_env, "StateBuildWrapper")
        if command_wrapper is None or state_builder is None:
            raise RuntimeError("Reference teacher environment is missing its standard observation/command wrappers.")
        base_dim = int(reference_wrapper.env.state_dim)
        command_dim = int(command_wrapper.command_dim)
        pre_command_dim = base_dim - command_dim
        if pre_command_dim != 90:
            raise RuntimeError(
                "The 276-D Isaac teacher requires the standard 90-D locomotion observation before its four commands; "
                f"current teacher base has {pre_command_dim} dimensions."
            )
        stacked_dim = int(state_builder.stack_size * state_builder._stacked_obs_dim)
        progress_index = stacked_dim
        for name in non_stacked:
            if name == "reference_progress":
                break
            try:
                progress_index += int(state_builder.obs_to_dim[name])
            except KeyError as exc:
                raise RuntimeError(f"Unknown observation before reference_progress: '{name}'.") from exc
        if not 0 <= progress_index <= pre_command_dim:
            raise RuntimeError(
                "Reference Progress placement does not fit the 90-D teacher observation. "
                "Keep the teacher-compatible locomotion observation layout and add only Reference Progress."
            )
        if progress_index != pre_command_dim:
            raise RuntimeError(
                "Place Reference Progress last in Non-Stacked Observation so it follows the standard "
                "90-D locomotion observation as the final student input."
            )
        expected_dim = pre_command_dim + 1
        return {
            "base_dim": base_dim,
            "command_dim": command_dim,
            "pre_command_dim": pre_command_dim,
            "progress_index": progress_index,
            "input_dim": expected_dim,
            "phase_frame_count": phase_frame_count,
        }

    @staticmethod
    def _student_observation(teacher_state: np.ndarray, reference_wrapper, layout: dict) -> np.ndarray:
        teacher_state = np.asarray(teacher_state, dtype=np.float32).reshape(-1)
        base_dim = int(layout["base_dim"])
        if teacher_state.size < base_dim:
            raise RuntimeError(f"Teacher state is shorter than its base observation: {teacher_state.shape}.")
        pre_command_dim = int(layout["pre_command_dim"])
        progress_index = int(layout["progress_index"])
        progress = reference_wrapper.motion.trajectory_progress(
            reference_wrapper.control_step,
            reference_wrapper.control_dt,
        )
        base = teacher_state[:base_dim]
        observation = np.concatenate(
            (base[:progress_index], progress, base[progress_index:pre_command_dim]),
            dtype=np.float32,
        )
        if observation.shape != (int(layout["input_dim"]),):
            raise RuntimeError(
                f"Distilled student observation must be {layout['input_dim']}-D, got {observation.shape}."
            )
        return observation

    def _collect_nominal_teacher_trajectory(
        self, config: dict, student_config: dict, session, input_name: str, output_names: list[str]
    ):
        """Collect the complete no-noise teacher trajectory exactly once.

        The resulting per-control-tick action prior is part of the student
        parameters.  It lets the student retain the teacher's nominal stable
        solution while DAgger learns a proprioceptive residual around it.
        """

        env = None
        try:
            env = build_env(config)
            reference_wrapper = self._find_reference_wrapper(env)
            student_layout = self._student_layout(env, student_config)
            state, _ = env.reset()
            table: list[np.ndarray] = []
            inputs: list[np.ndarray] = []
            rows: list[dict] = []
            terminated = truncated = False
            while not (terminated or truncated):
                if reference_wrapper.control_step != len(table):
                    raise RuntimeError(
                        "Reference control clock is not monotonic while collecting the trajectory prior: "
                        f"step={reference_wrapper.control_step}, table={len(table)}."
                    )
                teacher_action = self._teacher_action(session, input_name, output_names, state)
                table.append(teacher_action)
                inputs.append(self._student_observation(state, reference_wrapper, student_layout))
                state, terminated, truncated, info = env.step(teacher_action)
                rows.append(dict(info))
            summary = self._summarize_rollout(rows, terminated, truncated)
            if not summary["stable"]:
                raise RuntimeError(
                    "The selected reference teacher did not complete its nominal MuJoCo rollout; "
                    f"cannot construct a stable student prior: {summary}"
                )
            expected_steps = int(np.floor(reference_wrapper.motion.duration_s / reference_wrapper.control_dt + 1.0e-6))
            if len(table) != expected_steps:
                raise RuntimeError(
                    "Nominal teacher trajectory length does not match the reference control clock: "
                    f"got {len(table)}, expected {expected_steps}."
                )
            return (
                np.asarray(table, dtype=np.float32),
                np.asarray(inputs, dtype=np.float32),
                summary,
                student_layout,
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _collect_dagger_round(
        self,
        config: dict,
        student_config: dict,
        session,
        input_name: str,
        output_names: list[str],
        model,
        samples: int,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Collect on-policy configured-student states, labeled only by the teacher."""

        env = None
        try:
            env = build_env(config)
            reference_wrapper = self._find_reference_wrapper(env)
            student_layout = self._student_layout(env, student_config)
            state, _ = env.reset()
            inputs: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            collected = 0
            terminated = truncated = False
            while collected < samples and not self._stopped():
                teacher_action = self._teacher_action(session, input_name, output_names, state)
                student_state = self._student_observation(state, reference_wrapper, student_layout)
                inputs.append(student_state)
                labels.append(teacher_action)
                model.eval()
                with torch.no_grad():
                    student_action = model(torch.from_numpy(student_state).unsqueeze(0)).cpu().numpy()[0]
                rollout_action = float(beta) * teacher_action + (1.0 - float(beta)) * student_action
                state, terminated, truncated, _ = env.step(np.asarray(rollout_action, dtype=np.float32))
                collected += 1
                if terminated or truncated:
                    state, _ = env.reset()
                    terminated = truncated = False
            return np.asarray(inputs, dtype=np.float32), np.asarray(labels, dtype=np.float32), collected
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _train_epoch_block(self, model, inputs: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int, lr: float, rng):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        # Small raw-action errors are dynamically significant after the PD
        # target transform.  MSE keeps high-error joints from being flattened
        # by Huber's linear tail during the final imitation fit.
        criterion = nn.MSELoss()
        x = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
        y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        total_samples = int(x.shape[0])
        last_loss = float("nan")
        for epoch in range(max(1, int(epochs))):
            if self._stopped():
                break
            order = rng.permutation(total_samples)
            weighted_loss = 0.0
            seen = 0
            for start in range(0, total_samples, max(1, int(batch_size))):
                indices = order[start:start + max(1, int(batch_size))]
                xb = x[indices]
                yb = y[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                count = int(indices.size)
                weighted_loss += float(loss.detach().item()) * count
                seen += count
            last_loss = weighted_loss / max(1, seen)
            self._log(f"[reference-distill] train epoch {epoch + 1}/{epochs}: mse={last_loss:.6f}")
        return last_loss

    def _export(self, model, output_path: Path, input_dim: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        dummy = torch.zeros((1, int(input_dim)), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_dim = self._onnx_shape(session.get_inputs()[0])
        output_dim = self._onnx_shape(session.get_outputs()[0])
        if input_dim != int(model.input_dim) or output_dim != self.ACTION_DIM:
            raise RuntimeError(f"Exported student ONNX shape is invalid: {input_dim} -> {output_dim}.")

    @staticmethod
    def _distilled_student_config(student_config: dict, reset_perturbation: bool = False) -> dict:
        config = copy.deepcopy(student_config)
        reference_cfg = config.setdefault("reference_motion", {})
        reference_cfg["inference_mode"] = "distilled_locomotion"
        reference_cfg["reset_perturbation"] = bool(reset_perturbation)
        return config

    @staticmethod
    def _summarize_rollout(rows: list[dict], terminated: bool, truncated: bool) -> dict:
        if rows:
            joint_rmse = np.asarray([row["reference_joint_pos_rmse"] for row in rows], dtype=np.float64)
            root_height = np.asarray([abs(row["reference_root_height_error"]) for row in rows], dtype=np.float64)
            orientation = np.asarray([row["reference_root_orientation_error_rad"] for row in rows], dtype=np.float64)
        else:
            joint_rmse = root_height = orientation = np.zeros((0,), dtype=np.float64)
        return {
            "stable": bool((not terminated) and truncated),
            "steps": int(len(rows)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mean_joint_pos_rmse": float(joint_rmse.mean()) if joint_rmse.size else float("inf"),
            "max_joint_pos_rmse": float(joint_rmse.max()) if joint_rmse.size else float("inf"),
            "max_root_height_error": float(root_height.max()) if root_height.size else float("inf"),
            "max_root_orientation_error_rad": float(orientation.max()) if orientation.size else float("inf"),
        }

    def _evaluate_policy_stability(self, student_config: dict, policy_fn, reset_perturbation: bool = False) -> dict:
        env = None
        try:
            env = build_env(self._distilled_student_config(student_config, reset_perturbation=reset_perturbation))
            state, _ = env.reset()
            rows: list[dict] = []
            terminated = truncated = False
            while not (terminated or truncated):
                action = np.asarray(policy_fn(state), dtype=np.float32).reshape(-1)
                if action.shape != (self.ACTION_DIM,):
                    raise RuntimeError(f"Stability-gate policy returned {action.shape}; expected ({self.ACTION_DIM},).")
                state, terminated, truncated, info = env.step(action)
                rows.append(dict(info))
            return self._summarize_rollout(rows, terminated, truncated)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _evaluate_model_stability(self, config: dict, model, reset_perturbation: bool = False) -> dict:
        model.eval()

        def predict(state):
            with torch.no_grad():
                return model(torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)).cpu().numpy()[0]

        return self._evaluate_policy_stability(config, predict, reset_perturbation=reset_perturbation)

    def _evaluate_onnx_stability(self, config: dict, policy_path: Path, reset_perturbation: bool = False) -> dict:
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        def predict(state):
            output = session.run(output_names, {input_name: np.asarray(state, dtype=np.float32).reshape(1, -1)})[0]
            return np.asarray(output, dtype=np.float32).reshape(-1)

        return self._evaluate_policy_stability(config, predict, reset_perturbation=reset_perturbation)

    def _evaluate_onnx_stability_trials(self, config: dict, policy_path: Path, trials: int, seed: int) -> dict:
        """Gate export on deterministic nominal and reset-perturbation tests."""

        nominal = self._evaluate_onnx_stability(config, policy_path, reset_perturbation=False)
        perturbed: list[dict] = []
        for trial in range(max(1, int(trials))):
            # Reference reset noise is sampled through NumPy.  Seed each trial
            # explicitly so an exported manifest records a reproducible gate.
            np.random.seed(int(seed) + 10_000 + trial)
            perturbed.append(self._evaluate_onnx_stability(config, policy_path, reset_perturbation=True))
        return {
            "stable": bool(nominal["stable"] and all(row["stable"] for row in perturbed)),
            "nominal": nominal,
            "reset_perturbation_trials": perturbed,
            "reset_perturbation_trial_count": len(perturbed),
            "reset_perturbation_seed_base": int(seed) + 10_000,
        }

    def train_and_export(self) -> dict:
        self._require_dependencies()
        teacher_config = copy.deepcopy(self.settings.get("teacher_config", {}))
        student_config = copy.deepcopy(self.settings.get("student_config", {}))
        teacher_policy_path, teacher_session, input_name, output_names = self._load_teacher(
            str(self.settings.get("teacher_policy_path", ""))
        )
        if teacher_config.get("env", {}).get("id") != "humanoid_light_v2":
            raise RuntimeError("Reference locomotion distillation is available only for humanoid_light_v2.")
        if student_config.get("env", {}).get("id") != "humanoid_light_v2":
            raise RuntimeError("Student observation settings must belong to humanoid_light_v2.")
        reference_cfg = teacher_config.setdefault("reference_motion", {})
        if not bool(reference_cfg.get("enabled", False)):
            raise RuntimeError("Enable Reference Imitation and select a motion before distilling a locomotion policy.")
        reference_cfg["inference_mode"] = "teacher"
        teacher_config.setdefault("env", {})["render"] = False
        teacher_config["env"]["render_mode"] = "none"
        # Training labels/prior cover exactly one teacher clip.  This is
        # deliberately independent of the user-visible Max Duration, which
        # may be longer during Start Test to inspect post-clip behaviour.
        teacher_config["env"]["max_duration"] = self._reference_duration_from_config(teacher_config)
        teacher_config["fine_tune"] = {"enabled": False, "ridge_lambda": 1e-4, "max_samples": 1}
        student_config.setdefault("env", {})["render"] = False
        student_config["env"]["render_mode"] = "none"
        student_config["fine_tune"] = {"enabled": False, "ridge_lambda": 1e-4, "max_samples": 1}
        student_config.setdefault("reference_motion", {})["inference_mode"] = "distilled_locomotion"
        teacher_motion_path = Path(str(reference_cfg.get("motion", ""))).expanduser()
        if not teacher_motion_path.is_file():
            teacher_motion_path = Path(str(reference_cfg.get("directory", ""))).expanduser() / teacher_motion_path
        if not teacher_motion_path.is_file():
            raise RuntimeError("Could not resolve the reference teacher clip used for distillation.")

        samples_per_round = max(1, int(self.settings.get("samples_per_round", 6000)))
        dagger_rounds = max(1, int(self.settings.get("dagger_rounds", 6)))
        epochs_per_round = max(1, int(self.settings.get("epochs_per_round", 25)))
        batch_size = max(1, int(self.settings.get("batch_size", 512)))
        hidden_dim = max(16, int(self.settings.get("hidden_dim", 512)))
        fourier_frequencies = max(0, int(self.settings.get("fourier_frequencies", 16)))
        learning_rate = float(self.settings.get("learning_rate", 5e-4))
        seed = int(self.settings.get("seed", 42))
        stability_trials = max(1, int(self.settings.get("stability_trials", 5)))
        residual_gate_scale = max(1.0e-6, float(self.settings.get("residual_gate_scale", 0.05)))
        torch_num_threads = max(1, int(self.settings.get("torch_num_threads", 4)))
        output_value = str(self.settings.get("output_path", "")).strip()
        if not output_value:
            raise RuntimeError("Choose an output ONNX path for the distilled policy.")
        output_path = Path(output_value).expanduser().resolve()
        if output_path.suffix.lower() != ".onnx":
            output_path = output_path.with_suffix(".onnx")

        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(torch_num_threads)
        action_prior, nominal_inputs, nominal_stability, student_layout = self._collect_nominal_teacher_trajectory(
            teacher_config, student_config, teacher_session, input_name, output_names
        )
        student_obs_dim = int(student_layout["input_dim"])
        self._log(
            f"[reference-distill] teacher 276→26; student {student_obs_dim}→26; "
            f"progress index={student_layout['progress_index']}, phase=[reference_progress], scale=[1]."
        )
        self._log(
            "[reference-distill] collected nominal teacher trajectory prior: "
            f"{len(action_prior)} control ticks; max_joint_rmse={nominal_stability['max_joint_pos_rmse']:.4f}."
        )
        model = ReferenceTrajectoryStudentNet(
            action_prior,
            nominal_inputs[:, :90],
            student_obs_dim,
            int(student_layout["progress_index"]),
            self.ACTION_DIM,
            hidden_dim=hidden_dim,
            fourier_frequencies=fourier_frequencies,
            residual_gate_scale=residual_gate_scale,
        )
        inputs: list[np.ndarray] = [nominal_inputs]
        labels: list[np.ndarray] = [action_prior]
        history: list[dict] = []
        dagger_config = copy.deepcopy(teacher_config)
        dagger_config.setdefault("reference_motion", {})["reset_perturbation"] = True
        for round_index in range(dagger_rounds):
            if self._stopped():
                break
            if round_index > 0:
                # The prior makes the first student rollout nominally stable.
                # Decrease the teacher mixture thereafter so the residual sees
                # the state distribution induced by the deployed student.
                beta = max(0.0, 1.0 - round_index / max(1, dagger_rounds - 1))
                round_inputs, round_labels, collected = self._collect_dagger_round(
                    dagger_config,
                    student_config,
                    teacher_session,
                    input_name,
                    output_names,
                    model,
                    samples_per_round,
                    beta,
                )
                if collected:
                    inputs.append(round_inputs)
                    labels.append(round_labels)
            else:
                beta = 1.0
                collected = len(nominal_inputs)
            x = np.concatenate(inputs, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(labels, axis=0).astype(np.float32, copy=False)
            loss = self._train_epoch_block(model, x, y, epochs_per_round, batch_size, learning_rate, rng)
            history.append({
                "round": round_index + 1,
                "beta": float(beta),
                "collected": int(collected),
                "total_samples": int(x.shape[0]),
                "mse_loss": float(loss),
            })
            self._log(
                f"[reference-distill] DAgger {round_index + 1}/{dagger_rounds}: "
                f"beta={beta:.2f}, collected={collected}, total={x.shape[0]}."
            )

        if not inputs or self._stopped():
            raise RuntimeError("Reference distillation stopped before collecting any teacher labels.")
        # The policy selected by Start Test must be the learned student
        # itself.  Export it under a diagnostic candidate name first and run
        # the ONNX (not the PyTorch model) through the full MuJoCo clip before
        # ever replacing the user's deployable output path.
        candidate_path = output_path.with_name(f"{output_path.stem}.candidate.onnx")
        self._export(model, candidate_path, student_obs_dim)
        candidate_stability = self._evaluate_onnx_stability_trials(
            student_config, candidate_path, stability_trials, seed
        )
        self._log(
            "[reference-distill] learned-student ONNX gate: "
            f"nominal_steps={candidate_stability['nominal']['steps']}, "
            f"nominal_terminated={candidate_stability['nominal']['terminated']}, "
            f"perturbed_passes={sum(row['stable'] for row in candidate_stability['reset_perturbation_trials'])}/"
            f"{candidate_stability['reset_perturbation_trial_count']}."
        )
        if not candidate_stability["stable"]:
            raise RuntimeError(
                f"Learned {student_obs_dim}-D student ONNX did not pass the full MuJoCo reference rollout; "
                "no deployable locomotion policy was exported. "
                f"Diagnostic candidate retained at: {candidate_path}\n"
                f"Stability result: {candidate_stability}"
            )
        os.replace(candidate_path, output_path)
        deployment_strategy = "learned_distilled_trajectory_residual"
        deployment_stability = candidate_stability

        resolved_reference_cfg = teacher_config.get("reference_motion", {}) or {}
        reference_path = Path(str(resolved_reference_cfg.get("motion", ""))).expanduser().resolve()
        if not reference_path.is_file():
            raise RuntimeError("Could not resolve the reference motion used for distillation.")
        manifest = {
            "schema": SCHEMA,
            "mode": "distilled_locomotion",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "onnx_path": str(output_path),
            "onnx_sha256": sha256_file(output_path),
            "obs_dim": student_obs_dim,
            "action_dim": self.ACTION_DIM,
            "proprio_dim": 90,
            "phase_scale": 1.0,
            "phase_layout": PHASE_LAYOUT,
            "phase_input_mode": "user_configured_frame_count",
            "phase_frame_count": int(student_layout["phase_frame_count"]),
            "observation_contract": self._observation_contract(
                student_config.get("settings", student_config.get("observation", {}))
            ),
            "architecture": {
                "type": "trajectory_prior_plus_proprioceptive_residual_mlp",
                "hidden_dim": hidden_dim,
                "fourier_frequencies": fourier_frequencies,
                "action_prior_control_steps": int(action_prior.shape[0]),
                "residual_gate": "clamp(||proprio - nominal_proprio(control_progress)||_2 / scale, 0, 1)",
                "residual_gate_scale": residual_gate_scale,
                "torch_num_threads": torch_num_threads,
            },
            "distillation_phase_label": "teacher_control_step/(teacher_clip_control_steps-1), clamped",
            "reference_motion": {
                "path": str(reference_path),
                "filename": reference_path.name,
                "sha256": sha256_file(reference_path),
            },
            "teacher": {"path": str(teacher_policy_path), "sha256": sha256_file(teacher_policy_path)},
            "action_scale": list(teacher_config.get("action_scales", [])),
            "training": {
                "samples_per_round": samples_per_round,
                "dagger_rounds": dagger_rounds,
                "epochs_per_round": epochs_per_round,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "fourier_frequencies": fourier_frequencies,
                "learning_rate": learning_rate,
                "seed": seed,
                "stability_trials": stability_trials,
                "residual_gate_scale": residual_gate_scale,
                "history": history,
                "stopped": self._stopped(),
            },
            "deployment_strategy": deployment_strategy,
            "stability_gate": deployment_stability,
        }
        manifest_path = manifest_path_for(output_path)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        checkpoint_path = output_path.with_suffix(".pt")
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": student_obs_dim,
            "action_dim": self.ACTION_DIM,
            "hidden_dim": hidden_dim,
            "fourier_frequencies": fourier_frequencies,
            "action_prior_control_steps": int(action_prior.shape[0]),
            "residual_gate_scale": residual_gate_scale,
            "manifest_path": str(manifest_path),
        }, checkpoint_path)
        self._log(f"[reference-distill] exported ONNX: {output_path}")
        return {
            "onnx_path": str(output_path),
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "samples": int(sum(len(batch) for batch in inputs)),
            "history": history,
            "stopped": self._stopped(),
            "deployment_strategy": deployment_strategy,
            "stability_gate": deployment_stability,
        }
