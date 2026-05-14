import json
import os
import shutil
import time
import copy
from dataclasses import dataclass

import numpy as np
import yaml

from core.moe_trainer import OnnxExpertPolicy
from envs.build import build_env

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


class HomingDataset(Dataset):
    def __init__(self, dataset_paths):
        x_parts = []
        y_parts = []
        self.dataset_paths = list(dataset_paths or [])
        if not self.dataset_paths:
            raise RuntimeError("Select at least one Homing dataset.")
        for path in self.dataset_paths:
            with np.load(path) as payload:
                input_key = "state" if "state" in payload.files else "input"
                x_parts.append(payload[input_key].astype(np.float32))
                y_parts.append(payload["action_label"].astype(np.float32))
        self.x = np.concatenate(x_parts, axis=0)
        self.y = np.concatenate(y_parts, axis=0)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, index):
        return {
            "input": torch.from_numpy(self.x[index]),
            "action": torch.from_numpy(self.y[index]),
        }


class HomingPolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, action_mask=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        mask = torch.ones((int(action_dim),), dtype=torch.float32)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.float32).reshape(-1)
            if int(mask.numel()) != int(action_dim):
                mask = torch.ones((int(action_dim),), dtype=torch.float32)
        self.register_buffer("action_mask", mask)

    def forward(self, x):
        return self.net(x) * self.action_mask


class HomingActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, init_log_std=-3.0, action_mask=None):
        super().__init__()
        self.actor = HomingPolicyNet(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_mask)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std), dtype=torch.float32))

    def forward(self, x):
        return self.actor(x)

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def distribution(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)
        return torch.distributions.Normal(mean, std)

    def act(self, x):
        dist = self.distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(x)
        return action, log_prob, entropy, value


@dataclass
class HomingArtifacts:
    run_dir: str
    checkpoint_path: str
    onnx_path: str
    manifest_path: str
    summary_path: str


class HomingTrainer:
    def __init__(self, repo_root, settings, log_callback=None, stop_callback=None, switch_callback=None, command_callback=None):
        self.repo_root = os.path.abspath(repo_root)
        self.settings = dict(settings or {})
        self.env_id = str(self.settings.get("env_id", ""))
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.switch_callback = switch_callback
        self.command_callback = command_callback

    def _log(self, message):
        text = str(message)
        if not text.endswith("\n"):
            text += "\n"
        if callable(self.log_callback):
            self.log_callback(text)
        print(text, end="")

    def _stop_requested(self):
        return bool(callable(self.stop_callback) and self.stop_callback())

    def _switch_requested(self):
        return bool(callable(self.switch_callback) and self.switch_callback())

    def _current_command(self, command_dim):
        if callable(self.command_callback):
            try:
                values = self.command_callback()
                return self._pad_or_trim(values, command_dim, fill=0.0)
            except Exception:
                pass
        return self._pad_or_trim(self.settings.get("command_values", []), command_dim, fill=0.0)

    def _require_torch(self):
        if torch is None or nn is None or DataLoader is None:
            raise RuntimeError("Homing policy training requires torch. Install dependencies from requirements.txt.")

    def _load_yaml(self, rel_path):
        with open(os.path.join(self.repo_root, rel_path), "r", encoding="utf-8") as handle:
            return yaml.full_load(handle)

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _as_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return bool(default)

    def _make_settings_cfg(self, env_cfg):
        raw = env_cfg.get("settings", env_cfg) if isinstance(env_cfg, dict) else {}
        cmd_cfg = raw.get("command", {}) if isinstance(raw.get("command", {}), dict) else {}
        obs_scales = raw.get("obs_scales", {}) if isinstance(raw.get("obs_scales", {}), dict) else {}
        stacked = list(raw.get("stacked_obs_order", []) or [])
        non_stacked = list(raw.get("non_stacked_obs_order", []) or [])
        obs_names = [
            "dof_pos",
            "dof_vel",
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel",
            "projected_gravity",
            "height_map",
            "last_action",
        ]
        settings_cfg = {
            "stacked_obs_order": stacked,
            "non_stacked_obs_order": non_stacked,
            "stack_size": self._as_int(raw.get("stack_size", 3), 3),
        }
        for obs_name in obs_names:
            if obs_name in stacked or obs_name in non_stacked:
                settings_cfg[obs_name] = {
                    "freq": 50,
                    "scale": self._as_float(obs_scales.get(obs_name, 1.0), 1.0),
                }
            else:
                settings_cfg[obs_name] = None
        cmd_dim = self._as_int(cmd_cfg.get("command_dim", raw.get("command_dim", 6)), 6)
        settings_cfg["command_dim"] = cmd_dim
        raw_scales = raw.get("command_scales", {}) if isinstance(raw.get("command_scales", {}), dict) else {}
        settings_cfg["command_scales"] = {
            str(i): self._as_float(raw_scales.get(str(i), raw_scales.get(i, 1.0)), 1.0)
            for i in range(cmd_dim)
        }
        height_map_cfg = raw.get("height_map", {}) if isinstance(raw.get("height_map", {}), dict) else {}
        settings_cfg["height_map"] = {
            "freq": 50,
            "scale": self._as_float(obs_scales.get("height_map", 1.0), 1.0),
            "size_x": self._as_float(height_map_cfg.get("size_x", 1.0), 1.0),
            "size_y": self._as_float(height_map_cfg.get("size_y", 0.6), 0.6),
            "res_x": self._as_int(height_map_cfg.get("res_x", 15), 15),
            "res_y": self._as_int(height_map_cfg.get("res_y", 9), 9),
        }
        return settings_cfg

    def _base_config(self, terrain, render=False):
        env_table = self._load_yaml("config/env_table.yaml")
        random_table = self._load_yaml("config/random_table.yaml")
        env_cfg = dict(env_table[self.env_id])
        settings_cfg = self._make_settings_cfg(env_cfg)
        overrides = self.settings.get("env_overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}
        random_overrides = overrides.get("random", {})
        if not isinstance(random_overrides, dict):
            random_overrides = {}
        hardware = overrides.get("hardware", env_cfg.get("hardware", {}))
        action_scales = overrides.get("action_scales", env_cfg.get("action_scales", []))
        action_clippings = overrides.get("action_clippings", env_cfg.get("action_clippings", []))
        initial_positions = overrides.get("initial_positions", {"joints": {}})
        actuator = overrides.get("actuator", env_cfg.get("actuator", {}))
        return {
            "env": {
                "id": self.env_id,
                "terrain": terrain,
                "max_duration": 60.0,
                "position_command": False,
                "render": bool(render),
                "quiet": not bool(render),
            },
            "settings": settings_cfg,
            "observation": settings_cfg,
            "random": {
                "precision": random_overrides.get("precision", "medium"),
                "sensor_noise": random_overrides.get("sensor_noise", "none"),
                "init_noise": self._as_float(random_overrides.get("init_noise", 0.0), 0.0),
                "sliding_friction": self._as_float(random_overrides.get("sliding_friction", 0.8), 0.8),
                "torsional_friction": self._as_float(random_overrides.get("torsional_friction", 0.02), 0.02),
                "rolling_friction": self._as_float(random_overrides.get("rolling_friction", 0.01), 0.01),
                "friction_loss": self._as_float(random_overrides.get("friction_loss", 0.0), 0.0),
                "action_delay_prob": self._as_float(random_overrides.get("action_delay_prob", 0.0), 0.0),
                "mass_noise": self._as_float(random_overrides.get("mass_noise", 0.0), 0.0),
                "load": self._as_float(random_overrides.get("load", 0.0), 0.0),
            },
            "random_table": random_table["random_table"],
            "hardware": dict(hardware or {}),
            "action_scales": list(action_scales or []),
            "action_clippings": action_clippings or [],
            "initial_positions": initial_positions or {"joints": {}},
            "actuator": actuator or {},
        }

    def _dataset_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "dataset", "homing")

    def _weights_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "weights", "homing")

    def _make_artifacts(self, run_name="latest"):
        root = self._weights_root()
        os.makedirs(root, exist_ok=True)
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return HomingArtifacts(
            run_dir=run_dir,
            checkpoint_path=os.path.join(run_dir, "homing_policy.pt"),
            onnx_path=os.path.join(run_dir, "homing_policy.onnx"),
            manifest_path=os.path.join(run_dir, "homing_policy_manifest.json"),
            summary_path=os.path.join(run_dir, "train_summary.json"),
        )

    def _supervised_checkpoint_path(self):
        return os.path.join(self._weights_root(), "latest", "homing_policy_supervised.pt")

    def _ppo_checkpoint_path(self):
        return os.path.join(self._weights_root(), "latest", "homing_policy_ppo.pt")

    def _sample_command(self, rng, command_dim):
        def parse_vector(key, fallback):
            raw = self.settings.get(key, "")
            if isinstance(raw, str):
                values = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
            else:
                values = list(raw or [])
            parsed = []
            for value in values:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    pass
            if parsed:
                return self._pad_or_trim(parsed, command_dim, fill=fallback)
            return np.full((int(command_dim),), float(fallback), dtype=np.float32)

        fallback_lo = float(self.settings.get("command_min", -1.0))
        fallback_hi = float(self.settings.get("command_max", 1.0))
        lo = parse_vector("command_mins", fallback_lo)
        hi = parse_vector("command_maxs", fallback_hi)
        low = np.minimum(lo, hi)
        high = np.maximum(lo, hi)
        return rng.uniform(low, high).astype(np.float32)

    @staticmethod
    def _zero_command(command_dim):
        return np.zeros((int(command_dim),), dtype=np.float32)

    @staticmethod
    def _zero_base_velocity(env):
        try:
            data = env.get_data()
            if getattr(data, "qvel", None) is not None and data.qvel.size >= 6:
                data.qvel[:6] = 0.0
        except Exception:
            pass

    @staticmethod
    def _inject_applied_command(state, env):
        state = np.asarray(state, dtype=np.float32).copy()
        command = np.asarray(getattr(env, "applied_command", []), dtype=np.float32).reshape(-1)
        if command.size > 0 and state.size >= command.size:
            state[-command.size:] = command
        return state

    @staticmethod
    def _leaf_env(env):
        leaf = env
        while hasattr(leaf, "env"):
            leaf = leaf.env
        return leaf

    @staticmethod
    def _disable_leaf_termination(env):
        leaf = HomingTrainer._leaf_env(env)
        if hasattr(leaf, "_is_done"):
            try:
                leaf._is_done = lambda: False
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def _clear_wrapper_reset_flags(env):
        current = env
        while current is not None:
            if hasattr(current, "reset_flag"):
                try:
                    current.reset_flag = True
                except Exception:
                    pass
            current = getattr(current, "env", None)

    @staticmethod
    def _control_dt(env):
        leaf = HomingTrainer._leaf_env(env)
        dt = getattr(leaf, "dt_", None)
        frame_skip = getattr(leaf, "frame_skip", 1)
        try:
            return max(1e-6, float(dt) * float(frame_skip))
        except (TypeError, ValueError):
            return 0.02

    def _wheel_mask(self, env, action_dim):
        leaf = self._leaf_env(env)
        names = list(getattr(leaf, "initial_joint_names", []) or [])
        mask = np.zeros((int(action_dim),), dtype=bool)
        for i, name in enumerate(names[:action_dim]):
            if "wheel" in str(name).lower():
                mask[i] = True
        return mask

    def _action_output_mask(self, action_dim):
        mask = np.ones((int(action_dim),), dtype=np.float32)
        if not self.env_id:
            return mask
        env = None
        try:
            env = build_env(self._base_config("flat", render=False))
            wheel_mask = self._wheel_mask(env, action_dim)
            mask[wheel_mask] = 0.0
        except Exception:
            pass
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        return mask.astype(np.float32)

    @staticmethod
    def _pad_or_trim(values, dim, fill=0.0):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        out = np.full((int(dim),), float(fill), dtype=np.float32)
        n = min(out.size, arr.size)
        if n > 0:
            out[:n] = arr[:n]
        return out

    def _target_vector(self, key, dim, fill=0.0):
        raw = self.settings.get(key, "")
        if isinstance(raw, str):
            values = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
        else:
            values = list(raw or [])
        parsed = []
        for value in values:
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                pass
        return self._pad_or_trim(parsed, dim, fill=fill)

    def _current_joint_state(self, env, action_dim):
        last_obs = env.get_last_obs() or {}
        q_raw = np.asarray(last_obs.get("dof_pos", []), dtype=np.float32).reshape(-1)
        leaf = self._leaf_env(env)
        if hasattr(leaf, "_build_full_qpos_vector") and q_raw.size > 0:
            try:
                q_raw = np.asarray(leaf._build_full_qpos_vector(q_raw), dtype=np.float32).reshape(-1)
            except Exception:
                pass
        q = self._pad_or_trim(q_raw, action_dim)
        dq = self._pad_or_trim(last_obs.get("dof_vel", []), action_dim)
        if not np.any(q) or not np.any(dq):
            try:
                data = env.get_data()
                if not np.any(q):
                    q = self._pad_or_trim(np.asarray(data.qpos[7:], dtype=np.float32), action_dim)
                if not np.any(dq):
                    dq = self._pad_or_trim(np.asarray(data.qvel[6:], dtype=np.float32), action_dim)
            except Exception:
                pass
        return q, dq

    def _current_action_state(self, env, action_dim):
        last_obs = env.get_last_obs() or {}
        action = np.asarray(last_obs.get("last_action", []), dtype=np.float32).reshape(-1)
        if action.size == 0:
            try:
                action = np.asarray(getattr(self._leaf_env(env), "action", []), dtype=np.float32).reshape(-1)
            except Exception:
                action = np.zeros((0,), dtype=np.float32)
        return self._pad_or_trim(action, action_dim, fill=0.0)

    def _timed_trajectory_action(
        self,
        q_start,
        final_q,
        action_scales,
        step_index,
        total_steps,
        eligible_mask=None,
    ):
        q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
        final_q = np.asarray(final_q, dtype=np.float32).reshape(-1)
        dim = max(q_start.size, final_q.size)
        q_start = self._pad_or_trim(q_start, dim)
        final_q = self._pad_or_trim(final_q, dim)
        scales = self._pad_or_trim(action_scales, dim, fill=1.0)
        scales = np.maximum(np.abs(scales), 1e-6)

        progress = float(step_index + 1) / float(max(1, total_steps))
        progress = float(np.clip(progress, 0.0, 1.0))
        smooth = progress * progress * (3.0 - 2.0 * progress)
        start_action = q_start / scales
        final_action = final_q / scales
        action = start_action + smooth * (final_action - start_action)

        mask = np.ones((dim,), dtype=bool)
        if eligible_mask is not None:
            mask = self._pad_or_trim(np.asarray(eligible_mask, dtype=np.float32), dim, fill=0.0).astype(bool)
        action[~mask] = 0.0
        return action.astype(np.float32)

    @staticmethod
    def _blend_balance_action(stand_action, homing_action, balance_blend):
        blend = float(np.clip(balance_blend, 0.0, 1.0))
        stand_action = np.asarray(stand_action, dtype=np.float32).reshape(-1)
        homing_action = np.asarray(homing_action, dtype=np.float32).reshape(-1)
        dim = max(stand_action.size, homing_action.size)
        stand = np.zeros((dim,), dtype=np.float32)
        homing = np.zeros((dim,), dtype=np.float32)
        stand[:stand_action.size] = stand_action
        homing[:homing_action.size] = homing_action
        return (blend * stand + (1.0 - blend) * homing).astype(np.float32)

    def _make_input(self, state, q, dq, final_q, final_dq):
        _ = q, dq, final_q, final_dq
        return np.asarray(state, dtype=np.float32).reshape(-1).astype(np.float32)

    def collect(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for Homing data collection.")
        policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid stand-drive ONNX policy.")

        policy = OnnxExpertPolicy(policy_path)
        rng = np.random.default_rng(int(self.settings.get("seed", 42)))
        terrains = ["flat"]
        total_samples = max(1, int(self.settings.get("samples", 50000)))
        rollout_steps = max(1, int(self.settings.get("rollout_steps", 1000)))
        trajectory_seconds = max(0.02, float(self.settings.get("homing_trajectory_seconds", 3.0)))
        stand_warmup_steps = max(0, int(self.settings.get("homing_stand_warmup_steps", min(200, rollout_steps))))
        balance_blend = float(np.clip(float(self.settings.get("homing_balance_blend", 0.0)), 0.0, 1.0))
        per_terrain_target = int(np.ceil(total_samples / max(1, len(terrains))))

        run_name = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self._dataset_root(), run_name)
        os.makedirs(run_dir, exist_ok=True)
        self._log(f"[homing-collect] writing dataset to {run_dir}")

        input_all = []
        action_all = []
        state_all = []
        q_all = []
        dq_all = []
        terrain_all = []
        final_q = None
        final_dq = None

        for terrain in terrains:
            if self._stop_requested():
                self._log("[homing-collect] stop requested before next terrain.")
                break
            config = self._base_config(terrain)
            env = build_env(config)
            self._disable_leaf_termination(env)
            action_dim = int(env.action_dim)
            action_scales = np.asarray(config.get("action_scales", []), dtype=np.float32)
            action_scales = self._pad_or_trim(action_scales, action_dim, fill=1.0)
            final_q = self._target_vector("final_pos", action_dim, fill=0.0)
            final_dq = self._target_vector("final_vel", action_dim, fill=0.0)
            control_dt = self._control_dt(env)
            self.settings["teacher_dt"] = control_dt
            trajectory_steps = max(1, int(round(trajectory_seconds / max(control_dt, 1e-6))))
            eligible_mask = ~self._wheel_mask(env, action_dim)
            collected = 0
            self._log(
                f"[homing-collect] terrain={terrain} target={per_terrain_target} action_dim={action_dim} "
                f"trajectory={trajectory_seconds:.2f}s/{trajectory_steps}steps stand_warmup={stand_warmup_steps} "
                f"balance_blend={balance_blend:.2f}"
            )
            try:
                while collected < per_terrain_target and len(input_all) < total_samples:
                    if self._stop_requested():
                        self._log(f"[homing-collect] stop requested; terrain={terrain} collected={collected}")
                        break
                    state, _ = env.reset()
                    policy.reset()
                    command = self._sample_command(rng, env.command_dim)
                    env.receive_user_command(command)
                    state = self._inject_applied_command(state, env)

                    for _ in range(stand_warmup_steps):
                        if self._stop_requested():
                            break
                        action = policy.get_action(state)
                        state, terminated, truncated, _ = env.step(action)
                        state = self._inject_applied_command(state, env)
                        if truncated:
                            break
                        if terminated:
                            self._clear_wrapper_reset_flags(env)
                    if self._stop_requested():
                        break

                    q_start, _ = self._current_joint_state(env, action_dim)

                    for traj_step in range(trajectory_steps):
                        if self._stop_requested() or collected >= per_terrain_target or len(input_all) >= total_samples:
                            break
                        env.receive_user_command(command)
                        q, dq = self._current_joint_state(env, action_dim)
                        label = self._timed_trajectory_action(
                            q_start,
                            final_q,
                            action_scales,
                            traj_step,
                            trajectory_steps,
                            eligible_mask=eligible_mask,
                        )
                        if balance_blend > 0.0:
                            stand_action = policy.get_action(state)
                            label = self._blend_balance_action(stand_action, label, balance_blend)
                        model_input = self._make_input(state, q, dq, final_q, final_dq)
                        input_all.append(model_input)
                        action_all.append(label)
                        state_all.append(np.asarray(state, dtype=np.float32).copy())
                        q_all.append(q.copy())
                        dq_all.append(dq.copy())
                        terrain_all.append(str(terrain))
                        collected += 1

                        state, terminated, truncated, _ = env.step(label)
                        state = self._inject_applied_command(state, env)
                        if collected % 1000 == 0:
                            self._log(f"[homing-collect] terrain={terrain} collected={collected}/{per_terrain_target} total={len(input_all)}/{total_samples}")
                        if truncated:
                            break
                        if terminated:
                            self._clear_wrapper_reset_flags(env)
            finally:
                env.close()

        if not input_all:
            raise RuntimeError("No Homing samples were collected.")

        dataset_path = os.path.join(run_dir, "dataset.npz")
        np.savez_compressed(
            dataset_path,
            input=np.asarray(input_all, dtype=np.float32),
            action_label=np.asarray(action_all, dtype=np.float32),
            state=np.asarray(state_all, dtype=np.float32),
            q=np.asarray(q_all, dtype=np.float32),
            dq=np.asarray(dq_all, dtype=np.float32),
            final_pos=np.asarray(final_q, dtype=np.float32),
            final_vel=np.asarray(final_dq, dtype=np.float32),
            terrain=np.asarray(terrain_all),
        )
        metadata = {
            "env_id": self.env_id,
            "policy_path": policy_path,
            "samples": int(len(input_all)),
            "input_dim": int(np.asarray(input_all[0]).size),
            "action_dim": int(np.asarray(action_all[0]).size),
            "final_pos": np.asarray(final_q, dtype=float).tolist(),
            "final_vel": np.asarray(final_dq, dtype=float).tolist(),
            "trajectory_seconds": float(trajectory_seconds),
            "stand_warmup_steps": int(stand_warmup_steps),
            "balance_blend": float(balance_blend),
            "teacher": {
                "trajectory": "smoothstep action-space interpolation",
            },
            "input_mode": "obs",
            "dataset_path": dataset_path,
        }
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        self._log(f"[homing-collect] saved {len(input_all)} samples")
        return metadata

    def test_teacher(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for Homing teacher test.")
        policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid stand-drive ONNX policy.")

        policy = OnnxExpertPolicy(policy_path)
        rng = np.random.default_rng(int(self.settings.get("seed", 42)))
        warmup_steps = max(0, int(self.settings.get("test_warmup_steps", 200)))
        test_steps = max(1, int(self.settings.get("test_steps", 600)))
        config = self._base_config("flat", render=True)
        env = build_env(config)
        termination_disabled = self._disable_leaf_termination(env)
        action_dim = int(env.action_dim)
        action_scales = self._pad_or_trim(config.get("action_scales", []), action_dim, fill=1.0)
        final_q = self._target_vector("final_pos", action_dim, fill=0.0)
        final_dq = self._target_vector("final_vel", action_dim, fill=0.0)
        control_dt = self._control_dt(env)
        trajectory_seconds = max(0.02, float(self.settings.get("homing_trajectory_seconds", 3.0)))
        trajectory_steps = max(1, int(round(trajectory_seconds / max(control_dt, 1e-6))))
        balance_blend = float(np.clip(float(self.settings.get("homing_balance_blend", 0.0)), 0.0, 1.0))
        eligible_mask = ~self._wheel_mask(env, action_dim)
        total_steps = max(test_steps, trajectory_steps)
        self._log(
            f"[homing-test] flat render test warmup={warmup_steps} homing_steps={trajectory_steps} "
            f"total_steps={total_steps} trajectory={trajectory_seconds:.2f}s balance_blend={balance_blend:.2f}"
        )
        if termination_disabled:
            self._log("[homing-test] leaf env termination disabled; time-limit truncation still applies.")
        try:
            state, _ = env.reset()
            policy.reset()
            command = self._zero_command(env.command_dim)
            env.receive_user_command(command)
            state = self._inject_applied_command(state, env)
            for _ in range(warmup_steps):
                if self._stop_requested():
                    break
                action = policy.get_action(state)
                state, terminated, truncated, _ = env.step(action)
                env.render()
                state = self._inject_applied_command(state, env)
                if truncated:
                    self._log("[homing-test] env truncated during warmup.")
                    break
                if terminated:
                    self._clear_wrapper_reset_flags(env)
                    self._log("[homing-test] ignored env termination during warmup.")
                    continue

            q_start, _ = self._current_joint_state(env, action_dim)
            self._zero_base_velocity(env)
            last_wall_time = time.monotonic()
            for step in range(total_steps):
                if self._stop_requested():
                    self._log("[homing-test] stop requested.")
                    break
                q, dq = self._current_joint_state(env, action_dim)
                trajectory_step = min(step, trajectory_steps - 1)
                action = self._timed_trajectory_action(
                    q_start,
                    final_q,
                    action_scales,
                    trajectory_step,
                    trajectory_steps,
                    eligible_mask=eligible_mask,
                )
                if balance_blend > 0.0:
                    stand_action = policy.get_action(state)
                    action = self._blend_balance_action(stand_action, action, balance_blend)
                state, terminated, truncated, _ = env.step(action)
                env.render()
                if bool(self.settings.get("homing_realtime_test", True)):
                    now = time.monotonic()
                    sleep_time = control_dt - (now - last_wall_time)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    last_wall_time = time.monotonic()
                state = self._inject_applied_command(state, env)
                if step % 50 == 0:
                    pos_err = float(np.sqrt(np.mean((final_q - q) ** 2)))
                    vel_err = float(np.sqrt(np.mean((final_dq - dq) ** 2)))
                    self._log(f"[homing-test] step={step}/{total_steps} pos_rmse={pos_err:.4f} vel_rmse={vel_err:.4f}")
                if truncated:
                    self._log("[homing-test] env truncated.")
                    break
                if terminated:
                    self._clear_wrapper_reset_flags(env)
                    self._log("[homing-test] ignored env termination.")
        finally:
            env.close()
        return {"env_id": self.env_id, "mode": "test_teacher", "terrain": "flat", "steps": int(total_steps)}

    def test_export_policy(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for Homing policy test.")
        stand_policy_path = str(self.settings.get("policy_path", "")).strip()
        homing_policy_path = str(self.settings.get("output_path", "")).strip()
        if not os.path.isfile(stand_policy_path):
            raise RuntimeError("Select a valid stand-drive ONNX policy.")
        if not os.path.isfile(homing_policy_path):
            raise RuntimeError("Select a valid exported Homing ONNX policy.")

        stand_policy = OnnxExpertPolicy(stand_policy_path)
        homing_policy = OnnxExpertPolicy(homing_policy_path)
        test_steps = max(1, int(self.settings.get("test_steps", self.settings.get("rollout_steps", 1000))))
        config = self._base_config("flat", render=True)
        env = build_env(config)
        termination_disabled = self._disable_leaf_termination(env)
        active_policy = stand_policy
        active_name = "stand-drive"
        switched = False
        self._log(
            f"[homing-policy-test] running stand-drive policy. Click 'Switch Policy' to use exported homing policy. "
            f"steps={test_steps}"
        )
        if termination_disabled:
            self._log("[homing-policy-test] leaf env termination disabled; time-limit truncation still applies.")
        try:
            state, _ = env.reset()
            stand_policy.reset()
            homing_policy.reset()
            command = self._current_command(env.command_dim)
            env.receive_user_command(command)
            state = self._inject_applied_command(state, env)
            for step in range(test_steps):
                if self._stop_requested():
                    self._log("[homing-policy-test] stop requested.")
                    break

                command = self._current_command(env.command_dim)
                env.receive_user_command(command)

                if self._switch_requested() and not switched:
                    active_policy = homing_policy
                    active_name = "homing"
                    switched = True
                    homing_policy.reset()
                    self._zero_base_velocity(env)
                    self._log(f"[homing-policy-test] switched to exported homing policy at step={step}.")

                action = active_policy.get_action(state)
                state, terminated, truncated, _ = env.step(action)
                env.render()
                state = self._inject_applied_command(state, env)

                if step % 100 == 0:
                    self._log(f"[homing-policy-test] step={step}/{test_steps} active={active_name} command={command.tolist()}")
                if truncated:
                    self._log("[homing-policy-test] env truncated.")
                    break
                if terminated:
                    self._clear_wrapper_reset_flags(env)
                    self._log("[homing-policy-test] ignored env termination.")
        finally:
            env.close()
        return {
            "env_id": self.env_id,
            "mode": "test_policy",
            "terrain": "flat",
            "steps": int(test_steps),
            "switched": bool(switched),
            "onnx_path": homing_policy_path,
        }

    @staticmethod
    def _get_linear_acceleration(env):
        try:
            data = env.get_data()
            return np.asarray(data.sensor("linear-acceleration").data, dtype=np.float32).reshape(-1)
        except Exception:
            return np.zeros((3,), dtype=np.float32)

    @staticmethod
    def _contact_effort(env):
        try:
            data = env.get_data()
            cfrc = np.asarray(getattr(data, "cfrc_ext", []), dtype=np.float32)
            if cfrc.size == 0:
                return 0.0
            return float(np.mean(np.linalg.norm(cfrc.reshape(-1, cfrc.shape[-1]), axis=-1)))
        except Exception:
            return 0.0

    @staticmethod
    def _base_roll_pitch(env):
        quat_wxyz = None
        try:
            data = env.get_data()
            qpos = np.asarray(getattr(data, "qpos", []), dtype=np.float64).reshape(-1)
            if qpos.size >= 7:
                quat_wxyz = qpos[3:7]
        except Exception:
            quat_wxyz = None

        if quat_wxyz is None:
            try:
                leaf = HomingTrainer._leaf_env(env)
                model = getattr(leaf, "model", None)
                data = getattr(leaf, "data", None)
                if model is not None and data is not None:
                    for body_name in ("base_link", "pelvis_link"):
                        try:
                            body_id = model.body(body_name).id
                            quat_wxyz = np.asarray(data.xquat[body_id], dtype=np.float64).reshape(4)
                            break
                        except Exception:
                            continue
            except Exception:
                quat_wxyz = None

        if quat_wxyz is None:
            return 0.0, 0.0

        norm = float(np.linalg.norm(quat_wxyz))
        if norm <= 1e-8:
            return 0.0, 0.0
        w, x, y, z = (quat_wxyz / norm).tolist()
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(pitch_arg)
        return float(roll), float(pitch)

    @staticmethod
    def _fall_signal(env, min_height=0.12):
        try:
            data = env.get_data()
            if getattr(data, "qpos", None) is not None and data.qpos.size >= 3:
                if float(data.qpos[2]) < float(min_height):
                    return True
        except Exception:
            pass
        try:
            obs = env.get_last_obs() or {}
            gravity = np.asarray(obs.get("projected_gravity", []), dtype=np.float32).reshape(-1)
            if gravity.size >= 3 and abs(float(gravity[2])) < 0.45:
                return True
        except Exception:
            pass
        return False

    def _make_rl_config(self, terrain, rng, randomize_strength):
        config = self._base_config(terrain, render=False)
        strength = float(np.clip(randomize_strength, 0.0, 1.0))
        if strength <= 0.0:
            return config

        random_cfg = dict(config.get("random", {}))

        def scale_positive(key, lo, hi):
            base = max(1e-6, float(random_cfg.get(key, 0.0)))
            factor = rng.uniform(1.0 - strength * (1.0 - lo), 1.0 + strength * (hi - 1.0))
            random_cfg[key] = float(max(0.0, base * factor))

        scale_positive("sliding_friction", 0.65, 1.35)
        scale_positive("torsional_friction", 0.65, 1.35)
        scale_positive("rolling_friction", 0.65, 1.35)
        random_cfg["init_noise"] = float(max(random_cfg.get("init_noise", 0.0), 0.02 * strength))
        random_cfg["mass_noise"] = float(max(random_cfg.get("mass_noise", 0.0), 0.05 * strength))
        random_cfg["action_delay_prob"] = float(max(random_cfg.get("action_delay_prob", 0.0), 0.05 * strength))
        config["random"] = random_cfg
        return config

    def _load_supervised_actor_critic(self, checkpoint_path, input_dim, action_dim, hidden_dim, action_mask=None):
        checkpoint_path = str(checkpoint_path or "").strip()
        if not checkpoint_path:
            raise RuntimeError("Select a supervised Homing checkpoint before PPO fine-tuning.")
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Homing init checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_type = str(checkpoint.get("checkpoint_type", "supervised" if "model_state" in checkpoint else "unknown"))
        if checkpoint_type != "supervised" or "model_state" not in checkpoint:
            raise RuntimeError(
                "PPO fine-tune must start from a supervised Homing checkpoint. "
                "Run 'Train Policy' first or select homing_policy_supervised.pt."
            )
        if int(checkpoint.get("input_dim", input_dim)) != input_dim:
            raise RuntimeError("PPO fine-tune checkpoint input_dim does not match the selected environment observation.")
        if int(checkpoint.get("action_dim", action_dim)) != action_dim:
            raise RuntimeError("PPO fine-tune checkpoint action_dim does not match the selected environment.")
        hidden_dim = int(checkpoint.get("hidden_dim", hidden_dim))
        model = HomingActorCritic(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_mask)
        model.actor.load_state_dict(checkpoint["model_state"], strict=False)
        return model

    def _initialize_actor_from_stand_policy(
        self,
        model,
        envs,
        policies,
        rng,
        action_dim,
        action_output_mask,
        total_steps,
        lr,
    ):
        clone_steps = max(0, int(self.settings.get("ppo_stand_init_steps", min(4096, max(512, total_steps // 10)))))
        if clone_steps <= 0:
            return

        states = []
        actions = []
        env_count = max(1, len(envs))
        per_env_steps = max(1, int(np.ceil(clone_steps / float(env_count))))
        self._log(f"[homing-ppo] stand-drive actor init: cloning {clone_steps} samples")

        for env_index, env in enumerate(envs):
            policy = policies[env_index]
            state, _ = env.reset()
            policy.reset()
            command = self._sample_command(rng, env.command_dim)
            env.receive_user_command(command)
            state = self._inject_applied_command(state, env)
            for _ in range(per_env_steps):
                if len(states) >= clone_steps or self._stop_requested():
                    break
                action = self._pad_or_trim(policy.get_action(state), action_dim)
                action = np.clip(action, -1.0, 1.0) * action_output_mask
                states.append(np.asarray(state, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                state, terminated, truncated, _ = env.step(action)
                state = self._inject_applied_command(state, env)
                if terminated or truncated:
                    self._clear_wrapper_reset_flags(env)
                    state, _ = env.reset()
                    policy.reset()
                    command = self._sample_command(rng, env.command_dim)
                    env.receive_user_command(command)
                    state = self._inject_applied_command(state, env)

        if not states:
            return

        obs_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).float()
        act_t = torch.from_numpy(np.asarray(actions, dtype=np.float32)).float()
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=max(1e-8, float(lr)))
        batch_size = min(512, max(32, int(obs_t.shape[0])))
        indices = np.arange(int(obs_t.shape[0]))
        epochs = max(1, int(self.settings.get("ppo_stand_init_epochs", 3)))
        last_loss = 0.0
        for _epoch in range(epochs):
            rng.shuffle(indices)
            for start in range(0, indices.size, batch_size):
                mb_idx = indices[start:start + batch_size]
                pred = model.actor(obs_t[mb_idx])
                loss = (pred - act_t[mb_idx]).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
        self._log(f"[homing-ppo] stand-drive actor init done: samples={len(states)} loss={last_loss:.6f}")

    def _homing_reward(
        self,
        env,
        q,
        dq,
        action,
        prev_action,
        prev_prev_action,
        q_start,
        final_q,
        final_dq,
        action_scales,
        eligible_mask,
        step_index,
        trajectory_steps,
        terminated,
        truncated,
        weights,
        policy_ref_action=None,
    ):
        ref_action = self._timed_trajectory_action(
            q_start,
            final_q,
            action_scales,
            min(step_index, trajectory_steps - 1),
            trajectory_steps,
            eligible_mask=eligible_mask,
        )
        scales = self._pad_or_trim(action_scales, ref_action.size, fill=1.0)
        q_ref = ref_action * np.maximum(np.abs(scales), 1e-6)
        final_action = self._pad_or_trim(final_q, ref_action.size) / np.maximum(np.abs(scales), 1e-6)
        mask = self._pad_or_trim(np.asarray(eligible_mask, dtype=np.float32), ref_action.size, fill=0.0).astype(bool)
        if not np.any(mask):
            mask = np.ones((ref_action.size,), dtype=bool)

        q = self._pad_or_trim(q, ref_action.size)
        dq = self._pad_or_trim(dq, ref_action.size)
        final_dq = self._pad_or_trim(final_dq, ref_action.size)
        action = self._pad_or_trim(action, ref_action.size)
        prev_action = self._pad_or_trim(prev_action, ref_action.size)
        prev_prev_action = self._pad_or_trim(prev_prev_action, ref_action.size)
        policy_ref_action = ref_action if policy_ref_action is None else self._pad_or_trim(policy_ref_action, ref_action.size)

        pos_rmse = float(np.sqrt(np.mean((q[mask] - q_ref[mask]) ** 2)))
        final_rmse = float(np.sqrt(np.mean((q[mask] - final_q[mask]) ** 2)))
        vel_rmse = float(np.sqrt(np.mean((dq[mask] - final_dq[mask]) ** 2)))
        trajectory_action_rmse = float(np.sqrt(np.mean((action[mask] - ref_action[mask]) ** 2)))
        action_rmse = float(np.sqrt(np.mean((action[mask] - policy_ref_action[mask]) ** 2)))
        final_action_rmse = float(np.sqrt(np.mean((action[mask] - final_action[mask]) ** 2)))
        action_rate = float(np.sqrt(np.mean((action - prev_action) ** 2)))
        action_accel = float(np.sqrt(np.mean((action - 2.0 * prev_action + prev_prev_action) ** 2)))
        accel_norm = float(np.linalg.norm(self._get_linear_acceleration(env)))
        contact_effort = self._contact_effort(env)
        base_roll, base_pitch = self._base_roll_pitch(env)
        base_tilt = float(np.sqrt(base_roll * base_roll + base_pitch * base_pitch))
        fallen = bool(terminated or self._fall_signal(env, weights.get("fall_height", 0.12)))

        reward = 0.0
        track_weight = float(weights.get("track", 4.0))
        reward += float(weights.get("alive", 0.05))
        reward += track_weight * np.exp(-float(weights.get("final_sigma", 10.0)) * final_rmse)
        reward -= track_weight * float(weights.get("final_l2", 2.0)) * final_rmse
        reward += float(weights.get("trajectory", 0.3)) * np.exp(-float(weights.get("track_sigma", 4.0)) * pos_rmse)
        reward += float(weights.get("velocity", 0.4)) * np.exp(-float(weights.get("vel_sigma", 0.08)) * vel_rmse)
        reward -= float(weights.get("imitation", 0.4)) * action_rmse
        reward -= float(weights.get("final_action", 0.1)) * final_action_rmse
        reward -= float(weights.get("action_rate", 0.04)) * action_rate
        reward -= float(weights.get("action_accel", 0.02)) * action_accel
        reward -= float(weights.get("base_acc", 0.002)) * min(accel_norm, 80.0)
        reward -= float(weights.get("upright", 2.0)) * min(base_tilt, 1.5)
        reward -= float(weights.get("contact", 0.0005)) * min(contact_effort, 5000.0)
        if fallen:
            reward -= float(weights.get("fall", 8.0))
        if truncated:
            reward -= 0.5
        if step_index >= trajectory_steps - 1 and pos_rmse < float(weights.get("final_tolerance", 0.08)):
            reward += float(weights.get("final_bonus", 2.0))

        metrics = {
            "pos_rmse": pos_rmse,
            "final_rmse": final_rmse,
            "vel_rmse": vel_rmse,
            "action_rmse": action_rmse,
            "trajectory_action_rmse": trajectory_action_rmse,
            "final_action_rmse": final_action_rmse,
            "action_rate": action_rate,
            "action_accel": action_accel,
            "base_acc": accel_norm,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_tilt": base_tilt,
            "contact": contact_effort,
            "fallen": float(fallen),
        }
        return float(reward), bool(fallen or truncated or step_index >= trajectory_steps - 1), metrics

    def fine_tune_rl(self):
        self._require_torch()
        if not self.env_id:
            raise RuntimeError("Select a robot/env for Homing PPO fine-tune.")
        stand_policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(stand_policy_path):
            raise RuntimeError("Select a valid stand-drive ONNX policy.")
        supervised_init = self._as_bool(self.settings.get("ppo_supervised_init", True), True)
        use_trajectory_reward = self._as_bool(self.settings.get("ppo_use_trajectory_reward", True), True)
        mask_wheel_actions = self._as_bool(self.settings.get("ppo_mask_wheel_actions", True), True)
        init_checkpoint = str(self.settings.get("checkpoint_path", "")).strip()
        if supervised_init and not init_checkpoint:
            init_checkpoint = self._supervised_checkpoint_path()

        seed = int(self.settings.get("seed", 42))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        num_envs = max(1, int(self.settings.get("ppo_num_envs", 4)))
        total_steps = max(num_envs, int(self.settings.get("ppo_total_steps", 20000)))
        rollout_steps = max(8, int(self.settings.get("ppo_rollout_steps", 256)))
        ppo_epochs = max(1, int(self.settings.get("ppo_epochs", 4)))
        minibatch_size = max(8, int(self.settings.get("ppo_minibatch_size", 512)))
        hidden_dim = max(32, int(self.settings.get("hidden_dim", 256)))
        lr = max(1e-8, float(self.settings.get("ppo_learning_rate", self.settings.get("learning_rate", 3e-4))))
        gamma = float(np.clip(float(self.settings.get("ppo_gamma", 0.99)), 0.0, 0.9999))
        gae_lambda = float(np.clip(float(self.settings.get("ppo_gae_lambda", 0.95)), 0.0, 1.0))
        clip_ratio = float(np.clip(float(self.settings.get("ppo_clip_ratio", 0.2)), 0.01, 0.5))
        entropy_coef = max(0.0, float(self.settings.get("ppo_entropy_coef", 0.0)))
        value_coef = max(0.0, float(self.settings.get("ppo_value_coef", 0.5)))
        bc_coef = max(0.0, float(self.settings.get("ppo_bc_coef", 1.0)))
        max_grad_norm = max(0.0, float(self.settings.get("ppo_max_grad_norm", 0.5)))
        randomize_strength = float(np.clip(float(self.settings.get("ppo_domain_randomize", 0.3)), 0.0, 1.0))

        trajectory_seconds = max(0.02, float(self.settings.get("homing_trajectory_seconds", 3.0)))
        stand_warmup_steps = max(0, int(self.settings.get("homing_stand_warmup_steps", 200)))
        weights = {
            "track": float(self.settings.get("reward_track", 6.0)),
            "velocity": float(self.settings.get("reward_velocity", 0.4)),
            "imitation": float(self.settings.get("reward_imitation", 0.4)),
            "final_action": float(self.settings.get("reward_final_action", 0.1)),
            "action_rate": float(self.settings.get("reward_action_rate", 0.04)),
            "action_accel": float(self.settings.get("reward_action_accel", 0.02)),
            "base_acc": float(self.settings.get("reward_base_acc", 0.002)),
            "upright": float(self.settings.get("reward_upright", 2.0)),
            "contact": float(self.settings.get("reward_contact", 0.0005)),
            "fall": float(self.settings.get("reward_fall", 8.0)),
            "alive": float(self.settings.get("reward_alive", 0.05)),
            "track_sigma": float(self.settings.get("reward_track_sigma", 4.0)),
            "trajectory": float(self.settings.get("reward_trajectory", 0.3)),
            "final_sigma": float(self.settings.get("reward_final_sigma", 10.0)),
            "final_l2": float(self.settings.get("reward_final_l2", 2.0)),
            "vel_sigma": float(self.settings.get("reward_vel_sigma", 0.08)),
            "final_bonus": float(self.settings.get("reward_final_bonus", 2.0)),
            "final_tolerance": float(self.settings.get("reward_final_tolerance", 0.08)),
            "fall_height": float(self.settings.get("reward_fall_height", 0.12)),
        }
        if not use_trajectory_reward:
            for key in ("track", "trajectory", "velocity", "imitation", "final_action", "final_l2", "final_bonus"):
                weights[key] = 0.0
            bc_coef = 0.0

        envs = []
        policies = []
        contexts = []
        try:
            for env_index in range(num_envs):
                config = self._make_rl_config("flat", rng, randomize_strength)
                env = build_env(config)
                envs.append(env)
                policies.append(OnnxExpertPolicy(stand_policy_path))

            action_dim = int(envs[0].action_dim)
            input_dim = int(envs[0].state_dim)
            action_scales = self._pad_or_trim(self._base_config("flat").get("action_scales", []), action_dim, fill=1.0)
            final_q = self._target_vector("final_pos", action_dim, fill=0.0)
            final_dq = self._target_vector("final_vel", action_dim, fill=0.0)
            control_dt = self._control_dt(envs[0])
            trajectory_steps = max(1, int(round(trajectory_seconds / max(control_dt, 1e-6))))
            eligible_mask = ~self._wheel_mask(envs[0], action_dim)
            action_output_mask = np.ones((action_dim,), dtype=np.float32)
            if mask_wheel_actions:
                action_output_mask[~eligible_mask] = 0.0

            if supervised_init:
                model = self._load_supervised_actor_critic(init_checkpoint, input_dim, action_dim, hidden_dim, action_mask=action_output_mask)
                try:
                    hidden_dim = int(model.actor.net[0].out_features)
                except Exception:
                    pass
                init_mode = "supervised_homing"
            else:
                model = HomingActorCritic(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_output_mask)
                self._initialize_actor_from_stand_policy(
                    model,
                    envs,
                    policies,
                    rng,
                    action_dim,
                    action_output_mask,
                    total_steps,
                    lr,
                )
                init_checkpoint = ""
                init_mode = "stand_drive_actor_clone"
            base_actor = copy.deepcopy(model.actor)
            base_actor.eval()
            for param in base_actor.parameters():
                param.requires_grad_(False)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            def reset_task(env_index):
                env = envs[env_index]
                stand_policy = policies[env_index]
                state, _ = env.reset()
                stand_policy.reset()
                command = self._sample_command(rng, env.command_dim)
                env.receive_user_command(command)
                state = self._inject_applied_command(state, env)
                last_action = np.zeros((action_dim,), dtype=np.float32)
                for _ in range(stand_warmup_steps):
                    if self._stop_requested():
                        break
                    action = stand_policy.get_action(state)
                    last_action = self._pad_or_trim(action, action_dim)
                    state, terminated, truncated, _ = env.step(action)
                    state = self._inject_applied_command(state, env)
                    if terminated or truncated:
                        self._clear_wrapper_reset_flags(env)
                        state, _ = env.reset()
                        env.receive_user_command(command)
                        state = self._inject_applied_command(state, env)
                q_start, _ = self._current_joint_state(env, action_dim)
                return {
                    "state": np.asarray(state, dtype=np.float32),
                    "command": command,
                    "q_start": q_start,
                    "step": 0,
                    "prev_action": last_action.copy(),
                    "prev_prev_action": last_action.copy(),
                    "episode_reward": 0.0,
                    "episode_len": 0,
                }

            contexts = [reset_task(i) for i in range(num_envs)]
            self._log(
                f"[homing-ppo] envs={num_envs} total_steps={total_steps} rollout={rollout_steps} "
                f"trajectory={trajectory_seconds:.2f}s/{trajectory_steps}steps randomize={randomize_strength:.2f} "
                f"init={init_mode} traj_reward={use_trajectory_reward} wheel_mask={mask_wheel_actions}"
            )

            updates = int(np.ceil(total_steps / float(num_envs * rollout_steps)))
            history = []
            global_steps = 0
            best_score = float("inf")
            best_snapshot = None
            for update in range(updates):
                if self._stop_requested():
                    break
                obs_buf, act_buf, ref_act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], [], []
                metric_sums = {}
                metric_count = 0
                completed_returns = []

                for _ in range(rollout_steps):
                    obs_np = np.stack([ctx["state"] for ctx in contexts], axis=0).astype(np.float32)
                    obs_t = torch.from_numpy(obs_np).float()
                    with torch.no_grad():
                        action_t, logp_t, _, value_t = model.act(obs_t)
                        policy_ref_t = base_actor(obs_t)
                    action_raw_np = action_t.cpu().numpy().astype(np.float32)
                    action_raw_np = action_raw_np * action_output_mask.reshape(1, -1)
                    action_np = np.clip(action_raw_np, -1.0, 1.0).astype(np.float32) * action_output_mask.reshape(1, -1)
                    policy_ref_np = np.clip(policy_ref_t.cpu().numpy().astype(np.float32), -1.0, 1.0) * action_output_mask.reshape(1, -1)

                    rewards = []
                    dones = []
                    ref_actions = []
                    for env_index, env in enumerate(envs):
                        ctx = contexts[env_index]
                        env.receive_user_command(ctx["command"])
                        state, terminated, truncated, _ = env.step(action_np[env_index])
                        state = self._inject_applied_command(state, env)
                        q, dq = self._current_joint_state(env, action_dim)
                        ref_action = self._timed_trajectory_action(
                            ctx["q_start"],
                            final_q,
                            action_scales,
                            min(ctx["step"], trajectory_steps - 1),
                            trajectory_steps,
                            eligible_mask=eligible_mask,
                        )
                        reward, done, metrics = self._homing_reward(
                            env,
                            q,
                            dq,
                            action_np[env_index],
                            ctx["prev_action"],
                            ctx["prev_prev_action"],
                            ctx["q_start"],
                            final_q,
                            final_dq,
                            action_scales,
                            eligible_mask,
                            ctx["step"],
                            trajectory_steps,
                            terminated,
                            truncated,
                            weights,
                            policy_ref_action=policy_ref_np[env_index],
                        )
                        ctx["episode_reward"] += reward
                        ctx["episode_len"] += 1
                        for key, value in metrics.items():
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                        metric_count += 1
                        rewards.append(reward)
                        dones.append(float(done))
                        ref_actions.append(policy_ref_np[env_index])
                        ctx["prev_prev_action"] = ctx["prev_action"].copy()
                        ctx["prev_action"] = action_np[env_index].copy()
                        ctx["step"] += 1
                        ctx["state"] = np.asarray(state, dtype=np.float32)
                        if done:
                            completed_returns.append(ctx["episode_reward"])
                            contexts[env_index] = reset_task(env_index)

                    obs_buf.append(obs_np)
                    act_buf.append(action_raw_np)
                    ref_act_buf.append(np.asarray(ref_actions, dtype=np.float32))
                    logp_buf.append(logp_t.cpu().numpy().astype(np.float32))
                    rew_buf.append(np.asarray(rewards, dtype=np.float32))
                    done_buf.append(np.asarray(dones, dtype=np.float32))
                    val_buf.append(value_t.cpu().numpy().astype(np.float32))
                    global_steps += num_envs

                obs_arr = np.asarray(obs_buf, dtype=np.float32)
                act_arr = np.asarray(act_buf, dtype=np.float32)
                ref_act_arr = np.asarray(ref_act_buf, dtype=np.float32)
                logp_arr = np.asarray(logp_buf, dtype=np.float32)
                rew_arr = np.asarray(rew_buf, dtype=np.float32)
                done_arr = np.asarray(done_buf, dtype=np.float32)
                val_arr = np.asarray(val_buf, dtype=np.float32)

                with torch.no_grad():
                    last_obs = torch.from_numpy(np.stack([ctx["state"] for ctx in contexts], axis=0).astype(np.float32)).float()
                    last_values = model.value(last_obs).cpu().numpy().astype(np.float32)

                adv = np.zeros_like(rew_arr, dtype=np.float32)
                last_gae = np.zeros((num_envs,), dtype=np.float32)
                for t in reversed(range(rollout_steps)):
                    next_values = last_values if t == rollout_steps - 1 else val_arr[t + 1]
                    next_nonterminal = 1.0 - done_arr[t]
                    delta = rew_arr[t] + gamma * next_values * next_nonterminal - val_arr[t]
                    last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
                    adv[t] = last_gae
                ret = adv + val_arr

                flat_obs = torch.from_numpy(obs_arr.reshape(-1, input_dim)).float()
                flat_act = torch.from_numpy(act_arr.reshape(-1, action_dim)).float()
                flat_ref_act = torch.from_numpy(ref_act_arr.reshape(-1, action_dim)).float()
                flat_logp = torch.from_numpy(logp_arr.reshape(-1)).float()
                flat_adv = torch.from_numpy(adv.reshape(-1)).float()
                flat_ret = torch.from_numpy(ret.reshape(-1)).float()
                flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)

                batch_size = int(flat_obs.shape[0])
                indices = np.arange(batch_size)
                policy_loss_value = 0.0
                value_loss_value = 0.0
                bc_loss_value = 0.0
                entropy_value = 0.0
                for _epoch in range(ppo_epochs):
                    rng.shuffle(indices)
                    for start in range(0, batch_size, minibatch_size):
                        mb_idx = indices[start:start + minibatch_size]
                        mb_obs = flat_obs[mb_idx]
                        mb_act = flat_act[mb_idx]
                        mb_ref_act = flat_ref_act[mb_idx]
                        mb_old_logp = flat_logp[mb_idx]
                        mb_adv = flat_adv[mb_idx]
                        mb_ret = flat_ret[mb_idx]

                        dist = model.distribution(mb_obs)
                        new_logp = dist.log_prob(mb_act).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()
                        ratio = torch.exp(new_logp - mb_old_logp)
                        unclipped = ratio * mb_adv
                        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                        policy_loss = -torch.min(unclipped, clipped).mean()
                        value_loss = 0.5 * (model.value(mb_obs) - mb_ret).pow(2).mean()
                        bc_loss = (model.actor(mb_obs) - mb_ref_act).pow(2).mean()
                        loss = policy_loss + value_coef * value_loss + bc_coef * bc_loss - entropy_coef * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        if max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        policy_loss_value = float(policy_loss.item())
                        value_loss_value = float(value_loss.item())
                        bc_loss_value = float(bc_loss.item())
                        entropy_value = float(entropy.item())

                metric_means = {key: value / max(1, metric_count) for key, value in metric_sums.items()}
                mean_return = float(np.mean(completed_returns)) if completed_returns else float(np.mean(rew_arr) * trajectory_steps)
                entry = {
                    "update": update + 1,
                    "steps": int(global_steps),
                    "mean_reward": float(np.mean(rew_arr)),
                    "mean_episode_return": mean_return,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                    "bc_loss": bc_loss_value,
                    "entropy": entropy_value,
                    "action_std": float(torch.exp(model.log_std).mean().item()),
                    **metric_means,
                }
                # PPO can improve and then drift. Keep the best actor for export
                # instead of blindly saving the last update.
                if use_trajectory_reward:
                    score = (
                        float(entry.get("final_rmse", 1e6))
                        + 0.25 * float(entry.get("base_tilt", 0.0))
                        + 0.05 * float(entry.get("action_rmse", 0.0))
                        + 0.02 * float(entry.get("final_action_rmse", 0.0))
                        + 5.0 * float(entry.get("fallen", 0.0))
                    )
                else:
                    score = (
                        -float(entry.get("mean_episode_return", entry.get("mean_reward", 0.0)))
                        + 0.5 * float(entry.get("base_tilt", 0.0))
                        + 0.02 * float(entry.get("action_rate", 0.0))
                        + 5.0 * float(entry.get("fallen", 0.0))
                    )
                entry["selection_score"] = float(score)
                if score < best_score:
                    best_score = score
                    best_snapshot = {
                        "update": int(update + 1),
                        "steps": int(global_steps),
                        "selection_score": float(score),
                        "metrics": dict(entry),
                        "actor_state": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
                        "critic_state": {k: v.detach().cpu().clone() for k, v in model.critic.state_dict().items()},
                        "log_std": model.log_std.detach().cpu().clone(),
                    }
                history.append(entry)
                self._log(
                    f"[homing-ppo] update {update + 1}/{updates} steps={global_steps} "
                    f"reward={entry['mean_reward']:.3f} return={mean_return:.2f} "
                    f"pos={entry.get('pos_rmse', 0.0):.4f} final={entry.get('final_rmse', 0.0):.4f} "
                    f"tilt={entry.get('base_tilt', 0.0):.3f} "
                    f"act={entry.get('action_rmse', 0.0):.4f} fact={entry.get('final_action_rmse', 0.0):.4f} "
                    f"bc={entry.get('bc_loss', 0.0):.4f} std={entry.get('action_std', 0.0):.3f} "
                    f"fall={entry.get('fallen', 0.0):.3f} best={best_score:.4f}"
                )

            if best_snapshot is None:
                best_snapshot = {
                    "update": 0,
                    "steps": int(global_steps),
                    "selection_score": float("inf"),
                    "metrics": {},
                    "actor_state": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
                    "critic_state": {k: v.detach().cpu().clone() for k, v in model.critic.state_dict().items()},
                    "log_std": model.log_std.detach().cpu().clone(),
                }
            artifacts = self._make_artifacts("latest")
            ppo_checkpoint_path = self._ppo_checkpoint_path()
            torch.save({
                "checkpoint_type": "ppo_actor_critic",
                "actor_state": best_snapshot["actor_state"],
                "critic_state": best_snapshot["critic_state"],
                "log_std": best_snapshot["log_std"],
                "input_dim": input_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "action_mask": action_output_mask.astype(float).tolist(),
                "settings": dict(self.settings),
                "input_mode": "obs",
                "history": history,
                "init_checkpoint": init_checkpoint,
                "init_mode": init_mode,
                "use_trajectory_reward": bool(use_trajectory_reward),
                "mask_wheel_actions": bool(mask_wheel_actions),
                "selected_update": best_snapshot["update"],
                "selected_metrics": best_snapshot["metrics"],
            }, ppo_checkpoint_path)
            summary = {
                "env_id": self.env_id,
                "mode": "ppo_fine_tune",
                "checkpoint_path": ppo_checkpoint_path,
                "onnx_path": artifacts.onnx_path,
                "manifest_path": artifacts.manifest_path,
                "steps": int(global_steps),
                "input_dim": input_dim,
                "action_dim": action_dim,
                "selected_update": best_snapshot["update"],
                "selected_metrics": best_snapshot["metrics"],
                "history": history,
                "stopped": self._stop_requested(),
            }
            with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            self.export_onnx_from_checkpoint(ppo_checkpoint_path, artifacts.onnx_path)
            self._log(
                f"[homing-ppo] checkpoint saved: {ppo_checkpoint_path} "
                f"selected_update={best_snapshot['update']} score={best_snapshot['selection_score']:.4f}"
            )
            return summary
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    def train(self):
        self._require_torch()
        dataset_paths = list(self.settings.get("selected_datasets", []))
        if not dataset_paths:
            raise RuntimeError("Select at least one Homing dataset before training.")
        dataset = HomingDataset(dataset_paths)
        if len(dataset) < 2:
            raise RuntimeError("At least 2 Homing samples are required to train.")

        input_dim = int(dataset.x.shape[1])
        action_dim = int(dataset.y.shape[1])
        action_output_mask = self._action_output_mask(action_dim)
        epochs = max(1, int(self.settings.get("epochs", 30)))
        batch_size = max(1, int(self.settings.get("batch_size", 256)))
        lr = max(1e-8, float(self.settings.get("learning_rate", 1e-3)))
        val_ratio = min(0.9, max(0.0, float(self.settings.get("val_ratio", 0.1))))
        hidden_dim = max(32, int(self.settings.get("hidden_dim", 256)))

        val_len = int(round(len(dataset) * val_ratio))
        val_len = min(max(val_len, 1), len(dataset) - 1) if len(dataset) > 1 else 0
        train_len = len(dataset) - val_len
        generator = torch.Generator().manual_seed(int(self.settings.get("seed", 42)))
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator) if val_len else (dataset, None)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

        model = HomingPolicyNet(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_output_mask)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mse = nn.MSELoss()
        history = []
        self._log(f"[homing-train] samples={len(dataset)} input_dim={input_dim} action_dim={action_dim}")
        for epoch in range(epochs):
            if self._stop_requested():
                self._log("[homing-train] stop requested; ending after last completed epoch.")
                break
            model.train()
            train_loss = 0.0
            train_count = 0
            for batch in train_loader:
                pred = model(batch["input"].float())
                loss = mse(pred, batch["action"].float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * int(batch["input"].shape[0])
                train_count += int(batch["input"].shape[0])
            train_loss /= max(1, train_count)

            val_loss = None
            if val_loader is not None:
                model.eval()
                total = 0.0
                count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        pred = model(batch["input"].float())
                        loss = mse(pred, batch["action"].float())
                        total += float(loss.item()) * int(batch["input"].shape[0])
                        count += int(batch["input"].shape[0])
                val_loss = total / max(1, count)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            self._log(f"[homing-train] epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss if val_loss is not None else 0.0:.6f}")

        artifacts = self._make_artifacts("latest")
        supervised_checkpoint_path = self._supervised_checkpoint_path()
        checkpoint_payload = {
            "checkpoint_type": "supervised",
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "action_mask": action_output_mask.astype(float).tolist(),
            "settings": dict(self.settings),
            "input_mode": "obs",
            "history": history,
        }
        torch.save(checkpoint_payload, supervised_checkpoint_path)
        shutil.copyfile(supervised_checkpoint_path, artifacts.checkpoint_path)
        summary = {
            "env_id": self.env_id,
            "checkpoint_path": supervised_checkpoint_path,
            "latest_checkpoint_path": artifacts.checkpoint_path,
            "onnx_path": artifacts.onnx_path,
            "manifest_path": artifacts.manifest_path,
            "samples": int(len(dataset)),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "input_mode": "obs",
            "history": history,
            "stopped": self._stop_requested(),
        }
        with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.export_onnx_from_checkpoint(supervised_checkpoint_path, artifacts.onnx_path)
        self._log(f"[homing-train] checkpoint saved: {supervised_checkpoint_path}")
        return summary

    def export_onnx_from_checkpoint(self, checkpoint_path, output_path):
        self._require_torch()
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Homing checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        input_dim = int(checkpoint["input_dim"])
        action_dim = int(checkpoint["action_dim"])
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        action_mask = checkpoint.get("action_mask", None)
        if action_mask is None:
            action_mask = self._action_output_mask(action_dim)
        model = HomingPolicyNet(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_mask)
        if "actor_state" in checkpoint:
            model.load_state_dict(checkpoint["actor_state"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        dummy = torch.zeros((1, input_dim), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        manifest_path = os.path.splitext(output_path)[0] + ".json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "note": "Homing policy actor export. Input shape matches the stand-drive observation; PPO fine-tune does not append phase.",
                "env_id": self.env_id,
                "checkpoint_path": checkpoint_path,
                "onnx_path": output_path,
                "input_dim": input_dim,
                "action_dim": action_dim,
                "checkpoint_type": checkpoint.get("checkpoint_type", "supervised"),
                "input_mode": checkpoint.get("input_mode", "obs"),
                "onnx_inputs": ["obs"],
                "onnx_outputs": ["action"],
            }, handle, indent=2)
        latest_manifest = self._make_artifacts("latest").manifest_path
        if os.path.abspath(manifest_path) != os.path.abspath(latest_manifest):
            os.makedirs(os.path.dirname(latest_manifest), exist_ok=True)
            shutil.copyfile(manifest_path, latest_manifest)
        self._log(f"[homing-export] ONNX exported: {output_path}")
        return output_path
