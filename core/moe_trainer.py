import json
import copy
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import yaml

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


class GateNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ObsSubsetGate(nn.Module):
    def __init__(self, gate, indices, command_dim=0, command_scales=None):
        super().__init__()
        self.gate = gate
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.command_dim = int(command_dim)
        if command_scales is None:
            command_scales = []
        self.register_buffer("command_scales", torch.as_tensor(command_scales, dtype=torch.float32))

    def forward(self, obs):
        gate_obs = torch.index_select(obs, dim=-1, index=self.indices)
        if self.command_dim > 0 and self.command_scales.numel() >= self.command_dim:
            command_start = obs.shape[-1] - self.command_dim
            command_offsets = self.indices - command_start
            scale = torch.ones_like(gate_obs)
            for i in range(self.command_dim):
                scale = torch.where(
                    command_offsets.view(1, -1) == i,
                    torch.abs(self.command_scales[i]).clamp_min(1e-6),
                    scale,
                )
            gate_obs = gate_obs / scale
        return self.gate(gate_obs)


class MoEPolicy(nn.Module):
    def __init__(self, policy_a, policy_b, gate):
        super().__init__()
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.gate = gate

        for p in self.policy_a.parameters():
            p.requires_grad = False
        for p in self.policy_b.parameters():
            p.requires_grad = False

    def forward(self, obs):
        a_a = self.policy_a(obs)
        a_b = self.policy_b(obs)
        alpha = self.gate(obs)
        action = (1.0 - alpha) * a_a + alpha * a_b
        return action


class MoEGateDataset(Dataset):
    def __init__(self, dataset_paths):
        alpha_parts = []
        obs_parts = []
        self.dataset_paths = list(dataset_paths or [])
        if not self.dataset_paths:
            raise RuntimeError("Select at least one MoE dataset.")
        for path in self.dataset_paths:
            with np.load(path) as payload:
                alpha_parts.append(payload["alpha_label"].astype(np.float32).reshape(-1, 1))
                obs_parts.append(payload["obs"].astype(np.float32))
        self.obs = np.concatenate(obs_parts, axis=0)
        self.alpha = np.concatenate(alpha_parts, axis=0)
        self.gate_obs = self.obs

    def __len__(self):
        return int(self.obs.shape[0])

    def __getitem__(self, index):
        return {
            "obs": torch.from_numpy(self.obs[index]),
            "gate_obs": torch.from_numpy(self.gate_obs[index]),
            "alpha": torch.from_numpy(self.alpha[index]),
        }


class OnnxExpertPolicy:
    def __init__(self, policy_path):
        self.policy_path = policy_path
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = [output.name for output in self.session.get_outputs()]
        self.h_name = None
        self.c_name = None
        self.h_state = None
        self.c_state = None
        input_names = [item.name for item in self.inputs]
        if "h_in" in input_names and "c_in" in input_names:
            self.h_name = "h_in"
            self.c_name = "c_in"
            self.h_state = self._zero_state(self.inputs[input_names.index("h_in")])
            self.c_state = self._zero_state(self.inputs[input_names.index("c_in")])

    @staticmethod
    def _zero_state(input_meta):
        shape = []
        for dim in input_meta.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(1)
        if not shape:
            shape = [1, 1, 256]
        return np.zeros(tuple(shape), dtype=np.float32)

    def reset(self):
        if self.h_state is not None:
            self.h_state[:] = 0.0
        if self.c_state is not None:
            self.c_state[:] = 0.0

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        obs_name = self.inputs[0].name
        feed = {obs_name: np.expand_dims(state, axis=0)}
        if self.h_name and self.c_name:
            feed[self.h_name] = self.h_state
            feed[self.c_name] = self.c_state
        try:
            outputs = self.session.run(None, feed)
        except Exception:
            feed[obs_name] = state
            outputs = self.session.run(None, feed)
        if self.h_name and self.c_name and len(outputs) >= 3:
            action, self.h_state, self.c_state = outputs[0], outputs[1], outputs[2]
        else:
            action = outputs[0]
        if action.ndim >= 2 and action.shape[0] == 1:
            action = np.squeeze(action, axis=0)
        return np.asarray(action, dtype=np.float32)


@dataclass
class MoEArtifacts:
    run_dir: str
    checkpoint_path: str
    gate_onnx_path: str
    alpha_onnx_path: str
    manifest_path: str
    summary_path: str


class MoETrainer:
    def __init__(self, repo_root, settings, log_callback=None, stop_callback=None):
        self.repo_root = os.path.abspath(repo_root)
        self.settings = dict(settings or {})
        self.env_id = str(self.settings.get("env_id", ""))
        self.log_callback = log_callback
        self.stop_callback = stop_callback

    def _log(self, message):
        text = str(message)
        if not text.endswith("\n"):
            text += "\n"
        if callable(self.log_callback):
            self.log_callback(text)
        print(text, end="")

    def _stop_requested(self):
        return bool(callable(self.stop_callback) and self.stop_callback())

    def _require_torch(self):
        if torch is None or nn is None or DataLoader is None:
            raise RuntimeError("MoE gate training requires torch. Install dependencies from requirements.txt.")

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
        height_map_settings = {
            "freq": 50,
            "scale": self._as_float(obs_scales.get("height_map", 1.0), 1.0),
            "size_x": self._as_float(height_map_cfg.get("size_x", 1.0), 1.0),
            "size_y": self._as_float(height_map_cfg.get("size_y", 0.6), 0.6),
            "res_x": self._as_int(height_map_cfg.get("res_x", 15), 15),
            "res_y": self._as_int(height_map_cfg.get("res_y", 9), 9),
        }
        settings_cfg["height_map"] = height_map_settings
        return settings_cfg

    def _base_config(self, terrain):
        env_table = self._load_yaml("config/env_table.yaml")
        random_table = self._load_yaml("config/random_table.yaml")
        env_cfg = dict(env_table[self.env_id])
        settings_cfg = self._make_settings_cfg(env_cfg)
        hardware = dict(env_cfg.get("hardware", {}))
        action_scales = list(env_cfg.get("action_scales", []))
        config = {
            "env": {
                "id": self.env_id,
                "terrain": terrain,
                "max_duration": 60.0,
                "position_command": False,
                "render": False,
                "quiet": True,
            },
            "settings": settings_cfg,
            "observation": settings_cfg,
            "random": {
                "precision": "medium",
                "sensor_noise": "none",
                "init_noise": 0.0,
                "sliding_friction": 0.0,
                "torsional_friction": 0.0,
                "rolling_friction": 0.0,
                "friction_loss": 0.0,
                "action_delay_prob": 0.0,
                "mass_noise": 0.0,
                "load": 0.0,
            },
            "random_table": random_table["random_table"],
            "hardware": hardware,
            "action_scales": action_scales,
            "action_clippings": env_cfg.get("action_clippings", []),
            "initial_positions": {"joints": {}},
            "actuator": env_cfg.get("actuator", {}),
        }
        return config

    def _dataset_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "dataset", "moe_gate")

    def _weights_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "weights", "moe_gate")

    def _make_artifacts(self, run_name="latest"):
        root = self._weights_root()
        os.makedirs(root, exist_ok=True)
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return MoEArtifacts(
            run_dir=run_dir,
            checkpoint_path=os.path.join(run_dir, "moe_gate.pt"),
            gate_onnx_path=os.path.join(run_dir, "moe_gate.onnx"),
            alpha_onnx_path=os.path.join(run_dir, "moe_alpha.onnx"),
            manifest_path=os.path.join(run_dir, "moe_policy_manifest.json"),
            summary_path=os.path.join(run_dir, "train_summary.json"),
        )

    def _obs_dim_map(self, settings_cfg, action_dim):
        height_cfg = settings_cfg.get("height_map", {}) if isinstance(settings_cfg.get("height_map", {}), dict) else {}
        dof_pos_dim = int(action_dim)
        dof_vel_dim = int(action_dim)
        env_id = str(self.env_id)
        if env_id.startswith("wheeldog") or env_id.startswith("bon"):
            dof_pos_dim = 12
            dof_vel_dim = int(action_dim)
        elif env_id.startswith("flamingo_light"):
            dof_pos_dim = 2
            dof_vel_dim = int(action_dim)
        elif env_id.startswith("flamingo"):
            dof_pos_dim = 6
            dof_vel_dim = int(action_dim)
        return {
            "dof_pos": dof_pos_dim,
            "dof_vel": dof_vel_dim,
            "last_action": int(action_dim),
            "ang_vel": 3,
            "projected_gravity": 3,
            "lin_vel_x": 1,
            "lin_vel_y": 1,
            "lin_vel_z": 1,
            "height_map": int(height_cfg.get("res_x", 0) or 0) * int(height_cfg.get("res_y", 0) or 0),
        }

    def _gate_feature_names(self):
        names = self.settings.get("gate_feature_names", None)
        if names:
            return [str(name) for name in names]
        return ["dof_pos", "dof_vel", "projected_gravity", "command_1", "command_2"]

    def _command_dim(self):
        env_table = self._load_yaml("config/env_table.yaml")
        env_cfg = dict(env_table[self.env_id])
        settings_cfg = self._make_settings_cfg(env_cfg)
        return int(settings_cfg.get("command_dim", 0) or 0)

    def _command_scales(self, command_dim):
        env_table = self._load_yaml("config/env_table.yaml")
        env_cfg = dict(env_table[self.env_id])
        settings_cfg = self._make_settings_cfg(env_cfg)
        raw_scales = settings_cfg.get("command_scales", {}) or {}
        scales = []
        for i in range(int(command_dim)):
            value = float(raw_scales.get(str(i), raw_scales.get(i, 1.0)) or 1.0)
            scales.append(max(abs(value), 1e-6))
        return np.asarray(scales, dtype=np.float32)

    def _gate_input_indices(self, obs_dim):
        env_table = self._load_yaml("config/env_table.yaml")
        env_cfg = dict(env_table[self.env_id])
        settings_cfg = self._make_settings_cfg(env_cfg)
        action_dim = len(env_cfg.get("action_scales", []) or [])
        dims = self._obs_dim_map(settings_cfg, action_dim)
        selected = set(self._gate_feature_names())
        indices = []
        offset = 0

        for _ in range(int(settings_cfg.get("stack_size", 1))):
            for name in settings_cfg.get("stacked_obs_order", []):
                dim = int(dims.get(name, 0))
                if dim <= 0:
                    continue
                if name in selected:
                    indices.extend(range(offset, offset + dim))
                offset += dim

        for name in settings_cfg.get("non_stacked_obs_order", []):
            dim = int(dims.get(name, 0))
            if dim <= 0:
                continue
            if name in selected:
                indices.extend(range(offset, offset + dim))
            offset += dim

        command_dim = int(settings_cfg.get("command_dim", 0) or 0)
        expected_obs_dim = offset + command_dim
        command_start = offset
        for command_index in range(command_dim):
            if f"command_{command_index}" in selected or "command" in selected:
                indices.append(command_start + command_index)
        if not indices:
            raise RuntimeError("Gate feature selection produced no input indices.")
        if expected_obs_dim != int(obs_dim):
            non_command_dim = max(1, int(obs_dim) - max(0, command_dim))
            self._log(
                f"[moe-train] warning: obs layout mismatch expected={expected_obs_dim} actual={obs_dim}; "
                "falling back to non-command obs features."
            )
            indices = list(range(non_command_dim))
        return np.asarray(indices, dtype=np.int64)

    def _make_gate_obs(self, obs, gate_indices, command_dim, command_scales):
        gate_obs = np.asarray(obs, dtype=np.float32)[:, gate_indices].copy()
        if command_dim <= 0:
            return gate_obs
        command_scales = np.asarray(command_scales, dtype=np.float32).reshape(-1)
        command_start = int(obs.shape[1]) - int(command_dim)
        for gate_col, obs_index in enumerate(gate_indices):
            command_index = int(obs_index) - command_start
            if 0 <= command_index < command_dim and command_index < command_scales.size:
                gate_obs[:, gate_col] = gate_obs[:, gate_col] / max(abs(float(command_scales[command_index])), 1e-6)
        return gate_obs.astype(np.float32)

    @staticmethod
    def _prefixed_name(name, prefix, external_map):
        if not name:
            return name
        if name in external_map:
            return external_map[name]
        return f"{prefix}{name}"

    def _append_prefixed_graph(self, target_nodes, target_initializers, source_model, prefix, external_input_name):
        graph = source_model.graph
        initializer_names = {init.name for init in graph.initializer}
        graph_inputs = [item for item in graph.input if item.name not in initializer_names]
        graph_input_names = [item.name for item in graph_inputs]
        if not graph_inputs:
            raise RuntimeError(f"ONNX graph '{prefix}' has no runtime input.")
        if len(graph_inputs) != 1:
            raise RuntimeError(
                f"Final fused MoE export currently supports single-input expert/gate ONNX graphs. "
                f"Graph '{prefix}' has inputs: {graph_input_names}"
            )
        runtime_input = graph_inputs[0]
        external_map = {runtime_input.name: external_input_name}

        for initializer in graph.initializer:
            copied = copy.deepcopy(initializer)
            copied.name = self._prefixed_name(copied.name, prefix, external_map)
            target_initializers.append(copied)

        for node in graph.node:
            copied = copy.deepcopy(node)
            copied.name = self._prefixed_name(copied.name, prefix, external_map) if copied.name else ""
            copied.input[:] = [self._prefixed_name(name, prefix, external_map) for name in copied.input]
            copied.output[:] = [self._prefixed_name(name, prefix, external_map) for name in copied.output]
            target_nodes.append(copied)

        if not graph.output:
            raise RuntimeError(f"ONNX graph '{prefix}' has no output.")
        return self._prefixed_name(graph.output[0].name, prefix, external_map), runtime_input, graph.output[0]

    @staticmethod
    def _renamed_value_info(value_info, name):
        copied = copy.deepcopy(value_info)
        copied.name = name
        return copied

    def export_fused_moe_onnx(self, policy_a_path, policy_b_path, gate_onnx_path, output_path):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception as exc:
            raise RuntimeError("Final MoE ONNX export requires the 'onnx' package.") from exc

        if not os.path.isfile(policy_a_path):
            raise RuntimeError(f"Policy A ONNX not found: {policy_a_path}")
        if not os.path.isfile(policy_b_path):
            raise RuntimeError(f"Policy B ONNX not found: {policy_b_path}")
        if not os.path.isfile(gate_onnx_path):
            raise RuntimeError(f"Gate ONNX not found: {gate_onnx_path}")

        policy_a = onnx.load(policy_a_path)
        policy_b = onnx.load(policy_b_path)
        gate = onnx.load(gate_onnx_path)

        nodes = []
        initializers = []
        a_out, obs_info, action_info = self._append_prefixed_graph(nodes, initializers, policy_a, "policy_a/", "obs")
        b_out, _, _ = self._append_prefixed_graph(nodes, initializers, policy_b, "policy_b/", "obs")
        alpha_raw, _, _ = self._append_prefixed_graph(nodes, initializers, gate, "gate/", "obs")

        one_const = helper.make_tensor("moe/one_const", TensorProto.FLOAT, [1], [1.0])
        initializers.append(one_const)
        nodes.extend([
            helper.make_node("Sub", ["moe/one_const", alpha_raw], ["moe/one_minus_alpha"], name="moe/one_minus_alpha"),
            helper.make_node("Mul", ["moe/one_minus_alpha", a_out], ["moe/weighted_a"], name="moe/weighted_a"),
            helper.make_node("Mul", [alpha_raw, b_out], ["moe/weighted_b"], name="moe/weighted_b"),
            helper.make_node("Add", ["moe/weighted_a", "moe/weighted_b"], ["action"], name="moe/action"),
        ])

        opsets = {}
        for model in (policy_a, policy_b, gate):
            for opset in model.opset_import:
                domain = opset.domain
                opsets[domain] = max(opsets.get(domain, 0), int(opset.version))
        opsets[""] = max(opsets.get("", 0), 17)

        graph = helper.make_graph(
            nodes,
            "MoEPolicy",
            [
                self._renamed_value_info(obs_info, "obs"),
            ],
            [
                self._renamed_value_info(action_info, "action"),
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            producer_name="cosim_act_net_moe_export",
            opset_imports=[helper.make_operatorsetid(domain, version) for domain, version in sorted(opsets.items())],
        )
        model.ir_version = max(policy_a.ir_version, policy_b.ir_version, gate.ir_version)
        onnx.checker.check_model(model)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model, output_path)
        self._log(f"[moe-export] fused MoE ONNX exported: {output_path}")
        return output_path

    def export_manual_moe_onnx(self, policy_a_path, policy_b_path, alpha, output_path):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception as exc:
            raise RuntimeError("Manual MoE ONNX export requires the 'onnx' package.") from exc

        if not os.path.isfile(policy_a_path):
            raise RuntimeError(f"Policy A ONNX not found: {policy_a_path}")
        if not os.path.isfile(policy_b_path):
            raise RuntimeError(f"Policy B ONNX not found: {policy_b_path}")

        alpha_value = min(1.0, max(0.0, float(alpha)))
        policy_a = onnx.load(policy_a_path)
        policy_b = onnx.load(policy_b_path)

        nodes = []
        initializers = []
        a_out, obs_info, action_info = self._append_prefixed_graph(nodes, initializers, policy_a, "policy_a/", "obs")
        b_out, _, _ = self._append_prefixed_graph(nodes, initializers, policy_b, "policy_b/", "obs")

        initializers.extend([
            helper.make_tensor("moe/manual_alpha", TensorProto.FLOAT, [1], [alpha_value]),
            helper.make_tensor("moe/manual_one_minus_alpha", TensorProto.FLOAT, [1], [1.0 - alpha_value]),
        ])
        nodes.extend([
            helper.make_node("Mul", ["moe/manual_one_minus_alpha", a_out], ["moe/weighted_a"], name="moe/weighted_a"),
            helper.make_node("Mul", ["moe/manual_alpha", b_out], ["moe/weighted_b"], name="moe/weighted_b"),
            helper.make_node("Add", ["moe/weighted_a", "moe/weighted_b"], ["action"], name="moe/action"),
        ])

        opsets = {}
        for model in (policy_a, policy_b):
            for opset in model.opset_import:
                domain = opset.domain
                opsets[domain] = max(opsets.get(domain, 0), int(opset.version))
        opsets[""] = max(opsets.get("", 0), 17)

        graph = helper.make_graph(
            nodes,
            "ManualMoEPolicy",
            [
                self._renamed_value_info(obs_info, "obs"),
            ],
            [
                self._renamed_value_info(action_info, "action"),
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            producer_name="cosim_act_net_manual_moe_export",
            opset_imports=[helper.make_operatorsetid(domain, version) for domain, version in sorted(opsets.items())],
        )
        model.ir_version = max(policy_a.ir_version, policy_b.ir_version)
        onnx.checker.check_model(model)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model, output_path)
        manifest_path = os.path.splitext(output_path)[0] + ".json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "note": "Manual MoE ONNX graph with fixed alpha and explicit expert mixture.",
                "policy_a_path": policy_a_path,
                "policy_b_path": policy_b_path,
                "manual_alpha": alpha_value,
                "moe_onnx_path": output_path,
                "formula": "action = (1 - alpha) * policy_A(obs) + alpha * policy_B(obs)",
                "onnx_inputs": ["obs"],
                "onnx_outputs": ["action"],
            }, handle, indent=2)
        self._log(f"[moe-manual] exported alpha={alpha_value:.6f}: {output_path}")
        self._log(f"[moe-manual] manifest written: {manifest_path}")
        return output_path

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _alpha_label_from_height(self, height_map, terrain):
        hm = np.asarray(height_map, dtype=np.float32).reshape(-1)
        if hm.size == 0:
            hm = np.zeros((1,), dtype=np.float32)
        roughness = float(np.std(hm))
        height_range = float(np.max(hm) - np.min(hm))
        stairs_flag = 1.0 if "stairs" in str(terrain) else 0.0
        score = 10.0 * roughness + 6.0 * height_range + 2.0 * stairs_flag - 3.0
        return float(self._sigmoid(score))

    def _cmd_label_settings(self):
        threshold = max(0.0, float(self.settings.get("cmd_label_threshold", 0.2)))
        alpha = min(1.0, max(0.0, float(self.settings.get("cmd_label_alpha", 0.0))))
        flat_threshold = min(1.0, max(0.0, float(self.settings.get("cmd_label_flat_threshold", 0.15))))
        return threshold, alpha, flat_threshold

    def _adjust_alpha_label_for_flat_command(self, alpha_label, command, command_scales):
        threshold, alpha_cap, flat_threshold = self._cmd_label_settings()
        if threshold <= 0.0:
            threshold = 0.0
        if float(alpha_label) >= flat_threshold:
            return float(alpha_label)
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        command_scales = np.asarray(command_scales, dtype=np.float32).reshape(-1)
        if command.size < 3 or command_scales.size < 3:
            return float(alpha_label)
        command_unit = command[: command_scales.size] / np.maximum(np.abs(command_scales), 1e-6)
        cmd_motion = max(abs(float(command_unit[1])), abs(float(command_unit[2])))
        if cmd_motion >= threshold:
            return min(float(alpha_label), float(alpha_cap))
        return float(alpha_label)

    def _adjust_dataset_labels_for_flat_command(self, obs, alpha, command_dim, command_scales):
        threshold, alpha_cap, flat_threshold = self._cmd_label_settings()
        if command_dim < 3 or obs.shape[1] < command_dim:
            return alpha, 0
        command = obs[:, -command_dim:].astype(np.float32)
        command_scales = np.asarray(command_scales, dtype=np.float32).reshape(1, -1)
        if command_scales.shape[1] < 3:
            return alpha, 0
        command_unit = command / np.maximum(np.abs(command_scales), 1e-6)
        cmd_motion = np.maximum(np.abs(command_unit[:, 1]), np.abs(command_unit[:, 2])).reshape(-1, 1)
        mask = (alpha < flat_threshold) & (cmd_motion >= threshold)
        if not np.any(mask):
            return alpha, 0
        adjusted = alpha.copy()
        adjusted[mask] = np.minimum(adjusted[mask], float(alpha_cap))
        return adjusted, int(np.count_nonzero(mask))

    def _sample_command(self, rng, command_dim):
        lo = float(self.settings.get("command_min", -1.0))
        hi = float(self.settings.get("command_max", 1.0))
        if lo > hi:
            lo, hi = hi, lo
        return rng.uniform(lo, hi, size=(command_dim,)).astype(np.float32)

    @staticmethod
    def _inject_applied_command(state, env):
        state = np.asarray(state, dtype=np.float32).copy()
        command = np.asarray(getattr(env, "applied_command", []), dtype=np.float32).reshape(-1)
        if command.size > 0 and state.size >= command.size:
            state[-command.size:] = command
        return state

    def _is_near_boundary(self, env, boundary_m):
        try:
            data = env.get_data()
            x = float(data.qpos[0])
            y = float(data.qpos[1])
            return abs(x) > boundary_m or abs(y) > boundary_m
        except Exception:
            return False

    def collect(self):
        env_id = self.env_id
        if not env_id:
            raise RuntimeError("Select a robot/env for MoE data collection.")
        policy_a_path = str(self.settings.get("policy_a_path", "")).strip()
        policy_b_path = str(self.settings.get("policy_b_path", "")).strip()
        if not os.path.isfile(policy_a_path) or not os.path.isfile(policy_b_path):
            raise RuntimeError("Select valid ONNX files for Policy A and Policy B.")

        policy_a = OnnxExpertPolicy(policy_a_path)
        policy_b = OnnxExpertPolicy(policy_b_path)
        rng = np.random.default_rng(int(self.settings.get("seed", 42)))
        terrains = list(self.settings.get("terrains", [])) or ["flat"]
        total_samples = max(1, int(self.settings.get("samples", 200000)))
        rollout_steps = max(1, int(self.settings.get("rollout_steps", 1000)))
        boundary_m = max(1.0, float(self.settings.get("boundary_m", 8.0)))
        per_terrain_target = int(np.ceil(total_samples / max(1, len(terrains))))

        run_name = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self._dataset_root(), run_name)
        os.makedirs(run_dir, exist_ok=True)
        self._log(f"[moe-collect] writing dataset to {run_dir}")

        obs_all = []
        alpha_all = []
        terrain_all = []

        for terrain in terrains:
            if self._stop_requested():
                self._log("[moe-collect] stop requested before next terrain.")
                break
            self._log(f"[moe-collect] terrain={terrain} target={per_terrain_target}")
            config = self._base_config(terrain)
            env = build_env(config)
            command_scales = self._command_scales(env.command_dim)
            collected = 0
            try:
                state, _ = env.reset()
                policy_a.reset()
                policy_b.reset()
                command = self._sample_command(rng, env.command_dim)
                env.receive_user_command(command)
                state = self._inject_applied_command(state, env)
                steps_since_reset = 0
                while collected < per_terrain_target and len(obs_all) < total_samples:
                    if self._stop_requested():
                        self._log(f"[moe-collect] stop requested; terrain={terrain} collected={collected}")
                        break
                    last_obs = env.get_last_obs() or {}
                    alpha_label = self._alpha_label_from_height(last_obs.get("height_map", []), terrain)
                    alpha_label = self._adjust_alpha_label_for_flat_command(
                        alpha_label,
                        getattr(env, "applied_command", command),
                        command_scales,
                    )

                    action_a = policy_a.get_action(state)
                    action_b = policy_b.get_action(state)
                    action = (1.0 - alpha_label) * action_a + alpha_label * action_b

                    obs_all.append(np.asarray(state, dtype=np.float32).copy())
                    alpha_all.append(alpha_label)
                    terrain_all.append(str(terrain))
                    collected += 1

                    next_state, terminated, truncated, _ = env.step(action)
                    steps_since_reset += 1
                    if (
                        terminated
                        or truncated
                        or steps_since_reset >= rollout_steps
                        or self._is_near_boundary(env, boundary_m)
                    ):
                        state, _ = env.reset()
                        policy_a.reset()
                        policy_b.reset()
                        command = self._sample_command(rng, env.command_dim)
                        env.receive_user_command(command)
                        state = self._inject_applied_command(state, env)
                        steps_since_reset = 0
                    else:
                        state = next_state
                        if steps_since_reset % 50 == 0:
                            command = self._sample_command(rng, env.command_dim)
                            env.receive_user_command(command)
                            state = self._inject_applied_command(state, env)

                    if collected % 5000 == 0:
                        self._log(f"[moe-collect] terrain={terrain} collected={collected}/{per_terrain_target} total={len(obs_all)}/{total_samples}")
            finally:
                env.close()
            self._log(f"[moe-collect] terrain={terrain} done samples={collected}")
            if self._stop_requested():
                break

        if not obs_all:
            return {
                "env_id": env_id,
                "samples": 0,
                "stopped": True,
                "dataset_path": "",
            }

        dataset_path = os.path.join(run_dir, "dataset.npz")
        np.savez_compressed(
            dataset_path,
            obs=np.asarray(obs_all, dtype=np.float32),
            alpha_label=np.asarray(alpha_all, dtype=np.float32),
            terrain=np.asarray(terrain_all),
        )
        metadata = {
            "env_id": env_id,
            "samples": int(len(obs_all)),
            "terrains": terrains,
            "policy_a_path": policy_a_path,
            "policy_b_path": policy_b_path,
            "gate_input": "obs; privileged height_map is used only to create alpha_label during collection",
            "cmd_label_threshold": self._cmd_label_settings()[0],
            "cmd_label_alpha": self._cmd_label_settings()[1],
            "cmd_label_rule": "If alpha_label is flat and max(|command[1]|, |command[2]|) exceeds threshold, cap alpha_label.",
            "dataset_path": dataset_path,
            "stopped": self._stop_requested(),
        }
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        self._log(f"[moe-collect] saved {len(obs_all)} samples")
        return metadata

    def train(self):
        self._require_torch()
        dataset_paths = list(self.settings.get("selected_datasets", []))
        if not dataset_paths:
            raise RuntimeError("Select at least one MoE dataset before training.")
        dataset = MoEGateDataset(dataset_paths)
        if len(dataset) < 2:
            raise RuntimeError("At least 2 MoE samples are required to train the gate.")
        gate_indices = self._gate_input_indices(dataset.obs.shape[1])
        gate_feature_names = self._gate_feature_names()
        command_dim = self._command_dim()
        command_scales = self._command_scales(command_dim) if command_dim > 0 else np.ones((0,), dtype=np.float32)
        adjusted_alpha, adjusted_count = self._adjust_dataset_labels_for_flat_command(
            dataset.obs,
            dataset.alpha,
            command_dim,
            command_scales,
        )
        dataset.alpha = adjusted_alpha
        dataset.gate_obs = self._make_gate_obs(dataset.obs, gate_indices, command_dim, command_scales)
        lambda_cmd_alpha = max(0.0, float(self.settings.get("cmd_alpha_penalty", 0.0)))
        cmd_label_threshold, cmd_label_alpha, cmd_label_flat_threshold = self._cmd_label_settings()
        self._log(
            f"[moe-train] loaded samples={len(dataset)} obs_dim={dataset.obs.shape[1]} "
            f"gate_dim={len(gate_indices)} gate_features={gate_feature_names} "
            f"smoothness={float(self.settings.get('lambda_smooth', 0.0))} "
            f"cmd_alpha_penalty={lambda_cmd_alpha} command_scales={command_scales.tolist()} "
            f"cmd_label_threshold={cmd_label_threshold} cmd_label_alpha={cmd_label_alpha} "
            f"cmd_label_adjusted={adjusted_count}"
        )

        val_ratio = min(0.4, max(0.01, float(self.settings.get("val_ratio", 0.15))))
        val_count = max(1, int(len(dataset) * val_ratio))
        train_count = max(1, len(dataset) - val_count)
        generator = torch.Generator().manual_seed(int(self.settings.get("seed", 42)))
        train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)
        batch_size = max(1, int(self.settings.get("batch_size", 256)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        gate_indices_tensor = torch.as_tensor(gate_indices, dtype=torch.long)
        command_scales_tensor = torch.as_tensor(command_scales, dtype=torch.float32)

        gate = GateNet(int(len(gate_indices)))
        optimizer = torch.optim.Adam(gate.parameters(), lr=float(self.settings.get("learning_rate", 3e-4)))
        mse = nn.MSELoss()
        lambda_smooth = max(0.0, float(self.settings.get("lambda_smooth", 0.0)))
        epochs = max(1, int(self.settings.get("epochs", 50)))
        best_val = None
        best_state = None
        history = []

        for epoch in range(epochs):
            if self._stop_requested():
                self._log("[moe-train] stop requested; ending after last completed epoch.")
                break
            gate.train()
            train_loss_total = 0.0
            train_batches = 0
            for batch in train_loader:
                if self._stop_requested():
                    self._log("[moe-train] stop requested during training batch.")
                    break
                obs = batch["obs"].float()
                gate_obs = batch["gate_obs"].float()
                pred = gate(gate_obs)
                target = batch["alpha"].float()
                loss = mse(pred, target)
                if lambda_smooth > 0.0 and pred.shape[0] > 1:
                    loss = loss + lambda_smooth * torch.mean((pred[1:] - pred[:-1]) ** 2)
                if lambda_cmd_alpha > 0.0 and command_dim >= 3 and obs.shape[-1] >= command_dim:
                    command = obs[:, -command_dim:]
                    command_unit = command / command_scales_tensor
                    cmd_motion = torch.abs(command_unit[:, 1:3]).sum(dim=1, keepdim=True)
                    flat_mask = (target < 0.15).float()
                    weights = flat_mask * cmd_motion
                    denom = torch.clamp(weights.sum(), min=1e-6)
                    cmd_loss = torch.sum(weights * (pred ** 2)) / denom
                    loss = loss + lambda_cmd_alpha * cmd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.item())
                train_batches += 1

            if self._stop_requested():
                break

            gate.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if self._stop_requested():
                        self._log("[moe-train] stop requested during validation batch.")
                        break
                    gate_obs = batch["gate_obs"].float()
                    pred = gate(gate_obs)
                    loss = mse(pred, batch["alpha"].float())
                    val_loss_total += float(loss.item())
                    val_batches += 1
            if self._stop_requested():
                break
            train_loss = train_loss_total / max(1, train_batches)
            val_loss = val_loss_total / max(1, val_batches)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            self._log(f"[moe-train] epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in gate.state_dict().items()}

        if best_state is None:
            if self._stop_requested():
                return {
                    "env_id": self.env_id,
                    "samples": len(dataset),
                    "stopped": True,
                    "history": history,
                }
            raise RuntimeError("Gate training failed to produce a checkpoint.")
        gate.load_state_dict(best_state)
        artifacts = self._make_artifacts("latest")
        checkpoint = {
            "state_dict": gate.state_dict(),
            "gate_obs_dim": int(len(gate_indices)),
            "obs_dim": int(dataset.obs.shape[1]),
            "gate_input_indices": gate_indices.astype(int).tolist(),
            "gate_feature_names": gate_feature_names,
            "gate_command_dim": int(command_dim),
            "gate_command_scales": command_scales.astype(float).tolist(),
            "lambda_smooth": float(lambda_smooth),
            "cmd_alpha_penalty": float(lambda_cmd_alpha),
            "cmd_alpha_penalty_note": "Applied only when alpha_label < 0.15; penalizes alpha for nonzero command[1]/command[2].",
            "cmd_alpha_penalty_uses_unscaled_command": True,
            "cmd_label_threshold": float(cmd_label_threshold),
            "cmd_label_alpha": float(cmd_label_alpha),
            "cmd_label_flat_threshold": float(cmd_label_flat_threshold),
            "cmd_label_adjusted_count": int(adjusted_count),
            "history": history,
            "dataset_paths": dataset_paths,
            "policy_a_path": self.settings.get("policy_a_path", ""),
            "policy_b_path": self.settings.get("policy_b_path", ""),
        }
        torch.save(checkpoint, artifacts.checkpoint_path)
        summary = {
            "env_id": self.env_id,
            "samples": len(dataset),
            "best_val_loss": float(best_val),
            "checkpoint_path": artifacts.checkpoint_path,
            "gate_onnx_path": artifacts.gate_onnx_path,
            "alpha_onnx_path": artifacts.alpha_onnx_path,
            "moe_onnx_path": os.path.join(artifacts.run_dir, "moe_policy.onnx"),
            "manifest_path": artifacts.manifest_path,
            "gate_feature_names": gate_feature_names,
            "gate_dim": int(len(gate_indices)),
            "cmd_alpha_penalty": float(lambda_cmd_alpha),
            "history": history,
            "stopped": self._stop_requested(),
        }
        with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.export_onnx_from_checkpoint(artifacts.checkpoint_path, summary["moe_onnx_path"])
        return summary

    def export_onnx_from_checkpoint(self, checkpoint_path, output_path):
        self._require_torch()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        gate = GateNet(int(checkpoint["gate_obs_dim"]))
        gate.load_state_dict(checkpoint["state_dict"])
        gate.eval()
        obs_dim = int(checkpoint.get("obs_dim", checkpoint["gate_obs_dim"]))
        gate_indices = checkpoint.get("gate_input_indices", None)
        if gate_indices is None:
            if int(checkpoint["gate_obs_dim"]) != obs_dim:
                raise RuntimeError(
                    "This MoE checkpoint does not contain gate input indices. Retrain the gate with the current code."
                )
            gate_indices = list(range(obs_dim))
        gate_for_export = ObsSubsetGate(
            gate,
            gate_indices,
            command_dim=int(checkpoint.get("gate_command_dim", 0)),
            command_scales=checkpoint.get("gate_command_scales", []),
        )
        gate_for_export.eval()
        dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
        artifacts = self._make_artifacts("latest")
        gate_onnx_path = artifacts.gate_onnx_path
        if os.path.abspath(gate_onnx_path) == os.path.abspath(output_path):
            gate_onnx_path = os.path.join(artifacts.run_dir, "moe_gate_internal.onnx")
        torch.onnx.export(
            gate_for_export,
            dummy,
            gate_onnx_path,
            input_names=["obs"],
            output_names=["alpha"],
            opset_version=17,
        )
        if os.path.abspath(gate_onnx_path) != os.path.abspath(artifacts.alpha_onnx_path):
            shutil.copyfile(gate_onnx_path, artifacts.alpha_onnx_path)
        policy_a_path = checkpoint.get("policy_a_path", self.settings.get("policy_a_path", ""))
        policy_b_path = checkpoint.get("policy_b_path", self.settings.get("policy_b_path", ""))
        fused_path = self.export_fused_moe_onnx(policy_a_path, policy_b_path, gate_onnx_path, output_path)
        manifest = {
            "note": "Fused ONNX graph with Policy A, Policy B, GateNet, and explicit expert mixture.",
            "policy_a_path": policy_a_path,
            "policy_b_path": policy_b_path,
            "gate_onnx_path": gate_onnx_path,
            "alpha_onnx_path": artifacts.alpha_onnx_path,
            "moe_onnx_path": fused_path,
            "formula": "action = (1 - alpha) * policy_A(obs) + alpha * policy_B(obs)",
            "gate_feature_names": checkpoint.get("gate_feature_names", ["full_obs"]),
            "gate_input_indices": list(gate_indices),
            "gate_command_dim": int(checkpoint.get("gate_command_dim", 0)),
            "gate_command_scales": checkpoint.get("gate_command_scales", []),
            "onnx_inputs": ["obs"],
            "onnx_outputs": ["action"],
        }
        with open(artifacts.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self._log(f"[moe-export] gate ONNX exported: {gate_onnx_path}")
        self._log(f"[moe-export] alpha ONNX exported: {artifacts.alpha_onnx_path}")
        self._log(f"[moe-export] manifest written: {artifacts.manifest_path}")
        return fused_path
