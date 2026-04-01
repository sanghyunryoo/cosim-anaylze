import copy
import os

import numpy as np
import onnxruntime as ort


class MLPPolicy:
    def __init__(self, policy_path):
        self.policy_path = policy_path
        self.ort_session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def get_action(self, state: np.ndarray):
        state = state.astype(np.float32)
        try:
            batched_state = np.expand_dims(state, axis=0)
            action = self.ort_session.run(self.output_names, {self.input_name: batched_state})[0]
            action = np.squeeze(action, axis=0)
        except Exception:
            action = self.ort_session.run(self.output_names, {self.input_name: state})[0]
        return np.asarray(action, dtype=np.float32)


class LSTMPolicy:
    def __init__(self, config, policy_path):
        self.policy_path = policy_path
        self.ort_session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_names = [self.ort_session.get_inputs()[0].name, "h_in", "c_in"]
        assert self.ort_session.get_inputs()[1].name == "h_in" and self.ort_session.get_inputs()[2].name == "c_in", (
            "The input names of ONNX policy must include 'h_in' and 'c_in'"
        )

        self.h_in = np.zeros((1, 1, config["policy"]["h_in_dim"]), dtype=np.float32)
        self.c_in = np.zeros((1, 1, config["policy"]["c_in_dim"]), dtype=np.float32)

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        state = np.expand_dims(state, axis=0)
        policy_input = {
            self.input_names[0]: state,
            "h_in": self.h_in,
            "c_in": self.c_in,
        }
        action, h_out, c_out = self.ort_session.run(None, policy_input)
        self.h_in = h_out
        self.c_in = c_out

        action = np.squeeze(action, axis=0)
        return np.asarray(action, dtype=np.float32)


class EncoderPolicy:
    def __init__(self, encoder_path, policy_path):
        self.encoder_path = encoder_path
        self.policy_path = policy_path
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self.policy_sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self.enc_in_name = self.encoder_sess.get_inputs()[0].name
        self.enc_out_names = [o.name for o in self.encoder_sess.get_outputs()]

        self.pol_in_name = self.policy_sess.get_inputs()[0].name
        self.pol_out_names = [o.name for o in self.policy_sess.get_outputs()]

    def _ensure_batch(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.expand_dims(arr, axis=0)
        return arr

    def _run_encoder(self, obs_batched: np.ndarray) -> np.ndarray:
        out = self.encoder_sess.run(self.enc_out_names, {self.enc_in_name: obs_batched})[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        return out.astype(np.float32)

    def _concat(self, z: np.ndarray, obs_batched: np.ndarray) -> np.ndarray:
        if obs_batched.ndim > 2:
            obs_flat = obs_batched.reshape(obs_batched.shape[0], -1)
        else:
            obs_flat = obs_batched
        policy_in = np.concatenate([z, obs_flat], axis=-1)
        return policy_in.astype(np.float32)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = state.astype(np.float32)
        obs_batched = self._ensure_batch(state)
        z = self._run_encoder(obs_batched)
        policy_in = self._concat(z, obs_batched)
        try:
            action = self.policy_sess.run(self.pol_out_names, {self.pol_in_name: policy_in})[0]
            if action.ndim >= 2 and action.shape[0] == 1:
                action = np.squeeze(action, axis=0)
        except Exception:
            action = self.policy_sess.run(self.pol_out_names, {self.pol_in_name: policy_in.squeeze(axis=0)})[0]
        return np.asarray(action, dtype=np.float32)


class ResidualFineTunePolicy:
    def __init__(self, base_policy, config, policy_path, encoder_path=None):
        self.base_policy = base_policy
        self.config = config
        self.policy_path = policy_path
        self.encoder_path = encoder_path
        self.policy_type = str(config.get("policy", {}).get("policy_type", "MLP"))
        fine_tune_cfg = config.get("fine_tune", {}) if isinstance(config.get("fine_tune", {}), dict) else {}

        self.enabled = bool(fine_tune_cfg.get("enabled", False))
        self.max_samples = max(1, int(fine_tune_cfg.get("max_samples", 5000)))
        self.default_ridge_lambda = float(fine_tune_cfg.get("ridge_lambda", 1e-4))
        self.action_dim = max(0, int(config.get("hardware", {}).get("action_dim", 0)))

        self.manual_bias = np.zeros((self.action_dim,), dtype=np.float32)
        self.weight = None
        self.bias = None
        self.state_dim = None
        self.sample_states = []
        self.sample_targets = []
        self.sample_base_actions = []
        self._last_base_action = None
        self._last_final_action = None

    def _ensure_action_dim(self, action_dim: int):
        if self.action_dim <= 0:
            self.action_dim = int(action_dim)
        if self.manual_bias.shape[0] != self.action_dim:
            bias = np.zeros((self.action_dim,), dtype=np.float32)
            n = min(len(self.manual_bias), self.action_dim)
            if n > 0:
                bias[:n] = self.manual_bias[:n]
            self.manual_bias = bias
        if self.bias is not None and self.bias.shape[0] != self.action_dim:
            new_bias = np.zeros((self.action_dim,), dtype=np.float32)
            n = min(len(self.bias), self.action_dim)
            if n > 0:
                new_bias[:n] = self.bias[:n]
            self.bias = new_bias
        if self.weight is not None and self.weight.shape[1] != self.action_dim:
            self.weight = None
            self.bias = None

    def _ensure_state_dim(self, state_dim: int):
        if self.state_dim is None:
            self.state_dim = int(state_dim)
        if self.state_dim != int(state_dim):
            raise RuntimeError(
                f"Fine-tune state dimension changed from {self.state_dim} to {state_dim}. "
                "Please clear the session and start again."
            )

    def _predict_residual(self, state_vec: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if self.weight is None or self.bias is None:
            return np.zeros((self.action_dim,), dtype=np.float32)
        return (state_vec @ self.weight + self.bias).astype(np.float32)

    def _append_sample(self, state_vec: np.ndarray, residual_target: np.ndarray, base_action: np.ndarray):
        if not self.enabled:
            return
        self._ensure_state_dim(state_vec.shape[0])
        if len(self.sample_states) >= self.max_samples:
            self.sample_states.pop(0)
            self.sample_targets.pop(0)
            self.sample_base_actions.pop(0)
        self.sample_states.append(np.asarray(state_vec, dtype=np.float32).copy())
        self.sample_targets.append(np.asarray(residual_target, dtype=np.float32).copy())
        self.sample_base_actions.append(np.asarray(base_action, dtype=np.float32).copy())

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        base_action = np.asarray(self.base_policy.get_action(state), dtype=np.float32).reshape(-1)
        self._ensure_action_dim(base_action.shape[0])
        self._ensure_state_dim(state_vec.shape[0])

        learned_residual = self._predict_residual(state_vec)
        manual_bias = self.manual_bias[:self.action_dim]
        target_residual = learned_residual + manual_bias
        final_action = base_action + target_residual

        self._append_sample(state_vec, target_residual, base_action)
        self._last_base_action = base_action.copy()
        self._last_final_action = final_action.copy()
        return final_action.astype(np.float32)

    def set_fine_tune_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_max_samples(self, max_samples: int):
        self.max_samples = max(1, int(max_samples))
        while len(self.sample_states) > self.max_samples:
            self.sample_states.pop(0)
            self.sample_targets.pop(0)
            self.sample_base_actions.pop(0)

    def set_manual_bias(self, bias):
        arr = np.asarray(bias, dtype=np.float32).reshape(-1)
        self._ensure_action_dim(arr.shape[0])
        self.manual_bias[:] = 0.0
        n = min(arr.shape[0], self.action_dim)
        if n > 0:
            self.manual_bias[:n] = arr[:n]

    def clear_manual_bias(self):
        self.manual_bias[:] = 0.0

    def clear_samples(self):
        self.sample_states.clear()
        self.sample_targets.clear()
        self.sample_base_actions.clear()

    def fit_residual_head(self, ridge_lambda=None):
        if len(self.sample_states) == 0:
            raise RuntimeError("No fine-tune samples have been collected yet.")

        lam = self.default_ridge_lambda if ridge_lambda is None else float(ridge_lambda)
        x = np.asarray(self.sample_states, dtype=np.float32)
        y = np.asarray(self.sample_targets, dtype=np.float32)
        n_samples, state_dim = x.shape
        self._ensure_state_dim(state_dim)
        self._ensure_action_dim(y.shape[1])

        x_aug = np.concatenate([x, np.ones((n_samples, 1), dtype=np.float32)], axis=1)
        gram = x_aug.T @ x_aug
        reg = np.eye(gram.shape[0], dtype=np.float32) * lam
        reg[-1, -1] = 0.0
        try:
            beta = np.linalg.solve(gram + reg, x_aug.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + reg) @ (x_aug.T @ y)
        self.weight = beta[:-1, :].astype(np.float32)
        self.bias = beta[-1, :].astype(np.float32)
        learned = x @ self.weight + self.bias
        rmse = float(np.sqrt(np.mean((learned - y) ** 2)))

        fit_info = {
            "samples": int(n_samples),
            "state_dim": int(state_dim),
            "action_dim": int(self.action_dim),
            "ridge_lambda": float(lam),
            "rmse": rmse,
        }

        self.clear_samples()
        self.clear_manual_bias()
        return fit_info

    def get_fine_tune_status(self):
        return {
            "enabled": bool(self.enabled),
            "samples": len(self.sample_states),
            "trained": self.weight is not None and self.bias is not None,
            "state_dim": int(self.state_dim or 0),
            "action_dim": int(self.action_dim),
            "max_samples": int(self.max_samples),
            "manual_bias": self.manual_bias.astype(np.float32).copy(),
        }

    def export_merged_onnx(self, output_path: str):
        if self.weight is None or self.bias is None:
            raise RuntimeError("No trained fine-tune layer exists. Fit the residual head first.")
        if self.policy_type not in {"MLP", "LSTM"}:
            raise RuntimeError("Merged ONNX export currently supports only single-file MLP/LSTM policies.")

        try:
            import onnx
            from onnx import helper, numpy_helper
        except Exception as exc:
            raise RuntimeError(
                "The 'onnx' package is required to export a merged ONNX policy. "
                "Please install it first."
            ) from exc

        if not self.policy_path or not os.path.isfile(self.policy_path):
            raise RuntimeError("The original policy ONNX file could not be found.")

        model = onnx.load(self.policy_path)
        graph = model.graph
        if len(graph.input) == 0 or len(graph.output) == 0:
            raise RuntimeError("The ONNX graph must have at least one input and one output.")

        input_name = graph.input[0].name
        output_info = graph.output[0]
        base_output_name = output_info.name

        existing_value_names = {init.name for init in graph.initializer}
        for node in graph.node:
            existing_value_names.update(node.output)
        for value in graph.input:
            existing_value_names.add(value.name)
        for value in graph.output:
            existing_value_names.add(value.name)

        def _unique_name(seed: str) -> str:
            candidate = seed
            suffix = 0
            while candidate in existing_value_names:
                suffix += 1
                candidate = f"{seed}_{suffix}"
            existing_value_names.add(candidate)
            return candidate

        merged_output_name = _unique_name(f"{base_output_name}_merged")
        weight_name = _unique_name("fine_tune_residual_weight")
        bias_name = _unique_name("fine_tune_residual_bias")
        matmul_out = _unique_name("fine_tune_residual_matmul")
        residual_out = _unique_name("fine_tune_residual")

        graph.output[0].name = merged_output_name

        graph.initializer.extend([
            numpy_helper.from_array(np.asarray(self.weight, dtype=np.float32), name=weight_name),
            numpy_helper.from_array(np.asarray(self.bias, dtype=np.float32), name=bias_name),
        ])
        graph.node.extend([
            helper.make_node("MatMul", inputs=[input_name, weight_name], outputs=[matmul_out], name=_unique_name("FineTuneResidualMatMul")),
            helper.make_node("Add", inputs=[matmul_out, bias_name], outputs=[residual_out], name=_unique_name("FineTuneResidualBiasAdd")),
            helper.make_node("Add", inputs=[base_output_name, residual_out], outputs=[merged_output_name], name=_unique_name("FineTuneResidualMerge")),
        ])

        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        return os.path.abspath(output_path)


class UnsupportedFineTunePolicy:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.base_policy.get_action(state)

    def set_fine_tune_enabled(self, enabled: bool):
        return None

    def set_max_samples(self, max_samples: int):
        return None

    def set_manual_bias(self, bias):
        return None

    def clear_manual_bias(self):
        return None

    def clear_samples(self):
        return None

    def fit_residual_head(self, ridge_lambda=None):
        raise RuntimeError("Fine-tuning is not available for the current policy.")

    def get_fine_tune_status(self):
        return {
            "enabled": False,
            "samples": 0,
            "trained": False,
            "state_dim": 0,
            "action_dim": 0,
            "max_samples": 0,
            "manual_bias": np.zeros((0,), dtype=np.float32),
        }

    def export_merged_onnx(self, output_path: str):
        raise RuntimeError("Merged ONNX export is not available for the current policy.")


def build_policy(config, policy_path, encoder_path=None):
    policy_type = config["policy"]["policy_type"]
    if policy_type == "MLP":
        base_policy = MLPPolicy(policy_path)
    elif policy_type == "LSTM":
        base_policy = LSTMPolicy(config, policy_path)
    elif policy_type == "Encoder+MLP":
        base_policy = EncoderPolicy(encoder_path=encoder_path, policy_path=policy_path)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    try:
        return ResidualFineTunePolicy(
            base_policy=base_policy,
            config=copy.deepcopy(config),
            policy_path=policy_path,
            encoder_path=encoder_path,
        )
    except Exception:
        return UnsupportedFineTunePolicy(base_policy)
