import copy
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from envs.build import build_env


def _unique_name(existing_names, seed):
    candidate = seed
    suffix = 0
    while candidate in existing_names:
        suffix += 1
        candidate = f"{seed}_{suffix}"
    existing_names.add(candidate)
    return candidate


def _find_state_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "stacked_obs_order") and hasattr(current, "_stacked_obs_dim"):
            return current
        next_env = getattr(current, "env", None)
        if next_env is current:
            break
        current = next_env
    raise RuntimeError("Could not locate the environment observation wrapper.")


def _last_action_layout(config):
    env_config = copy.deepcopy(config or {})
    env_config.setdefault("env", {})
    env_config["env"]["render"] = False
    env_config["env"]["quiet"] = True
    env = build_env(env_config)
    try:
        state_wrapper = _find_state_wrapper(env)
        order = list(state_wrapper.stacked_obs_order)
        if "last_action" not in order:
            raise RuntimeError(
                "This environment does not include 'last_action' in its stacked observation, "
                "so a same-input stateless smooth ONNX cannot be exported safely."
            )
        action_dim = int(env.action_dim)
        source_env = state_wrapper.env
        offset = 0
        for name in order:
            dim = int(source_env.obs_to_dim.get(name, 0))
            if name == "last_action":
                break
            offset += dim
        last_action_dim = int(source_env.obs_to_dim.get("last_action", 0))
        if last_action_dim != action_dim:
            raise RuntimeError(
                f"last_action dimension {last_action_dim} does not match environment action dimension {action_dim}."
            )
        scale_config = state_wrapper.settings_cfg.get("last_action", {})
        if isinstance(scale_config, dict):
            last_action_scale = float(scale_config.get("scale", 1.0))
        else:
            last_action_scale = 1.0
        if abs(last_action_scale) < 1e-8:
            raise RuntimeError("The last_action observation scale must not be zero.")
        return {
            "observation_dim": int(env.state_dim),
            "action_dim": action_dim,
            "last_action_start": int(offset),
            "last_action_end": int(offset + last_action_dim),
            "last_action_scale": float(last_action_scale),
        }
    finally:
        env.close()


def _verify_export(source_path, output_path, input_name, last_action_start, last_action_end, last_action_scale, alpha):
    source_session = ort.InferenceSession(source_path, providers=["CPUExecutionProvider"])
    smooth_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    source_input = source_session.get_inputs()[0]
    input_shape = list(source_input.shape)
    obs_dim = input_shape[-1]
    if not isinstance(obs_dim, int) or obs_dim <= 0:
        obs_dim = int(last_action_end)
    observation = np.linspace(-0.25, 0.25, int(obs_dim), dtype=np.float32)[None, :]
    raw_action = np.asarray(source_session.run(None, {input_name: observation})[0], dtype=np.float32)
    smooth_action = np.asarray(smooth_session.run(None, {input_name: observation})[0], dtype=np.float32)
    previous_action = observation[:, last_action_start:last_action_end] / float(last_action_scale)
    expected = float(alpha) * raw_action + (1.0 - float(alpha)) * previous_action
    max_abs_error = float(np.max(np.abs(smooth_action - expected)))
    if not np.allclose(smooth_action, expected, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"Exported smooth ONNX verification failed (max absolute error {max_abs_error:.8f}).")
    return max_abs_error


def export_smoothed_onnx(policy_path, config, alpha, output_path=None):
    """Wrap a single-input policy with exact EMA using last_action already in obs."""
    policy_path = os.path.abspath(str(policy_path))
    if not os.path.isfile(policy_path):
        raise RuntimeError(f"Source ONNX policy was not found: {policy_path}")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise RuntimeError("Alpha must be greater than 0 and no greater than 1.")

    layout = _last_action_layout(config)
    model = onnx.load(policy_path)
    graph = model.graph
    if len(graph.input) != 1 or not graph.output:
        raise RuntimeError("Smooth ONNX export supports a policy with one observation input and at least one output.")
    input_info = graph.input[0]
    output_info = graph.output[0]
    input_shape = list(input_info.type.tensor_type.shape.dim)
    if len(input_shape) != 2:
        raise RuntimeError("Smooth ONNX export supports [batch, observation] policy inputs only.")
    input_dim = input_shape[-1].dim_value
    if input_dim and int(input_dim) != layout["observation_dim"]:
        raise RuntimeError(
            f"Observation dimension mismatch: environment={layout['observation_dim']}, ONNX={input_dim}."
        )
    output_shape = list(output_info.type.tensor_type.shape.dim)
    if len(output_shape) != 2:
        raise RuntimeError("Smooth ONNX export supports [batch, action] policy outputs only.")
    output_dim = output_shape[-1].dim_value
    if output_dim and int(output_dim) != layout["action_dim"]:
        raise RuntimeError(f"Action dimension mismatch: environment={layout['action_dim']}, ONNX={output_dim}.")
    if input_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError("Smooth ONNX export currently supports float32 observation inputs only.")

    existing_names = {item.name for item in graph.initializer}
    for node in graph.node:
        existing_names.update(node.input)
        existing_names.update(node.output)
    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        existing_names.add(item.name)

    action_output_name = output_info.name
    raw_action_name = _unique_name(existing_names, f"{action_output_name}_raw")
    for node in graph.node:
        node.output[:] = [raw_action_name if name == action_output_name else name for name in node.output]
        node.input[:] = [raw_action_name if name == action_output_name else name for name in node.input]

    starts_name = _unique_name(existing_names, "smooth_last_action_starts")
    ends_name = _unique_name(existing_names, "smooth_last_action_ends")
    axes_name = _unique_name(existing_names, "smooth_last_action_axes")
    steps_name = _unique_name(existing_names, "smooth_last_action_steps")
    alpha_name = _unique_name(existing_names, "smooth_raw_alpha")
    previous_weight_name = _unique_name(existing_names, "smooth_previous_alpha")
    previous_obs_name = _unique_name(existing_names, "smooth_previous_action_obs")
    raw_weighted_name = _unique_name(existing_names, "smooth_raw_weighted")
    previous_weighted_name = _unique_name(existing_names, "smooth_previous_weighted")

    graph.initializer.extend([
        numpy_helper.from_array(np.asarray([layout["last_action_start"]], dtype=np.int64), name=starts_name),
        numpy_helper.from_array(np.asarray([layout["last_action_end"]], dtype=np.int64), name=ends_name),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=axes_name),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=steps_name),
        numpy_helper.from_array(np.asarray([alpha], dtype=np.float32), name=alpha_name),
        numpy_helper.from_array(
            np.asarray([(1.0 - alpha) / layout["last_action_scale"]], dtype=np.float32),
            name=previous_weight_name,
        ),
    ])
    graph.node.extend([
        helper.make_node(
            "Slice",
            inputs=[input_info.name, starts_name, ends_name, axes_name, steps_name],
            outputs=[previous_obs_name],
            name=_unique_name(existing_names, "SmoothExtractLastAction"),
        ),
        helper.make_node(
            "Mul",
            inputs=[raw_action_name, alpha_name],
            outputs=[raw_weighted_name],
            name=_unique_name(existing_names, "SmoothWeightRawAction"),
        ),
        helper.make_node(
            "Mul",
            inputs=[previous_obs_name, previous_weight_name],
            outputs=[previous_weighted_name],
            name=_unique_name(existing_names, "SmoothWeightPreviousAction"),
        ),
        helper.make_node(
            "Add",
            inputs=[raw_weighted_name, previous_weighted_name],
            outputs=[action_output_name],
            name=_unique_name(existing_names, "SmoothActionEMA"),
        ),
    ])

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    onnx.checker.check_model(model)
    if output_path is None:
        root, _ = os.path.splitext(policy_path)
        output_path = root + "_smoothed.onnx"
    output_path = os.path.abspath(str(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)

    verification_error = _verify_export(
        policy_path,
        output_path,
        input_info.name,
        layout["last_action_start"],
        layout["last_action_end"],
        layout["last_action_scale"],
        alpha,
    )
    manifest_path = os.path.splitext(output_path)[0] + ".json"
    summary = {
        "source_onnx_path": policy_path,
        "onnx_path": output_path,
        "manifest_path": manifest_path,
        "alpha": alpha,
        "onnx_inputs": [input_info.name],
        "onnx_outputs": [action_output_name],
        "last_action_observation_slice": [layout["last_action_start"], layout["last_action_end"]],
        "last_action_scale": layout["last_action_scale"],
        "verification_max_abs_error": verification_error,
        "note": "Exact EMA graph wrapper: action = alpha * source_policy(obs) + (1-alpha) * previous_action_from_obs. No policy training was performed.",
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
