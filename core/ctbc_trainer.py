import json
import os
import copy
import tempfile

import numpy as np

from core.homing_trainer import HomingActorCritic, HomingArtifacts, HomingPolicyNet, HomingTrainer, torch
from core.moe_trainer import OnnxExpertPolicy
from envs.build import build_env


class CtbcTrainer(HomingTrainer):
    def _weights_root(self):
        return os.path.join(self.repo_root, "envs", self.env_id, "weights", "ctbc")

    def _make_artifacts(self, run_name="latest"):
        root = self._weights_root()
        os.makedirs(root, exist_ok=True)
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return HomingArtifacts(
            run_dir=run_dir,
            checkpoint_path=os.path.join(run_dir, "ctbc_policy.pt"),
            onnx_path=os.path.join(run_dir, "ctbc_policy.onnx"),
            manifest_path=os.path.join(run_dir, "ctbc_policy_manifest.json"),
            summary_path=os.path.join(run_dir, "train_summary.json"),
        )

    def _ppo_checkpoint_path(self):
        return os.path.join(self._weights_root(), "latest", "ctbc_policy_ppo.pt")

    @staticmethod
    def _prefixed_onnx_name(name, prefix, external_map):
        if not name:
            return name
        if name in external_map:
            return external_map[name]
        return f"{prefix}{name}"

    def _append_prefixed_onnx_graph(self, target_nodes, target_initializers, source_model, prefix, external_input_name):
        graph = source_model.graph
        initializer_names = {init.name for init in graph.initializer}
        graph_inputs = [item for item in graph.input if item.name not in initializer_names]
        if len(graph_inputs) != 1:
            names = [item.name for item in graph_inputs]
            raise RuntimeError(f"CTBC standalone export supports single-input ONNX policies only. Graph '{prefix}' inputs={names}")
        runtime_input = graph_inputs[0]
        external_map = {runtime_input.name: external_input_name}
        for initializer in graph.initializer:
            copied = copy.deepcopy(initializer)
            copied.name = self._prefixed_onnx_name(copied.name, prefix, external_map)
            target_initializers.append(copied)
        for node in graph.node:
            copied = copy.deepcopy(node)
            copied.name = self._prefixed_onnx_name(copied.name, prefix, external_map) if copied.name else ""
            copied.input[:] = [self._prefixed_onnx_name(name, prefix, external_map) for name in copied.input]
            copied.output[:] = [self._prefixed_onnx_name(name, prefix, external_map) for name in copied.output]
            target_nodes.append(copied)
        if not graph.output:
            raise RuntimeError(f"ONNX graph '{prefix}' has no output.")
        return self._prefixed_onnx_name(graph.output[0].name, prefix, external_map), runtime_input, graph.output[0]

    @staticmethod
    def _renamed_onnx_value_info(value_info, name):
        copied = copy.deepcopy(value_info)
        copied.name = name
        return copied

    def _wheel_contact_forces(self, env):
        try:
            import mujoco
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None:
                return np.zeros((2,), dtype=np.float32)
            wheel_body_ids = []
            for side in ("left", "right"):
                body_id = -1
                for name in (f"{side}_wheel_link", f"{side}_foot_link"):
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    except Exception:
                        body_id = -1
                    if body_id >= 0:
                        break
                wheel_body_ids.append(int(body_id))
            forces = np.zeros((2,), dtype=np.float32)
            force6 = np.zeros(6, dtype=np.float64)
            for contact_index in range(int(getattr(data, "ncon", 0))):
                contact = data.contact[contact_index]
                body1 = int(model.geom_bodyid[int(contact.geom1)])
                body2 = int(model.geom_bodyid[int(contact.geom2)])
                try:
                    mujoco.mj_contactForce(model, data, contact_index, force6)
                    effort = float(np.linalg.norm(force6[:3]))
                except Exception:
                    effort = 0.0
                for side_index, body_id in enumerate(wheel_body_ids):
                    if body_id >= 0 and (body1 == body_id or body2 == body_id):
                        forces[side_index] += effort
            return forces
        except Exception:
            return np.zeros((2,), dtype=np.float32)

    def _non_wheel_contact_effort(self, env):
        try:
            import mujoco
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None:
                return 0.0
            safe_body_ids = set()
            for side in ("left", "right"):
                for name in (f"{side}_wheel_link", f"{side}_foot_link"):
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    except Exception:
                        body_id = -1
                    if body_id >= 0:
                        safe_body_ids.add(int(body_id))
            ground_id = -1
            for name in ("ground", "floor"):
                try:
                    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                except Exception:
                    ground_id = -1
                if ground_id >= 0:
                    break
            force6 = np.zeros(6, dtype=np.float64)
            effort = 0.0
            for contact_index in range(int(getattr(data, "ncon", 0))):
                contact = data.contact[contact_index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if ground_id >= 0 and geom1 != ground_id and geom2 != ground_id:
                    continue
                body1 = int(model.geom_bodyid[geom1])
                body2 = int(model.geom_bodyid[geom2])
                other_body = body2 if geom1 == ground_id else body1
                if other_body in safe_body_ids:
                    continue
                try:
                    mujoco.mj_contactForce(model, data, contact_index, force6)
                    effort += float(np.linalg.norm(force6[:3]))
                except Exception:
                    effort += 1.0
            return float(effort)
        except Exception:
            return 0.0

    def _set_stair_height(self, env, terrain, height):
        try:
            import mujoco
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None or not hasattr(model, "hfield_size"):
                return False
            hfield_id = -1
            try:
                obj_hfield = getattr(mujoco.mjtObj, "mjOBJ_HFIELD")
                hfield_id = mujoco.mj_name2id(model, obj_hfield, str(terrain))
            except Exception:
                hfield_id = -1
            if hfield_id < 0:
                for geom_name in ("ground", "floor"):
                    try:
                        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    except Exception:
                        geom_id = -1
                    if geom_id >= 0:
                        data_id = int(model.geom_dataid[int(geom_id)])
                        if data_id >= 0:
                            hfield_id = data_id
                            break
            if hfield_id < 0:
                return False
            model.hfield_size[int(hfield_id), 2] = float(height)
            mujoco.mj_forward(model, data)
            return True
        except Exception:
            return False

    def _side_action_indices(self, env, action_dim):
        leaf = self._leaf_env(env)
        names = list(getattr(leaf, "initial_joint_names", []) or [])
        groups = {
            0: {"hip": [], "shoulder": [], "leg": [], "wheel": [], "all": []},
            1: {"hip": [], "shoulder": [], "leg": [], "wheel": [], "all": []},
        }
        for index, name in enumerate(names[:action_dim]):
            lower = str(name).lower()
            side = None
            if lower.startswith("left") or lower.startswith("fl") or lower.startswith("rl"):
                side = 0
            elif lower.startswith("right") or lower.startswith("fr") or lower.startswith("rr"):
                side = 1
            if side is None:
                continue
            groups[side]["all"].append(index)
            if "wheel" in lower:
                groups[side]["wheel"].append(index)
            elif "shoulder" in lower:
                groups[side]["shoulder"].append(index)
            elif "leg" in lower or "knee" in lower:
                groups[side]["leg"].append(index)
            elif "hip" in lower:
                groups[side]["hip"].append(index)
        for side, fallback in ((0, [0]), (1, [1 if action_dim > 1 else 0])):
            if not groups[side]["all"]:
                groups[side]["all"] = list(fallback)
            for key in ("hip", "shoulder", "leg"):
                if not groups[side][key]:
                    groups[side][key] = list(groups[side]["all"][:1])
        return groups

    def _action_scales(self, env, action_dim):
        try:
            leaf = self._leaf_env(env)
            scales = np.asarray(getattr(leaf, "action_scaler", []), dtype=np.float32).reshape(-1)
            if scales.size:
                return self._pad_or_trim(scales, action_dim, fill=1.0)
        except Exception:
            pass
        return np.ones((int(action_dim),), dtype=np.float32)

    def _stacked_obs_slices(self, env):
        try:
            settings_cfg = getattr(env, "settings_cfg", {}) or {}
            leaf = self._leaf_env(env)
            obs_to_dim = dict(getattr(leaf, "obs_to_dim", {}) or {})
            order = list(settings_cfg.get("stacked_obs_order", []) or [])
            offset = 0
            slices = {}
            for name in order:
                dim = int(obs_to_dim.get(name, 0) or 0)
                slices[name] = slice(offset, offset + dim)
                offset += dim
            return slices, int(offset)
        except Exception:
            return {}, 0

    def _proprio_stair_trigger(self, state, env):
        obs = np.asarray(state, dtype=np.float32).reshape(-1)
        slices, frame_dim = self._stacked_obs_slices(env)
        if frame_dim <= 0 or obs.size < frame_dim:
            return False
        current = obs[:frame_dim]
        previous = obs[frame_dim:2 * frame_dim] if obs.size >= 2 * frame_dim else current
        grav_slice = slices.get("projected_gravity")
        ang_slice = slices.get("ang_vel")
        vel_slice = slices.get("dof_vel")
        tilt_proxy = 0.0
        grav_delta = 0.0
        ang_mag = 0.0
        vel_delta = 0.0
        if grav_slice is not None:
            grav = np.asarray(current[grav_slice], dtype=np.float32).reshape(-1)
            prev_grav = np.asarray(previous[grav_slice], dtype=np.float32).reshape(-1)
            if grav.size >= 2:
                tilt_proxy = float(np.linalg.norm(grav[:2]))
                grav_delta = float(np.linalg.norm(grav[:min(3, grav.size)] - prev_grav[:min(3, prev_grav.size)]))
        if ang_slice is not None:
            ang = np.asarray(current[ang_slice], dtype=np.float32).reshape(-1)
            ang_mag = float(np.linalg.norm(ang))
        if vel_slice is not None:
            vel = np.asarray(current[vel_slice], dtype=np.float32).reshape(-1)
            prev_vel = np.asarray(previous[vel_slice], dtype=np.float32).reshape(-1)
            n = min(int(vel.size), int(prev_vel.size))
            if n > 0:
                vel_delta = float(np.linalg.norm(vel[:n] - prev_vel[:n]) / max(1.0, np.sqrt(float(n))))
        return bool(
            tilt_proxy > float(self.settings.get("ctbc_trigger_tilt_proxy", 0.075))
            or grav_delta > float(self.settings.get("ctbc_trigger_gravity_delta", 0.018))
            or ang_mag > float(self.settings.get("ctbc_trigger_ang_vel", 0.055))
            or vel_delta > float(self.settings.get("ctbc_trigger_dof_vel_delta", 0.035))
        )

    def _proprio_gate_config(self, env):
        slices, _frame_dim = self._stacked_obs_slices(env)
        grav_slice = slices.get("projected_gravity", slice(0, 0))
        ang_slice = slices.get("ang_vel", slice(0, 0))
        gravity_indices = list(range(int(grav_slice.start or 0), int(min(grav_slice.stop or 0, (grav_slice.start or 0) + 2))))
        ang_indices = list(range(int(ang_slice.start or 0), int(ang_slice.stop or 0)))
        return {
            "gravity_xy_indices": gravity_indices,
            "ang_vel_indices": ang_indices,
            "threshold": float(self.settings.get("ctbc_onnx_gate_threshold", 0.16)),
            "softness": max(1e-4, float(self.settings.get("ctbc_onnx_gate_softness", 0.025))),
            "ang_gain": float(self.settings.get("ctbc_onnx_gate_ang_gain", 1.0)),
        }

    def _primitive_action_clip(self):
        return max(0.01, float(self.settings.get("ctbc_ff_clip", self.settings.get("ctbc_action_clip", 4.0))))

    def _ctbc_action_clip(self):
        return max(1.0, float(self.settings.get("ctbc_action_clip", self.settings.get("ctbc_ff_clip", 4.0))))

    def _ctbc_safety_scale(self, env):
        roll, pitch = self._base_roll_pitch(env)
        tilt = float(np.sqrt(roll * roll + pitch * pitch))
        warn = max(0.01, float(self.settings.get("ctbc_safe_tilt", 0.22)))
        emergency = max(warn + 1e-3, float(self.settings.get("ctbc_emergency_tilt", 0.34)))
        if tilt <= warn:
            scale = 1.0
        elif tilt >= emergency:
            scale = 0.0
        else:
            scale = (emergency - tilt) / (emergency - warn)
        return float(np.clip(scale, 0.0, 1.0)), tilt

    def _wheel_body_heights(self, env):
        heights = np.zeros((2,), dtype=np.float32)
        try:
            import mujoco
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None:
                return heights
            for side_index, prefixes in enumerate((("left", "fl", "rl"), ("right", "fr", "rr"))):
                z_values = []
                for prefix in prefixes:
                    for suffix in ("wheel_link", "foot_link", "leg_link"):
                        name = f"{prefix}_{suffix}" if prefix in ("left", "right") else f"{prefix.upper()}_{suffix}"
                        try:
                            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                        except Exception:
                            body_id = -1
                        if body_id >= 0:
                            z_values.append(float(data.xpos[int(body_id)][2]))
                if z_values:
                    heights[side_index] = float(np.max(z_values))
        except Exception:
            pass
        return heights

    def _base_position(self, env):
        try:
            import mujoco
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None:
                return np.zeros((3,), dtype=np.float32)
            body_id = -1
            for name in ("base_link", "trunk", "torso"):
                try:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                except Exception:
                    body_id = -1
                if body_id >= 0:
                    break
            if body_id >= 0:
                return np.asarray(data.xpos[int(body_id)][:3], dtype=np.float32)
        except Exception:
            pass
        return np.zeros((3,), dtype=np.float32)

    def _sample_ctbc_command(self, rng, command_dim):
        command_dim = int(command_dim)
        command = np.zeros((command_dim,), dtype=np.float32)
        if command_dim <= 0:
            return command
        x_min = float(self.settings.get("ctbc_command_x_min", 0.35))
        x_max = float(self.settings.get("ctbc_command_x_max", 0.70))
        lo, hi = min(x_min, x_max), max(x_min, x_max)
        command[0] = float(rng.uniform(lo, hi))
        if command_dim > 1:
            y_abs = max(0.0, float(self.settings.get("ctbc_command_y_abs", 0.03)))
            command[1] = float(rng.uniform(-y_abs, y_abs))
        if command_dim > 2:
            yaw_abs = max(0.0, float(self.settings.get("ctbc_command_yaw_abs", 0.05)))
            command[2] = float(rng.uniform(-yaw_abs, yaw_abs))
        return command

    def _heightmap_obstacle(self, env):
        try:
            obs = env.get_last_obs() or {}
            hm = np.asarray(obs.get("height_map", []), dtype=np.float32).reshape(-1)
            if hm.size == 0:
                return 0.0, -1
            leaf = self._leaf_env(env)
            res_x = int(getattr(leaf, "res_x", 0) or 0)
            res_y = int(getattr(leaf, "res_y", 0) or 0)
            if res_x <= 0 or res_y <= 0 or res_x * res_y != hm.size:
                res_x = int(round(np.sqrt(float(hm.size))))
                res_x = max(1, res_x)
                res_y = max(1, int(hm.size // res_x))
                hm = hm[:res_x * res_y]
            grid = hm.reshape(res_y, res_x)
            front_start = max(0, int(0.62 * res_x))
            near_end = max(1, int(0.45 * res_x))
            near = grid[:, :near_end]
            front = grid[:, front_start:]
            if front.size == 0:
                return 0.0, -1
            baseline = float(np.median(near)) if near.size else float(np.median(grid))
            obstacle = max(0.0, baseline - float(np.min(front)))

            mid_y = max(1, res_y // 2)
            right_band = front[:mid_y, :]
            left_band = front[mid_y:, :]
            left_obstacle = max(0.0, baseline - float(np.min(left_band))) if left_band.size else obstacle
            right_obstacle = max(0.0, baseline - float(np.min(right_band))) if right_band.size else obstacle
            side_hint = 0 if left_obstacle >= right_obstacle else 1
            return float(obstacle), int(side_hint)
        except Exception:
            return 0.0, -1

    def _stair_gate(self, env, contact_forces=None):
        _ = contact_forces
        obstacle, _side_hint = self._heightmap_obstacle(env)
        threshold = float(self.settings.get("ctbc_gate_height_threshold", 0.06))
        softness = max(1e-6, float(self.settings.get("ctbc_gate_height_softness", 0.025)))
        height_gate = 1.0 / (1.0 + np.exp(-(obstacle - threshold) / softness))
        return float(np.clip(height_gate, 0.0, 1.0))

    def _contact_spike_side(self, ctx, forces):
        forces = np.asarray(forces, dtype=np.float32).reshape(2)
        baseline = np.asarray(ctx.get("contact_baseline", forces), dtype=np.float32).reshape(2)
        alpha = float(np.clip(float(self.settings.get("ctbc_contact_baseline_alpha", 0.02)), 0.0, 1.0))
        baseline = (1.0 - alpha) * baseline + alpha * forces
        ctx["contact_baseline"] = baseline
        excess = forces - baseline
        threshold = float(self.settings.get("ctbc_contact_spike_threshold", 80.0))
        if float(np.max(excess)) > threshold:
            return int(np.argmax(excess))
        return -1

    def _smooth_gate(self, ctx, target):
        previous = float(ctx.get("gate", 0.0))
        rise = float(np.clip(float(self.settings.get("ctbc_gate_rise", 0.35)), 0.0, 1.0))
        fall = float(np.clip(float(self.settings.get("ctbc_gate_fall", 0.03)), 0.0, 1.0))
        rate = rise if float(target) > previous else fall
        gate = previous + rate * (float(target) - previous)
        gate = float(np.clip(gate, 0.0, 1.0))
        ctx["gate"] = gate
        return gate

    @staticmethod
    def _zero_actor_output(model):
        try:
            last = model.actor.net[-1]
            if hasattr(last, "weight"):
                last.weight.data.zero_()
            if hasattr(last, "bias") and last.bias is not None:
                last.bias.data.zero_()
        except Exception:
            pass

    def _feedforward_lift_action(self, ctx, action_dim, control_dt):
        side = int(ctx.get("active_lift_side", -1))
        if side < 0:
            return np.zeros((action_dim,), dtype=np.float32)
        elapsed = int(ctx.get("lift_step", 0)) * float(control_dt)
        period = max(0.02, float(self.settings.get("ctbc_lift_period", 0.75)))
        if elapsed >= period:
            ctx["active_lift_side"] = -1
            ctx["lift_step"] = 0
            cooldown_steps = max(1, int(float(self.settings.get("ctbc_lift_cooldown", 0.35)) / max(float(control_dt), 1e-6)))
            ctx["next_lift_step"] = int(ctx.get("step", 0)) + cooldown_steps
            return np.zeros((action_dim,), dtype=np.float32)
        amplitude = float(self.settings.get("ctbc_lift_amplitude", 0.90))
        progress = float(np.clip(elapsed / period, 0.0, 1.0))
        lift_phase = np.sin(np.pi * progress)
        sweep_phase = np.sin(np.pi * np.clip((progress - 0.15) / 0.70, 0.0, 1.0))
        retract_phase = np.sin(np.pi * np.clip(progress / 0.55, 0.0, 1.0))
        push_phase = np.sin(np.pi * np.clip((progress - 0.45) / 0.55, 0.0, 1.0))
        ff_target = np.zeros((action_dim,), dtype=np.float32)
        groups = ctx.get("side_action_indices", {})
        active = groups.get(side, {}) if isinstance(groups, dict) else {}
        stance = groups.get(1 - side, {}) if isinstance(groups, dict) else {}

        shoulder_gain = float(self.settings.get("ctbc_shoulder_gain", 0.50))
        leg_retract_gain = float(self.settings.get("ctbc_leg_gain", 0.0))
        leg_push_gain = float(self.settings.get("ctbc_leg_push_gain", 1.75))
        hip_gain = float(self.settings.get("ctbc_hip_gain", 0.0))
        stance_gain = float(self.settings.get("ctbc_stance_gain", 0.30))

        def add(indices, value):
            for action_index in indices or []:
                if 0 <= int(action_index) < action_dim:
                    ff_target[int(action_index)] += float(value)

        add(active.get("shoulder", []), amplitude * shoulder_gain * (0.35 * lift_phase + 0.65 * sweep_phase))
        add(active.get("leg", []), amplitude * (leg_retract_gain * retract_phase + leg_push_gain * push_phase))
        add(active.get("hip", []), amplitude * hip_gain * (0.5 * lift_phase + 0.5 * sweep_phase))
        add(stance.get("shoulder", []), -amplitude * stance_gain * lift_phase)
        add(stance.get("hip", []), -amplitude * 0.5 * stance_gain * lift_phase)
        scales = np.asarray(ctx.get("action_scales", np.ones((action_dim,), dtype=np.float32)), dtype=np.float32)
        scales = self._pad_or_trim(scales, action_dim, fill=1.0)
        if str(self.settings.get("ctbc_compensate_action_scale", "1")).strip().lower() not in ("0", "false", "no", "off"):
            ff = ff_target / np.maximum(np.abs(scales), 1e-6)
        else:
            ff = ff_target
        ff = np.clip(ff, -self._primitive_action_clip(), self._primitive_action_clip())
        ctx["lift_step"] = int(ctx.get("lift_step", 0)) + 1
        return ff

    def _reward(
        self,
        env,
        action,
        prev_action,
        prev_prev_action,
        base_action,
        ff_action,
        command,
        contact_forces,
        terminated,
        truncated,
        gate=0.0,
        clearance_baseline=None,
        base_height_baseline=0.0,
        prev_base_position=None,
        stair_height=0.0,
    ):
        dim = len(action)
        action = self._pad_or_trim(action, dim)
        prev_action = self._pad_or_trim(prev_action, dim)
        prev_prev_action = self._pad_or_trim(prev_prev_action, dim)
        base_action = self._pad_or_trim(base_action, dim)
        ff_action = self._pad_or_trim(ff_action, dim)
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        cmd_x = float(command[0]) if command.size else 0.0
        try:
            obs = env.get_last_obs() or {}
            vx = float(np.asarray(obs.get("lin_vel_x", [0.0])).reshape(-1)[0])
            vy = float(np.asarray(obs.get("lin_vel_y", [0.0])).reshape(-1)[0])
        except Exception:
            vx = 0.0
            vy = 0.0
        cmd_y = float(command[1]) if command.size > 1 else 0.0
        roll, pitch = self._base_roll_pitch(env)
        tilt = float(np.sqrt(roll * roll + pitch * pitch))
        contact_active = bool(float(gate) > float(self.settings.get("ctbc_gate_reward_threshold", 0.35)))
        lift_active = bool(np.any(np.abs(ff_action) > 1e-6))
        action_rate = float(np.sqrt(np.mean((action - prev_action) ** 2)))
        action_accel = float(np.sqrt(np.mean((action - 2.0 * prev_action + prev_prev_action) ** 2)))
        base_rmse = float(np.sqrt(np.mean((action - base_action) ** 2)))
        action_clip = self._ctbc_action_clip()
        ff_rmse = float(np.sqrt(np.mean((action - np.clip(base_action + ff_action, -action_clip, action_clip)) ** 2)))
        non_wheel_contact = self._non_wheel_contact_effort(env)
        wheel_heights = self._wheel_body_heights(env)
        base_pos = self._base_position(env)
        stair_height = max(0.0, float(stair_height))
        clearance_ratio = max(0.0, float(self.settings.get("ctbc_clearance_stair_ratio", 0.90)))
        climb_ratio = max(0.0, float(self.settings.get("ctbc_climb_stair_ratio", 0.75)))
        clearance_target = max(1e-6, float(self.settings.get("ctbc_clearance_target", 0.14)), clearance_ratio * stair_height)
        if clearance_baseline is None:
            clearance_baseline = np.zeros_like(wheel_heights)
        clearance_baseline = self._pad_or_trim(clearance_baseline, wheel_heights.size, fill=0.0)
        wheel_clearance = float(np.max(wheel_heights - clearance_baseline)) if wheel_heights.size else 0.0
        wheel_clearance = max(0.0, wheel_clearance)
        clearance_score = float(np.clip(wheel_clearance / clearance_target, 0.0, 2.0))
        base_height_gain = max(0.0, float(base_pos[2]) - float(base_height_baseline))
        height_target = max(1e-6, float(self.settings.get("ctbc_base_height_target", 0.14)), climb_ratio * stair_height)
        height_gain_score = float(np.clip(base_height_gain / height_target, 0.0, 2.0))
        stair_clear_score = float(np.clip(wheel_clearance / max(clearance_ratio * stair_height, 1e-6), 0.0, 2.0)) if stair_height > 1e-6 else 0.0
        stair_climb_score = float(np.clip(base_height_gain / max(climb_ratio * stair_height, 1e-6), 0.0, 2.0)) if stair_height > 1e-6 else 0.0
        stair_success = float(stair_height > 1e-6 and stair_clear_score >= 1.0 and stair_climb_score >= 1.0)
        if prev_base_position is None:
            prev_base_position = base_pos
        prev_base_position = self._pad_or_trim(prev_base_position, 3, fill=0.0)
        forward_progress = max(0.0, float(base_pos[0]) - float(prev_base_position[0]))
        height_progress = max(0.0, float(base_pos[2]) - float(prev_base_position[2]))
        min_forward_progress = max(1e-6, float(self.settings.get("ctbc_min_forward_progress", 0.010)))
        stair_forward_score = float(np.clip(forward_progress / min_forward_progress, 0.0, 2.0))
        stair_motion_score = min(stair_clear_score, stair_climb_score, stair_forward_score)
        unsafe_tilt = tilt > float(self.settings.get("ctbc_terminate_tilt", 0.42))
        bad_contact_threshold = max(0.0, float(self.settings.get("ctbc_bad_contact_threshold", 1.0)))
        bad_contact = non_wheel_contact > bad_contact_threshold
        fallen = bool(
            terminated
            or unsafe_tilt
            or bad_contact
            or self._fall_signal(env, float(self.settings.get("reward_fall_height", 0.12)))
        )

        reward = 0.05
        reward += float(self.settings.get("reward_track", 1.2)) * np.exp(-20.0 * (cmd_x - vx) ** 2)
        reward += 0.5 * float(self.settings.get("reward_track", 1.2)) * np.exp(-20.0 * (cmd_y - vy) ** 2)
        reward -= float(self.settings.get("ctbc_base_imitation", 0.5)) * base_rmse
        reward -= float(self.settings.get("reward_upright", 2.0)) * min(tilt, 1.5)
        reward -= float(self.settings.get("reward_action_rate", 0.04)) * action_rate
        reward -= float(self.settings.get("reward_action_accel", 0.02)) * action_accel
        reward -= float(self.settings.get("ctbc_non_wheel_contact_penalty", 4.0)) * min(non_wheel_contact, 1000.0) / 100.0
        if bad_contact:
            reward -= float(self.settings.get("ctbc_bad_contact_penalty", 20.0))
        reward -= float(self.settings.get("ctbc_tilt_guard_penalty", 8.0)) * max(0.0, tilt - float(self.settings.get("ctbc_safe_tilt", 0.22)))
        if contact_active or lift_active:
            reward += float(self.settings.get("ctbc_reward_lift", 2.0)) * float(lift_active)
            reward += float(self.settings.get("ctbc_reward_clearance", 1.0)) * np.exp(-4.0 * ff_rmse)
            reward += float(self.settings.get("ctbc_reward_wheel_clearance", 4.0)) * clearance_score
            reward += float(self.settings.get("ctbc_reward_base_height", 4.0)) * height_gain_score
            reward += float(self.settings.get("ctbc_reward_forward_progress", 35.0)) * forward_progress
            reward += float(self.settings.get("ctbc_reward_stair_forward", 2.0)) * stair_forward_score
            reward += float(self.settings.get("ctbc_reward_stair_motion", 4.0)) * stair_motion_score
            reward += float(self.settings.get("ctbc_reward_height_progress", 30.0)) * height_progress
            reward += float(self.settings.get("ctbc_reward_balance_on_stair", 0.7)) * np.exp(-3.0 * tilt * tilt)
            reward += float(self.settings.get("ctbc_reward_stair_success", 5.0)) * stair_success
            if stair_height >= float(self.settings.get("ctbc_hard_stair_threshold", 0.14)):
                reward -= float(self.settings.get("ctbc_hard_stair_fail_penalty", 1.5)) * (1.0 - min(stair_clear_score, stair_climb_score, 1.0))
                reward -= float(self.settings.get("ctbc_no_progress_penalty", 1.0)) * max(0.0, 1.0 - stair_forward_score)
            if base_height_gain < float(self.settings.get("ctbc_min_climb_height", 0.015)):
                reward -= float(self.settings.get("ctbc_no_climb_penalty", 0.12)) * float(contact_active)
        if fallen:
            reward -= float(self.settings.get("reward_fall", 8.0))
        if truncated:
            reward -= 0.5
        metrics = {
            "lin_vel_x": vx,
            "lin_vel_y": vy,
            "vel_error": abs(cmd_x - vx),
            "lat_vel_error": abs(cmd_y - vy),
            "base_tilt": tilt,
            "base_rmse": base_rmse,
            "ff_rmse": ff_rmse,
            "action_rate": action_rate,
            "non_wheel_contact": non_wheel_contact,
            "contact_left": float(contact_forces[0]),
            "contact_right": float(contact_forces[1]),
            "contact_active": float(contact_active),
            "lift_active": float(lift_active),
            "wheel_clearance": wheel_clearance,
            "clearance_score": clearance_score,
            "base_height_gain": base_height_gain,
            "height_gain_score": height_gain_score,
            "stair_clear_score": stair_clear_score,
            "stair_climb_score": stair_climb_score,
            "stair_success": stair_success,
            "forward_progress": forward_progress,
            "stair_forward_score": stair_forward_score,
            "stair_motion_score": stair_motion_score,
            "height_progress": height_progress,
            "unsafe_tilt": float(unsafe_tilt),
            "bad_contact": float(bad_contact),
            "fallen": float(fallen),
        }
        return float(reward), bool(fallen or truncated), metrics

    def fine_tune(self):
        self._require_torch()
        if not self.env_id:
            raise RuntimeError("Select a robot/env for CTBC fine-tune.")
        policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid source ONNX policy.")

        seed = int(self.settings.get("seed", 42))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        terrain = str(self.settings.get("ctbc_terrain", "stairs_up_easy")).strip() or "stairs_up_easy"
        num_envs = max(1, int(self.settings.get("ppo_num_envs", 4)))
        total_steps = max(num_envs, int(self.settings.get("ppo_total_steps", 20000)))
        rollout_steps = max(8, int(self.settings.get("ppo_rollout_steps", 256)))
        ppo_epochs = max(1, int(self.settings.get("ppo_epochs", 4)))
        minibatch_size = max(8, int(self.settings.get("ppo_minibatch_size", 512)))
        hidden_dim = max(32, int(self.settings.get("hidden_dim", 256)))
        lr = max(1e-8, float(self.settings.get("ppo_learning_rate", 5e-5)))
        gamma = float(np.clip(float(self.settings.get("ppo_gamma", 0.99)), 0.0, 0.9999))
        gae_lambda = float(np.clip(float(self.settings.get("ppo_gae_lambda", 0.95)), 0.0, 1.0))
        clip_ratio = float(np.clip(float(self.settings.get("ppo_clip_ratio", 0.2)), 0.01, 0.5))
        entropy_coef = max(0.0, float(self.settings.get("ppo_entropy_coef", 0.0)))
        value_coef = max(0.0, float(self.settings.get("ppo_value_coef", 0.5)))
        bc_coef = max(0.0, float(self.settings.get("ppo_bc_coef", 0.5)))
        residual_limit = max(0.01, float(self.settings.get("ctbc_residual_limit", 4.0)))
        action_clip = self._ctbc_action_clip()
        max_grad_norm = max(0.0, float(self.settings.get("ppo_max_grad_norm", 0.5)))
        randomize_strength = float(np.clip(float(self.settings.get("ppo_domain_randomize", 0.05)), 0.0, 1.0))
        anneal_ratio = float(np.clip(float(self.settings.get("ctbc_anneal_ratio", 0.7)), 0.0, 1.0))
        max_episode_steps = max(64, int(self.settings.get("ctbc_episode_steps", rollout_steps * 4)))
        gate_height_threshold = float(self.settings.get("ctbc_gate_height_threshold", 0.06))
        lift_gate_threshold = float(self.settings.get("ctbc_gate_lift_threshold", 0.25))
        contact_spike_threshold = float(self.settings.get("ctbc_contact_spike_threshold", 80.0))
        lift_cooldown = float(self.settings.get("ctbc_lift_cooldown", 0.35))
        assist_trigger = float(self.settings.get("ctbc_assist_trigger_gate", 0.12))
        assist_gate_floor = float(self.settings.get("ctbc_assist_gate_floor", 0.85))
        assist_min = float(np.clip(float(self.settings.get("ctbc_assist_min", 0.0)), 0.0, 1.0))
        gate_residual = str(self.settings.get("ctbc_gate_residual_runtime", "0")).strip().lower() not in ("0", "false", "no", "off")
        anneal_bc = str(self.settings.get("ctbc_anneal_bc_with_assist", "1")).strip().lower() not in ("0", "false", "no", "off")
        distill_primitive = str(self.settings.get("ctbc_distill_primitive", "1")).strip().lower() not in ("0", "false", "no", "off")
        bc_weight_min = float(np.clip(float(self.settings.get("ctbc_bc_weight_min", 0.15)), 0.0, 1.0))
        force_alternating = str(self.settings.get("ctbc_force_alternating_lift", "1")).strip().lower() not in ("0", "false", "no", "off")
        stair_curriculum = str(self.settings.get("ctbc_curriculum_enabled", "1")).strip().lower() not in ("0", "false", "no", "off")
        stair_height_min = max(0.0, float(self.settings.get("ctbc_stair_height_min", 0.025)))
        stair_height_max = max(stair_height_min, float(self.settings.get("ctbc_stair_height_max", 0.20)))
        reflex_only = str(self.settings.get("ctbc_reflex_only", "1")).strip().lower() not in ("0", "false", "no", "off")
        reflex_samples = max(0, int(self.settings.get("ctbc_reflex_samples", 8192)))
        reflex_epochs = max(1, int(self.settings.get("ctbc_reflex_epochs", 12)))
        reflex_batch = max(16, int(self.settings.get("ctbc_reflex_batch", 256)))
        reflex_lr = max(1e-7, float(self.settings.get("ctbc_reflex_lr", 3e-4)))
        reflex_flat_ratio = float(np.clip(float(self.settings.get("ctbc_reflex_flat_ratio", 0.35)), 0.0, 0.95))
        reflex_gain = max(0.0, float(self.settings.get("ctbc_reflex_gain", 1.0)))
        reflex_teacher = str(self.settings.get("ctbc_reflex_teacher", "primitive")).strip().lower()
        reflex_proprio_trigger = str(self.settings.get("ctbc_reflex_proprio_trigger", "0")).strip().lower() not in ("0", "false", "no", "off")
        fast_teacher_steps = max(0, int(self.settings.get("ctbc_fast_teacher_steps", 4096)))
        fast_teacher_epochs = max(1, int(self.settings.get("ctbc_fast_teacher_epochs", 6)))
        fast_teacher_batch = max(16, int(self.settings.get("ctbc_fast_teacher_batch", 256)))
        fast_teacher_lr = max(1e-7, float(self.settings.get("ctbc_fast_teacher_lr", 2e-4)))
        fast_teacher_gain = max(0.0, float(self.settings.get("ctbc_fast_teacher_gain", 1.0)))
        fast_teacher_height = max(stair_height_min, float(self.settings.get("ctbc_fast_teacher_stair_height", min(stair_height_max, 0.12))))
        curriculum_ratio = max(1e-6, float(self.settings.get("ctbc_curriculum_ratio", 0.60)))
        select_after_ratio = float(np.clip(float(self.settings.get("ctbc_select_after_ratio", 0.65)), 0.0, 1.0))

        envs, policies, contexts = [], [], []
        try:
            for _ in range(num_envs):
                env = build_env(self._make_rl_config(terrain, rng, randomize_strength))
                envs.append(env)
                policies.append(OnnxExpertPolicy(policy_path))

            action_dim = int(envs[0].action_dim)
            input_dim = int(envs[0].state_dim)
            action_mask = np.ones((action_dim,), dtype=np.float32)
            model = HomingActorCritic(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=action_mask)
            self._zero_actor_output(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            control_dt = self._control_dt(envs[0])
            updates = int(np.ceil(total_steps / float(num_envs * rollout_steps)))

            def apply_stair_height(height):
                applied = 0
                for env in envs:
                    if self._set_stair_height(env, terrain, height):
                        applied += 1
                return applied

            def reset_task(env_index):
                env = envs[env_index]
                policy = policies[env_index]
                state, _ = env.reset()
                policy.reset()
                command = self._sample_ctbc_command(rng, env.command_dim)
                env.receive_user_command(command)
                state = self._inject_applied_command(state, env)
                last_action = np.zeros((action_dim,), dtype=np.float32)
                base_position = self._base_position(env)
                return {
                    "state": np.asarray(state, dtype=np.float32),
                    "command": command,
                    "step": 0,
                    "prev_action": last_action.copy(),
                    "prev_prev_action": last_action.copy(),
                    "episode_reward": 0.0,
                    "contact_baseline": self._wheel_contact_forces(env),
                    "active_lift_side": -1,
                    "lift_step": 0,
                    "next_lift_step": 0,
                    "lift_cycle_side": 0,
                    "side_action_indices": self._side_action_indices(env, action_dim),
                    "action_scales": self._action_scales(env, action_dim),
                    "clearance_baseline": self._wheel_body_heights(env),
                    "base_height_baseline": float(base_position[2]),
                    "prev_base_position": base_position.copy(),
                    "gate": 0.0,
                    "proprio_triggered": False,
                }

            current_stair_height = stair_height_min if stair_curriculum else stair_height_max
            stair_height_applied = apply_stair_height(current_stair_height)
            contexts = [reset_task(i) for i in range(num_envs)]
            self._log(
                f"[ctbc] envs={num_envs} terrain={terrain} total_steps={total_steps} rollout={rollout_steps} "
                f"gate_h={gate_height_threshold:.3f} lift_gate={lift_gate_threshold:.2f} "
                f"assist={assist_gate_floor:.2f}@{assist_trigger:.2f} "
                f"assist_min={assist_min:.2f} spike={contact_spike_threshold:.1f} "
                f"cooldown={lift_cooldown:.2f} anneal={anneal_ratio:.2f} "
                f"stair_h={current_stair_height:.3f}->{stair_height_max:.3f} applied={stair_height_applied}/{num_envs} "
                f"alt={int(force_alternating)} gate_res={int(gate_residual)} "
                f"bc_anneal={int(anneal_bc)} distill={int(distill_primitive)} bc_min={bc_weight_min:.2f} "
                f"reflex={int(reflex_only)} samples={reflex_samples} fast_teacher={fast_teacher_steps}@{fast_teacher_height:.3f}"
            )

            history, global_steps = [], 0
            best_score, best_snapshot = float("inf"), None
            if reflex_only:
                applied = apply_stair_height(stair_height_min)
                reflex_ctx = reset_task(0)
                reflex_policy = policies[0]
                reflex_env = envs[0]
                obs_samples, ref_samples = [], []
                trigger_samples = 0
                reflex_side = 0
                controller_params = {}
                if reflex_teacher in ("controller", "stair_controller"):
                    params_path = str(self.settings.get("ctbc_controller_params_path", "")).strip()
                    if not params_path:
                        params_path = os.path.join(self._weights_root(), "latest", "stair_controller_params.json")
                    if os.path.isfile(params_path):
                        with open(params_path, "r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        best_payload = payload.get("best", payload) if isinstance(payload, dict) else {}
                        controller_params = dict(best_payload.get("params", best_payload) if isinstance(best_payload, dict) else {})
                        self.settings.update({key: str(value) for key, value in controller_params.items() if str(key).startswith("ctbc_")})
                        self._log(f"[ctbc-reflex] using controller teacher params: {params_path}")
                    else:
                        self._log(f"[ctbc-reflex] controller teacher params not found, using primitive teacher: {params_path}")
                        reflex_teacher = "primitive"
                self._log(
                    f"[ctbc-reflex] collecting supervised detector/corrector samples={reflex_samples} "
                    f"flat_ratio={reflex_flat_ratio:.2f} stair_h={stair_height_min:.3f}->{stair_height_max:.3f} "
                    f"applied={applied}/{num_envs} teacher={reflex_teacher} proprio_trigger={int(reflex_proprio_trigger)}"
                )
                reflex_segment_steps = max(16, int(self.settings.get("ctbc_reflex_segment_steps", 128)))
                reflex_teacher_warmup_steps = max(0, int(self.settings.get("ctbc_reflex_teacher_warmup_steps", 24)))
                reflex_use_flat = True
                for sample_index in range(reflex_samples):
                    if sample_index == 0 or bool(reflex_ctx["step"] >= reflex_segment_steps):
                        reflex_use_flat = rng.random() < reflex_flat_ratio
                        target_height = 0.0 if reflex_use_flat else float(rng.uniform(max(1e-4, stair_height_min), stair_height_max))
                        apply_stair_height(target_height)
                        reflex_ctx = reset_task(0)
                        reflex_ctx["proprio_triggered"] = False
                        reflex_policy.reset()
                        if not reflex_use_flat and "ctbc_command_x" in controller_params and reflex_ctx["command"].size:
                            reflex_ctx["command"][0] = float(controller_params.get("ctbc_command_x", reflex_ctx["command"][0]))
                            if reflex_ctx["command"].size > 1:
                                reflex_ctx["command"][1] = 0.0
                            if reflex_ctx["command"].size > 2:
                                reflex_ctx["command"][2] = 0.0
                    reflex_env.receive_user_command(reflex_ctx["command"])
                    obs = np.asarray(reflex_ctx["state"], dtype=np.float32)
                    base = np.clip(
                        self._pad_or_trim(reflex_policy.get_action(obs), action_dim),
                        -1.0,
                        1.0,
                    )
                    if (not reflex_use_flat) and reflex_proprio_trigger:
                        if not bool(reflex_ctx.get("proprio_triggered", False)) and self._proprio_stair_trigger(obs, reflex_env):
                            reflex_ctx["proprio_triggered"] = True
                        trigger_samples += int(bool(reflex_ctx.get("proprio_triggered", False)))
                    teacher_enabled = (not reflex_use_flat) and (
                        (not reflex_proprio_trigger) or bool(reflex_ctx.get("proprio_triggered", False))
                    )
                    if reflex_use_flat or int(reflex_ctx.get("step", 0)) < reflex_teacher_warmup_steps or not teacher_enabled:
                        ref = np.zeros((action_dim,), dtype=np.float32)
                    else:
                        if int(reflex_ctx.get("active_lift_side", -1)) < 0:
                            reflex_ctx["active_lift_side"] = int(reflex_side)
                            reflex_ctx["lift_step"] = 0
                            reflex_ctx["next_lift_step"] = 0
                            reflex_side = 1 - int(reflex_side)
                        teacher_ref = self._feedforward_lift_action(reflex_ctx, action_dim, control_dt)
                        if reflex_teacher in ("controller", "stair_controller"):
                            teacher_ref = teacher_ref + self._controller_wheel_push(
                                reflex_ctx,
                                action_dim,
                                float(controller_params.get("ctbc_wheel_push_gain", self.settings.get("ctbc_wheel_push_gain", 0.0))),
                            )
                        ref = np.clip(reflex_gain * teacher_ref, -residual_limit, residual_limit).astype(np.float32)
                    action = np.clip(base + ref, -action_clip, action_clip)
                    state, terminated, truncated, _ = reflex_env.step(action)
                    state = self._inject_applied_command(state, reflex_env)
                    obs_samples.append(obs)
                    ref_samples.append(ref)
                    reflex_ctx["prev_prev_action"] = reflex_ctx["prev_action"].copy()
                    reflex_ctx["prev_action"] = action.copy()
                    reflex_ctx["prev_base_position"] = self._base_position(reflex_env).copy()
                    reflex_ctx["step"] += 1
                    reflex_ctx["state"] = np.asarray(state, dtype=np.float32)
                    if bool(terminated or truncated):
                        reflex_ctx = reset_task(0)
                        reflex_ctx["step"] = reflex_segment_steps
                        reflex_ctx["proprio_triggered"] = False
                        reflex_policy.reset()
                apply_stair_height(current_stair_height)
                if not obs_samples:
                    raise RuntimeError("CTBC reflex collection produced no samples.")
                obs_t = torch.from_numpy(np.asarray(obs_samples, dtype=np.float32)).float()
                ref_t = torch.from_numpy(np.asarray(ref_samples, dtype=np.float32)).float()
                reflex_optimizer = torch.optim.Adam(model.actor.parameters(), lr=reflex_lr)
                reflex_indices = np.arange(int(obs_t.shape[0]))
                reflex_loss_value = 0.0
                for _epoch in range(reflex_epochs):
                    rng.shuffle(reflex_indices)
                    for start in range(0, reflex_indices.size, reflex_batch):
                        mb_idx = reflex_indices[start:start + reflex_batch]
                        pred = model.actor(obs_t[mb_idx])
                        loss = (pred - ref_t[mb_idx]).pow(2).mean()
                        reflex_optimizer.zero_grad()
                        loss.backward()
                        if max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_grad_norm)
                        reflex_optimizer.step()
                        reflex_loss_value = float(loss.item())
                with torch.no_grad():
                    pred_t = model.actor(obs_t)
                    residual_rms = float(torch.sqrt((pred_t.pow(2)).mean()).item())
                    target_rms = float(torch.sqrt((ref_t.pow(2)).mean()).item())
                    zero_mask = torch.sqrt((ref_t.pow(2)).mean(dim=-1)) < 1e-6
                    flat_leak = float(torch.sqrt((pred_t[zero_mask].pow(2)).mean()).item()) if bool(zero_mask.any()) else 0.0
                best_snapshot = {
                    "update": 0,
                    "selection_score": float(reflex_loss_value),
                    "metrics": {
                        "reflex_loss": float(reflex_loss_value),
                        "reflex_samples": int(obs_t.shape[0]),
                        "target_residual_rmse": float(target_rms),
                        "residual_rmse": float(residual_rms),
                        "flat_leak_rmse": float(flat_leak),
                        "trigger_fraction": float(trigger_samples) / max(1, int(obs_t.shape[0])),
                    },
                    "actor_state": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
                    "critic_state": {k: v.detach().cpu().clone() for k, v in model.critic.state_dict().items()},
                    "log_std": model.log_std.detach().cpu().clone(),
                }
                history.append({
                    "update": 0,
                    "steps": int(reflex_samples),
                    "mean_reward": 0.0,
                    "mean_episode_return": 0.0,
                    "bc_loss": float(reflex_loss_value),
                    "residual_rmse": float(residual_rms),
                    "flat_leak_rmse": float(flat_leak),
                    "target_residual_rmse": float(target_rms),
                    "trigger_fraction": float(trigger_samples) / max(1, int(obs_t.shape[0])),
                    "stair_height": float(stair_height_max),
                    "selectable": 1.0,
                    "selection_score": float(reflex_loss_value),
                })
                global_steps = int(reflex_samples)
                self._log(
                    f"[ctbc-reflex] trained loss={reflex_loss_value:.5f} target={target_rms:.4f} "
                    f"res={residual_rms:.4f} flat_leak={flat_leak:.4f} "
                    f"trigger={float(trigger_samples) / max(1, int(obs_t.shape[0])):.3f}"
                )
                artifacts = self._make_artifacts("latest")
                checkpoint_path = self._ppo_checkpoint_path()
                torch.save({
                    "checkpoint_type": "ctbc_reflex_actor",
                    "actor_state": best_snapshot["actor_state"],
                    "critic_state": best_snapshot["critic_state"],
                    "log_std": best_snapshot["log_std"],
                    "input_dim": input_dim,
                    "action_dim": action_dim,
                    "hidden_dim": hidden_dim,
                    "action_mask": action_mask.astype(float).tolist(),
                    "residual_limit": float(residual_limit),
                    "source_policy_path": policy_path,
                    "settings": dict(self.settings),
                    "input_mode": "obs_residual",
                    "ctbc_reflex_teacher": reflex_teacher,
                    "ctbc_reflex_proprio_trigger": bool(reflex_proprio_trigger),
                    "ctbc_proprio_gate": self._proprio_gate_config(envs[0]) if bool(reflex_proprio_trigger) else None,
                    "ctbc_controller_params": controller_params,
                    "history": history,
                    "selected_update": 0,
                    "selected_metrics": best_snapshot["metrics"],
                }, checkpoint_path)
                summary = {
                    "env_id": self.env_id,
                    "mode": "ctbc_reflex",
                    "checkpoint_path": checkpoint_path,
                    "onnx_path": artifacts.onnx_path,
                    "manifest_path": artifacts.manifest_path,
                    "steps": int(global_steps),
                    "input_dim": input_dim,
                    "action_dim": action_dim,
                    "selected_update": 0,
                    "selected_metrics": best_snapshot["metrics"],
                    "history": history,
                    "stopped": self._stop_requested(),
                }
                with open(artifacts.summary_path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
                onnx_path = str(self.settings.get("output_path", "")).strip() or artifacts.onnx_path
                self.export_onnx_from_checkpoint(checkpoint_path, onnx_path)
                summary["onnx_path"] = onnx_path
                self._log(f"[ctbc-reflex] checkpoint saved: {checkpoint_path} loss={reflex_loss_value:.5f}")
                return summary
            if fast_teacher_steps > 0 and fast_teacher_gain > 0.0:
                applied = apply_stair_height(min(stair_height_max, fast_teacher_height))
                teacher_ctx = reset_task(0)
                teacher_policy = policies[0]
                teacher_env = envs[0]
                teacher_obs, teacher_ref = [], []
                teacher_loss_value = 0.0
                teacher_side = 0
                self._log(
                    f"[ctbc-teacher] warm-start steps={fast_teacher_steps} epochs={fast_teacher_epochs} "
                    f"height={min(stair_height_max, fast_teacher_height):.3f} applied={applied}/{num_envs} gain={fast_teacher_gain:.2f}"
                )
                for teacher_step in range(fast_teacher_steps):
                    teacher_env.receive_user_command(teacher_ctx["command"])
                    if int(teacher_ctx.get("active_lift_side", -1)) < 0:
                        teacher_ctx["active_lift_side"] = int(teacher_side)
                        teacher_ctx["lift_step"] = 0
                        teacher_ctx["next_lift_step"] = 0
                        teacher_side = 1 - int(teacher_side)
                    obs = np.asarray(teacher_ctx["state"], dtype=np.float32)
                    base = np.clip(
                        self._pad_or_trim(teacher_policy.get_action(obs), action_dim),
                        -1.0,
                        1.0,
                    )
                    ff = self._feedforward_lift_action(teacher_ctx, action_dim, control_dt)
                    ref = np.clip(fast_teacher_gain * ff, -residual_limit, residual_limit).astype(np.float32)
                    action = np.clip(base + ref, -action_clip, action_clip)
                    state, terminated, truncated, _ = teacher_env.step(action)
                    state = self._inject_applied_command(state, teacher_env)
                    teacher_obs.append(obs)
                    teacher_ref.append(ref)
                    teacher_ctx["prev_prev_action"] = teacher_ctx["prev_action"].copy()
                    teacher_ctx["prev_action"] = action.copy()
                    teacher_ctx["prev_base_position"] = self._base_position(teacher_env).copy()
                    teacher_ctx["step"] += 1
                    teacher_ctx["state"] = np.asarray(state, dtype=np.float32)
                    if bool(terminated or truncated or teacher_ctx["step"] >= max_episode_steps):
                        teacher_ctx = reset_task(0)
                if teacher_obs:
                    obs_t = torch.from_numpy(np.asarray(teacher_obs, dtype=np.float32)).float()
                    ref_t = torch.from_numpy(np.asarray(teacher_ref, dtype=np.float32)).float()
                    teacher_optimizer = torch.optim.Adam(model.actor.parameters(), lr=fast_teacher_lr)
                    teacher_indices = np.arange(int(obs_t.shape[0]))
                    for _epoch in range(fast_teacher_epochs):
                        rng.shuffle(teacher_indices)
                        for start in range(0, teacher_indices.size, fast_teacher_batch):
                            mb_idx = teacher_indices[start:start + fast_teacher_batch]
                            pred = model.actor(obs_t[mb_idx])
                            loss = (pred - ref_t[mb_idx]).pow(2).mean()
                            teacher_optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm > 0.0:
                                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_grad_norm)
                            teacher_optimizer.step()
                            teacher_loss_value = float(loss.item())
                    with torch.no_grad():
                        residual_rms = float(torch.sqrt((model.actor(obs_t).pow(2)).mean()).item())
                    self._log(
                        f"[ctbc-teacher] learned samples={int(obs_t.shape[0])} loss={teacher_loss_value:.5f} "
                        f"res={residual_rms:.4f}"
                    )
                apply_stair_height(current_stair_height)
                contexts = [reset_task(i) for i in range(num_envs)]
            for update in range(updates):
                if self._stop_requested():
                    break
                if stair_curriculum:
                    progress = min(1.0, float(update) / max(1.0, float(updates - 1) * curriculum_ratio))
                    current_stair_height = stair_height_min + (stair_height_max - stair_height_min) * progress
                    apply_stair_height(current_stair_height)
                obs_buf, act_buf, ref_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], [], []
                metric_sums, metric_count, completed_returns = {}, 0, []
                for _ in range(rollout_steps):
                    obs_np = np.stack([ctx["state"] for ctx in contexts], axis=0).astype(np.float32)
                    obs_t = torch.from_numpy(obs_np).float()
                    with torch.no_grad():
                        action_t, logp_t, _, value_t = model.act(obs_t)
                    raw_np = action_t.cpu().numpy().astype(np.float32)
                    delta_np = np.clip(raw_np, -residual_limit, residual_limit)
                    base_np = np.zeros_like(delta_np, dtype=np.float32)
                    for env_index, policy in enumerate(policies):
                        base_np[env_index] = np.clip(
                            self._pad_or_trim(policy.get_action(contexts[env_index]["state"]), action_dim),
                            -1.0,
                            1.0,
                        )
                    exec_np = base_np.copy()
                    ref_np = np.zeros_like(delta_np, dtype=np.float32)
                    anneal_steps = max(1, int(total_steps * anneal_ratio))
                    kff_raw = max(0.0, 1.0 - float(global_steps) / float(anneal_steps)) if anneal_ratio > 0.0 else 1.0
                    kff = max(assist_min, kff_raw)
                    bc_scale = kff if anneal_bc else 1.0
                    if distill_primitive:
                        bc_scale = max(bc_weight_min, bc_scale)
                    bc_weight = bc_coef * bc_scale
                    ff_np = np.zeros_like(delta_np)
                    gate_np = np.zeros((num_envs,), dtype=np.float32)
                    assist_np = np.zeros((num_envs,), dtype=np.float32)

                    for env_index, env in enumerate(envs):
                        ctx = contexts[env_index]
                        contact_forces = self._wheel_contact_forces(env)
                        gate = self._smooth_gate(ctx, self._stair_gate(env, contact_forces))
                        gate_np[env_index] = gate
                        safety_scale, _tilt_now = self._ctbc_safety_scale(env)
                        assist_gate = max(gate, assist_gate_floor) if gate > assist_trigger else 0.0
                        assist_gate *= safety_scale
                        assist_np[env_index] = assist_gate
                        side = -1 if force_alternating else self._contact_spike_side(ctx, contact_forces)
                        if side < 0 and gate > lift_gate_threshold:
                            if force_alternating:
                                side = int(ctx.get("lift_cycle_side", 0))
                            else:
                                heights = self._wheel_body_heights(env)
                                if heights.size >= 2 and np.any(heights):
                                    side = int(np.argmin(heights))
                                else:
                                    side = int(ctx.get("lift_cycle_side", 0))
                            if int(ctx.get("step", 0)) < int(ctx.get("next_lift_step", 0)):
                                side = -1
                        if side >= 0 and int(ctx.get("active_lift_side", -1)) < 0:
                            ctx["active_lift_side"] = int(side)
                            ctx["lift_step"] = 0
                            ctx["lift_cycle_side"] = 1 - int(side)
                        ff = self._feedforward_lift_action(ctx, action_dim, control_dt)
                        ff_np[env_index] = ff
                        ref_gain = 1.0 if distill_primitive else kff
                        ref_np[env_index] = assist_gate * ref_gain * ff
                        delta_scale = safety_scale * (gate if gate_residual else 1.0)
                        exec_np[env_index] = np.clip(
                            base_np[env_index] + delta_scale * delta_np[env_index] + assist_gate * kff * ff,
                            -action_clip,
                            action_clip,
                        )

                    rewards, dones = [], []
                    for env_index, env in enumerate(envs):
                        ctx = contexts[env_index]
                        env.receive_user_command(ctx["command"])
                        state, terminated, truncated, _ = env.step(exec_np[env_index])
                        state = self._inject_applied_command(state, env)
                        contact_forces = self._wheel_contact_forces(env)
                        reward, done, metrics = self._reward(
                            env,
                            exec_np[env_index],
                            ctx["prev_action"],
                            ctx["prev_prev_action"],
                            base_np[env_index],
                            assist_np[env_index] * kff * ff_np[env_index],
                            ctx["command"],
                            contact_forces,
                            terminated,
                            truncated,
                            gate=gate_np[env_index],
                            clearance_baseline=ctx.get("clearance_baseline", None),
                            base_height_baseline=ctx.get("base_height_baseline", 0.0),
                            prev_base_position=ctx.get("prev_base_position", None),
                            stair_height=current_stair_height,
                        )
                        done = bool(done or ctx["step"] >= max_episode_steps)
                        ctx["episode_reward"] += reward
                        metrics["ctbc_gate"] = float(gate_np[env_index])
                        metrics["residual_rmse"] = float(np.sqrt(np.mean(delta_np[env_index] ** 2)))
                        for key, value in metrics.items():
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                        metric_count += 1
                        rewards.append(reward)
                        dones.append(float(done))
                        ctx["prev_prev_action"] = ctx["prev_action"].copy()
                        ctx["prev_action"] = exec_np[env_index].copy()
                        ctx["prev_base_position"] = self._base_position(env).copy()
                        ctx["step"] += 1
                        ctx["state"] = np.asarray(state, dtype=np.float32)
                        if done:
                            completed_returns.append(ctx["episode_reward"])
                            contexts[env_index] = reset_task(env_index)

                    obs_buf.append(obs_np)
                    act_buf.append(raw_np)
                    ref_buf.append(ref_np)
                    logp_buf.append(logp_t.cpu().numpy().astype(np.float32))
                    rew_buf.append(np.asarray(rewards, dtype=np.float32))
                    done_buf.append(np.asarray(dones, dtype=np.float32))
                    val_buf.append(value_t.cpu().numpy().astype(np.float32))
                    global_steps += num_envs

                obs_arr = np.asarray(obs_buf, dtype=np.float32)
                act_arr = np.asarray(act_buf, dtype=np.float32)
                ref_arr = np.asarray(ref_buf, dtype=np.float32)
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
                flat_ref = torch.from_numpy(ref_arr.reshape(-1, action_dim)).float()
                flat_logp = torch.from_numpy(logp_arr.reshape(-1)).float()
                flat_adv = torch.from_numpy(adv.reshape(-1)).float()
                flat_ret = torch.from_numpy(ret.reshape(-1)).float()
                flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
                indices = np.arange(int(flat_obs.shape[0]))
                policy_loss_value = value_loss_value = bc_loss_value = entropy_value = 0.0
                for _epoch in range(ppo_epochs):
                    rng.shuffle(indices)
                    for start in range(0, indices.size, minibatch_size):
                        mb_idx = indices[start:start + minibatch_size]
                        dist = model.distribution(flat_obs[mb_idx])
                        new_logp = dist.log_prob(flat_act[mb_idx]).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()
                        ratio = torch.exp(new_logp - flat_logp[mb_idx])
                        unclipped = ratio * flat_adv[mb_idx]
                        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * flat_adv[mb_idx]
                        policy_loss = -torch.min(unclipped, clipped).mean()
                        value_loss = 0.5 * (model.value(flat_obs[mb_idx]) - flat_ret[mb_idx]).pow(2).mean()
                        bc_loss = (model.actor(flat_obs[mb_idx]) - flat_ref[mb_idx]).pow(2).mean()
                        loss = policy_loss + value_coef * value_loss + bc_weight * bc_loss - entropy_coef * entropy
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
                mean_return = float(np.mean(completed_returns)) if completed_returns else float(np.mean(rew_arr) * max_episode_steps)
                entry = {
                    "update": update + 1,
                    "steps": int(global_steps),
                    "mean_reward": float(np.mean(rew_arr)),
                    "mean_episode_return": mean_return,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                    "bc_loss": bc_loss_value,
                    "bc_weight": float(bc_weight),
                    "entropy": entropy_value,
                    "action_std": float(torch.exp(model.log_std).mean().item()),
                    "stair_height": float(current_stair_height),
                    **metric_means,
                }
                score = (
                    float(entry.get("vel_error", 1e6))
                    + 0.5 * float(entry.get("base_tilt", 0.0))
                    + 0.02 * float(entry.get("action_rate", 0.0))
                    - 0.60 * float(entry.get("clearance_score", 0.0))
                    - 1.20 * float(entry.get("height_gain_score", 0.0))
                    - 1.50 * float(entry.get("stair_clear_score", 0.0))
                    - 2.00 * float(entry.get("stair_climb_score", 0.0))
                    - 3.00 * float(entry.get("stair_success", 0.0))
                    - 1.00 * float(entry.get("stair_forward_score", 0.0))
                    - 2.00 * float(entry.get("stair_motion_score", 0.0))
                    - 18.00 * float(entry.get("forward_progress", 0.0))
                    - 0.60 * float(entry.get("stair_height", 0.0)) / max(stair_height_max, 1e-6)
                    - 0.05 * float(entry.get("contact_active", 0.0))
                    + 3.0 * float(entry.get("unsafe_tilt", 0.0))
                    + 12.0 * float(entry.get("bad_contact", 0.0))
                    + 20.0 * float(entry.get("fallen", 0.0))
                )
                entry["selection_score"] = float(score)
                can_select = (float(update + 1) / max(1.0, float(updates))) >= select_after_ratio
                entry["selectable"] = float(can_select)
                if can_select and score < best_score:
                    best_score = score
                    best_snapshot = {
                        "update": int(update + 1),
                        "selection_score": float(score),
                        "metrics": dict(entry),
                        "actor_state": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
                        "critic_state": {k: v.detach().cpu().clone() for k, v in model.critic.state_dict().items()},
                        "log_std": model.log_std.detach().cpu().clone(),
                    }
                history.append(entry)
                self._log(
                    f"[ctbc] update {update + 1}/{updates} steps={global_steps} reward={entry['mean_reward']:.3f} "
                    f"return={mean_return:.2f} contact={entry.get('contact_active', 0.0):.3f} "
                    f"gate={entry.get('ctbc_gate', 0.0):.3f} lift={entry.get('lift_active', 0.0):.3f} tilt={entry.get('base_tilt', 0.0):.3f} "
                    f"clear={entry.get('wheel_clearance', 0.0):.3f} climb={entry.get('base_height_gain', 0.0):.3f} "
                    f"hscore={entry.get('height_gain_score', 0.0):.2f} sclear={entry.get('stair_clear_score', 0.0):.2f} "
                    f"sclimb={entry.get('stair_climb_score', 0.0):.2f} succ={entry.get('stair_success', 0.0):.3f} "
                    f"prog={entry.get('forward_progress', 0.0):.4f} fscore={entry.get('stair_forward_score', 0.0):.2f} "
                    f"motion={entry.get('stair_motion_score', 0.0):.2f} "
                    f"unsafe={entry.get('unsafe_tilt', 0.0):.3f} bad={entry.get('bad_contact', 0.0):.3f} "
                    f"fall={entry.get('fallen', 0.0):.3f} stair_h={entry.get('stair_height', 0.0):.3f} sel={int(can_select)} "
                    f"base={entry.get('base_rmse', 0.0):.4f} ff={entry.get('ff_rmse', 0.0):.4f} "
                    f"res={entry.get('residual_rmse', 0.0):.4f} bc={entry.get('bc_loss', 0.0):.4f} bcw={entry.get('bc_weight', 0.0):.3f} "
                    f"std={entry.get('action_std', 0.0):.3f} best={best_score:.4f}"
                )

            if best_snapshot is None:
                best_snapshot = {
                    "update": 0,
                    "selection_score": float("inf"),
                    "metrics": {},
                    "actor_state": {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()},
                    "critic_state": {k: v.detach().cpu().clone() for k, v in model.critic.state_dict().items()},
                    "log_std": model.log_std.detach().cpu().clone(),
                }
            artifacts = self._make_artifacts("latest")
            checkpoint_path = self._ppo_checkpoint_path()
            torch.save({
                "checkpoint_type": "ctbc_residual_ppo_actor_critic",
                "actor_state": best_snapshot["actor_state"],
                "critic_state": best_snapshot["critic_state"],
                "log_std": best_snapshot["log_std"],
                "input_dim": input_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "action_mask": action_mask.astype(float).tolist(),
                "residual_limit": float(residual_limit),
                "source_policy_path": policy_path,
                "settings": dict(self.settings),
                "input_mode": "obs_residual",
                "history": history,
                "selected_update": best_snapshot["update"],
                "selected_metrics": best_snapshot["metrics"],
            }, checkpoint_path)
            summary = {
                "env_id": self.env_id,
                "mode": "ctbc_fine_tune",
                "checkpoint_path": checkpoint_path,
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
            onnx_path = str(self.settings.get("output_path", "")).strip() or artifacts.onnx_path
            self.export_onnx_from_checkpoint(checkpoint_path, onnx_path)
            summary["onnx_path"] = onnx_path
            self._log(f"[ctbc] checkpoint saved: {checkpoint_path} selected_update={best_snapshot['update']} score={best_score:.4f}")
            return summary
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    def _export_residual_onnx_from_checkpoint(self, checkpoint, output_path):
        self._require_torch()
        input_dim = int(checkpoint["input_dim"])
        action_dim = int(checkpoint["action_dim"])
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        model = HomingPolicyNet(input_dim, action_dim, hidden_dim=hidden_dim, action_mask=checkpoint.get("action_mask", None))
        model.load_state_dict(checkpoint["actor_state"], strict=False)
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        dummy = torch.zeros((1, input_dim), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["delta_action"],
            dynamic_axes={"obs": {0: "batch"}, "delta_action": {0: "batch"}},
            opset_version=17,
        )

    def _export_standalone_onnx(self, source_policy_path, residual_policy_path, output_path, residual_limit, action_clip, proprio_gate=None):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception as exc:
            raise RuntimeError("CTBC standalone ONNX export requires the 'onnx' package.") from exc
        if not os.path.isfile(source_policy_path):
            raise RuntimeError(f"CTBC source policy ONNX not found: {source_policy_path}")
        source_model = onnx.load(source_policy_path)
        residual_model = onnx.load(residual_policy_path)
        nodes = []
        initializers = []
        source_out, obs_info, action_info = self._append_prefixed_onnx_graph(nodes, initializers, source_model, "ctbc/source/", "obs")
        residual_out, _, _ = self._append_prefixed_onnx_graph(nodes, initializers, residual_model, "ctbc/residual/", "obs")

        initializers.extend([
            helper.make_tensor("ctbc/residual_min", TensorProto.FLOAT, [1], [-float(residual_limit)]),
            helper.make_tensor("ctbc/residual_max", TensorProto.FLOAT, [1], [float(residual_limit)]),
            helper.make_tensor("ctbc/action_min", TensorProto.FLOAT, [1], [-float(action_clip)]),
            helper.make_tensor("ctbc/action_max", TensorProto.FLOAT, [1], [float(action_clip)]),
        ])
        residual_for_add = "ctbc/residual_clipped"
        nodes.extend([
            helper.make_node(
                "Clip",
                [residual_out, "ctbc/residual_min", "ctbc/residual_max"],
                ["ctbc/residual_clipped"],
                name="ctbc/clip_residual",
            ),
        ])
        if isinstance(proprio_gate, dict):
            gravity_indices = [int(v) for v in proprio_gate.get("gravity_xy_indices", []) if int(v) >= 0]
            ang_indices = [int(v) for v in proprio_gate.get("ang_vel_indices", []) if int(v) >= 0]
            if gravity_indices or ang_indices:
                initializers.extend([
                    helper.make_tensor("ctbc/gate_axes", TensorProto.INT64, [1], [1]),
                    helper.make_tensor("ctbc/gate_threshold", TensorProto.FLOAT, [1, 1], [float(proprio_gate.get("threshold", 0.16))]),
                    helper.make_tensor("ctbc/gate_softness", TensorProto.FLOAT, [1, 1], [max(1e-4, float(proprio_gate.get("softness", 0.025)))]),
                    helper.make_tensor("ctbc/gate_ang_gain", TensorProto.FLOAT, [1, 1], [float(proprio_gate.get("ang_gain", 1.0))]),
                ])
                metric_terms = []
                if gravity_indices:
                    initializers.append(helper.make_tensor("ctbc/gate_gravity_indices", TensorProto.INT64, [len(gravity_indices)], gravity_indices))
                    nodes.extend([
                        helper.make_node("Gather", ["obs", "ctbc/gate_gravity_indices"], ["ctbc/gate_gravity_xy"], name="ctbc/gather_gravity_xy", axis=1),
                        helper.make_node("Mul", ["ctbc/gate_gravity_xy", "ctbc/gate_gravity_xy"], ["ctbc/gate_gravity_sq"], name="ctbc/gravity_square"),
                        helper.make_node("ReduceSum", ["ctbc/gate_gravity_sq", "ctbc/gate_axes"], ["ctbc/gate_gravity_sum"], name="ctbc/gravity_sum", keepdims=1),
                        helper.make_node("Sqrt", ["ctbc/gate_gravity_sum"], ["ctbc/gate_tilt_metric"], name="ctbc/tilt_metric"),
                    ])
                    metric_terms.append("ctbc/gate_tilt_metric")
                if ang_indices:
                    initializers.append(helper.make_tensor("ctbc/gate_ang_indices", TensorProto.INT64, [len(ang_indices)], ang_indices))
                    nodes.extend([
                        helper.make_node("Gather", ["obs", "ctbc/gate_ang_indices"], ["ctbc/gate_ang"], name="ctbc/gather_ang_vel", axis=1),
                        helper.make_node("Mul", ["ctbc/gate_ang", "ctbc/gate_ang"], ["ctbc/gate_ang_sq"], name="ctbc/ang_square"),
                        helper.make_node("ReduceSum", ["ctbc/gate_ang_sq", "ctbc/gate_axes"], ["ctbc/gate_ang_sum"], name="ctbc/ang_sum", keepdims=1),
                        helper.make_node("Sqrt", ["ctbc/gate_ang_sum"], ["ctbc/gate_ang_norm"], name="ctbc/ang_norm"),
                        helper.make_node("Mul", ["ctbc/gate_ang_norm", "ctbc/gate_ang_gain"], ["ctbc/gate_ang_metric"], name="ctbc/ang_metric"),
                    ])
                    metric_terms.append("ctbc/gate_ang_metric")
                metric_name = metric_terms[0]
                if len(metric_terms) > 1:
                    nodes.append(helper.make_node("Add", metric_terms[:2], ["ctbc/gate_metric"], name="ctbc/gate_metric_add"))
                    metric_name = "ctbc/gate_metric"
                nodes.extend([
                    helper.make_node("Sub", [metric_name, "ctbc/gate_threshold"], ["ctbc/gate_centered"], name="ctbc/gate_centered"),
                    helper.make_node("Div", ["ctbc/gate_centered", "ctbc/gate_softness"], ["ctbc/gate_logit"], name="ctbc/gate_logit"),
                    helper.make_node("Sigmoid", ["ctbc/gate_logit"], ["ctbc/proprio_gate"], name="ctbc/proprio_gate"),
                    helper.make_node("Mul", ["ctbc/residual_clipped", "ctbc/proprio_gate"], ["ctbc/residual_gated"], name="ctbc/apply_proprio_gate"),
                ])
                residual_for_add = "ctbc/residual_gated"
        nodes.extend([
            helper.make_node("Add", [source_out, residual_for_add], ["ctbc/action_raw"], name="ctbc/add_source_residual"),
            helper.make_node(
                "Clip",
                ["ctbc/action_raw", "ctbc/action_min", "ctbc/action_max"],
                ["action"],
                name="ctbc/clip_action",
            ),
        ])
        opsets = {}
        for model in (source_model, residual_model):
            for opset in model.opset_import:
                opsets[opset.domain] = max(opsets.get(opset.domain, 0), int(opset.version))
        opsets[""] = max(opsets.get("", 0), 17)
        graph = helper.make_graph(
            nodes,
            "CTBCStandalonePolicy",
            [self._renamed_onnx_value_info(obs_info, "obs")],
            [self._renamed_onnx_value_info(action_info, "action")],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            producer_name="cosim_act_net_ctbc_export",
            opset_imports=[helper.make_operatorsetid(domain, version) for domain, version in sorted(opsets.items())],
        )
        model.ir_version = max(source_model.ir_version, residual_model.ir_version)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass
        onnx.checker.check_model(model)
        onnx.save(model, output_path)

    def export_onnx_from_checkpoint(self, checkpoint_path, output_path):
        self._require_torch()
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"CTBC checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source_policy_path = checkpoint.get("source_policy_path", checkpoint.get("settings", {}).get("policy_path", ""))
        residual_limit = float(checkpoint.get("residual_limit", self.settings.get("ctbc_residual_limit", 4.0)))
        action_clip = float(checkpoint.get("settings", {}).get("ctbc_action_clip", self.settings.get("ctbc_action_clip", 4.0)))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="ctbc_residual_", suffix=".onnx", delete=False) as tmp:
            residual_path = tmp.name
        try:
            self._export_residual_onnx_from_checkpoint(checkpoint, residual_path)
            self._export_standalone_onnx(
                source_policy_path,
                residual_path,
                output_path,
                residual_limit,
                action_clip,
                proprio_gate=checkpoint.get("ctbc_proprio_gate", None),
            )
        finally:
            try:
                os.remove(residual_path)
            except Exception:
                pass
        input_dim = int(checkpoint["input_dim"])
        action_dim = int(checkpoint["action_dim"])
        manifest_path = os.path.splitext(output_path)[0] + ".json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "note": "CTBC standalone actor export. Output is final action = clipped(source_policy(obs) + clipped(ctbc_delta(obs))).",
                "env_id": self.env_id,
                "checkpoint_path": checkpoint_path,
                "onnx_path": output_path,
                "source_policy_path": source_policy_path,
                "input_dim": input_dim,
                "action_dim": action_dim,
                "residual_limit": residual_limit,
                "action_clip": action_clip,
                "ctbc_gate_height_threshold": float(checkpoint.get("settings", {}).get("ctbc_gate_height_threshold", self.settings.get("ctbc_gate_height_threshold", 0.06))),
                "ctbc_gate_height_softness": float(checkpoint.get("settings", {}).get("ctbc_gate_height_softness", self.settings.get("ctbc_gate_height_softness", 0.025))),
                "ctbc_gate_rise": float(checkpoint.get("settings", {}).get("ctbc_gate_rise", self.settings.get("ctbc_gate_rise", 0.35))),
                "ctbc_gate_fall": float(checkpoint.get("settings", {}).get("ctbc_gate_fall", self.settings.get("ctbc_gate_fall", 0.08))),
                "checkpoint_type": "ctbc_standalone_actor",
                "source_checkpoint_type": checkpoint.get("checkpoint_type", "ctbc_residual_ppo_actor_critic"),
                "input_mode": "obs_to_final_action",
                "onnx_inputs": ["obs"],
                "onnx_outputs": ["action"],
            }, handle, indent=2)
        latest_manifest = self._make_artifacts("latest").manifest_path
        if os.path.abspath(manifest_path) != os.path.abspath(latest_manifest):
            os.makedirs(os.path.dirname(latest_manifest), exist_ok=True)
            with open(latest_manifest, "w", encoding="utf-8") as handle:
                json.dump({"onnx_path": output_path, "checkpoint_path": checkpoint_path}, handle, indent=2)
        self._log(f"[ctbc] exported onnx: {output_path}")
        return output_path

    def _freeze_base_state(self, env, root_qpos=None):
        try:
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            data = getattr(leaf, "data", None)
            if model is None or data is None:
                return root_qpos
            if root_qpos is None:
                root_qpos = np.asarray(data.qpos[:7], dtype=np.float64).copy() if data.qpos.size >= 7 else np.asarray(data.qpos, dtype=np.float64).copy()
            n_root = min(int(np.asarray(root_qpos).size), int(data.qpos.size))
            if n_root > 0:
                data.qpos[:n_root] = np.asarray(root_qpos, dtype=np.float64)[:n_root]
            if data.qvel.size >= 6:
                data.qvel[:6] = 0.0
            elif data.qvel.size > 0:
                data.qvel[:] = 0.0
            try:
                import mujoco
                mujoco.mj_forward(model, data)
            except Exception:
                pass
            return root_qpos
        except Exception:
            return root_qpos

    def _disable_gravity(self, env):
        try:
            leaf = self._leaf_env(env)
            model = getattr(leaf, "model", None)
            if model is not None and hasattr(model, "opt"):
                model.opt.gravity[:] = 0.0
                return True
        except Exception:
            pass
        return False

    def test_primitive(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for CTBC primitive test.")
        terrain = str(self.settings.get("ctbc_terrain", "flat")).strip() or "flat"
        test_steps = max(1, int(self.settings.get("test_steps", self.settings.get("ctbc_primitive_steps", 1200))))
        config = self._base_config(terrain, render=True)
        env = build_env(config)
        termination_disabled = self._disable_leaf_termination(env)
        gravity_disabled = self._disable_gravity(env)
        self._log(
            f"[ctbc-primitive-test] rendering CTBC primitive only. terrain={terrain} steps={test_steps} "
            f"gravity_off={gravity_disabled} base_frozen=True"
        )
        if termination_disabled:
            self._log("[ctbc-primitive-test] leaf env termination disabled.")
        try:
            state, _ = env.reset()
            _ = state
            command = np.zeros((int(env.command_dim),), dtype=np.float32)
            env.receive_user_command(command)
            action_dim = int(env.action_dim)
            control_dt = self._control_dt(env)
            ctx = {
                "active_lift_side": 0,
                "lift_step": 0,
                "step": 0,
                "next_lift_step": 0,
                "lift_cycle_side": 1,
                "side_action_indices": self._side_action_indices(env, action_dim),
                "action_scales": self._action_scales(env, action_dim),
            }
            self._log(
                f"[ctbc-primitive-test] action_scales={np.asarray(ctx['action_scales']).round(3).tolist()} "
                f"groups={ctx['side_action_indices']}"
            )
            root_qpos = self._freeze_base_state(env, None)
            for step in range(test_steps):
                if self._stop_requested():
                    self._log("[ctbc-primitive-test] stop requested.")
                    break
                if int(ctx.get("active_lift_side", -1)) < 0 and step >= int(ctx.get("next_lift_step", 0)):
                    side = int(ctx.get("lift_cycle_side", 0))
                    ctx["active_lift_side"] = side
                    ctx["lift_step"] = 0
                    ctx["lift_cycle_side"] = 1 - side
                ff = self._feedforward_lift_action(ctx, action_dim, control_dt)
                _, terminated, truncated, _info = env.step(np.clip(ff, -self._primitive_action_clip(), self._primitive_action_clip()))
                root_qpos = self._freeze_base_state(env, root_qpos)
                env.render()
                ctx["step"] = int(ctx.get("step", 0)) + 1
                if step % 60 == 0:
                    side = int(ctx.get("active_lift_side", -1))
                    self._log(
                        f"[ctbc-primitive-test] step={step}/{test_steps} side={side} "
                        f"ff_norm={float(np.sqrt(np.mean(ff ** 2))):.4f} ff_max={float(np.max(np.abs(ff))):.3f}"
                    )
                if terminated or truncated:
                    self._clear_wrapper_reset_flags(env)
        finally:
            env.close()
        return {
            "env_id": self.env_id,
            "mode": "ctbc_test_primitive",
            "terrain": terrain,
            "steps": int(test_steps),
        }

    def _controller_wheel_push(self, ctx, action_dim, gain):
        push = np.zeros((action_dim,), dtype=np.float32)
        groups = ctx.get("side_action_indices", {})
        scales = np.asarray(ctx.get("action_scales", np.ones((action_dim,), dtype=np.float32)), dtype=np.float32)
        scales = self._pad_or_trim(scales, action_dim, fill=1.0)
        for side in (0, 1):
            side_group = groups.get(side, {}) if isinstance(groups, dict) else {}
            for action_index in side_group.get("wheel", []) or []:
                if 0 <= int(action_index) < action_dim:
                    value = float(gain)
                    if str(self.settings.get("ctbc_compensate_action_scale", "1")).strip().lower() not in ("0", "false", "no", "off"):
                        value /= max(abs(float(scales[int(action_index)])), 1e-6)
                    push[int(action_index)] += value
        return np.clip(push, -self._primitive_action_clip(), self._primitive_action_clip())

    def tune_stair_controller(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for CTBC stair controller tuning.")
        policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid base ONNX policy.")
        terrain = str(self.settings.get("ctbc_terrain", "stairs_up_easy")).strip() or "stairs_up_easy"
        rng = np.random.default_rng(int(self.settings.get("seed", 42)))
        base_seed = int(self.settings.get("seed", 42))
        candidates = max(1, int(self.settings.get("ctbc_controller_candidates", 64)))
        test_steps = max(64, int(self.settings.get("ctbc_episode_steps", self.settings.get("test_steps", 1200))))
        stair_height = max(0.0, float(self.settings.get("ctbc_stair_height_max", 0.20)))
        trigger_step = max(0, int(self.settings.get("ctbc_controller_trigger_step", 0)))
        base_settings = dict(self.settings)
        config = self._base_config(terrain, render=False)
        env = build_env(config)
        policy = OnnxExpertPolicy(policy_path)
        applied = self._set_stair_height(env, terrain, stair_height)
        self._log(
            f"[ctbc-controller] tuning deterministic stair controller. candidates={candidates} "
            f"terrain={terrain} stair_h={stair_height:.3f} applied={int(applied)} steps={test_steps}"
        )
        self._log("[ctbc-controller] stair detection uses MuJoCo privileged terrain/stair state; no height-map input is used.")
        try:
            action_dim = int(env.action_dim)
            control_dt = self._control_dt(env)
            default_params = {
                "ctbc_lift_amplitude": float(base_settings.get("ctbc_lift_amplitude", 0.90)),
                "ctbc_lift_period": float(base_settings.get("ctbc_lift_period", 0.75)),
                "ctbc_shoulder_gain": float(base_settings.get("ctbc_shoulder_gain", 0.50)),
                "ctbc_leg_gain": float(base_settings.get("ctbc_leg_gain", 0.0)),
                "ctbc_leg_push_gain": float(base_settings.get("ctbc_leg_push_gain", 1.75)),
                "ctbc_hip_gain": float(base_settings.get("ctbc_hip_gain", 0.0)),
                "ctbc_stance_gain": float(base_settings.get("ctbc_stance_gain", 0.30)),
                "ctbc_wheel_push_gain": float(base_settings.get("ctbc_wheel_push_gain", 0.0)),
                "ctbc_command_x": float(base_settings.get("ctbc_command_x", base_settings.get("ctbc_command_x_min", 0.20))),
            }

            def sample_params(index):
                if index == 0:
                    return dict(default_params)
                params = dict(default_params)
                params["ctbc_lift_amplitude"] = float(np.clip(rng.normal(default_params["ctbc_lift_amplitude"], 0.20), 0.35, 1.60))
                params["ctbc_lift_period"] = float(np.clip(rng.normal(default_params["ctbc_lift_period"], 0.18), 0.35, 1.20))
                if str(self.settings.get("ctbc_controller_broad_search", "1")).strip().lower() not in ("0", "false", "no", "off"):
                    params["ctbc_shoulder_gain"] = float(rng.uniform(-1.5, 2.2))
                    params["ctbc_leg_gain"] = float(rng.uniform(-2.5, 2.0))
                    params["ctbc_leg_push_gain"] = float(rng.uniform(-1.0, 3.5))
                    params["ctbc_hip_gain"] = float(rng.uniform(-1.5, 1.5))
                    params["ctbc_stance_gain"] = float(rng.uniform(-1.0, 1.5))
                    params["ctbc_wheel_push_gain"] = float(rng.uniform(-1.0, 1.0))
                    params["ctbc_command_x"] = float(rng.uniform(0.03, 0.45))
                else:
                    params["ctbc_shoulder_gain"] = float(np.clip(rng.normal(default_params["ctbc_shoulder_gain"], 0.35), -1.0, 1.8))
                    params["ctbc_leg_gain"] = float(np.clip(rng.normal(default_params["ctbc_leg_gain"], 0.55), -2.2, 1.2))
                    params["ctbc_leg_push_gain"] = float(np.clip(rng.normal(default_params["ctbc_leg_push_gain"], 0.55), -0.5, 3.0))
                    params["ctbc_hip_gain"] = float(np.clip(rng.normal(default_params["ctbc_hip_gain"], 0.35), -1.2, 1.2))
                    params["ctbc_stance_gain"] = float(np.clip(rng.normal(default_params["ctbc_stance_gain"], 0.25), -0.6, 1.2))
                    params["ctbc_wheel_push_gain"] = float(np.clip(rng.normal(default_params["ctbc_wheel_push_gain"], 0.20), -0.7, 0.7))
                return params

            best = None
            history = []
            for candidate_index in range(candidates):
                if self._stop_requested():
                    self._log("[ctbc-controller] stop requested.")
                    break
                params = sample_params(candidate_index)
                eval_seed = int(base_seed + candidate_index)
                params["ctbc_eval_seed"] = eval_seed
                self.settings.update({key: str(value) for key, value in params.items()})
                np.random.seed(eval_seed)
                state, _ = env.reset()
                self._set_stair_height(env, terrain, stair_height)
                policy.reset()
                command = self._sample_ctbc_command(rng, env.command_dim)
                if "ctbc_command_x" in params and command.size:
                    command[0] = float(params["ctbc_command_x"])
                    if command.size > 1:
                        command[1] = 0.0
                    if command.size > 2:
                        command[2] = 0.0
                env.receive_user_command(command)
                state = self._inject_applied_command(state, env)
                start_pos = self._base_position(env)
                max_pos = start_pos.copy()
                max_clear = 0.0
                max_tilt = 0.0
                fallen = False
                bad_contact = False
                success = False
                action_rms_sum = 0.0
                ctx = {
                    "active_lift_side": -1,
                    "lift_step": 0,
                    "step": 0,
                    "next_lift_step": 0,
                    "lift_cycle_side": 0,
                    "side_action_indices": self._side_action_indices(env, action_dim),
                    "action_scales": self._action_scales(env, action_dim),
                }
                clearance_baseline = self._wheel_body_heights(env)
                for step in range(test_steps):
                    privileged_stair = bool(str(terrain).startswith("stairs") and step >= trigger_step)
                    base = np.clip(
                        self._pad_or_trim(policy.get_action(state), action_dim),
                        -1.0,
                        1.0,
                    )
                    correction = np.zeros((action_dim,), dtype=np.float32)
                    if privileged_stair:
                        if int(ctx.get("active_lift_side", -1)) < 0 and step >= int(ctx.get("next_lift_step", 0)):
                            side = int(ctx.get("lift_cycle_side", 0))
                            ctx["active_lift_side"] = side
                            ctx["lift_step"] = 0
                            ctx["lift_cycle_side"] = 1 - side
                        correction += self._feedforward_lift_action(ctx, action_dim, control_dt)
                        correction += self._controller_wheel_push(ctx, action_dim, params.get("ctbc_wheel_push_gain", 0.0))
                    action = np.clip(base + correction, -self._ctbc_action_clip(), self._ctbc_action_clip())
                    env.receive_user_command(command)
                    state, terminated, truncated, _ = env.step(action)
                    state = self._inject_applied_command(state, env)
                    pos = self._base_position(env)
                    max_pos = np.maximum(max_pos, pos)
                    max_clear = max(max_clear, float(np.max(self._wheel_body_heights(env) - clearance_baseline)))
                    current_progress = max(0.0, float(pos[0] - start_pos[0]))
                    current_climb = max(0.0, float(pos[2] - start_pos[2]))
                    roll, pitch = self._base_roll_pitch(env)
                    tilt = float(np.sqrt(roll * roll + pitch * pitch))
                    max_tilt = max(max_tilt, tilt)
                    action_rms_sum += float(np.sqrt(np.mean(action ** 2)))
                    if (
                        current_progress > float(self.settings.get("ctbc_success_min_progress", 0.05))
                        and current_climb >= float(self.settings.get("ctbc_success_climb_ratio", 0.70)) * max(stair_height, 1e-6)
                        and max_clear >= float(self.settings.get("ctbc_success_clear_ratio", 0.80)) * max(stair_height, 1e-6)
                    ):
                        success = True
                        break
                    if self._non_wheel_contact_effort(env) > float(self.settings.get("ctbc_bad_contact_threshold", 1.0)):
                        bad_contact = True
                    if bool(terminated or truncated or self._fall_signal(env, float(self.settings.get("reward_fall_height", 0.12)))):
                        fallen = True
                        break
                    ctx["step"] = int(ctx.get("step", 0)) + 1
                progress = max(0.0, float(max_pos[0] - start_pos[0]))
                climb = max(0.0, float(max_pos[2] - start_pos[2]))
                clear = max(0.0, float(max_clear))
                action_rms = action_rms_sum / max(1, step + 1)
                score = (
                    2.0 * progress
                    + 35.0 * climb
                    + 12.0 * clear
                    + (80.0 if success else 0.0)
                    - 4.0 * max_tilt
                    - 0.15 * action_rms
                    - (30.0 if fallen else 0.0)
                    - (15.0 if bad_contact else 0.0)
                )
                entry = {
                    "candidate": int(candidate_index),
                    "score": float(score),
                    "progress": float(progress),
                    "climb": float(climb),
                    "clear": float(clear),
                    "tilt": float(max_tilt),
                    "success": bool(success),
                    "fallen": bool(fallen),
                    "bad_contact": bool(bad_contact),
                    "action_rms": float(action_rms),
                    "params": dict(params),
                }
                history.append(entry)
                if best is None or score > best["score"]:
                    best = entry
                    self._log(
                        f"[ctbc-controller] best#{candidate_index} score={score:.3f} prog={progress:.3f} "
                        f"climb={climb:.3f} clear={clear:.3f} tilt={max_tilt:.3f} "
                        f"succ={int(success)} fall={int(fallen)} bad={int(bad_contact)}"
                    )
                elif candidate_index % 10 == 0:
                    self._log(
                        f"[ctbc-controller] cand={candidate_index}/{candidates} score={score:.3f} "
                        f"best={best['score']:.3f}"
                    )
            self.settings.clear()
            self.settings.update(base_settings)
            if best is None:
                raise RuntimeError("Controller tuning produced no evaluated candidates.")
            artifacts = self._make_artifacts("latest")
            params_path = os.path.join(artifacts.run_dir, "stair_controller_params.json")
            payload = {
                "env_id": self.env_id,
                "terrain": terrain,
                "stair_height": float(stair_height),
                "best": best,
                "history": history,
                "note": "Deterministic stair controller parameters tuned with MuJoCo privileged stair state. No height-map input is used.",
            }
            with open(params_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self._log(f"[ctbc-controller] saved best params: {params_path}")
            return {
                "env_id": self.env_id,
                "mode": "ctbc_stair_controller_tune",
                "params_path": params_path,
                "selected_metrics": best,
                "history": history,
                "stopped": self._stop_requested(),
            }
        finally:
            self.settings.clear()
            self.settings.update(base_settings if "base_settings" in locals() else self.settings)
            try:
                env.close()
            except Exception:
                pass

    def test_stair_controller(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for CTBC stair controller test.")
        policy_path = str(self.settings.get("policy_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid base ONNX policy.")
        terrain = str(self.settings.get("ctbc_terrain", "stairs_up_easy")).strip() or "stairs_up_easy"
        test_steps = max(1, int(self.settings.get("test_steps", self.settings.get("ctbc_episode_steps", 1200))))
        stair_height = max(0.0, float(self.settings.get("ctbc_stair_height_max", 0.20)))
        trigger_step = max(0, int(self.settings.get("ctbc_controller_trigger_step", 0)))
        config = self._base_config(terrain, render=True)
        env = build_env(config)
        policy = OnnxExpertPolicy(policy_path)
        applied = self._set_stair_height(env, terrain, stair_height)
        self._log(
            f"[ctbc-controller-test] base ONNX + deterministic controller. terrain={terrain} "
            f"stair_h={stair_height:.3f} applied={int(applied)} steps={test_steps}"
        )
        self._log("[ctbc-controller-test] stair trigger uses MuJoCo privileged terrain/stair state; no height-map input is used.")
        try:
            state, _ = env.reset()
            self._set_stair_height(env, terrain, stair_height)
            state = self._inject_applied_command(state, env)
            command = self._current_command(env.command_dim)
            env.receive_user_command(command)
            action_dim = int(env.action_dim)
            control_dt = self._control_dt(env)
            ctx = {
                "active_lift_side": -1,
                "lift_step": 0,
                "step": 0,
                "next_lift_step": 0,
                "lift_cycle_side": 0,
                "side_action_indices": self._side_action_indices(env, action_dim),
                "action_scales": self._action_scales(env, action_dim),
            }
            start_pos = self._base_position(env)
            clearance_baseline = self._wheel_body_heights(env)
            max_clear = 0.0
            success = False
            for step in range(test_steps):
                if self._stop_requested():
                    self._log("[ctbc-controller-test] stop requested.")
                    break
                command = self._current_command(env.command_dim)
                env.receive_user_command(command)
                base = np.clip(self._pad_or_trim(policy.get_action(state), action_dim), -1.0, 1.0)
                privileged_stair = bool(str(terrain).startswith("stairs") and step >= trigger_step)
                correction = np.zeros((action_dim,), dtype=np.float32)
                if privileged_stair:
                    if int(ctx.get("active_lift_side", -1)) < 0 and step >= int(ctx.get("next_lift_step", 0)):
                        side = int(ctx.get("lift_cycle_side", 0))
                        ctx["active_lift_side"] = side
                        ctx["lift_step"] = 0
                        ctx["lift_cycle_side"] = 1 - side
                    correction += self._feedforward_lift_action(ctx, action_dim, control_dt)
                    correction += self._controller_wheel_push(
                        ctx,
                        action_dim,
                        float(self.settings.get("ctbc_wheel_push_gain", 0.0)),
                    )
                action = np.clip(base + correction, -self._ctbc_action_clip(), self._ctbc_action_clip())
                state, terminated, truncated, _ = env.step(action)
                state = self._inject_applied_command(state, env)
                env.render()
                pos = self._base_position(env)
                max_clear = max(max_clear, float(np.max(self._wheel_body_heights(env) - clearance_baseline)))
                progress = max(0.0, float(pos[0] - start_pos[0]))
                climb = max(0.0, float(pos[2] - start_pos[2]))
                roll, pitch = self._base_roll_pitch(env)
                tilt = float(np.sqrt(roll * roll + pitch * pitch))
                ctx["step"] = int(ctx.get("step", 0)) + 1
                if step % 60 == 0:
                    self._log(
                        f"[ctbc-controller-test] step={step}/{test_steps} x={progress:.3f} "
                        f"climb={climb:.3f} clear={max_clear:.3f} "
                        f"tilt={tilt:.3f} corr={float(np.sqrt(np.mean(correction ** 2))):.4f}"
                    )
                if (
                    progress > float(self.settings.get("ctbc_success_min_progress", 0.05))
                    and climb >= float(self.settings.get("ctbc_success_climb_ratio", 0.70)) * max(stair_height, 1e-6)
                    and max_clear >= float(self.settings.get("ctbc_success_clear_ratio", 0.80)) * max(stair_height, 1e-6)
                ):
                    success = True
                    self._log(
                        f"[ctbc-controller-test] success at step={step}: "
                        f"x={progress:.3f} climb={climb:.3f} clear={max_clear:.3f} tilt={tilt:.3f}"
                    )
                    break
                if terminated or truncated:
                    self._log(f"[ctbc-controller-test] env terminated/truncated at step={step}.")
                    break
            end_pos = self._base_position(env)
            return {
                "env_id": self.env_id,
                "mode": "ctbc_stair_controller_test",
                "terrain": terrain,
                "steps": int(step + 1),
                "progress": float(end_pos[0] - start_pos[0]),
                "climb": float(end_pos[2] - start_pos[2]),
                "clear": float(max_clear),
                "success": bool(success),
            }
        finally:
            env.close()

    def test_export_policy(self):
        if not self.env_id:
            raise RuntimeError("Select a robot/env for CTBC policy test.")
        policy_path = str(self.settings.get("output_path", "")).strip()
        if not os.path.isfile(policy_path):
            raise RuntimeError("Select a valid exported standalone CTBC ONNX policy.")

        policy = OnnxExpertPolicy(policy_path)
        terrain = str(self.settings.get("ctbc_terrain", "stairs_up_easy")).strip() or "stairs_up_easy"
        test_steps = max(1, int(self.settings.get("test_steps", self.settings.get("ctbc_episode_steps", 1000))))
        config = self._base_config(terrain, render=True)
        env = build_env(config)
        test_stair_height = float(self.settings.get("ctbc_test_stair_height", self.settings.get("ctbc_stair_height_max", 0.20)))
        stair_height_applied = self._set_stair_height(env, terrain, test_stair_height)
        disable_termination = str(self.settings.get("ctbc_test_disable_termination", "0")).strip().lower() not in ("0", "false", "no", "off")
        termination_disabled = self._disable_leaf_termination(env) if disable_termination else False
        self._log(
            f"[ctbc-policy-test] running standalone CTBC ONNX policy. terrain={terrain} "
            f"steps={test_steps} stair_h={test_stair_height:.3f} applied={int(stair_height_applied)}"
        )
        if termination_disabled:
            self._log("[ctbc-policy-test] leaf env termination disabled; time-limit truncation still applies.")
        try:
            state, _ = env.reset()
            policy.reset()
            command = self._current_command(env.command_dim)
            env.receive_user_command(command)
            state = self._inject_applied_command(state, env)
            for step in range(test_steps):
                if self._stop_requested():
                    self._log("[ctbc-policy-test] stop requested.")
                    break

                command = self._current_command(env.command_dim)
                env.receive_user_command(command)
                _safety_scale, tilt = self._ctbc_safety_scale(env)
                action = self._pad_or_trim(policy.get_action(state), env.action_dim)
                state, terminated, truncated, _ = env.step(action)
                env.render()
                state = self._inject_applied_command(state, env)

                if step % 100 == 0:
                    self._log(
                        f"[ctbc-policy-test] step={step}/{test_steps} "
                        f"tilt={tilt:.3f} "
                        f"action={float(np.sqrt(np.mean(action ** 2))):.4f} command={command.tolist()}"
                    )
                if truncated:
                    self._log("[ctbc-policy-test] env truncated.")
                    break
                if terminated:
                    if termination_disabled:
                        self._clear_wrapper_reset_flags(env)
                        self._log("[ctbc-policy-test] ignored env termination.")
                    else:
                        self._log(f"[ctbc-policy-test] env terminated at step={step}.")
                        break
        finally:
            env.close()
        return {
            "env_id": self.env_id,
            "mode": "ctbc_test_policy",
            "terrain": terrain,
            "steps": int(test_steps),
            "onnx_path": policy_path,
        }
