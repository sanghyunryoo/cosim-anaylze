import copy
import random
from collections import deque

import numpy as np
import torch


class ControlManager:
    # -------------------------------------------------------
    # Action/state order assumed for Wheeldog (16-dim):
    # [FL_hip, FR_hip,
    #  FL_shoulder, FR_shoulder,
    #  FL_leg, FR_leg,
    #  FL_wheel, FR_wheel,
    #  RL_hip, RR_hip,
    #  RL_shoulder, RR_shoulder,
    #  RL_leg, RR_leg,
    #  RL_wheel, RR_wheel]
    # -------------------------------------------------------
    HIP_NET_PATH = ""
    SHOULDER_NET_PATH = ""
    LEG_NET_PATH = ""
    WHEEL_NET_PATH = ""
    HIP_INDICES = [0, 1, 8, 9]
    SHOULDER_INDICES = [2, 3, 10, 11]
    LEG_INDICES = [4, 5, 12, 13]
    WHEEL_INDICES = [6, 7, 14, 15]

    HISTORY_LEN = 3
    INPUT_ORDER = "pos_vel"

    # shoulder net options
    SHOULDER_NET_INPUT_IN_MOTOR_SPACE = False
    SHOULDER_NET_OUTPUT_IN_MOTOR_SPACE = False
    SHOULDER_GEAR_RATIO = 1.0
    SHOULDER_GAMMA = 1.0
    SHOULDER_TORQUE_SCALE = 1.0

    # leg net options
    LEG_NET_INPUT_IN_MOTOR_SPACE = False
    LEG_NET_OUTPUT_IN_MOTOR_SPACE = True
    LEG_GEAR_RATIO = 1.0
    LEG_GAMMA = 0.98
    LEG_TORQUE_SCALE = 1.0

    def __init__(self, config):
        self.prev_action = None
        self.filtered_action = None
        self.prob = config["random"]["action_delay_prob"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.group_models = {}
        self.group_modes = {}
        self.group_indices = {
            "hip": self.HIP_INDICES,
            "shoulder": self.SHOULDER_INDICES,
            "leg": self.LEG_INDICES,
            "wheel": self.WHEEL_INDICES,
        }
        self.group_defaults = {
            "hip": {"mode": "pd", "path": self.HIP_NET_PATH},
            "shoulder": {"mode": "pd", "path": self.SHOULDER_NET_PATH},
            "leg": {"mode": "pd", "path": self.LEG_NET_PATH},
            "wheel": {"mode": "pd", "path": self.WHEEL_NET_PATH},
        }
        self.group_options = {}

        actuator_cfg = config.get("actuator", {}) if isinstance(config.get("actuator", {}), dict) else {}
        global_mode = str(actuator_cfg.get("mode", "")).strip().lower()
        if global_mode:
            for group in self.group_defaults:
                actuator_cfg.setdefault(f"{group}_mode", global_mode)

        # gear settings: keep previous style (hardware -> leg defaults)
        if "hardware" in config:
            self.LEG_GEAR_RATIO = self._to_float(config["hardware"].get("gear_ratio", self.LEG_GEAR_RATIO), self.LEG_GEAR_RATIO)
            self.LEG_GAMMA = self._to_float(config["hardware"].get("gamma", self.LEG_GAMMA), self.LEG_GAMMA)

        for group, defaults in self.group_defaults.items():
            mode_key = f"{group}_mode"
            path_key = f"{group}_net_path"
            mode_raw = str(actuator_cfg.get(mode_key, defaults["mode"])).strip().lower()
            mode = "actuator_net" if mode_raw == "actuator_net" else "pd"
            path = str(actuator_cfg.get(path_key, defaults["path"])).strip()
            self.group_modes[group] = mode

            if group == "shoulder":
                options = self._build_group_options(
                    actuator_cfg,
                    group,
                    self.group_indices[group],
                    default_torque_scale=self.SHOULDER_TORQUE_SCALE,
                    default_input_in_motor=self.SHOULDER_NET_INPUT_IN_MOTOR_SPACE,
                    default_output_in_motor=self.SHOULDER_NET_OUTPUT_IN_MOTOR_SPACE,
                    default_gear_ratio=self.SHOULDER_GEAR_RATIO,
                    default_gamma=self.SHOULDER_GAMMA,
                )
            elif group == "leg":
                options = self._build_group_options(
                    actuator_cfg,
                    group,
                    self.group_indices[group],
                    default_torque_scale=self.LEG_TORQUE_SCALE,
                    default_input_in_motor=self.LEG_NET_INPUT_IN_MOTOR_SPACE,
                    default_output_in_motor=self.LEG_NET_OUTPUT_IN_MOTOR_SPACE,
                    default_gear_ratio=self.LEG_GEAR_RATIO,
                    default_gamma=self.LEG_GAMMA,
                )
            else:
                options = self._build_group_options(
                    actuator_cfg,
                    group,
                    self.group_indices[group],
                )
            self.group_options[group] = options

            if mode != "actuator_net":
                self.group_models[group] = None
                continue

            if not path:
                raise RuntimeError(
                    f"Actuator mode for '{group}' is actuator_net, but '{path_key}' is empty."
                )
            try:
                model = torch.jit.load(path, map_location=self.device)
                model.eval()
                self.group_models[group] = model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {group} actuator net: '{path}'.\n{e}"
                ) from e

        self._reset_all_histories()

    @staticmethod
    def pd_controller(kp, tq, q, kd, td, d):
        return kp * (tq - q) + kd * (td - d)

    @staticmethod
    def _to_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "t", "yes", "y", "on"):
                return True
            if text in ("0", "false", "f", "no", "n", "off"):
                return False
        return default

    @staticmethod
    def _to_float(value, default=1.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _parse_axis_signs(self, raw_value, dim):
        if raw_value is None:
            return np.ones(dim, dtype=np.float32)
        if isinstance(raw_value, (int, float, str)):
            s = self._to_float(raw_value, 1.0)
            return np.full(dim, s, dtype=np.float32)
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            vals = [self._to_float(v, 1.0) for v in list(raw_value)]
            if len(vals) == dim:
                return np.asarray(vals, dtype=np.float32)
        return np.ones(dim, dtype=np.float32)

    def _resolve_axis_signs(self, actuator_cfg, group, indices):
        key_list = f"{group}_axis_signs"
        key_single = f"{group}_axis_sign"
        if key_list in actuator_cfg:
            return self._parse_axis_signs(actuator_cfg.get(key_list), len(indices))
        if key_single in actuator_cfg:
            return self._parse_axis_signs(actuator_cfg.get(key_single), len(indices))

        global_signs = actuator_cfg.get("axis_signs")
        if isinstance(global_signs, (list, tuple, np.ndarray)) and len(global_signs) > max(indices):
            return self._parse_axis_signs([global_signs[i] for i in indices], len(indices))
        return np.ones(len(indices), dtype=np.float32)

    def _build_group_options(
        self,
        actuator_cfg,
        group,
        indices,
        default_torque_scale=1.0,
        default_input_in_motor=False,
        default_output_in_motor=False,
        default_gear_ratio=1.0,
        default_gamma=1.0,
    ):
        return {
            "torque_scale": self._to_float(actuator_cfg.get(f"{group}_torque_scale", default_torque_scale), default_torque_scale),
            "input_in_motor_space": self._to_bool(
                actuator_cfg.get(f"{group}_input_in_motor_space", default_input_in_motor),
                default_input_in_motor,
            ),
            "output_in_motor_space": self._to_bool(
                actuator_cfg.get(f"{group}_output_in_motor_space", default_output_in_motor),
                default_output_in_motor,
            ),
            "gear_ratio": self._to_float(actuator_cfg.get(f"{group}_gear_ratio", default_gear_ratio), default_gear_ratio),
            "gamma": self._to_float(actuator_cfg.get(f"{group}_gamma", default_gamma), default_gamma),
            "axis_signs": self._resolve_axis_signs(actuator_cfg, group, indices),
        }

    def delay_filter(self, action):
        v = random.uniform(0, 1)
        delay_flag = True if self.prob > v else False
        if not delay_flag or self.prev_action is None:
            self.prev_action = copy.deepcopy(action)
            return action
        output = copy.deepcopy(self.prev_action)
        self.prev_action = copy.deepcopy(action)
        return output

    def _reset_group_history(self, name):
        setattr(self, f"{name}_pos_err_hist", deque(maxlen=self.HISTORY_LEN))
        setattr(self, f"{name}_vel_hist", deque(maxlen=self.HISTORY_LEN))

    def _reset_all_histories(self):
        for group in self.group_indices.keys():
            self._reset_group_history(group)

    def _update_group_history(
        self,
        name,
        target,
        q,
        d,
        indices,
        axis_signs,
        input_in_motor_space=False,
        gear_ratio=1.0,
    ):
        target = np.asarray(target, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)

        tgt_g = target[indices].copy() * axis_signs
        q_g = q[indices].copy() * axis_signs
        d_g = d[indices].copy() * axis_signs

        if input_in_motor_space:
            tgt_g = tgt_g * gear_ratio
            q_g = q_g * gear_ratio
            d_g = d_g * gear_ratio

        pos_err_g = tgt_g - q_g

        pos_hist = getattr(self, f"{name}_pos_err_hist")
        vel_hist = getattr(self, f"{name}_vel_hist")

        pos_hist.append(pos_err_g)
        vel_hist.append(d_g)

        while len(pos_hist) < self.HISTORY_LEN:
            pos_hist.appendleft(pos_err_g.copy())
        while len(vel_hist) < self.HISTORY_LEN:
            vel_hist.appendleft(d_g.copy())

    def _build_group_net_input(self, name):
        pos_hist = getattr(self, f"{name}_pos_err_hist")
        vel_hist = getattr(self, f"{name}_vel_hist")

        e_t = pos_hist[-1]
        e_t1 = pos_hist[-2]
        e_t2 = pos_hist[-3]

        v_t = vel_hist[-1]
        v_t1 = vel_hist[-2]
        v_t2 = vel_hist[-3]

        if self.INPUT_ORDER == "vel_pos":
            return np.stack([v_t, v_t1, v_t2, e_t, e_t1, e_t2], axis=1).astype(np.float32)
        return np.stack([e_t, e_t1, e_t2, v_t, v_t1, v_t2], axis=1).astype(np.float32)

    def _infer_group_torque(self, group, target, q, d):
        model = self.group_models[group]
        if model is None:
            return None

        opt = self.group_options[group]
        axis_signs = opt["axis_signs"]
        indices = self.group_indices[group]

        self._update_group_history(
            name=group,
            target=target,
            q=q,
            d=d,
            indices=indices,
            axis_signs=axis_signs,
            input_in_motor_space=opt["input_in_motor_space"],
            gear_ratio=opt["gear_ratio"],
        )

        net_input = self._build_group_net_input(group)
        x = torch.from_numpy(net_input).to(self.device)

        with torch.no_grad():
            tau = model(x)

        tau = tau.detach().cpu().numpy().reshape(-1).astype(np.float32)
        tau = tau * opt["torque_scale"]

        if opt["output_in_motor_space"]:
            tau = tau * opt["gear_ratio"] * opt["gamma"]

        return tau * axis_signs

    def compute_torque(self, kp, tq, q, kd, td, d):
        tq = np.asarray(tq, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)

        if np.isscalar(td):
            td = np.full_like(d, fill_value=td, dtype=np.float32)
        else:
            td = np.asarray(td, dtype=np.float32)

        tq_pd = self.delay_filter(tq)
        tq_pd = np.asarray(tq_pd, dtype=np.float32)

        tau = self.pd_controller(kp, tq_pd, q, kd, td, d)
        tau = np.asarray(tau, dtype=np.float32)

        for group, mode in self.group_modes.items():
            if mode != "actuator_net":
                continue
            group_tau = self._infer_group_torque(group, tq, q, d)
            if group_tau is not None:
                tau[self.group_indices[group]] = group_tau

        return tau

    def reset(self):
        self.prev_action = None
        self.filtered_action = None
        self._reset_all_histories()
