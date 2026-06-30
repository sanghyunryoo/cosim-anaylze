import mujoco
import numpy as np


def get_actuator_mode(config, default="torque"):
    mode = str(config.get("hardware", {}).get("actuator_mode", default)).strip().lower()
    aliases = {
        "motor": "torque",
        "mit": "torque",
        "pd": "torque",
        "pos": "position",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("torque", "position"):
        raise ValueError("hardware.actuator_mode must be 'torque' or 'position'.")
    return mode


def mj_step_once(env):
    mujoco.mj_step(env.model, env.data)


def simulate_dynamic_ctrl(env, ctrl_fn):
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = ctrl_fn()
        mj_step_once(env)
    mujoco.mj_rnePostConstraint(env.model, env.data)


def _joint_gain(hw, joint_name, prefix):
    name = joint_name.lower()
    candidates = []

    for side in ("left_", "right_", "fl_", "fr_", "rl_", "rr_"):
        name = name.replace(side, "")

    if "hip_pitch" in name:
        candidates.append(f"{prefix}_hip_pitch")
    if "hip_roll" in name:
        candidates.append(f"{prefix}_hip_roll")
    if "hip_yaw" in name:
        candidates.append(f"{prefix}_hip_yaw")
    if "hip" in name:
        candidates.append(f"{prefix}_hip")
    if "torso_yaw" in name:
        candidates.append(f"{prefix}_torso_yaw")
    if "torso_pitch" in name:
        candidates.append(f"{prefix}_torso_pitch")
    if "torso_roll" in name:
        candidates.append(f"{prefix}_torso_roll")
    if "torso" in name:
        candidates.append(f"{prefix}_torso")
    if "shoulder_pitch" in name:
        candidates.append(f"{prefix}_shoulder_pitch")
    if "shoulder_roll" in name:
        candidates.append(f"{prefix}_shoulder_roll")
    if "shoulder_yaw" in name:
        candidates.append(f"{prefix}_shoulder_yaw")
    if "shoulder" in name:
        candidates.append(f"{prefix}_shoulder")
    if "knee" in name:
        candidates.append(f"{prefix}_knee")
    if "ankle_pitch" in name:
        candidates.append(f"{prefix}_ankle_pitch")
    if "ankle_roll" in name:
        candidates.append(f"{prefix}_ankle_roll")
    if "elbow_pitch" in name:
        candidates.append(f"{prefix}_elbow_pitch")
    if "elbow_yaw" in name:
        candidates.append(f"{prefix}_elbow_yaw")
    if "elbow" in name:
        candidates.append(f"{prefix}_elbow")
    if "wrist" in name:
        candidates.append(f"{prefix}_wrist")
    if "head" in name:
        candidates.append(f"{prefix}_head")
    if "leg" in name:
        candidates.append(f"{prefix}_leg")

    fallback = 0.0 if prefix == "Kd" else 1.0
    for key in candidates:
        if key in hw:
            return float(hw[key])
    return fallback


def _joint_torque_limit(hw, joint_name):
    name = joint_name.lower()
    specific = f"{joint_name}_max_torque"
    if specific in hw:
        return float(hw[specific])

    for side in ("left_", "right_", "fl_", "fr_", "rl_", "rr_"):
        name = name.replace(side, "")

    candidates = []
    if "hip_pitch" in name:
        candidates.append("hip_pitch_joint_max_torque")
    if "hip_roll" in name:
        candidates.append("hip_roll_joint_max_torque")
    if "hip_yaw" in name:
        candidates.append("hip_yaw_joint_max_torque")
    if "hip" in name:
        candidates.append("hip_max_torque")
    if "torso_yaw" in name:
        candidates.append("torso_yaw_joint_max_torque")
    if "torso_pitch" in name:
        candidates.append("torso_pitch_joint_max_torque")
    if "torso_roll" in name:
        candidates.append("torso_roll_joint_max_torque")
    if "torso" in name:
        candidates.append("torso_joint_max_torque")
    if "shoulder_pitch" in name:
        candidates.append("shoulder_pitch_joint_max_torque")
    if "shoulder_roll" in name:
        candidates.append("shoulder_roll_joint_max_torque")
    if "shoulder_yaw" in name:
        candidates.append("shoulder_yaw_joint_max_torque")
    if "shoulder" in name:
        candidates.append("shoulder_max_torque")
    if "knee" in name:
        candidates.append("knee_joint_max_torque")
    if "ankle_pitch" in name:
        candidates.append("ankle_pitch_joint_max_torque")
    if "ankle_roll" in name:
        candidates.append("ankle_roll_joint_max_torque")
    if "elbow_pitch" in name:
        candidates.append("elbow_pitch_joint_max_torque")
    if "elbow_yaw" in name:
        candidates.append("elbow_yaw_joint_max_torque")
    if "elbow" in name:
        candidates.append("elbow_joint_max_torque")
    if "wrist" in name:
        candidates.append("wrist_joint_max_torque")
    if "head" in name:
        candidates.append("head_joint_max_torque")
    if "wheel" in name:
        candidates.append("wheel_max_torque")
    if "leg" in name:
        candidates.append("leg_max_torque")

    for key in candidates:
        if key in hw:
            return float(hw[key])
    return 100.0


def configure_actuator_xml(root, config, position_exclude_names=("wheel",)):
    hw = config.get("hardware", {})
    mode = get_actuator_mode(config, default="torque")
    position_excludes = tuple(str(name).lower() for name in position_exclude_names)

    actuated_joints = []
    for actuator in root.findall(".//actuator/*"):
        joint_name = actuator.attrib.get("joint")
        if joint_name:
            actuated_joints.append(joint_name)

    joint_by_name = {joint.attrib.get("name"): joint for joint in root.findall(".//joint") if joint.attrib.get("name")}
    for joint_name in actuated_joints:
        joint = joint_by_name.get(joint_name)
        if joint is None:
            continue
        if any(excluded in joint_name.lower() for excluded in position_excludes):
            continue
        kd = _joint_gain(hw, joint_name, "Kd")
        joint.attrib["damping"] = str(kd if mode == "position" else 0.0)

    for actuator in list(root.findall(".//actuator/*")):
        joint_name = actuator.attrib.get("joint")
        if not joint_name:
            continue
        torque_limit = _joint_torque_limit(hw, joint_name)
        is_position = mode == "position" and not any(
            excluded in joint_name.lower() for excluded in position_excludes
        )
        actuator.tag = "position" if is_position else "motor"
        if is_position:
            actuator.attrib["kp"] = str(_joint_gain(hw, joint_name, "Kp"))
            joint_range = joint_by_name.get(joint_name, {}).attrib.get("range") if joint_by_name.get(joint_name) is not None else None
            actuator.attrib["ctrlrange"] = joint_range if joint_range else "-3.141592653589793 3.141592653589793"
        else:
            actuator.attrib.pop("kp", None)
            actuator.attrib["ctrlrange"] = f"{-torque_limit} {torque_limit}"


def position_actuator_mask(model):
    return np.asarray(model.actuator_biastype != 0, dtype=bool)


def simulate_actuator_mode(env, torque_ctrl_fn, position_ctrl=None):
    position_mask = position_actuator_mask(env.model)
    uses_position = bool(np.any(position_mask))
    if not uses_position:
        simulate_dynamic_ctrl(env, torque_ctrl_fn)
        return

    if position_ctrl is None:
        raise ValueError("position_ctrl is required when position actuators are present.")

    position_ctrl = np.asarray(position_ctrl, dtype=np.float64)
    motor_mask = ~position_mask
    if not np.any(motor_mask):
        env.do_simulation(position_ctrl, env.frame_skip)
        return

    def ctrl_fn():
        ctrl = position_ctrl.copy()
        torque_ctrl = np.asarray(torque_ctrl_fn(), dtype=np.float64)
        ctrl[motor_mask] = torque_ctrl[motor_mask]
        return ctrl

    simulate_dynamic_ctrl(env, ctrl_fn)
