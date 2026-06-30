import numpy as np


INITIAL_POSE_METADATA = {
    "wheeldog_p_v2": {
        "base_z": 0.47957,
        "joint_names": [
            "FL_hip_joint", "FR_hip_joint",
            "FL_shoulder_joint", "FR_shoulder_joint",
            "FL_leg_joint", "FR_leg_joint",
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_hip_joint", "RR_hip_joint",
            "RL_shoulder_joint", "RR_shoulder_joint",
            "RL_leg_joint", "RR_leg_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ],
    },
    "bon_p_v1": {
        "base_z": 0.5,
        "joint_names": [
            "FL_hip_joint", "FR_hip_joint",
            "FL_shoulder_joint", "FR_shoulder_joint",
            "FL_leg_joint", "FR_leg_joint",
            "FL_wheel_joint", "FR_wheel_joint",
            "RL_hip_joint", "RR_hip_joint",
            "RL_shoulder_joint", "RR_shoulder_joint",
            "RL_leg_joint", "RR_leg_joint",
            "RL_wheel_joint", "RR_wheel_joint",
        ],
    },
    "flamingo_p_v3": {
        "base_z": 0.4757,
        "joint_names": [
            "left_hip_joint", "right_hip_joint",
            "left_shoulder_joint", "right_shoulder_joint",
            "left_leg_joint", "right_leg_joint",
            "left_wheel_joint", "right_wheel_joint",
        ],
    },
    "flamingo_p_v3_2": {
        "base_z": 0.4757,
        "joint_names": [
            "left_hip_joint", "right_hip_joint",
            "left_shoulder_joint", "right_shoulder_joint",
            "left_leg_joint", "right_leg_joint",
            "left_wheel_joint", "right_wheel_joint",
        ],
    },
    "flamingo_p_v3_1": {
        "base_z": 0.642,
        "joint_names": [
            "left_hip_joint", "right_hip_joint",
            "left_shoulder_joint", "right_shoulder_joint",
            "left_leg_joint", "right_leg_joint",
            "left_wheel_joint", "right_wheel_joint",
        ],
    },
    "flamingo_p_10dof": {
        "base_z": 0.615,
        "joint_names": [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_leg_joint", "right_leg_joint",
            "left_wheel_joint", "right_wheel_joint",
        ],
    },
    "wheeldog_p_v0": {
        "base_z": 0.6,
        "joint_names": [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_shoulder_joint", "FR_shoulder_joint", "RL_shoulder_joint", "RR_shoulder_joint",
            "FL_leg_joint", "FR_leg_joint", "RL_leg_joint", "RR_leg_joint",
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        ],
    },
    "flamingo_light_p_v3": {
        "base_z": 0.13,
        "joint_names": [
            "left_shoulder_joint", "right_shoulder_joint",
            "left_wheel_joint", "right_wheel_joint",
        ],
    },
    "humanoid_p_v0": {
        "base_z": 1.105,
        "joint_names": [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "torso_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            "left_knee_joint", "right_knee_joint",
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_elbow_pitch_joint", "right_elbow_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            "left_elbow_yaw_joint", "right_elbow_yaw_joint",
        ],
    },
    "humanoid_light_v2": {
        "base_z": 0.75,
        "joint_names": [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_hip_roll_joint",
            "left_ankle_pitch_joint",
            "left_hip_yaw_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_hip_roll_joint",
            "right_ankle_pitch_joint",
            "right_hip_yaw_joint",
            "right_ankle_roll_joint",
            "torso_yaw_joint",
            "head_joint",
            "torso_pitch_joint",
            "torso_roll_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_joint",
            "right_wrist_joint",
        ],
    },
}


def get_initial_pose_metadata(env_id):
    return INITIAL_POSE_METADATA.get(env_id, {"base_z": 0.3, "joint_names": []})


def get_initial_pose_joint_names(env_id):
    return list(get_initial_pose_metadata(env_id)["joint_names"])


def get_default_initial_joint_map(env_id):
    return {name: 0.0 for name in get_initial_pose_joint_names(env_id)}


def get_default_initial_pose(env_id):
    metadata = get_initial_pose_metadata(env_id)
    return {
        "base_z": float(metadata["base_z"]),
        "joints": get_default_initial_joint_map(env_id),
    }


def _initial_pose_config(config):
    for key in ("initial_positions", "initial_pose", "joint_offsets"):
        raw = config.get(key, {}) or {}
        if isinstance(raw, dict) and raw:
            return raw
    return {}


def _normalize_joint_overrides(config):
    raw = _initial_pose_config(config)
    if not isinstance(raw, dict):
        return {}
    joints = raw.get("joints", raw)
    if not isinstance(joints, dict):
        return {}
    normalized = {}
    for name, value in joints.items():
        try:
            normalized[str(name)] = float(value)
        except Exception:
            continue
    return normalized


def _base_z_override(config, default_base_z):
    raw = _initial_pose_config(config)
    if not isinstance(raw, dict):
        return float(default_base_z)
    for key in ("base_z", "z"):
        if key in raw:
            try:
                return float(raw[key])
            except Exception:
                break
    return float(default_base_z)


def build_initial_qpos(model, mujoco_utils, config, env_id, init_noise, joint_names=None):
    metadata = get_initial_pose_metadata(env_id)
    resolved_joint_names = list(joint_names or metadata["joint_names"])
    joint_overrides = _normalize_joint_overrides(config)

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[2] = _base_z_override(config, metadata["base_z"])
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if not resolved_joint_names:
        return qpos

    q_indices = mujoco_utils.get_qpos_joint_indices_by_name(resolved_joint_names)
    joint_values = np.array(
        [joint_overrides.get(name, 0.0) for name in resolved_joint_names],
        dtype=np.float64,
    )
    if init_noise > 0.0:
        joint_values = joint_values + np.random.uniform(-init_noise, init_noise, size=joint_values.shape)
    qpos[q_indices] = joint_values
    return qpos
