from envs.flamingo_light_p_v3.flamingo_light_p_v3 import FlamingoLightPV3
from envs.flamingo_p_v3.flamingo_p_v3 import FlamingoPV3
from envs.flamingo_p_v3_1.flamingo_p_v3 import FlamingoPV31
from envs.flamingo_p_v3_2.flamingo_p_v3 import FlamingoPV32
from envs.flamingo_p_10dof.flamingo_p_10dof import FlamingoP10dof
from envs.bon_p_v1.bon_p_v1 import BonPV1
from envs.hd_dog.hd_dog import HDDog
from envs.wheeldog_p_v0.wheeldog_p_v0 import WheelDogPV0
from envs.wheeldog_p_v2.wheeldog_p_v2 import WheelDogPV2
from envs.humanoid_p_v0.humanoid_p_v0 import HumanoidPV0
from envs.humanoid_light_v2.humanoid_light_v2 import HumanoidLightV2
from envs.humanoid_light_v2.reference_motion_loader import (
    HumanoidLightReferenceMotion,
    resolve_reference_motion,
)
from envs.wrappers import (
    CommandWrapper,
    ReferenceMotionProgressWrapper,
    ReferenceMotionResetWrapper,
    ReferenceMotionTargetWrapper,
    StateBuildWrapper,
    TimeLimitWrapper,
)


def build_env(config):
    # Normalize legacy/new config keys.
    if "settings" in config and "observation" not in config:
      config["observation"] = config["settings"]
    elif "observation" in config and "settings" not in config:
      config["settings"] = config["observation"]

    render_flag = bool(config.get("env", {}).get("render", True))
    render_mode = config.get("env", {}).get("render_mode", "human")

    reference_cfg = config.get("reference_motion", {}) or {}
    reference_enabled = bool(reference_cfg.get("enabled", False))
    reference_motion = None

    if config["env"]['id'] == "flamingo_p_v3":
      env = FlamingoPV3(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "flamingo_p_v3_1":
      env = FlamingoPV31(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "flamingo_p_v3_2":
      env = FlamingoPV32(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "flamingo_p_10dof":
      env = FlamingoP10dof(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "flamingo_light_p_v3":
      env = FlamingoLightPV3(config, render_flag=render_flag, render_mode=render_mode)  
    elif config["env"]['id'] == "bon_p_v1":
      env = BonPV1(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "hd_dog":
      env = HDDog(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "wheeldog_p_v0":
      env = WheelDogPV0(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "wheeldog_p_v2":
      env = WheelDogPV2(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "humanoid_p_v0":
      env = HumanoidPV0(config, render_flag=render_flag, render_mode=render_mode)
    elif config["env"]['id'] == "humanoid_light_v2":
      env = HumanoidLightV2(config, render_flag=render_flag, render_mode=render_mode)
    else:
      raise NameError(f"Please select a valid environment id. Received '{config['env']['id']}'.")
    
    if reference_enabled:
      if config["env"]['id'] != "humanoid_light_v2":
        raise ValueError("Reference-motion inference is currently available only for humanoid_light_v2.")
      reference_dir = reference_cfg.get("directory")
      if not reference_dir:
        raise ValueError("reference_motion.directory is required for Humanoid Light reference inference.")
      reference_path = resolve_reference_motion(reference_cfg.get("motion"), reference_dir)
      reference_motion = HumanoidLightReferenceMotion(reference_path, env.joint_names_in_order)
      # Do not overwrite env.max_duration with the clip duration here.  The
      # GUI initialises it from the selected clip, but an explicit user value
      # must reach TimeLimitWrapper so post-clip behaviour can be inspected.
      # Reference values/commands are clamped to the last clip frame after
      # the clip ends.
      config["reference_motion"] = {
        **reference_cfg,
        "enabled": True,
        "motion": str(reference_path),
      }
      env = ReferenceMotionResetWrapper(env, reference_motion, config)

    env = StateBuildWrapper(env, config)
    env = TimeLimitWrapper(env, config)
    # A zero command dimension is a supported no-command configuration. This
    # keeps the standard locomotion wrapper flow unchanged for the 91-D
    # reference student: StateBuild(90 + progress) -> CommandWrapper(0).
    env = CommandWrapper(env, config)
    if reference_motion is not None:
      inference_mode = str(reference_cfg.get("inference_mode", "teacher")).strip().lower()
      if inference_mode == "teacher":
        env = ReferenceMotionTargetWrapper(env, reference_motion, control_freq=50.0, config=config)
      elif inference_mode == "distilled_locomotion":
        env = ReferenceMotionProgressWrapper(env, reference_motion, control_freq=50.0, config=config)
      else:
        raise ValueError(f"Unknown Humanoid Light reference inference mode: {inference_mode}")

    return env
