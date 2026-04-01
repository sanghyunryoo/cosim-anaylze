from envs.flamingo_light_p_v3.flamingo_light_p_v3 import FlamingoLightPV3
from envs.flamingo_p_v3.flamingo_p_v3 import FlamingoPV3
from envs.flamingo_p_v3_2.flamingo_p_v3 import FlamingoPV32
from envs.flamingo_p_10dof.flamingo_p_10dof import FlamingoP10dof
from envs.bon_p_v1.bon_p_v1 import BonPV1
from envs.wheeldog_p_v0.wheeldog_p_v0 import WheelDogPV0
from envs.wheeldog_p_v2.wheeldog_p_v2 import WheelDogPV2
from envs.humanoid_p_v0.humanoid_p_v0 import HumanoidPV0
from envs.humanoid_light_v1.humanoid_light_v1 import HumanoidLightV1
from envs.wrappers import StateBuildWrapper, TimeLimitWrapper, CommandWrapper


def build_env(config):
    # Normalize legacy/new config keys.
    if "settings" in config and "observation" not in config:
      config["observation"] = config["settings"]
    elif "observation" in config and "settings" not in config:
      config["settings"] = config["observation"]

    if config["env"]['id'] == "flamingo_p_v3":
      env = FlamingoPV3(config)
    elif config["env"]['id'] == "flamingo_p_v3_2":
      env = FlamingoPV32(config)
    elif config["env"]['id'] == "flamingo_p_10dof":
      env = FlamingoP10dof(config)
    elif config["env"]['id'] == "flamingo_light_p_v3":
      env = FlamingoLightPV3(config)  
    elif config["env"]['id'] == "bon_p_v1":
      env = BonPV1(config)
    elif config["env"]['id'] == "wheeldog_p_v0":
      env = WheelDogPV0(config)
    elif config["env"]['id'] == "wheeldog_p_v2":
      env = WheelDogPV2(config)
    elif config["env"]['id'] == "humanoid_p_v0":
      env = HumanoidPV0(config)
    elif config["env"]['id'] == "humanoid_light_v1":
      env = HumanoidLightV1(config)
    else:
      raise NameError(f"Please select a valid environment id. Received '{config['env']['id']}'.")
    
    env = StateBuildWrapper(env, config)
    env = TimeLimitWrapper(env, config)
    env = CommandWrapper(env, config)

    return env
