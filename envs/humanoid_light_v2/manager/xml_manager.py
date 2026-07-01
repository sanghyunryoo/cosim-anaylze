import os
import xml.etree.ElementTree as ET

import numpy as np


def _config_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


class XMLManager:
    def __init__(self, config):
        self.config = config
        self.cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.body_components =["pelvis_link", "torso_roll_link", "torso_pitch_link", "torso_yaw_link",
                               
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", 
            "left_elbow_link", "left_wrist_link",

            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
            "right_elbow_link", "right_wrist_link",

            "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
            "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",

            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
            "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
            
            "head_link", "head_cam_link", "lower_cam_link",
            "lower_imu_link", "upper_imu_link"]

        self.precision_attr_map = config["random_table"]["precision"]

    def _actuator_mode(self):
        mode = str(
            self.config["hardware"].get(
                "actuator_mode",
                self.config["hardware"].get("control_mode", "position"),
            )
        ).strip().lower()
        if mode == "motor":
            mode = "torque"
        if mode not in ("position", "torque"):
            raise ValueError("humanoid_light_v2 hardware.actuator_mode must be 'position' or 'torque'.")
        return mode

    def _pd_gains_for_joint(self, joint_name):
        hw = self.config["hardware"]
        if "hip_pitch" in joint_name:
            return hw.get("Kp_hip_pitch", 150), hw.get("Kd_hip_pitch", 2)
        if "hip_roll" in joint_name:
            return hw.get("Kp_hip_roll", 150), hw.get("Kd_hip_roll", 2)
        if "hip_yaw" in joint_name:
            return hw.get("Kp_hip_yaw", 150), hw.get("Kd_hip_yaw", 2)
        if "knee" in joint_name:
            return hw.get("Kp_knee", 200), hw.get("Kd_knee", 4)
        if "ankle_pitch" in joint_name:
            return hw.get("Kp_ankle_pitch", 40), hw.get("Kd_ankle_pitch", 2)
        if "ankle_roll" in joint_name:
            return hw.get("Kp_ankle_roll", 40), hw.get("Kd_ankle_roll", 2)
        if "torso_yaw" in joint_name:
            return hw.get("Kp_torso_yaw", hw.get("Kp_torso", 300)), hw.get("Kd_torso_yaw", hw.get("Kd_torso", 6))
        if "torso_pitch" in joint_name:
            return hw.get("Kp_torso_pitch", hw.get("Kp_torso", 300)), hw.get("Kd_torso_pitch", hw.get("Kd_torso", 6))
        if "torso_roll" in joint_name:
            return hw.get("Kp_torso_roll", hw.get("Kp_torso", 300)), hw.get("Kd_torso_roll", hw.get("Kd_torso", 6))
        if "shoulder_pitch" in joint_name:
            return hw.get("Kp_shoulder_pitch", 100), hw.get("Kd_shoulder_pitch", 2)
        if "shoulder_roll" in joint_name:
            return hw.get("Kp_shoulder_roll", 100), hw.get("Kd_shoulder_roll", 2)
        if "shoulder_yaw" in joint_name:
            return hw.get("Kp_shoulder_yaw", 50), hw.get("Kd_shoulder_yaw", 2)
        if "elbow" in joint_name:
            return hw.get("Kp_elbow", 50), hw.get("Kd_elbow", 2)
        if "wrist" in joint_name:
            return hw.get("Kp_wrist", 50), hw.get("Kd_wrist", 2)
        if "head" in joint_name:
            return hw.get("Kp_head", 50), hw.get("Kd_head", 2)
        return 100, 2

    def _torque_limit_for_joint(self, joint_name):
        hw = self.config["hardware"]
        if "hip_pitch" in joint_name:
            return float(hw.get("hip_pitch_joint_max_torque", 120))
        if "hip_roll" in joint_name:
            return float(hw.get("hip_roll_joint_max_torque", 60))
        if "hip_yaw" in joint_name:
            return float(hw.get("hip_yaw_joint_max_torque", 60))
        if "knee" in joint_name:
            return float(hw.get("knee_joint_max_torque", 120))
        if "ankle_pitch" in joint_name:
            return float(hw.get("ankle_pitch_joint_max_torque", 14))
        if "ankle_roll" in joint_name:
            return float(hw.get("ankle_roll_joint_max_torque", 14))
        if "torso_yaw" in joint_name:
            return float(hw.get("torso_yaw_joint_max_torque", hw.get("torso_joint_max_torque", 60)))
        if "torso_pitch" in joint_name:
            return float(hw.get("torso_pitch_joint_max_torque", hw.get("torso_joint_max_torque", 60)))
        if "torso_roll" in joint_name:
            return float(hw.get("torso_roll_joint_max_torque", hw.get("torso_joint_max_torque", 60)))
        if "shoulder_pitch" in joint_name:
            return float(hw.get("shoulder_pitch_joint_max_torque", 60))
        if "shoulder_roll" in joint_name:
            return float(hw.get("shoulder_roll_joint_max_torque", 60))
        if "shoulder_yaw" in joint_name:
            return float(hw.get("shoulder_yaw_joint_max_torque", 17))
        if "elbow" in joint_name:
            return float(hw.get("elbow_joint_max_torque", 36))
        if "wrist" in joint_name:
            return float(hw.get("wrist_joint_max_torque", 14))
        if "head" in joint_name:
            return float(hw.get("head_joint_max_torque", 5.5))
        return 100.0

    def _coupled_control_joint_names(self):
        hw = self.config["hardware"]
        if not _config_bool(hw.get("coupled_control_enabled", hw.get("coupled_enabled", False)), False):
            return set()

        coupled_cfg = hw.get("coupled_actuators", {})

        def pair_enabled(pair_name, legacy_key):
            pair_cfg = coupled_cfg.get(pair_name, {}) if isinstance(coupled_cfg, dict) else {}
            return _config_bool(pair_cfg.get("enabled", hw.get(legacy_key, True)), True)

        joint_names = set()
        if pair_enabled("left_ankle", "left_ankle_coupled"):
            joint_names.update(("left_ankle_pitch_joint", "left_ankle_roll_joint"))
        if pair_enabled("right_ankle", "right_ankle_coupled"):
            joint_names.update(("right_ankle_pitch_joint", "right_ankle_roll_joint"))
        if pair_enabled("torso_pitch_roll", "torso_coupled"):
            joint_names.update(("torso_pitch_joint", "torso_roll_joint"))
        return joint_names

    def _coupled_position_gains_by_joint(self):
        hw = self.config["hardware"]
        if not _config_bool(hw.get("coupled_control_enabled", hw.get("coupled_enabled", False)), False):
            return {}

        coupled_cfg = hw.get("coupled_actuators", {})
        pair_defs = [
            ("left_ankle", ("left_ankle_pitch_joint", "left_ankle_roll_joint"), -2.0, -2.0, False, "left_ankle"),
            ("right_ankle", ("right_ankle_pitch_joint", "right_ankle_roll_joint"), -2.0, -2.0, False, "right_ankle"),
            ("torso_pitch_roll", ("torso_pitch_joint", "torso_roll_joint"), 1.0, 1.0, False, "torso"),
        ]

        gains_by_joint = {}
        for pair_name, joint_names, default_g1, default_g2, default_mirror, prefix in pair_defs:
            pair_cfg = coupled_cfg.get(pair_name, {}) if isinstance(coupled_cfg, dict) else {}
            if not _config_bool(pair_cfg.get("enabled", hw.get(f"{prefix}_coupled", True)), True):
                continue

            if pair_name in ("left_ankle", "right_ankle"):
                shared_g1 = hw.get("ankle_gear_ratio_1", hw.get("ankle_gear_ratio", default_g1))
                shared_g2 = hw.get("ankle_gear_ratio_2", hw.get("ankle_gear_ratio", default_g2))
            else:
                shared_g1 = hw.get("torso_pitch_roll_gear_ratio_1", hw.get("torso_pitch_roll_gear_ratio", default_g1))
                shared_g2 = hw.get("torso_pitch_roll_gear_ratio_2", hw.get("torso_pitch_roll_gear_ratio", default_g2))

            g1 = float(pair_cfg.get("gear_ratio_1", hw.get(f"{prefix}_gear_ratio_1", shared_g1)))
            g2 = float(pair_cfg.get("gear_ratio_2", hw.get(f"{prefix}_gear_ratio_2", shared_g2)))
            gamma = float(pair_cfg.get("gamma", hw.get(f"{prefix}_gamma", hw.get("coupled_gamma", 1.0))))
            if g1 == 0.0 or g2 == 0.0:
                raise ValueError(f"{pair_name} coupled actuator gear ratios must be non-zero.")
            if not np.isclose(abs(g1), abs(g2)):
                raise ValueError(
                    f"{pair_name} requires |gear_ratio_1| == |gear_ratio_2| for position-mode coupled control."
                )

            kp1, kd1 = self._pd_gains_for_joint(joint_names[0])
            kp2, kd2 = self._pd_gains_for_joint(joint_names[1])
            if not np.isclose(float(kp1), float(kp2)):
                raise ValueError(f"{pair_name} requires equal Kp for both motor slots for position-mode coupled control.")
            if not np.isclose(float(kd1), float(kd2)):
                raise ValueError(f"{pair_name} requires equal Kd for both motor slots for position-mode coupled control.")

            gain_scale = 2.0 * abs(g1) * abs(g1) * gamma
            equivalent_gains = (gain_scale * float(kp1), gain_scale * float(kd1))
            gains_by_joint[joint_names[0]] = equivalent_gains
            gains_by_joint[joint_names[1]] = equivalent_gains
        return gains_by_joint

    @staticmethod
    def _ensure_imu_links_and_sensors(root):
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        existing_meshes = {mesh.attrib.get("name") for mesh in asset.findall("mesh")}
        for mesh_name in ("lower_imu_link", "upper_imu_link"):
            if mesh_name not in existing_meshes:
                ET.SubElement(
                    asset,
                    "mesh",
                    {"name": mesh_name, "file": f"../mesh/visual/{mesh_name}.STL"},
                )
                existing_meshes.add(mesh_name)

        imu_links = {
            "lower_imu_link": {
                "parent": "pelvis_link",
                "pos": "0 0 0",
                "mesh": "lower_imu_link",
                "sensor_prefix": "lower_imu",
            },
            "upper_imu_link": {
                "parent": "torso_roll_link",
                "pos": "0.00025 0.41400 0",
                "quat": "0.5 -0.5 -0.5 -0.5",
                "mesh": "upper_imu_link",
                "sensor_prefix": "upper_imu",
            },
        }

        for parent in root.findall(".//body"):
            for child in list(parent.findall("./body")):
                if child.attrib.get("name") in {"lower_imu", "upper_imu", *imu_links.keys()}:
                    parent.remove(child)

        for body_name, cfg in imu_links.items():
            parent = root.find(f".//body[@name='{cfg['parent']}']")
            if parent is None:
                continue

            body_attrib = {"name": body_name, "pos": cfg["pos"]}
            if "quat" in cfg:
                body_attrib["quat"] = cfg["quat"]
            body = ET.Element("body", body_attrib)
            body.append(
                ET.Element(
                    "geom",
                    {
                        "name": f"{cfg['mesh']}_visual",
                        "type": "mesh",
                        "mesh": cfg["mesh"],
                        "class": "visual",
                    },
                )
            )
            body.append(
                ET.Element(
                    "site",
                    {
                        "name": f"{cfg['sensor_prefix']}_site",
                        "size": "0.01",
                        "pos": "0 0 0",
                        "quat": "1 0 0 0",
                    },
                )
            )
            parent.insert(0, body)

        sensor = root.find("sensor")
        if sensor is None:
            sensor = ET.SubElement(root, "sensor")

        existing_names = {elem.attrib.get("name") for elem in sensor}
        for imu_name in ("lower_imu", "upper_imu"):
            site_name = f"{imu_name}_site"
            for tag, attrib in [
                ("framequat", {"name": f"{imu_name}_orientation", "objtype": "site", "objname": site_name}),
                ("gyro", {"name": f"{imu_name}_angular_velocity", "site": site_name, "cutoff": "34.9"}),
                ("velocimeter", {"name": f"{imu_name}_linear_velocity", "site": site_name, "cutoff": "30"}),
                ("accelerometer", {"name": f"{imu_name}_linear_acceleration", "site": site_name, "cutoff": "157"}),
            ]:
                if attrib["name"] not in existing_names:
                    sensor.append(ET.Element(tag, attrib))
                    existing_names.add(attrib["name"])

    def get_model_path(self):
        # Use the URDF-converted MuJoCo model with the unified mesh directory.
        original_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'humanoid_light_v2.xml')
        tree = ET.parse(original_model_path)
        root = tree.getroot()
        self._ensure_imu_links_and_sensors(root)

        # Keep source meshes for collision, and use *_modified meshes from the
        # unified mesh directory for visuals when available.
        asset = root.find("asset")
        visual_mesh_names = set()
        if asset is not None:
            for mesh in list(asset.findall("mesh")):
                mesh_name = mesh.attrib.get("name")
                mesh_file = mesh.attrib.get("file")
                if not mesh_name or not mesh_file:
                    continue
                if "/collision/" not in mesh_file:
                    continue
                _, ext = os.path.splitext(os.path.basename(mesh_file))
                modified = os.path.join(self.cur_dir, "..", "assets", "mesh", "visual", f"{mesh_name}_modified{ext}")
                plain = os.path.join(self.cur_dir, "..", "assets", "mesh", "visual", f"{mesh_name}{ext}")
                if os.path.isfile(modified):
                    visual_file = f"../mesh/visual/{mesh_name}_modified{ext}"
                elif os.path.isfile(plain):
                    visual_file = f"../mesh/visual/{mesh_name}{ext}"
                else:
                    continue
                visual_name = f"{mesh_name}_visual"
                ET.SubElement(asset, "mesh", {"name": visual_name, "file": visual_file})
                visual_mesh_names.add(mesh_name)

        for body in root.findall(".//body"):
            for geom in list(body.findall("geom")):
                if geom.attrib.get("type") != "mesh":
                    continue
                mesh_name = geom.attrib.get("mesh")
                geom_name = geom.attrib.get("name", "")
                if mesh_name not in visual_mesh_names:
                    continue
                geom.attrib["group"] = "3"
                geom.attrib["rgba"] = "1 1 1 0"

                if any(
                    existing.attrib.get("class") == "visual"
                    and existing.attrib.get("mesh") == f"{mesh_name}_visual"
                    for existing in body.findall("geom")
                ):
                    continue

                visual_attrib = {
                    "name": f"{geom_name}_mesh_visual",
                    "type": "mesh",
                    "mesh": f"{mesh_name}_visual",
                    "class": "visual",
                }
                for attr in ("pos", "quat"):
                    if attr in geom.attrib:
                        visual_attrib[attr] = geom.attrib[attr]
                if "rgba" in geom.attrib and geom.attrib["rgba"] != "1 1 1 0":
                    visual_attrib["rgba"] = geom.attrib["rgba"]
                body.append(ET.Element("geom", visual_attrib))

        # 1. Set the terrain
        terrain = self.config["env"]["terrain"]

        for geom in root.findall('.//geom'):
            if geom.attrib.get('name') == "ground":
                if terrain == "flat":
                    geom.attrib["type"] = "plane"
                    geom.attrib.pop("hfield", None)
                    geom.attrib["size"] = "100 100 0.1" 
                else:
                    geom.attrib["type"] = "hfield"
                    geom.attrib["hfield"] = terrain

        # 2. Set the precision of the simulation
        precision_level = self.config["random"]["precision"]
        if precision_level in self.precision_attr_map:
            precision_attrs = self.precision_attr_map[precision_level]
            option = root.find("option")
            if option is not None:
                option.attrib["timestep"] = str(precision_attrs["timestep"])
                option.attrib["iterations"] = str(precision_attrs["iterations"])

        # 3. Set the noisy mass & load
        for body in root.findall('.//body'):
            body_name = body.attrib.get('name')
            if body_name in self.body_components:
                for inertial in body.findall('inertial'):
                    if 'mass' in inertial.attrib:
                        original_mass = float(inertial.attrib['mass'])
                        noise = np.random.uniform(-original_mass * self.config["random"]["mass_noise"],
                                                  original_mass * self.config["random"]["mass_noise"])
                        randomized_mass = original_mass + noise
                        if body_name == "pelvis_link":
                            randomized_mass += self.config["random"]["load"]
                        inertial.attrib['mass'] = str(randomized_mass)

        # 4. Set the friction of wheel geoms in left_ankle_roll_link and right_ankle_roll_link
        for body in root.findall('.//body'):
            if body.attrib.get('name') in ['left_ankle_roll_link', 'right_ankle_roll_link']:
                for geom in body.findall('geom'):
                    if 'friction' in geom.attrib:
                        geom.attrib['friction'] = (
                            f"{self.config['random']['sliding_friction']} "
                            f"{self.config['random']['torsional_friction']} "
                            f"{self.config['random']['rolling_friction']}"
                        )

        # 5. Set the friction of ground plane
        for geom in root.findall('.//geom'):
            geom_name = geom.attrib.get('name')
            if geom_name == "ground":
                if 'friction' in geom.attrib:
                    geom.attrib['friction'] = (str(self.config["random"]["sliding_friction"])
                                               + ' ' + str(self.config["random"]["torsional_friction"])
                                               + ' ' + str(self.config["random"]["rolling_friction"]))

        # 6. Set the friction loss
        for default in root.findall(".//default"):
            default_class = default.attrib.get("class")
            for joint in default.findall("joint"):
                if "damping" in joint.attrib:
                    joint.attrib["damping"] = "0.0"
                if "frictionloss" in joint.attrib:
                    joint.attrib["frictionloss"] = str(self.config["random"]["friction_loss"])
            if default_class == "joints":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])
            elif default_class == "wheels":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])

        actuator_mode = self._actuator_mode()
        coupled_position_gains = self._coupled_position_gains_by_joint() if actuator_mode == "position" else {}
        joint_ranges = {}
        for joint in root.findall(".//joint"):
            joint_name = joint.attrib.get("name")
            if not joint_name:
                continue
            if joint_name in coupled_position_gains:
                _, kd = coupled_position_gains[joint_name]
                joint.attrib["frictionloss"] = str(self.config["random"]["friction_loss"])
            else:
                _, kd = self._pd_gains_for_joint(joint_name)
            joint.attrib["damping"] = str(kd)
            joint_ranges[joint_name] = joint.attrib.get("range", "-3.14 3.14")

        actuator = root.find("actuator")
        if actuator is not None:
            for actuator_elem in list(actuator):
                joint_name = actuator_elem.attrib.get("joint")
                if not joint_name:
                    continue
                if actuator_mode == "position":
                    if joint_name in coupled_position_gains:
                        kp, _ = coupled_position_gains[joint_name]
                    else:
                        kp, _ = self._pd_gains_for_joint(joint_name)
                    actuator_elem.tag = "position"
                    actuator_elem.attrib.pop("gear", None)
                    actuator_elem.attrib["kp"] = str(kp)
                    actuator_elem.attrib["ctrllimited"] = "true"
                    actuator_elem.attrib["ctrlrange"] = joint_ranges.get(joint_name, "-3.14 3.14")
                else:
                    torque_limit = self._torque_limit_for_joint(joint_name)
                    actuator_elem.tag = "motor"
                    actuator_elem.attrib.pop("kp", None)
                    actuator_elem.attrib["gear"] = actuator_elem.attrib.get("gear", "1")
                    actuator_elem.attrib["ctrllimited"] = "true"
                    actuator_elem.attrib["ctrlrange"] = f"{-torque_limit} {torque_limit}"

        # 7. Initialize spheres for height map
        if self.config["observation"]["height_map"] is not None:
            res_x = self.config["observation"]["height_map"]["res_x"]
            res_y = self.config["observation"]["height_map"]["res_y"]

            # Find <worldbody> and then <body name="base_link">
            worldbody = root.find('worldbody')
            base_link = None
            for body in worldbody.findall('body'):
                if body.get('name') == 'pelvis_link':
                    base_link = body
                    break

            if base_link is None:
                raise ValueError("Could not find <body name='pelvis_link' (whch is the base link)> in the XML file.")

            # Add <site> elements
            for i in range(res_y):
                for j in range(res_x):
                    site_name = f"heightmap_site_{i}_{j}"
                    site_element = ET.Element('site', {
                        'name': site_name,
                        'type': 'sphere',
                        'size': '0.00000001',
                        'pos': '0 0 -1',
                        'rgba': '0 1 0 0.0000001',
                        'group': '0',   
                    })
                    base_link.append(site_element)

        monitoring_cfg = self.config.get("monitoring", {}) or {}
        hm_cfg = monitoring_cfg.get("height_map", {}) if isinstance(monitoring_cfg.get("height_map", {}), dict) else {}
        if bool(hm_cfg.get("inference_visualize", False)):
            res_x = int(hm_cfg.get("res_x", 0) or 0)
            res_y = int(hm_cfg.get("res_y", 0) or 0)
            frame_body_name = str(hm_cfg.get("frame_body", "camera_link"))

            worldbody = root.find('worldbody')
            target_body = None
            for body in worldbody.iter('body'):
                if body.get('name') == frame_body_name:
                    target_body = body
                    break

            if target_body is None:
                raise ValueError(f"Could not find <body name='{frame_body_name}'> in the XML file.")

            for i in range(res_y):
                for j in range(res_x):
                    site_name = f"inference_heightmap_site_{i}_{j}"
                    site_element = ET.Element('site', {
                        'name': site_name,
                        'type': 'sphere',
                        'size': '0.00000001',
                        'pos': '0 0 -1',
                        'rgba': '0.15 0.7 1 0.0000001',
                        'group': '0',
                    })
                    target_body.append(site_element)

        # Isaac training runs this robot with self-collisions disabled.
        # Keep robot-ground contacts, but exclude every robot body-body pair.
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for exclude in list(contact.findall("exclude")):
            contact.remove(exclude)

        body_names = [
            body.attrib["name"]
            for body in root.findall(".//body")
            if body.attrib.get("name")
        ]
        for i, body1 in enumerate(body_names):
            for body2 in body_names[i + 1:]:
                ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

        randomized_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'applied_humanoid_light_v2.xml')
        tree.write(randomized_model_path)
        return randomized_model_path
