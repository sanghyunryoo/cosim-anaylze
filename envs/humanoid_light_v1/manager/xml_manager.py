import os
import xml.etree.ElementTree as ET

import numpy as np


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
            
            "head_link", "head_cam_link", "lower_cam_link"]

        self.precision_attr_map = config["random_table"]["precision"]

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
        if "torso" in joint_name:
            return hw.get("Kp_torso", 300), hw.get("Kd_torso", 6)
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

    def get_model_path(self):
        # The current Isaac policy was trained from the URDF/USD model, not the
        # older hand-edited mesh_v2 XML.  Use the URDF-converted MuJoCo model so
        # principal-axis inertias, convex collision meshes, and fixed-joint
        # merges match the training asset as closely as this sim-to-sim path can.
        original_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'humanoid_light_v1_from_urdf.xml')
        tree = ET.parse(original_model_path)
        root = tree.getroot()

        # Keep URDF-converted convex meshes for collision, but use the decimated
        # mesh_v2 assets for visuals.  The mesh_v2 files are the MuJoCo-safe
        # Blender-decimated versions of the original URDF visuals.
        asset = root.find("asset")
        visual_mesh_names = set()
        if asset is not None:
            for mesh in list(asset.findall("mesh")):
                mesh_name = mesh.attrib.get("name")
                mesh_file = mesh.attrib.get("file")
                if not mesh_name or not mesh_file:
                    continue
                basename = os.path.basename(mesh_file)
                stem, ext = os.path.splitext(basename)
                modified = os.path.join(self.cur_dir, "..", "assets", "mesh_v2", f"{stem}_modified{ext}")
                plain = os.path.join(self.cur_dir, "..", "assets", "mesh_v2", basename)
                if os.path.isfile(modified):
                    visual_file = f"../mesh_v2/{stem}_modified{ext}"
                elif os.path.isfile(plain):
                    visual_file = f"../mesh_v2/{basename}"
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
                    "name": f"{geom_name}_mesh_v2_visual",
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

        # Use MuJoCo position actuators for this policy.  The Isaac action is a
        # joint-position target; this path matches the stable behavior of the
        # trained policy better than re-integrating the PD torque explicitly.
        joint_ranges = {}
        for joint in root.findall(".//joint"):
            joint_name = joint.attrib.get("name")
            if not joint_name:
                continue
            _, kd = self._pd_gains_for_joint(joint_name)
            joint.attrib["damping"] = str(kd)
            joint_ranges[joint_name] = joint.attrib.get("range", "-3.14 3.14")

        actuator = root.find("actuator")
        if actuator is not None:
            for actuator_elem in list(actuator):
                joint_name = actuator_elem.attrib.get("joint")
                if not joint_name:
                    continue
                kp, _ = self._pd_gains_for_joint(joint_name)
                actuator_elem.tag = "position"
                actuator_elem.attrib.pop("gear", None)
                actuator_elem.attrib["kp"] = str(kp)
                actuator_elem.attrib["ctrllimited"] = "true"
                actuator_elem.attrib["ctrlrange"] = joint_ranges.get(joint_name, "-3.14 3.14")

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

        randomized_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'applied_humanoid_p_v0.xml')
        tree.write(randomized_model_path)
        return randomized_model_path
