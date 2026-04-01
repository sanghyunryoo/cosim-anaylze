import os
import xml.etree.ElementTree as ET

import numpy as np


class XMLManager:
    def __init__(self, config, has_wheels=True, use_gear=True):
        self.config = config
        self.has_wheels = has_wheels
        self.use_gear = use_gear
        self.cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.body_components = [
            "base_link",
            "left_hip_pitch_link", "right_hip_pitch_link",
            "left_hip_roll_link", "right_hip_roll_link",
            "left_hip_yaw_link", "right_hip_yaw_link",
            "left_leg_link", "right_leg_link",
            "left_wheel_link", "right_wheel_link",
        ]

        self.precision_attr_map = config["random_table"]["precision"]

    def get_model_path(self):
        original_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'flamingo_p_10dof.xml')
        tree = ET.parse(original_model_path)
        root = tree.getroot()

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
                        if body_name == "base_link":
                            randomized_mass += self.config["random"]["load"]
                        inertial.attrib['mass'] = str(randomized_mass)

        # 4. Set the friction of wheel geoms in left_wheel_link and right_wheel_link
        for body in root.findall('.//body'):
            if body.attrib.get('name') in ['left_wheel_link', 'right_wheel_link']:
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
            if default_class == "joints":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])
            elif default_class == "wheels":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])

        # 7. Initialize spheres for height map
        if self.config["observation"]["height_map"] is not None:
            res_x = self.config["observation"]["height_map"]["res_x"]
            res_y = self.config["observation"]["height_map"]["res_y"]

            # Find <worldbody> and then <body name="base_link">
            worldbody = root.find('worldbody')
            base_link = None
            for body in worldbody.findall('body'):
                if body.get('name') == 'base_link':
                    base_link = body
                    break

            if base_link is None:
                raise ValueError("Could not find <body name='base_link'> in the XML file.")

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

        randomized_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'applied_flamingo_p_10dof.xml')
        tree.write(randomized_model_path)
        return randomized_model_path
