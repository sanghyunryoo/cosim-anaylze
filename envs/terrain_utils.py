import xml.etree.ElementTree as ET
from math import isfinite


_BEAM_EASY_DEFAULTS = {
    # Keep a clear approach area in front of the robot spawn at the world origin.
    "beam_start_x": 2.0,
    "beam_count": 12,
    "beam_spacing": 0.79,
    "beam_height": 0.35,
}
_BEAM_FLANGE_HALF_THICKNESS = 0.005


def _beam_easy_settings(settings):
    """Return validated beam settings with a clear approach area by default."""
    settings = settings if isinstance(settings, dict) else {}
    normalized = dict(_BEAM_EASY_DEFAULTS)

    for key in ("beam_start_x", "beam_spacing", "beam_height"):
        try:
            value = float(settings.get(key, normalized[key]))
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            normalized[key] = value

    try:
        normalized["beam_count"] = int(settings.get("beam_count", normalized["beam_count"]))
    except (TypeError, ValueError):
        pass

    # A beam must remain above the ground and its web must have positive height.
    normalized["beam_spacing"] = max(0.001, normalized["beam_spacing"])
    normalized["beam_height"] = max(2 * _BEAM_FLANGE_HALF_THICKNESS + 0.001, normalized["beam_height"])
    normalized["beam_count"] = max(1, normalized["beam_count"])
    return normalized


def add_beam_easy_geoms(root, beam_settings=None):
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Could not find <worldbody> in the XML file.")

    settings = _beam_easy_settings(beam_settings)
    beam_start_x = settings["beam_start_x"]
    beam_spacing = settings["beam_spacing"]
    beam_height = settings["beam_height"]
    beam_count = settings["beam_count"]
    beam_centers_x = [beam_start_x + idx * beam_spacing for idx in range(beam_count)]
    beam_y = -0.005
    web_half_height = (beam_height - 2 * _BEAM_FLANGE_HALF_THICKNESS) / 2
    flange_center_z = beam_height - _BEAM_FLANGE_HALF_THICKNESS

    for idx, center_x in enumerate(beam_centers_x):
        body = ET.Element("body", {
            "name": f"beam_easy_t_beam_{idx:02d}",
            "pos": f"{center_x:.6f} {beam_y:.6f} 0",
        })
        body.append(ET.Element("geom", {
            "name": f"beam_easy_t_beam_{idx:02d}_web",
            "type": "box",
            "pos": f"0 0 {web_half_height:.6f}",
            "size": f"0.005000 2.495000 {web_half_height:.6f}",
            "rgba": "0.48 0.50 0.54 1",
            "friction": "1.0 0.005 0.0001",
            "contype": "1",
            "conaffinity": "1",
            "group": "1",
        }))
        body.append(ET.Element("geom", {
            "name": f"beam_easy_t_beam_{idx:02d}_flange",
            "type": "box",
            "pos": f"0 0 {flange_center_z:.6f}",
            "size": "0.062500 2.495000 0.005000",
            "rgba": "0.48 0.50 0.54 1",
            "friction": "1.0 0.005 0.0001",
            "contype": "1",
            "conaffinity": "1",
            "group": "1",
        }))
        worldbody.append(body)


def configure_terrain_xml(root, terrain, beam_settings=None):
    for geom in root.findall(".//geom"):
        if geom.attrib.get("name") == "ground":
            if terrain in {"flat", "beam_easy"}:
                geom.attrib["type"] = "plane"
                geom.attrib.pop("hfield", None)
                geom.attrib["pos"] = "0 0 0"
                geom.attrib["size"] = "100 100 0.1"
            else:
                geom.attrib["type"] = "hfield"
                geom.attrib["hfield"] = terrain
                geom.attrib["pos"] = "0 0 0"

    if terrain == "beam_easy":
        add_beam_easy_geoms(root, beam_settings)
