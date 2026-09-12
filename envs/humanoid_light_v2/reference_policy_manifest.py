"""Contract metadata for distilled Humanoid Light reference policies.

The distilled trajectory interface is the standard 90-D locomotion
observation with one explicitly configured, final non-stacked normalized
progress value. A sidecar manifest records that contract so a policy is never
run against a merely same-sized but differently ordered observation vector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "humanoid_light_reference_distilled_trajectory_v3"
PHASE_LAYOUT = ["control_progress"]


def manifest_path_for(policy_path: str | Path) -> Path:
    """Return the unambiguous sidecar path for a distilled policy."""

    return Path(policy_path).expanduser().resolve().with_suffix(".reference_imitation.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_distilled_locomotion_manifest(policy_path: str | Path) -> dict[str, Any] | None:
    """Return a valid distilled-policy manifest, or ``None`` for other ONNXes."""

    path = manifest_path_for(policy_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != SCHEMA or payload.get("mode") != "distilled_locomotion":
        return None
    if int(payload.get("obs_dim", -1)) != 91 or int(payload.get("action_dim", -1)) != 26:
        return None
    if payload.get("phase_scale") != 1.0:
        return None
    if payload.get("phase_layout") != PHASE_LAYOUT:
        return None
    observation_contract = payload.get("observation_contract")
    if not isinstance(observation_contract, dict):
        return None
    non_stacked = observation_contract.get("non_stacked_obs_order", [])
    if not isinstance(non_stacked, list) or non_stacked.count("reference_progress") != 1:
        return None
    if non_stacked[-1] != "reference_progress":
        return None
    if int(observation_contract.get("command_dim", -1)) != 0:
        return None
    progress_cfg = (observation_contract.get("per_observation", {}) or {}).get("reference_progress", {})
    if int(progress_cfg.get("freq", 0)) != 50 or float(progress_cfg.get("scale", float("nan"))) != 1.0:
        return None
    phase_source = payload.get("phase_source")
    if not isinstance(phase_source, dict) or not str(phase_source.get("sha256", "")):
        return None
    if payload.get("deployment_strategy") != "learned_distilled_trajectory_residual":
        return None
    return payload
