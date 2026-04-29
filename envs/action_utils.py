import numpy as np


def normalize_action_clippings(config, action_dim):
    raw = config.get("action_clippings", [])
    mins = np.full(action_dim, -np.inf, dtype=np.float64)
    maxs = np.full(action_dim, np.inf, dtype=np.float64)

    if not isinstance(raw, (list, tuple)) or len(raw) != action_dim:
        return mins, maxs

    for i, item in enumerate(raw):
        if not item:
            continue
        if isinstance(item, dict):
            if not bool(item.get("enabled", False)):
                continue
            lo = item.get("min", mins[i])
            hi = item.get("max", maxs[i])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lo, hi = item[0], item[1]
        else:
            continue

        try:
            lo = float(lo)
            hi = float(hi)
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        mins[i] = lo
        maxs[i] = hi

    return mins, maxs


def scale_and_clip_action(action, scaler, clip_min, clip_max):
    return np.clip(action * scaler, clip_min, clip_max)
