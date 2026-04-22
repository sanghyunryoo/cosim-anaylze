import json
import os
from collections import deque
from datetime import datetime

import numpy as np


class HeightMapDatasetWriter:
    def __init__(self, env_id: str, output_root: str, height_map_shape, depth_shape=None, depth_scale: int = 1):
        self.env_id = str(env_id)
        self.height_map_shape = tuple(int(v) for v in height_map_shape)
        self.depth_shape = tuple(int(v) for v in (depth_shape or (0, 0)))
        self.depth_scale = max(1, int(depth_scale))
        self.depth_history = deque(maxlen=3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_root, "height_map_supervision", timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        self._depth_histories = []
        self._projected_gravities = []
        self._height_maps = []
        self._sample_count = 0

        metadata = {
            "env_id": self.env_id,
            "format": "depth_history_to_height_map_v1",
            "depth_history_length": 3,
            "depth_shape": list(self.depth_shape),
            "depth_scale": self.depth_scale,
            "height_map_shape": list(self.height_map_shape),
        }
        with open(os.path.join(self.run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def add_sample(self, depth_frame, projected_gravity, height_map):
        depth_frame = np.asarray(depth_frame, dtype=np.uint8)
        projected_gravity = np.asarray(projected_gravity, dtype=np.float32).reshape(3)
        height_map = np.asarray(height_map, dtype=np.float32).reshape(self.height_map_shape)

        self.depth_history.append(depth_frame)
        if len(self.depth_history) < 3:
            return False

        self._depth_histories.append(np.stack(tuple(self.depth_history), axis=0))
        self._projected_gravities.append(projected_gravity)
        self._height_maps.append(height_map)
        self._sample_count += 1
        return True

    def flush(self):
        if not self._depth_histories:
            return

        output_path = os.path.join(self.run_dir, "dataset.npz")
        np.savez_compressed(
            output_path,
            depth_history=np.asarray(self._depth_histories, dtype=np.uint8),
            projected_gravity=np.asarray(self._projected_gravities, dtype=np.float32),
            height_map=np.asarray(self._height_maps, dtype=np.float32),
        )
        self._depth_histories.clear()
        self._projected_gravities.clear()
        self._height_maps.clear()

    def close(self):
        self.flush()

    @property
    def sample_count(self):
        return int(self._sample_count)
