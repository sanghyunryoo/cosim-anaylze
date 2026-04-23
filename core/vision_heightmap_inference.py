from collections import deque

import numpy as np


try:
    import onnxruntime as ort
except Exception:
    ort = None


def _normalize_dim(value, fallback):
    try:
        dim = int(value)
        if dim > 0:
            return dim
    except Exception:
        pass
    return int(fallback)


class VisionHeightMapInferencer:
    def __init__(self, onnx_path: str):
        if ort is None:
            raise RuntimeError(
                "Vision height-map inference requires the 'onnxruntime' package. "
                "Install the packages from requirements.txt first."
            )

        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        inputs = list(self.session.get_inputs())
        if len(inputs) < 2:
            raise RuntimeError(
                "Vision height-map ONNX must expose two inputs: depth_history and projected_gravity."
            )

        self.depth_input = inputs[0]
        self.state_input = inputs[1]
        self.output = self.session.get_outputs()[0]

        depth_shape = list(getattr(self.depth_input, "shape", []) or [])
        state_shape = list(getattr(self.state_input, "shape", []) or [])
        self.history_len = _normalize_dim(depth_shape[1] if len(depth_shape) > 1 else None, 3)
        self.depth_height = _normalize_dim(depth_shape[2] if len(depth_shape) > 2 else None, 60)
        self.depth_width = _normalize_dim(depth_shape[3] if len(depth_shape) > 3 else None, 80)
        self.state_dim = _normalize_dim(state_shape[1] if len(state_shape) > 1 else None, 3)

        self.depth_history = deque(maxlen=self.history_len)
        gravity_history_len = max(1, self.state_dim // 3) if (self.state_dim % 3) == 0 else 1
        self.gravity_history = deque(maxlen=gravity_history_len)

    def reset(self):
        self.depth_history.clear()
        self.gravity_history.clear()

    def _prepare_state_vec(self, projected_gravity):
        gravity = np.asarray(projected_gravity, dtype=np.float32).reshape(-1)
        if gravity.size == 0:
            raise RuntimeError("Projected gravity input is empty.")

        self.gravity_history.append(gravity[:3] if gravity.size >= 3 else np.pad(gravity, (0, 3 - gravity.size)))
        if self.state_dim == 3:
            return np.asarray(self.gravity_history[-1], dtype=np.float32).reshape(1, 3)

        if (self.state_dim % 3) == 0:
            required = self.state_dim // 3
            chunks = list(self.gravity_history)
            if not chunks:
                chunks = [np.zeros(3, dtype=np.float32)]
            while len(chunks) < required:
                chunks.insert(0, np.asarray(chunks[0], dtype=np.float32))
            merged = np.concatenate(chunks[-required:], axis=0).astype(np.float32, copy=False)
            return merged.reshape(1, self.state_dim)

        vec = np.zeros((self.state_dim,), dtype=np.float32)
        count = min(self.state_dim, gravity.size)
        vec[:count] = gravity[:count]
        return vec.reshape(1, self.state_dim)

    def predict(self, depth_frame, projected_gravity):
        depth = np.asarray(depth_frame, dtype=np.float32)
        if depth.ndim != 2:
            raise RuntimeError("Depth frame must have shape [H, W].")
        if depth.shape != (self.depth_height, self.depth_width):
            raise RuntimeError(
                f"Depth frame shape mismatch. Expected {(self.depth_height, self.depth_width)}, got {tuple(depth.shape)}."
            )

        self.depth_history.append(depth / 255.0)
        if len(self.depth_history) < self.history_len:
            return None

        depth_history = np.stack(tuple(self.depth_history), axis=0).astype(np.float32, copy=False)
        state_vec = self._prepare_state_vec(projected_gravity)
        outputs = self.session.run(
            [self.output.name],
            {
                self.depth_input.name: depth_history.reshape(1, self.history_len, self.depth_height, self.depth_width),
                self.state_input.name: state_vec,
            },
        )
        height_map = np.asarray(outputs[0], dtype=np.float32)
        if height_map.ndim == 4:
            height_map = height_map[0, 0]
        elif height_map.ndim == 3:
            height_map = height_map[0]
        else:
            raise RuntimeError(f"Unexpected height-map output shape: {tuple(height_map.shape)}")
        return height_map.astype(np.float32, copy=False)
