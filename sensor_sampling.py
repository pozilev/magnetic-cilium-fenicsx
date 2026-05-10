from __future__ import annotations

import numpy as np


def make_sensor_sample_points(sensor_point: np.ndarray, average: bool, radius: float, n: int) -> np.ndarray:
    if not average:
        return np.asarray([sensor_point], dtype=np.float64)
    if radius <= 0.0:
        raise RuntimeError("--sensor-average-radius must be positive when --sensor-average is used.")
    if n < 1:
        raise RuntimeError("--sensor-average-n must be >= 1.")
    if n == 1:
        return np.asarray([sensor_point], dtype=np.float64)

    offsets = np.linspace(-radius, radius, n)
    points = []
    for dx in offsets:
        for dy in offsets:
            if dx * dx + dy * dy <= radius * radius + 1e-30:
                points.append([sensor_point[0] + dx, sensor_point[1] + dy, sensor_point[2]])
    if not points:
        raise RuntimeError("Sensor averaging produced no sample points.")
    return np.asarray(points, dtype=np.float64)
