from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def make_sensor_sample_points(sensor_point, average, radius, n):
    return legacy_attr("sensor_sampling", "make_sensor_sample_points")(sensor_point, average, radius, n)


def sensor_average_requested_points(average, n):
    return legacy_attr("sensor_sampling", "sensor_average_requested_points")(average, n)


def sensor_effective_area(average, radius):
    return legacy_attr("sensor_sampling", "sensor_effective_area")(average, radius)
