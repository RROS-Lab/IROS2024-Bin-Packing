import json

import numpy as np


def is_outside_container(package_body_array, container_size, height_check=False):
    min_x, max_x = np.min(package_body_array[:, 0]), np.max(package_body_array[:, 0])
    min_y, max_y = np.min(package_body_array[:, 1]), np.max(package_body_array[:, 1])
    min_z, max_z = np.min(package_body_array[:, 2]), np.max(package_body_array[:, 2])

    min_container_x, max_container_x = 0, container_size[0]
    min_container_y, max_container_y = 0, container_size[1]
    min_container_z, max_container_z = 0, container_size[2]

    outside_x = (max_container_x < min_x) or (max_x < min_container_x)
    outside_y = (max_container_y < min_y) or (max_y < min_container_y)
    outside_z = (max_container_z < min_z) or (max_z < min_container_z)

    return (
        outside_x or outside_y
        if not height_check
        else outside_x or outside_y or outside_z
    )


def is_outside_paddle(package_body_array, paddle_pos):
    min_x = np.min(package_body_array[:, 0])
    max_z = np.max(package_body_array[:, 2])

    outside_x = paddle_pos[0] < min_x and max_z > paddle_pos[2]

    return outside_x


def packing_score(
    bin_state_x: np.float32,
    bin_state_theta: np.float32,
    bin_state_z: np.float32,
    bin_state_thickness: np.float32,
    package_positions: str,
    package_size=[0.29, 0.21, 0.02],
    container_size=[0.4, 0.32, 0.25],
) -> np.float32:
    package_body_array = np.array(json.loads(package_positions))

    if is_outside_container(package_body_array, container_size):
        print("outside container")
        return [0.0]

    p_bottom = package_body_array[[19, 21, 23], :]
    p_top = package_body_array[[1, 3, 5], :]

    x1 = np.average(p_bottom[:, 0])
    y1 = np.average(p_bottom[:, 2])
    x2 = np.average(p_top[:, 0])
    y2 = np.average(p_top[:, 2])

    angle = np.arctan((y2 - y1) / (x2 - x1)) if x2 != x1 else -np.pi / 2

    packing_theta = -(np.pi / 2 + angle) if angle < 0 else (np.pi / 2 - angle)

    angle_score = (
        1 / (1 + np.abs(packing_theta - bin_state_theta) / np.abs(bin_state_theta))
        if bin_state_theta != 0
        else 1 / (1 + np.abs(packing_theta - bin_state_theta))
    )
    x_score = (
        1 / (1 + np.abs(x1 - bin_state_x - package_size[2]) / bin_state_x)
        if bin_state_x != 0
        else 1 / (1 + np.abs(x1 - package_size[2]))
    )
    thickness_score = 1 / (
        1
        + np.abs(bin_state_thickness + package_size[2] - x1)
        / (bin_state_thickness + package_size[2])
    )
    z_score = 1 / (1 + np.abs(bin_state_z - (y2 - y1) / 2) / bin_state_z)

    packing_score = (angle_score + x_score + thickness_score + z_score) / 4

    return [angle_score, x_score, thickness_score, z_score, packing_score]
