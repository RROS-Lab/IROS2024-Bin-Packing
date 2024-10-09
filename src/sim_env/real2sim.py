import csv
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.sim_env.models.specific_scene_xml import SpecificSceneXML
from src.sim_env.utils.container import Package
from src.sim_env.utils.conversion import quat_to_mujoco_quat
from src.sim_env.utils.score import packing_score, is_outside_container, is_outside_paddle

lock = threading.Lock()


def parse_robot_action(
    start, end, current_time, robot_action, rotation=True, reverse=False
):
    total_time = end - start
    t = 0
    if current_time < start:
        t = 0
    elif current_time > end:
        t = 1
    else:
        t = (current_time - start) / total_time

    action = 0.0
    if rotation:
        theta = (1 - t) * robot_action[0][1] + t * robot_action[1][1]
        action = quat_to_mujoco_quat(
            (R.from_rotvec(np.deg2rad(theta) * np.array([0, 1, 0]))).as_quat()
        )
    else:
        if reverse:
            action = (1 - t) * robot_action[1][0] + t * robot_action[0][0]
        else:
            action = (1 - t) * robot_action[0][0] + t * robot_action[1][0]

    return action


def get_ideal_bouding_box(package_size, center_point, angle):
    width, height, depth = package_size
    half_width, half_height, half_depth = width / 2, height / 2, depth / 2

    corners = np.array(
        [
            [-half_width, -half_height, -half_depth],
            [half_width, -half_height, -half_depth],
            [half_width, half_height, -half_depth],
            [-half_width, half_height, -half_depth],
            [-half_width, -half_height, half_depth],
            [half_width, -half_height, half_depth],
            [half_width, half_height, half_depth],
            [-half_width, half_height, half_depth],
        ]
    )

    rotated_corners = [
        np.array(center_point) + np.dot(R.from_euler("y", angle).as_matrix(), corner)
        for corner in corners
    ]

    return np.array2string(np.array(rotated_corners), precision=4, separator=",")


def generate_packages(csv_filename):
    package_dict = {}

    package_df = pd.read_csv(csv_filename)
    package_names = package_df["name"].to_list()

    package_ids = package_df["package_id"].to_numpy(dtype=np.int8)
    package_weights = package_df["weight"].to_numpy(dtype=np.float32) / 1000.0
    package_thickness = package_df["thickness"].to_numpy(dtype=np.float32) / 1000.0
    package_stiffness = package_df["stiffness"].to_numpy(dtype=np.float32)

    for package_indx in range(len(package_names)):
        package_size = [0.29, 0.21, package_thickness[package_indx]]
        new_package = Package(
            name=package_names[package_indx],
            package_id=package_ids[package_indx],
            package_size=package_size,
            mass=package_weights[package_indx],
            stiffness=package_stiffness[package_indx],
        )
        package_dict[package_ids[package_indx]] = new_package

    return package_dict


def create_scene(
    stack_packages,
    dropped_package,
    suction_robot: np.ndarray,
    paddle_robot: np.ndarray,
    bin_state: np.ndarray,
    exp_scores: np.ndarray,
):
    """
    bin_state: [x, theta, z, thickness]
    """
    suction_x, suction_theta = suction_robot
    paddle_x, paddle_theta, paddle_z = paddle_robot
    bin_state_x, bin_state_theta, bin_state_z, bin_state_thickness = bin_state
    thickness = dropped_package.package_size[2]

    scene = SpecificSceneXML(
        stack_packages,
        90 + bin_state_theta,
        dropped_package,
        [bin_state_x + suction_x, 0.16, 0.5],
        suction_theta,
        [
            bin_state_x + thickness + paddle_x,
            0.16,
            0.096 + paddle_z,
        ],
        paddle_theta,
    )

    robot_action = [
        [
            np.array(
                [
                    bin_state_x + thickness + paddle_x,
                    0.16,
                    0.096 + paddle_z,
                ]
            ),
            paddle_theta,
        ],
        [np.array([0.39, 0.16, 0.096 + paddle_z]), 0.0],
    ]

    rotation_start_time = 3
    rotation_end_time = 5
    translation_start_time = 5
    translation_back_start_time = 609
    translation_back_end_time = 613
    force_data = []

    xml = scene.get_xml()

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    dropping_recorded = False
    pkg_name = dropped_package.name

    drop_pkg_grid = []
    stow_pkg_grid = []

    start = time.time()
    while time.time() - start < translation_back_end_time + 2:
        step_start = time.time()
        current_time = step_start - start

        if current_time >= rotation_start_time and not dropping_recorded:
            for i in range(4 * 3 * 2):
                drop_pkg_grid.append(d.body(f"{pkg_name}_{i}").xpos)
            dropping_recorded = True

        d.mocap_quat = parse_robot_action(
            rotation_start_time,
            rotation_end_time,
            current_time,
            robot_action,
            rotation=True,
        )
        if current_time < translation_start_time:
            d.mocap_pos = robot_action[0][0]

        if (
            current_time >= translation_start_time
            and current_time < translation_back_start_time - 2
        ):
            pos_x = robot_action[0][0][0] - 0.02 * (
                current_time - translation_start_time
            )
            d.mocap_pos = [pos_x, robot_action[0][0][1], robot_action[0][0][2]]

        if current_time >= translation_back_start_time:
            d.mocap_pos = parse_robot_action(
                translation_back_start_time,
                translation_back_end_time,
                current_time,
                robot_action,
                rotation=False,
                reverse=True,
            )

        if (
            current_time >= translation_start_time
            and current_time < translation_back_start_time - 2
        ):
            force_data.append(d.sensordata[0])
            if force_data[-1] >= 6:
                translation_back_start_time = current_time + 2
                translation_back_end_time = current_time + 6
                robot_action[0][0][0] = d.site_xpos[0][0] + 0.03
                robot_action[1][0][0] = d.site_xpos[0][0]

        mujoco.mj_step(m, d)

        impossible_position = False

        for pkg in stack_packages:
            pkg_grid = []
            for i in range(4 * 3 * 2):
                pkg_grid.append(d.body(f"{pkg.name}_{i}").xpos)

            pkg_bounding_box = np.array(pkg_grid)
            if is_outside_container(pkg_bounding_box, [0.4, 0.32, 0.25], True):
                impossible_position = True
                break

        drop_pkg = []
        for i in range(4 * 3 * 2):
            drop_pkg.append(d.body(f"{pkg_name}_{i}").xpos)

        if is_outside_container(
            np.array(drop_pkg), [0.4, 0.32, 0.25]
        ) or is_outside_paddle(np.array(drop_pkg), np.array(d.site_xpos[0])):
            impossible_position = True

        if impossible_position:
            return []

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    for i in range(4 * 3 * 2):
        stow_pkg_grid.append(deepcopy(d.body(f"{pkg_name}_{i}").xpos))

    drop_bounding_box = np.array2string(
        np.array(
            [
                drop_pkg_grid[18],
                drop_pkg_grid[19],
                drop_pkg_grid[23],
                drop_pkg_grid[22],
                drop_pkg_grid[0],
                drop_pkg_grid[1],
                drop_pkg_grid[5],
                drop_pkg_grid[4],
            ]
        ),
        precision=4,
        separator=",",
    )
    stow_bounding_box = np.array2string(
        np.array(
            [
                stow_pkg_grid[18],
                stow_pkg_grid[19],
                stow_pkg_grid[23],
                stow_pkg_grid[22],
                stow_pkg_grid[0],
                stow_pkg_grid[1],
                stow_pkg_grid[5],
                stow_pkg_grid[4],
            ]
        ),
        precision=4,
        separator=",",
    )

    drop_pkg_body_string = np.array2string(
        np.array(drop_pkg_grid), precision=4, separator=","
    )
    stow_pkg_body_string = np.array2string(
        np.array(stow_pkg_grid), precision=4, separator=","
    )

    packing_scores = packing_score(
        bin_state_x,
        bin_state_theta,
        bin_state_z,
        bin_state_thickness,
        stow_pkg_body_string,
        dropped_package.package_size,
    )

    dropping_scores = packing_score(
        bin_state_x,
        bin_state_theta,
        bin_state_z,
        bin_state_thickness,
        drop_pkg_body_string,
        dropped_package.package_size,
    )

    data_row = [
        suction_x,
        suction_theta,
        paddle_x,
        paddle_z,
        paddle_theta,
        bin_state_x,
        bin_state_theta,
        bin_state_z,
        bin_state_thickness,
        dropped_package.package_size[2],
        dropped_package.mass,
        dropped_package.stiffness,
        get_ideal_bouding_box(
            dropped_package.package_size,
            [bin_state_x + thickness / 2, 0.16, dropped_package.package_size[0] / 2],
            np.pi / 2,
        ),
        drop_bounding_box,
        stow_bounding_box,
        stow_pkg_body_string,
        dropping_scores[-1],
        packing_scores[-1],
        exp_scores[0],
        exp_scores[1],
        xml,
        robot_action,
    ]

    return data_row


def parse_and_execute(row, package_dict, data_filename):
    if row["in_bin_pkg"] == "[]":
        stack_packages = []
    else:
        stack_packages_str = row["in_bin_pkg"][1:-1].split(", ")
        stack_packages = [
            package_dict[int(package_id)] for package_id in stack_packages_str
        ]
    dropped_package = package_dict[row["drop_pkg"]]

    overall_thickness = sum([package.package_size[2] for package in stack_packages])

    suction_robot = [row["suction_x"], np.rad2deg(row["suction_theta"])]
    paddle_robot = [
        row["paddle_x"],
        np.rad2deg(row["paddle_theta"]),
        row["paddle_z"],
    ]
    bin_state = [
        row["bin_state_x"],
        np.rad2deg(row["bin_state_theta"]),
        row["bin_state_z"],
        overall_thickness,
    ]

    exp_scores = [row["dropping_score"], row["packing_score"]]

    data_row = create_scene(
        stack_packages,
        dropped_package,
        suction_robot,
        paddle_robot,
        bin_state,
        exp_scores,
    )

    if len(data_row) == 0:
        return

    with lock:
        with open(data_filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(data_row)

    return


def main():
    workspace_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    package_filename = os.path.join(workspace_folder, "sim_env", "assets", "package_definitions.csv")
    package_dict = generate_packages(package_filename)

    exp_filename = os.path.join(workspace_folder, "data", "physic_exp.csv")
    exp_df = pd.read_csv(exp_filename)
    exp_df.reset_index()

    data_filename = sys.argv[1] if len(sys.argv) > 1 else "data_exp.csv"

    with open(data_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "suction_x",
                "suction_theta",
                "paddle_x",
                "paddle_z",
                "paddle_theta",
                "bin_state_x",
                "bin_state_theta",
                "bin_state_z",
                "bin_state_thickness",
                "dropped_package_thickness",
                "dropped_package_mass",
                "dropped_package_stiffness",
                "ideal_bounding_box",
                "drop_bounding_box",
                "stow_bounding_box",
                "stow_pkg_body_string",
                "dropping_score",
                "packing_score",
                "exp_dropping_score",
                "exp_packing_score",
                "xml",
                "robot_action",
            ]
        )

    for _, row in exp_df.iterrows():
        pool = ThreadPoolExecutor(max_workers=8)
        for _ in range(8):
            pool.submit(parse_and_execute, row, package_dict, data_filename)

        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
