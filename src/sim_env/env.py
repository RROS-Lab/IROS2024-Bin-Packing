import csv
import copy
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from src.sim_env.models.scene_xml import SceneXML
from src.sim_env.utils.conversion import quat_to_mujoco_quat
from src.sim_env.utils.container import Package
from src.sim_env.utils.score import (
    packing_score,
    is_outside_container,
    is_outside_paddle,
)


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


def create_scene(packages, data_filename, scene_num=1, visualize=False):
    current_scene = 0
    while current_scene < scene_num:
        impossible_position = False
        print("Current scene count: ", current_scene)
        pkg_num = np.random.randint(0, 8)
        stack_packages, dropped_package = generate_feasible_sample(
            packages, pkg_num, 0.4
        )
        try:
            scene = SceneXML(stack_packages, dropped_package, np.random.randint(70, 90))
        except ValueError as e:
            print(e)
            continue

        robot_action = scene.get_robot_action()
        end_robot_action = [
            [copy.deepcopy(robot_action[0][0]), 0],
            [copy.deepcopy(robot_action[0][0]), 0],
        ]
        end_robot_action[1][0][0] = 0.39

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
        pkg_name = scene.record_data["package_name"]

        drop_pkg_grid = []
        stow_pkg_grid = []

        if visualize:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                # Close the viewer automatically after 30 wall-seconds.
                start = time.time()
                viewer.cam.lookat[0] = 0.2
                viewer.cam.lookat[1] = 0.15
                viewer.cam.lookat[2] = 0.2
                viewer.cam.distance = 2
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -40

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

                    if (
                        current_time >= translation_start_time
                        and current_time < translation_back_start_time - 2
                    ):
                        pos_x = robot_action[0][0][0] - 0.02 * (
                            current_time - translation_start_time
                        )
                        d.mocap_pos = [
                            pos_x,
                            robot_action[0][0][1],
                            robot_action[0][0][2],
                        ]

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
                        if force_data[-1] >= 10:
                            translation_back_start_time = current_time + 2
                            translation_back_end_time = current_time + 6
                            robot_action[1][0][0] = d.site_xpos[0][0]

                    mujoco.mj_step(m, d)

                    for pkg in stack_packages:
                        pkg_grid = []
                        for i in range(4 * 3 * 2):
                            pkg_grid.append(d.body(f"{pkg.name}_{i}").xpos)

                        pkg_bounding_box = np.array(pkg_grid)
                        if is_outside_container(
                            pkg_bounding_box, [0.4, 0.32, 0.25], True
                        ):
                            impossible_position = True
                            break

                    drop_pkg = []
                    for i in range(4 * 3 * 2):
                        drop_pkg.append(d.body(f"{pkg_name}_{i}").xpos)

                    if is_outside_container(
                        np.array(drop_pkg), [0.4, 0.32, 0.25]
                    ) or is_outside_paddle(
                        np.array(drop_pkg), np.array(d.site_xpos[0])
                    ):
                        impossible_position = True

                    if impossible_position:
                        break

                    viewer.sync()

                    time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            start = time.time()
            while time.time() - start < translation_back_end_time + 2:
                step_start = time.time()
                current_time = step_start - start

                if current_time >= rotation_start_time and not dropping_recorded:
                    for i in range(4 * 3 * 2):
                        drop_pkg_grid.append(deepcopy(d.body(f"{pkg_name}_{i}").xpos))
                    dropping_recorded = True

                d.mocap_quat = parse_robot_action(
                    rotation_start_time,
                    rotation_end_time,
                    current_time,
                    robot_action,
                    rotation=True,
                )
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
                    if force_data[-1] >= 10:
                        translation_back_start_time = current_time + 2
                        translation_back_end_time = current_time + 6
                        robot_action[1][0][0] = d.site_xpos[0][0]

                mujoco.mj_step(m, d)

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
                    break

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        if impossible_position:
            print("Package place impossible position")
            continue

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

        record = True
        for warning in d.warning:
            if warning.number > 0:
                record = False
                break

        if not record:
            continue

        packing_scores = packing_score(
            scene.record_data["bin_state_x"],
            scene.record_data["bin_state_theta"],
            scene.record_data["bin_state_z"],
            scene.record_data["bin_state_thickness"],
            stow_pkg_body_string,
            [0.29, 0.21, scene.record_data["package_thickness"]],
        )

        dropping_scores = packing_score(
            scene.record_data["bin_state_x"],
            scene.record_data["bin_state_theta"],
            scene.record_data["bin_state_z"],
            scene.record_data["bin_state_thickness"],
            drop_pkg_body_string,
            [0.29, 0.21, scene.record_data["package_thickness"]],
        )

        data_row = [
            scene.record_data["suction_x"],
            scene.record_data["suction_theta"],
            scene.record_data["paddle_x"],
            scene.record_data["paddle_z"],
            scene.record_data["paddle_theta"],
            scene.record_data["bin_state_x"],
            scene.record_data["bin_state_theta"],
            scene.record_data["bin_state_z"],
            scene.record_data["bin_state_thickness"],
            scene.record_data["package_thickness"],
            scene.record_data["package_mass"],
            scene.record_data["package_stiffness"],
            scene.record_data["ideal_bounding_box"],
            drop_bounding_box,
            stow_bounding_box,
            stow_pkg_body_string,
            dropping_scores[-1],
            packing_scores[-1],
            np.array2string(np.array(dropping_scores), precision=4, separator=","),
            np.array2string(np.array(packing_scores), precision=4, separator=","),
            xml,
            robot_action,
        ]

        current_scene += 1

        with lock:
            with open(data_filename, "a") as f:
                writer = csv.writer(f)
                writer.writerow(data_row)

    print("Thread finished")


def generate_packages(csv_filename):
    package_list = []

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
        package_list.append(new_package)

    return package_list


def generate_feasible_sample(
    packages: list, num_packages: int, max_container_width: np.float32
) -> list:

    in_bin_packages = []

    sampling_indices = np.arange(0, 18, 1)
    np.random.shuffle(sampling_indices)
    dropped_package_id = np.random.randint(1, 18)
    dropped_package = packages[sampling_indices[dropped_package_id]]
    sampling_indices = np.delete(sampling_indices, dropped_package_id)

    current_package_width = 0.0
    count = 0
    while len(in_bin_packages) < num_packages:
        current_package_id = sampling_indices[count]
        current_package = packages[current_package_id]
        # print("In Bin Package ID: ", current_package.package_id)
        current_package_width += current_package.package_size[2] + 0.005

        if current_package_width >= (
            max_container_width - dropped_package.package_size[2]
        ):
            break
        in_bin_packages.append(current_package)
        count += 1

    return [in_bin_packages, dropped_package]


if __name__ == "__main__":
    workspace_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    package_filename = os.path.join(
        workspace_folder, "sim_env", "assets", "package_definitions.csv"
    )
    packages = generate_packages(package_filename)

    num_threads = 12
    num_scenes = 400

    data_folder = os.path.join(workspace_folder, "sim_env", "data", "sim_exp")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data_filename = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    data_filename = os.path.join(data_folder, data_filename)

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
                "package_thickness",
                "package_mass",
                "package_stiffness",
                "ideal_bounding_box",
                "place_bounding_box",
                "stow_bounding_box",
                "package_positions",
                "dropping_score",
                "packing_score",
                "dropping_scores",
                "packing_scores",
                "xml",
                "robot_action",
            ]
        )
    
    create_scene(
        packages,
        data_filename,
        scene_num=num_scenes,
        visualize=False,
    )
    exit(0)

    pool = ThreadPoolExecutor(max_workers=num_threads)

    for i in range(num_threads):
        pool.submit(
            create_scene,
            packages,
            data_filename,
            scene_num=num_scenes,
            visualize=False,
        )

    pool.shutdown(wait=True)
