import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.sim_env.utils.conversion import quat_to_mujoco_quat


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


def create_scene():

    rotation_start_time = 3
    rotation_end_time = 5
    translation_start_time = 5
    translation_back_start_time = 609
    translation_back_end_time = 613

    robot_action = [
        [np.array([0.28028859, 0.16, 0.08531272]), 1.8864241234218981],
        [np.array([0.25028859, 0.16, 0.08531272]), 0.0],
    ]
    force_data = []
    m = mujoco.MjModel.from_xml_path("./model.xml")
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
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
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        viewer.close()


if __name__ == "__main__":
    create_scene()
