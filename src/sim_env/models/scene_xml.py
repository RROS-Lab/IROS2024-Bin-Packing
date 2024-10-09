import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.sim_env.models.container_xml import ContainerXML
from src.sim_env.models.package_xml import PackageXML
from src.sim_env.utils.collision import Line, Point
from src.sim_env.utils.container import Package
from src.sim_env.utils.conversion import convert_array_to_string, quat_to_mujoco_quat

WORKING_FOLDER = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class SceneXML:
    def __init__(
        self, in_bin_packages: list, drop_package: Package, stack_angle=85
    ) -> ET.ElementTree:
        self._tree = ET.parse(os.path.join(WORKING_FOLDER, "assets", "base_model.xml"))

        self._container_xml = ContainerXML()

        default = self._tree.find("default")
        for idx in range(len(self._container_xml.default_element)):
            default.insert(idx, self._container_xml.default_element[idx])

        worldbody = self._tree.find("worldbody")
        for idx in range(len(self._container_xml.worldbody_element)):
            worldbody.insert(idx, self._container_xml.worldbody_element[idx])

        self.record_data = {}

        stack_package_xml = self.__generate_packages_stack_csv__(
            in_bin_packages, stack_angle
        )
        for package_xml in stack_package_xml:
            worldbody.insert(0, package_xml.element)

        drop_package_xml = self.__generate_drop_package_csv__(drop_package)
        worldbody.insert(0, drop_package_xml.element)

        self._robot_action = self.__generate_robot_action__(drop_package.package_size)

        mocap = self._tree.findall(".//body[@name='mocap']")[0]
        gripper = self._tree.findall(".//body[@name='gripper_base']")[0]

        mocap_trans = np.eye(4)
        mocap_rotation = R.from_rotvec(
            np.deg2rad(self._robot_action[0][1]) * np.array([0, 1, 0])
        )
        mocap_trans[:3, :3] = mocap_rotation.as_matrix()
        mocap_trans[:3, 3] = self._robot_action[0][0]

        mocap.set("pos", convert_array_to_string(mocap_trans[:3, 3]))
        mocap.set(
            "quat",
            convert_array_to_string(quat_to_mujoco_quat(mocap_rotation.as_quat())),
        )

        gripper_trans = np.array(
            mocap_trans
            @ np.array(
                [[0, 0, 1, -0.005], [-1, 0, 0, 0], [0, -1, 0, 0.31], [0, 0, 0, 1]]
            )
        )

        gripper_rotation = R.from_matrix(gripper_trans[:3, :3])
        gripper.set("pos", convert_array_to_string(gripper_trans[:3, 3]))
        gripper.set(
            "quat",
            convert_array_to_string(quat_to_mujoco_quat(gripper_rotation.as_quat())),
        )

    def __generate_packages_stack_csv__(
        self,
        in_bin_packages: list,
        stack_angle: float,
        package_size=[0.29, 0.21, 0.02],
    ):

        stack_package_xml = []

        stack_rad = np.deg2rad(stack_angle)

        package_num = len(in_bin_packages)
        overall_thickness = sum(
            [package.package_size[2] for package in in_bin_packages]
        )

        pos_x = package_size[0] / 2 * np.cos(stack_rad)

        for idx in range(package_num):
            current_package = in_bin_packages[idx]
            thickness = current_package.package_size[2]

            pos_x += thickness / 2 * np.sin(stack_rad)
            pos_y = self._container_xml._y_size / 2
            pos_z = package_size[0] / 2 * np.sin(stack_rad) + thickness / 2 * np.cos(
                stack_rad
            )

            mass = current_package.mass
            young = current_package.youngs_mod

            name = current_package.name

            package_xml = PackageXML(
                name,
                [package_size[0], package_size[1], thickness],
                [pos_x, pos_y, pos_z],
                stack_angle,
                mass,
                young,
            )

            stack_package_xml.append(package_xml)

            pos_x += thickness / 2 * np.sin(stack_rad) + 0.005

        self.record_data["bin_state_x"] = (
            pos_x + package_size[0] / 2 * np.cos(stack_rad) if package_num > 0 else 0
        )
        self.record_data["bin_state_theta"] = (
            -np.deg2rad(90 - stack_angle) if package_num > 0 else 0
        )
        self.record_data["bin_state_z"] = (
            package_size[0] / 2 * np.sin(stack_rad)
            if package_num > 0
            else package_size[0] / 2
        )
        self.record_data["bin_state_thickness"] = overall_thickness

        return stack_package_xml

    def __generate_drop_package_csv__(
        self, drop_package, package_size=[0.29, 0.21, 0.02]
    ):
        thickness = drop_package.package_size[2]

        suction_theta = np.random.uniform(-20, 0)
        angle = suction_theta + 90

        suction_x = np.random.uniform(-0.02, 0.02)

        # Add gap for the edge dropping
        collision = True
        count = 0
        while collision:
            pos_x = self.record_data["bin_state_x"] + thickness + suction_x

            if pos_x > -package_size[0] / 2 * np.sin(
                np.deg2rad(suction_theta)
            ) and pos_x < self._container_xml._x_size + package_size[0] / 2 * np.sin(
                np.deg2rad(suction_theta)
            ):
                collision = False

            count += 1
            if count > 1000 and collision:
                raise ValueError("Package collision not resolved after 1000 attempts.")

        pos_y = self._container_xml._y_size / 2
        pos_z = package_size[0] / 2 + self._container_xml._z_size

        self.pkg_lines = self.__generate_package_lines__(
            package_size, [pos_x, pos_z], suction_theta
        )

        mass = drop_package.mass
        name = drop_package.name

        stiffness = drop_package.stiffness
        youngs_mod = drop_package.youngs_mod

        self.record_data["package_name"] = name
        self.record_data["suction_x"] = suction_x
        self.record_data["suction_theta"] = np.deg2rad(suction_theta)
        self.record_data["package_thickness"] = thickness
        self.record_data["package_mass"] = mass
        self.record_data["package_stiffness"] = stiffness

        ideal_bbox = self.__generate_ideal_bounding_box__(
            [package_size[0], package_size[1], thickness],
            [
                self.record_data["bin_state_x"] + thickness / 2,
                self._container_xml._y_size / 2,
                package_size[0] / 2,
            ],
            np.pi / 2,
        )
        self.record_data["ideal_bounding_box"] = np.array2string(
            np.array(ideal_bbox), precision=4, separator=","
        )

        return PackageXML(
            name,
            [package_size[0], package_size[1], thickness],
            [pos_x, pos_y, pos_z],
            angle,
            mass,
            youngs_mod,
        )

    def __generate_ideal_bounding_box__(self, package_size, center_point, angle):
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
            np.array(center_point)
            + np.dot(R.from_euler("y", angle).as_matrix(), corner)
            for corner in corners
        ]
        return rotated_corners

    def __generate_robot_action__(self, package_size=[0.29, 0.21, 0.02]):
        collision = True
        container_line = Line(
            Point(self._container_xml._x_size, 0),
            Point(self._container_xml._x_size, self._container_xml._z_size),
        )
        pkg_lines = self.pkg_lines

        count = 0
        while collision:
            paddle_x = np.random.uniform(0, 0.03)
            paddle_z = np.random.uniform(-0.07, 0.05)
            paddle_theta = np.random.uniform(0, 20)

            place_x = (
                self.record_data["bin_state_x"]
                + self.record_data["package_thickness"]
                + paddle_x
            )

            paddle_x = (
                place_x
                - self.record_data["bin_state_x"]
                - self.record_data["package_thickness"]
            )
            place_y = self._container_xml._y_size / 2
            place_z = package_size[0] / 2 + paddle_z

            paddle_line = Line(
                Point(place_x, place_z),
                Point(
                    place_x + 0.31 * np.sin(np.deg2rad(paddle_theta)),
                    place_z + 0.31 * np.cos(np.deg2rad(paddle_theta)),
                ),
            )

            container_no_collision = not container_line.intersect(paddle_line)

            package_no_collision = True
            for pkg_line in pkg_lines:
                if pkg_line.intersect(paddle_line):
                    package_no_collision = False
                    break

            package_above_paddle = True
            for pkg_line in pkg_lines:
                if (
                    not pkg_line.above(paddle_line)
                    or pkg_line.p1.x > paddle_line.p2.x
                    or pkg_line.p2.x > paddle_line.p2.x
                ):
                    package_above_paddle = False
                    break

            place_outside = (
                place_x >= self._container_xml._x_size - self._container_xml._t_size
            )

            if (
                container_no_collision
                and package_no_collision
                and package_above_paddle
                and not place_outside
            ):
                collision = False

            count += 1
            if count > 1000 and collision:
                raise ValueError("Collision not resolved after 1000 attempts.")

        stow_theta = 0
        stow_x = min(
            (
                self.record_data["bin_state_x"]
                + self.record_data["package_thickness"] / 1.2
                if self.record_data["bin_state_thickness"] != 0
                else self.record_data["package_thickness"] + 0.03
            ),
            self._container_xml._x_size - self._container_xml._t_size,
        )
        stow_y = self._container_xml._y_size / 2
        stow_z = place_z

        place_action = [np.array([place_x, place_y, place_z]), paddle_theta]
        stow_action = [np.array([stow_x, stow_y, stow_z]), stow_theta]

        self.record_data["paddle_x"] = paddle_x
        self.record_data["paddle_z"] = paddle_z
        self.record_data["paddle_theta"] = np.deg2rad(paddle_theta)

        return [place_action, stow_action]

    def __generate_package_lines__(self, package_size, center_point, angle):
        pos_x, pos_z = center_point

        pkg_edge = Point(
            package_size[2] * np.cos(np.deg2rad(angle)),
            -package_size[2] * np.sin(np.deg2rad(angle)),
        )

        pkg_top = Point(
            pos_x + package_size[0] / 2 * np.sin(np.deg2rad(angle)),
            pos_z + package_size[0] / 2 * np.cos(np.deg2rad(angle)),
        )
        pkg_bottom = Point(
            pos_x - package_size[0] / 2 * np.sin(np.deg2rad(angle)),
            pos_z - package_size[0] / 2 * np.cos(np.deg2rad(angle)),
        )

        pkg_top_left = Point(pkg_top.x - pkg_edge.x, pkg_top.y - pkg_edge.y)
        pkg_top_right = Point(pkg_top.x + pkg_edge.x, pkg_top.y + pkg_edge.y)
        pkg_bottom_left = Point(pkg_bottom.x - pkg_edge.x, pkg_bottom.y - pkg_edge.y)
        pkg_bottom_right = Point(pkg_bottom.x + pkg_edge.x, pkg_bottom.y + pkg_edge.y)

        pkg_lines = [
            Line(pkg_top_left, pkg_top_right),
            Line(pkg_top_right, pkg_bottom_right),
            Line(pkg_bottom_right, pkg_bottom_left),
            Line(pkg_bottom_left, pkg_top_left),
        ]

        return pkg_lines

    def get_xml(self):
        return ET.tostring(self._tree.getroot(), encoding="unicode")

    def get_robot_action(self):
        return self._robot_action
