import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.sim_env.models.container_xml import ContainerXML
from src.sim_env.models.package_xml import PackageXML
from src.sim_env.utils.container import Package
from src.sim_env.utils.conversion import convert_array_to_string, quat_to_mujoco_quat


class SpecificSceneXML:
    def __init__(
        self,
        stack_packages: list[Package],
        stack_angle: np.float32,
        drop_package: Package,
        drop_pos: np.ndarray,
        drop_angle: np.float32,
        mocap_pos: np.ndarray = np.array([0.39, 0.16, 0.1]),
        mocap_angle: np.float32 = 0.0,
    ) -> ET.ElementTree:
        self._tree = ET.parse("./assets/base_model.xml")
        self._container_xml = ContainerXML()

        default = self._tree.find("default")
        for idx in range(len(self._container_xml.default_element)):
            default.insert(idx, self._container_xml.default_element[idx])

        worldbody = self._tree.find("worldbody")
        for idx in range(len(self._container_xml.worldbody_element)):
            worldbody.insert(idx, self._container_xml.worldbody_element[idx])

        stack_package_xml = self.__generate_packages_stack_csv__(
            stack_packages, stack_angle
        )
        for pkg_xml in stack_package_xml:
            worldbody.insert(0, pkg_xml.element)

        drop_package_xml = self.__generate_drop_package_csv__(
            drop_package, drop_pos, drop_angle
        )
        worldbody.insert(0, drop_package_xml.element)

        self.__generate_mocap_control__(mocap_pos, mocap_angle)

    def __generate_mocap_control__(self, mocap_pos=[0.39, 0.16, 0.1], mocap_angle = 0.0):
        mocap = self._tree.findall(".//body[@name='mocap']")[0]
        gripper = self._tree.findall(".//body[@name='gripper_base']")[0]

        mocap_trans = np.eye(4)
        mocap_rotation = R.from_rotvec(
            np.deg2rad(mocap_angle) * np.array([0, 1, 0])
        )
        mocap_trans[:3, :3] = mocap_rotation.as_matrix()
        mocap_trans[:3, 3] = mocap_pos

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
    ):

        stack_package_xml = []

        package_num = len(in_bin_packages)
        stack_rad = np.deg2rad(stack_angle) if package_num > 0 else np.pi / 2

        pos_x = (
            in_bin_packages[0].package_size[0] / 2 * np.cos(stack_rad)
            if package_num > 0
            else 0
        )

        pos_x = 0
        for pkg in in_bin_packages:
            temp_stack_rad = stack_rad if in_bin_packages.index(pkg) == len(in_bin_packages) - 1 else np.pi / 2

            x, y, z = np.array(pkg.package_size) 

            pos_x += z / 2 * np.sin(temp_stack_rad)
            pos_y = self._container_xml._y_size / 2
            pos_z = x / 2 * np.sin(temp_stack_rad) + z / 2 * np.cos(temp_stack_rad)

            if temp_stack_rad == stack_rad:
                pos_x += pkg.package_size[0] / 2 * np.cos(stack_rad)

            mass = pkg.mass
            young = pkg.youngs_mod
            name = pkg.name

            package_xml = PackageXML(
                name,
                [x, y, z],
                [pos_x, pos_y, pos_z],
                np.rad2deg(temp_stack_rad),
                mass,
                young,
            )

            stack_package_xml.append(package_xml)

            pos_x += z / 2 * np.sin(temp_stack_rad) + 0.005

        return stack_package_xml

    def __generate_drop_package_csv__(self, drop_package, pkg_pos, pkg_angle):
        angle = pkg_angle + 90

        mass = drop_package.mass
        name = drop_package.name
        youngs_mod = drop_package.youngs_mod

        return PackageXML(
            name,
            np.array(drop_package.package_size),
            pkg_pos,
            angle,
            mass,
            youngs_mod,
        )

    def get_xml(self):
        return ET.tostring(self._tree.getroot(), encoding="unicode")
