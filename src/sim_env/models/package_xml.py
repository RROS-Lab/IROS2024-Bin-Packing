import numpy as np
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation as R

from src.sim_env.utils.conversion import (
    convert_array_to_string,
    quat_to_mujoco_quat,
)


class PackageXML(object):
    def __init__(
        self,
        name="A",
        size=[0.29, 0.21, 0.04],
        pos=[0.2, 0.15, 0.148],
        angle=90,
        mass=1,
        young=20000,
    ) -> None:
        self._edge_damping = 1
        self._count = [4, 3, 2]

        self._size = size
        self._spacing = np.divide(size, [x - 1 for x in self._count])
        self._radius = np.min([np.min(self._spacing) / 2, 0.001])
        self._young = young

        self._name = name
        self._pos = pos

        quat = R.from_rotvec(np.deg2rad(angle) * np.array([0, 1, 0])).as_quat()
        self._quat = quat_to_mujoco_quat(quat)

        self._mass = mass

        self._rgba = [np.random.uniform(0, 1) for i in range(3)] + [1]

        self.element = ET.Element("flexcomp")

        self.__create_element_node__()

    def __create_element_node__(self):
        self.element.attrib = {
            "name": self._name,
            "type": "grid",
            "count": convert_array_to_string(self._count),
            "spacing": convert_array_to_string(self._spacing),
            "pos": convert_array_to_string(self._pos),
            "quat": convert_array_to_string(self._quat),
            "radius": str(self._radius),
            "rgba": convert_array_to_string(self._rgba),
            "dim": "3",
            "mass": str(self._mass),
        }

        ET.SubElement(
            self.element,
            "contact",
            {"condim": "3", "selfcollide": "none", "friction": "1 0.005 0.0005"},
        )

        ET.SubElement(
            self.element,
            "edge",
            {"damping": str(self._edge_damping), "equality": "true"},
        )

        plugin = ET.SubElement(
            self.element, "plugin", {"plugin": "mujoco.elasticity.solid"}
        )

        ET.SubElement(plugin, "config", {"key": "poisson", "value": "0.3"})
        ET.SubElement(plugin, "config", {"key": "young", "value": str(self._young)})
