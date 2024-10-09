from collections.abc import Iterable

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from scipy.spatial.transform import Rotation as R


def convert_array_to_string(array=[], delim=" ") -> str:
    """
    Convert the input array to string splitted by given delimiter
    """
    return delim.join([str(size) for size in array])


def convert_single_to_array(input) -> list:
    """
    Convert the single input variable to [input, input], do nothing if input=[low, high]
    """
    return input if isinstance(input, Iterable) and len(input) == 2 else [input] * 2


def mujoco_quat_to_quat(mujoco_quat=[1, 0, 0, 0]) -> R:
    """
    Convert the mujoco quat [w, x, y, z] to standard quat [x, y, z, w]
    """
    return R.from_quat([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])


def quat_to_mujoco_quat(quat=[0, 0, 0, 1]) -> list:
    """
    Convert the standard quat [x, y, z, w] to mujoco quat [w, x, y, z]
    """
    return [quat[3], quat[0], quat[1], quat[2]]


def convert_element_to_xml(element: ET.Element) -> str:
    return minidom.parseString(ET.tostring(element, encoding="unicode")).toprettyxml()
