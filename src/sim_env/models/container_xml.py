import xml.etree.ElementTree as ET

from src.sim_env.utils.conversion import convert_array_to_string


class ContainerXML(object):
    """
    Initialize the container with the given size parameters

    Attributes:
        size: [x_size, y_size, z_size, t_size]
            x_size: the length (range) of the container's x-side
            y_size: the length (range) of the container's y-side
            z_size: the length (range) of the container's z-size
            t_size: the length (range) of the container's thickness
        pos: [x, y, z]

    """

    def __init__(self, size=[0.4, 0.32, 0.255, 0.01]) -> None:
        self.size = size

        self._x_size = self.size[0]
        self._y_size = self.size[1]
        self._z_size = self.size[2]
        self._t_size = self.size[3]

        self.default_element = []
        self.worldbody_element = []

        self.__create_element_node__()

    def __create_element_node__(self):
        x_wall_element = ET.Element("default", {"class": "x_wall"})
        y_wall_element = ET.Element("default", {"class": "y_wall"})
        self.default_element = [x_wall_element, y_wall_element]

        x_wall_geom = ET.SubElement(
            x_wall_element,
            "geom",
            {"type": "box", "size": "", "rgba": "0 0 0 0.2"},
        )
        y_wall_geom = ET.SubElement(
            y_wall_element,
            "geom",
            {"type": "box", "size": "", "rgba": "0 0 0 0.2"},
        )

        x_wall_geom.set(
            "size",
            convert_array_to_string(
                [self._z_size / 2, self._y_size / 2 + self._t_size, self._t_size]
            ),
        )
        y_wall_geom.set(
            "size",
            convert_array_to_string([self._x_size / 2, self._z_size / 2, self._t_size]),
        )

        x_p_wall = ET.Element(
            "geom",
            {"name": "+x", "class": "x_wall", "zaxis": "1 0 0"},
        )
        x_n_wall = ET.Element(
            "geom",
            {"name": "-x", "class": "x_wall", "zaxis": "-1 0 0"},
        )
        y_p_wall = ET.Element(
            "geom",
            {"name": "+y", "class": "y_wall", "zaxis": "0 1 0"},
        )
        y_n_wall = ET.Element(
            "geom",
            {"name": "-y", "class": "y_wall", "zaxis": "0 -1 0"},
        )
        self.worldbody_element = [x_p_wall, x_n_wall, y_p_wall, y_n_wall]

        x_p_wall.set(
            "pos",
            convert_array_to_string(
                [-self._t_size / 2, self._y_size / 2, self._z_size / 2]
            ),
        )
        x_n_wall.set(
            "pos",
            convert_array_to_string(
                [self._x_size + self._t_size / 2, self._y_size / 2, self._z_size / 2]
            ),
        )
        y_p_wall.set(
            "pos",
            convert_array_to_string(
                [self._x_size / 2, -self._t_size / 2, self._z_size / 2]
            ),
        )
        y_n_wall.set(
            "pos",
            convert_array_to_string(
                [self._x_size / 2, self._y_size + self._t_size / 2, self._z_size / 2]
            ),
        )


if __name__ == "__main__":
    container = ContainerXML()
