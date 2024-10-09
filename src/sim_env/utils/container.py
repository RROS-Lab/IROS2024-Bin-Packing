class Package:

    def __init__(
        self,
        name: str,
        package_id: int,
        package_size: list,
        mass: float,
        stiffness: float,
    ) -> None:

        self.name = name
        self.mass = mass
        self.package_size = package_size
        self.mass = mass
        self.stiffness = stiffness
        self.package_id = package_id

        self.youngs_mod = (self.stiffness * self.package_size[2]) / (
            self.package_size[0] * self.package_size[1]
        )

        print(f"Package {self.name} created with youngs modulus {self.youngs_mod}.")
