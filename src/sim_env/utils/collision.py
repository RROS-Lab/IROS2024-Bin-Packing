import numpy as np


class Point:
    def __init__(self, x: np.float32, y: np.float32) -> None:
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def intersect(self, other):
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = other.p1.x, other.p1.y
        x4, y4 = other.p2.x, other.p2.y

        def ccw(x1, y1, x2, y2, x3, y3):
            return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

        return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(
            x1, y1, x2, y2, x3, y3
        ) != ccw(x1, y1, x2, y2, x4, y4)

    def above(self, other):
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = other.p1.x, other.p1.y
        x4, y4 = other.p2.x, other.p2.y

        def ccw(x1, y1, x2, y2, x3, y3):
            return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

        return ccw(x1, y1, x3, y3, x4, y4)

    def __str__(self):
        return f"Line from {self.p1.x, self.p1.y} to {self.p2.x, self.p2.y}"
