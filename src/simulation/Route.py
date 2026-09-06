from dataclasses import dataclass, field
from typing import List
from src.models.Coordinated_delivery import CoordinatedDelivery

@dataclass
class Route:
    points: List[CoordinatedDelivery] = field(default_factory=list) # ensures that each route has its own list

    def total_distance(self) -> float:
        """
        Sums the distance between each consecutive pair of points in the route.
        A route with 0 or 1 points has a total distance of 0.
        :return: total distance of the route.
        """
        if len(self.points) < 2:
            return 0.0

        distance = 0.0

        for current, next_point in zip(self.points, self.points[1:]):
            distance += current.distance(next_point)

        return distance

    def add_point(self, point: CoordinatedDelivery) -> None:
        """Appends a point to the end of the route."""
        self.points.append(point)

    def __repr__(self):
        names = " -> ".join(p.name for p in self.points)
        return f"Route([{names}], distancia_total={self.total_distance():.2f})"