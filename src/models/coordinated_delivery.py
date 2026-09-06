"""
It represents a delivery point within the simulated environment.
Each point has a location (x, y coordinates) and can be associated
to one or more orders that are to be delivered to that location.
"""

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CoordinatedDelivery:
    id: int
    name: str
    x: float
    y: float

    def distance(self, other: 'CoordinatedDelivery') -> float:
        """
        Calculate the distance between two coordinates.
        :param other: another point of delivery
        :return: distance between two points
        """
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f'Ponto de Entrega({self.id}, {self.x}, {self.y})'