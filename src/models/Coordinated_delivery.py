
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CoordinatedDelivery:
    id: int
    name: str
    x: float
    y: float

    def distance(self, other: 'CoordinatedDelivery') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f'Ponto de Entrega({self.id}, {self.x}, {self.y})'