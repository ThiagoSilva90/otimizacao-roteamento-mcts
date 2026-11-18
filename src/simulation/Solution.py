from dataclasses import dataclass, field
from typing import List
from src.models.Order import Order
from src.models.Vehicle import Vehicle
from src.models.Instance import RoutingInstance
from src.simulation.Route import Route

@dataclass
class Trip:
    vehicle: Vehicle
    orders: List[Order] = field(default_factory=list)

    @property
    def load(self) -> float:
        return sum(o.volume for o in self.orders)

    def route(self, depot) -> Route:
        pts = [depot] + [o.delivery_coordinate for o in self.orders] + [depot]
        return Route(points=pts)

    def distance(self, depot) -> float:
        return self.route(depot).total_distance()

    def cost(self, depot, fuel_price: float) -> float:
        return (self.vehicle.estimate_combustible_cost(self.distance(depot), fuel_price)
                + self.vehicle.fixed_cost)

@dataclass
class Solution:
    trips: List[Trip] = field(default_factory=list)

    def total_distance(self, depot) -> float:
        return sum(t.distance(depot) for t in self.trips)

    def total_cost(self, depot, fuel_price: float) -> float:
        return sum(t.cost(depot, fuel_price) for t in self.trips)

    def is_feasible(self, instance: RoutingInstance) -> bool:
        """Every order exactly once and no trip above its vehicle's capacity."""
        delivered = [o.id for t in self.trips for o in t.orders]
        if sorted(delivered) != sorted(o.id for o in instance.orders):
            return False
        return all(t.orders and t.vehicle.holds_load(t.load) for t in self.trips)

    def __repr__(self):
        lines = [f"  Viagem {i + 1} [{t.vehicle.name}, carga={t.load:.2f}]: "
                 + " -> ".join(str(o.id) for o in t.orders)
                 for i, t in enumerate(self.trips)]
        return "Solution(\n" + "\n".join(lines) + "\n)"