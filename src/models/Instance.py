
from typing import List
from src.models.Coordinated_delivery import CoordinatedDelivery
from src.models.Order import Order
from src.models.Vehicle import Vehicle

class RoutingInstance:
    """
        :param depot: start/end point of every trip.
        :param orders: orders to deliver (one node of the search per order).
        :param fleet: available vehicle types (each may be used on many trips).
        :param fuel_price: price of one liter of fuel (R$/L).
        :param allow_early_return: if False, a vehicle may only go back to the
            depot when it cannot carry any remaining order. Used by the TSP
            relaxation (legacy pipeline) so it does not explore useless trips.
    """

    def __init__(
        self,
        depot: CoordinatedDelivery,
        orders: List[Order],
        fleet: List[Vehicle],
        fuel_price: float,
        allow_early_return: bool = True,
    ):
        if not orders:
            raise ValueError("The instance needs at least one order.")
        if not fleet:
            raise ValueError("The instance needs at least one vehicle.")
        if fuel_price <= 0:
            raise ValueError("Fuel price must be greater than zero.")
        for v in fleet:
            if v.consumption <= 0:
                raise ValueError(f"Vehicle {v.name}: consumption must be greater than zero.")

        max_capacity = max(v.volumetric_capacity for v in fleet)
        for o in orders:
            if o.volume <= 0:
                raise ValueError(f"Order {o.id}: volume must be greater than zero.")
            if o.volume > max_capacity:
                raise ValueError(
                    f"Order {o.id} (volume {o.volume}) does not fit in any vehicle "
                    f"(largest capacity is {max_capacity}); orders are not splittable."
                )

        self.depot = depot
        self.orders = list(orders)
        self.fleet = list(fleet)
        self.fuel_price = fuel_price
        self.allow_early_return = allow_early_return

        self.n = len(self.orders)  # number of orders
        self.m = len(self.fleet)  # number of vehicle types
        points = [depot] + [o.delivery_coordinate for o in self.orders]
        self.dist = [[a.distance(b) for b in points] for a in points]
        self.volumes = [o.volume for o in self.orders]
        self.capacity = [v.volumetric_capacity for v in self.fleet]
        self.km_cost = [fuel_price / v.consumption for v in self.fleet]  # R$ per km
        self.fixed_cost = [v.fixed_cost for v in self.fleet]

        @property
        def total_volume(self) -> float:
            return sum(self.volumes)

        def __repr__(self):
            return (f"RoutingInstance(n={self.n}, frota={self.m}, "
                    f"volume_total={self.total_volume:.2f}, combustivel=R${self.fuel_price:.2f}/L)")