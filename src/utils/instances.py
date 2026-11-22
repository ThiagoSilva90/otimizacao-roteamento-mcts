import random
from typing import List, Optional
from src.models.Coordinated_delivery import CoordinatedDelivery
from src.models.Order import Order
from src.models.Vehicle import Vehicle
from src.models.Instance import RoutingInstance

GASOLINE_PRICE = 6.67
DIESEL_S10_PRICE = 7.39


def default_fleet() -> List[Vehicle]:
    return [
        Vehicle(id=1, name="Fiat Fiorino", volumetric_capacity=3.3, consumption=11.7,
                fuel_price=GASOLINE_PRICE),
        Vehicle(id=2, name="Fiat Ducato Cargo", volumetric_capacity=11.5, consumption=10.0,
                fuel_price=DIESEL_S10_PRICE),
        Vehicle(id=3, name="VUC (Delivery Express)", volumetric_capacity=20.0, consumption=6.0,
                fuel_price=DIESEL_S10_PRICE),
    ]

def generate_instance(n_orders: int, load_factor: float = 0.8, seed: Optional[int] = None,
                      fleet: Optional[List[Vehicle]] = None, fuel_price: float = 6.0,
                      area: float = 100.0) -> RoutingInstance:
    rng = random.Random(seed)
    fleet = fleet or default_fleet()
    depot = CoordinatedDelivery(0, "Deposito", area / 2, area / 2)

    raw = [rng.uniform(0.3, 1.0) for _ in range(n_orders)]
    target = load_factor * max(v.volumetric_capacity for v in fleet)
    scale = target / sum(raw)
    orders = []
    for i, r in enumerate(raw, start=1):
        point = CoordinatedDelivery(i, f"C{i}", rng.uniform(0, area), rng.uniform(0, area))
        orders.append(Order(id=i, delivery_coordinate=point, volume=round(r * scale, 3)))
    return RoutingInstance(depot, orders, fleet, fuel_price)