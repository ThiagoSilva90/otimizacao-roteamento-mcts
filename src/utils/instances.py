import random
from typing import List, Optional
from src.models.Coordinated_delivery import CoordinatedDelivery
from src.models.Order import Order
from src.models.Vehicle import Vehicle
from src.models.Instance import RoutingInstance

def default_fleet() -> List[Vehicle]:
    """
    PLACEHOLDER fleet (capacity in m^3, consumption in km/L).
    Replace by the real vehicles of Table I of the article.
    """
    return [
        Vehicle(id=1, name="Utilitario", volumetric_capacity=3.0, consumption=12.0),
        Vehicle(id=2, name="Van", volumetric_capacity=8.0, consumption=9.0),
        Vehicle(id=3, name="Caminhao VUC", volumetric_capacity=20.0, consumption=5.5),
    ]

def generate_instance(n_orders: int, load_factor: float = 0.8, seed: Optional[int] = None,
                      fleet: Optional[List[Vehicle]] = None, fuel_price: float = 6.0,
                      area: float = 100.0) -> RoutingInstance:
    """
    :param n_orders: number of orders (one delivery point each), uniformly scattered
        in an area x area square (km); the depot sits at the centre.
    :param load_factor: total volume / capacity of the LARGEST vehicle.
        <= 1.0 -> everything fits in one vehicle (legacy pipeline is applicable);
        >  1.0 -> several trips are unavoidable (legacy pipeline is infeasible).
    """
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