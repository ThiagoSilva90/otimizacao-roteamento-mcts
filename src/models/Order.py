"""
Represents an order to be delivered.
An order is linked to a delivery point and has a volume
"""

from dataclasses import dataclass
from coordinated_delivery import CoordinatedDelivery

@dataclass
class Order:
    id: int
    delivery_coordinate: CoordinatedDelivery
    volume: float # volume in m^3

    def __repr__(self):
        return (
            f"Pedido: (id={self.id}, "
            f"Ponto de Entrega={self.delivery_coordinate}, "
            f"Volume={self.volume})"
        )
