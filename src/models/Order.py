
from dataclasses import dataclass
from src.models.Coordinated_delivery import CoordinatedDelivery

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
