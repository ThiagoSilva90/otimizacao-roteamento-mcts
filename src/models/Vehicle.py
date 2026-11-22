from dataclasses import dataclass
from typing import Optional

@dataclass
class Vehicle:
    id: int
    name: str
    volumetric_capacity: float
    consumption: float
    fixed_cost: float = 0.0
    fuel_price: Optional[float] = None

    def price_or(self, default_price: float) -> float:
        """Price (R$/L) paid by this vehicle: its own fuel price if set, otherwise the default one."""
        return self.fuel_price if self.fuel_price is not None else default_price

    def holds_load(self, total_load: float) -> bool:
        """
        Checks if the vehicle has the capacity for the full volume of the load.
        :param total_load: Total volume of the load.
        :return: True if the vehicle has the capacity for the full volume of the load.
        """
        return self.volumetric_capacity >= total_load

    def estimate_combustible_cost(self, total_distance: float, combustible_cost: float) -> float:
        """
        Estimates the fuel cost (in R$) to travel the given distance.
        :param combustible_cost: Median cost of the combustible.
        :param total_distance: Total distance of the operation
        :return: Total coast
        """
        if self.consumption <= 0:
            raise ValueError("Consumption must be greater than zero.")
        total_liters = total_distance / self.consumption
        return total_liters * combustible_cost

    def __repr__(self):
        return (
            f"Veiculo(id={self.id}, nome='{self.name}', "
            f"Consumo={self.consumption} km/L, "
            f"Capacidade={self.volumetric_capacity}"
            + (f", Combustivel=R${self.fuel_price:.2f}/L" if self.fuel_price is not None else "") + ")"
        )