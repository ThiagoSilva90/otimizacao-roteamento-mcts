"""
Testes automatizados para os modelos CoordinatedDelivery, Order e Vehicle.

Rode com: pytest tests/
"""

import pytest
from src.models.Coordinated_delivery import CoordinatedDelivery
from src.models.Order import Order
from src.models.Vehicle import Vehicle


# ---------- CoordinatedDelivery ----------

def test_distance_known_triangle():
    # Triângulo 3-4-5: distância deve ser exatamente 5.0
    origin = CoordinatedDelivery(id=1, name="Origem", x=0, y=0)
    destination = CoordinatedDelivery(id=2, name="Destino", x=3, y=4)
    assert origin.distance(destination) == pytest.approx(5.0)


def test_distance_to_itself_is_zero():
    point = CoordinatedDelivery(id=1, name="A", x=10, y=10)
    assert point.distance(point) == 0.0

# ---------- Order ----------

def test_order_holds_volume_and_coordinate():
    point = CoordinatedDelivery(id=1, name="Cliente A", x=0, y=0)
    order = Order(id=1, delivery_coordinate=point, volume=3.0)
    assert order.volume == 3.0
    assert order.delivery_coordinate.name == "Cliente A"

# ---------- Vehicle ----------

@pytest.fixture
def default_vehicle():
    # Van que roda 8 km com 1 litro
    return Vehicle(
        id=1,
        name="Van 1",
        volumetric_capacity=10.0,
        consumption=8.0,
    )

def test_holds_load_within_capacity(default_vehicle):
    assert default_vehicle.holds_load(9.9) is True


def test_holds_load_at_exact_capacity(default_vehicle):
    # Limite exato deve ser aceito (>=)
    assert default_vehicle.holds_load(10.0) is True

def test_holds_load_above_capacity(default_vehicle):
    assert default_vehicle.holds_load(10.1) is False

def test_estimate_combustible_cost(default_vehicle):
    # 80 km / 8 km-por-litro = 10 litros; 10 litros * R$6,00 = R$60,00
    cost = default_vehicle.estimate_combustible_cost(
        total_distance=80, combustible_cost=6.0
    )
    assert cost == pytest.approx(60.0)

def test_estimate_combustible_cost_zero_distance(default_vehicle):
    cost = default_vehicle.estimate_combustible_cost(
        total_distance=0, combustible_cost=6.0
    )
    assert cost == pytest.approx(0.0)

def test_estimate_combustible_cost_with_zero_consumption_raises():
    broken_vehicle = Vehicle(
        id=2, name="Quebrada",
        volumetric_capacity=10, consumption=0,
    )
    with pytest.raises(ValueError):
        broken_vehicle.estimate_combustible_cost(total_distance=80, combustible_cost=6.0)