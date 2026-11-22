import pytest
from src.models.Coordinated_delivery import CoordinatedDelivery
from src.models.Order import Order
from src.models.Vehicle import Vehicle
from src.models.Instance import RoutingInstance
from src.simulation.Solution import Solution, Trip
from src.algorithms.vrp_state import VRPState, replay
from src.algorithms.solvers import (solve_mcts_vrp, solve_nearest_neighbor,
                                    solve_legacy_pipeline, solve_exact)
from src.utils.instances import generate_instance


def make_instance(volumes, capacities, coords=None):
    depot = CoordinatedDelivery(0, "D", 0, 0)
    coords = coords or [(i + 1, 0) for i in range(len(volumes))]
    orders = [Order(i + 1, CoordinatedDelivery(i + 1, f"C{i+1}", x, y), v)
              for i, ((x, y), v) in enumerate(zip(coords, volumes))]
    fleet = [Vehicle(i + 1, f"V{i+1}", cap, 10.0) for i, cap in enumerate(capacities)]
    return RoutingInstance(depot, orders, fleet, fuel_price=5.0)

# ------------------------------ instance ------------------------------ #
def test_instance_rejects_order_larger_than_any_vehicle():
    with pytest.raises(ValueError):
        make_instance(volumes=[12.0], capacities=[10.0])

def test_instance_rejects_empty_orders_and_fleet():
    depot = CoordinatedDelivery(0, "D", 0, 0)
    with pytest.raises(ValueError):
        RoutingInstance(depot, [], [Vehicle(1, "V", 10, 10)], 5.0)
    with pytest.raises(ValueError):
        RoutingInstance(depot, [Order(1, CoordinatedDelivery(1, "C", 1, 1), 1.0)], [], 5.0)

# ------------------------------ state ------------------------------ #
def test_initial_state_forces_dispatch_of_a_vehicle_that_fits():
    inst = make_instance([4.0, 4.0], capacities=[3.0, 10.0])
    s = VRPState.initial(inst)

    assert s.legal_actions() == [inst.n + 1]

def test_capacity_is_enforced_and_trip_ends_with_return():
    inst = make_instance([6.0, 6.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(inst.n + 0)
    s = s.apply(0)
    assert 1 not in s.legal_actions()
    assert s.return_action in s.legal_actions()
    s = s.apply(s.return_action)
    assert s.vehicle is None and s.loc == 0

def test_cannot_return_to_depot_without_delivering():
    inst = make_instance([1.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(inst.n + 0)
    assert s.return_action not in s.legal_actions()

def test_terminal_only_after_all_delivered_and_back_at_depot():
    inst = make_instance([1.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(1).apply(0)
    assert not s.is_terminal()
    s = s.apply(s.return_action)
    assert s.is_terminal() and s.legal_actions() == []

def test_state_cost_matches_hand_computation():
    # order at (3,4): round trip 10 km, 10 km/L, R$5/L -> R$5.00
    inst = make_instance([1.0], capacities=[10.0], coords=[(3, 4)])
    state, sol = replay(inst, [inst.n + 0, 0, inst.n + inst.m])
    assert state.cost == pytest.approx(5.0)
    assert sol.total_cost(inst.depot, inst.fuel_price) == pytest.approx(5.0)

def test_replay_rejects_illegal_or_incomplete_sequences():
    inst = make_instance([1.0, 1.0], capacities=[10.0])
    with pytest.raises(ValueError):
        replay(inst, [0])
    with pytest.raises(ValueError):
        replay(inst, [inst.n + 0, 0])

# ------------------------------ solvers ------------------------------ #
def test_mcts_solution_is_feasible_and_cost_is_consistent():
    inst = generate_instance(12, load_factor=2.0, seed=5)
    sol, cost, _ = solve_mcts_vrp(inst, iterations=1500, seed=1)
    assert sol.is_feasible(inst)
    assert sol.total_cost(inst.depot, inst.fuel_price) == pytest.approx(cost)

def test_mcts_is_reproducible_with_same_seed():
    inst = generate_instance(10, load_factor=1.5, seed=3)
    a = solve_mcts_vrp(inst, iterations=800, seed=42)[1]
    b = solve_mcts_vrp(inst, iterations=800, seed=42)[1]
    assert a == b

def test_multiple_trips_are_used_when_volume_exceeds_largest_vehicle():
    inst = make_instance([6.0, 6.0, 6.0], capacities=[10.0], coords=[(1, 0), (2, 0), (3, 0)])
    sol, cost, _ = solve_mcts_vrp(inst, iterations=500, seed=0)
    assert sol.is_feasible(inst) and len(sol.trips) == 3  # no two orders fit together

@pytest.mark.parametrize("load_factor", [0.8, 2.0])
def test_mcts_reaches_exact_optimum_on_tiny_instances(load_factor):
    inst = generate_instance(6, load_factor=load_factor, seed=1)
    _, optimum = solve_exact(inst)
    _, cost, _ = solve_mcts_vrp(inst, iterations=3000, seed=0)
    assert cost == pytest.approx(optimum, rel=1e-9)

def test_nearest_neighbor_is_feasible_and_never_better_than_exact():
    inst = generate_instance(6, load_factor=2.0, seed=2)
    sol, cost = solve_nearest_neighbor(inst)
    assert sol.is_feasible(inst)
    assert cost >= solve_exact(inst)[1] - 1e-9

def test_legacy_pipeline_infeasible_when_total_exceeds_every_vehicle():
    inst = generate_instance(10, load_factor=2.5, seed=1)
    assert solve_legacy_pipeline(inst, iterations=200, seed=0) is None

def test_legacy_pipeline_returns_single_trip_when_it_fits():
    inst = generate_instance(8, load_factor=0.8, seed=1)
    sol, cost, _ = solve_legacy_pipeline(inst, iterations=500, seed=0)
    assert len(sol.trips) == 1 and sol.is_feasible(inst)

def test_integrated_search_not_worse_than_legacy_on_tiny_instance():
    inst = generate_instance(6, load_factor=0.8, seed=4)
    legacy = solve_legacy_pipeline(inst, iterations=3000, seed=0)[1]
    _, cost, _ = solve_mcts_vrp(inst, iterations=3000, seed=0)
    assert cost <= legacy + 1e-9

# ------------------------------ fuel price per vehicle ------------------------------ #
def test_vehicle_own_fuel_price_overrides_instance_price():
    depot = CoordinatedDelivery(0, "D", 0, 0)
    orders = [Order(1, CoordinatedDelivery(1, "C1", 3, 4), 1.0)]
    fleet = [Vehicle(1, "Gasolina", 10.0, 10.0, fuel_price=8.0),   # own price
             Vehicle(2, "Padrao", 10.0, 10.0)]                      # uses the instance price (5.0)
    inst = RoutingInstance(depot, orders, fleet, fuel_price=5.0)
    assert inst.km_cost[0] == pytest.approx(0.8)   # 8.0 / 10
    assert inst.km_cost[1] == pytest.approx(0.5)   # 5.0 / 10

def test_invalid_vehicle_fuel_price_is_rejected():
    depot = CoordinatedDelivery(0, "D", 0, 0)
    orders = [Order(1, CoordinatedDelivery(1, "C1", 1, 1), 1.0)]
    with pytest.raises(ValueError):
        RoutingInstance(depot, orders, [Vehicle(1, "V", 10.0, 10.0, fuel_price=0.0)], 5.0)

def test_solution_cost_uses_each_vehicles_fuel_price():
    # round trip of 10 km: V1 (10 km/L, R$8/L) = R$8 ; V2 (10 km/L, instance R$5/L) = R$5
    depot = CoordinatedDelivery(0, "D", 0, 0)
    o = Order(1, CoordinatedDelivery(1, "C1", 3, 4), 1.0)
    v1 = Vehicle(1, "A", 10.0, 10.0, fuel_price=8.0)
    v2 = Vehicle(2, "B", 10.0, 10.0)
    assert Trip(v1, [o]).cost(depot, 5.0) == pytest.approx(8.0)
    assert Trip(v2, [o]).cost(depot, 5.0) == pytest.approx(5.0)

def test_search_cost_matches_solution_cost_with_mixed_fuels():
    inst = generate_instance(12, load_factor=2.0, seed=9)   # default fleet: gasoline + diesel
    sol, cost, _ = solve_mcts_vrp(inst, iterations=1000, seed=0)
    assert sol.is_feasible(inst)
    assert sol.total_cost(inst.depot, inst.fuel_price) == pytest.approx(cost)

def test_legacy_pipeline_cost_matches_solution_cost_with_mixed_fuels():
    inst = generate_instance(8, load_factor=0.8, seed=2)
    sol, cost, _ = solve_legacy_pipeline(inst, iterations=500, seed=0)
    assert sol.total_cost(inst.depot, inst.fuel_price) == pytest.approx(cost)