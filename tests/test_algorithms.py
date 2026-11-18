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
    # vehicle 0 (cap 3) cannot carry any remaining order -> only vehicle 1 is dispatchable
    assert s.legal_actions() == [inst.n + 1]

def test_capacity_is_enforced_and_trip_ends_with_return():
    inst = make_instance([6.0, 6.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(inst.n + 0)        # dispatch
    s = s.apply(0)                                       # deliver order 0 (cap_left 4)
    assert 1 not in s.legal_actions()                    # 6 > 4: cannot deliver order 1 now
    assert s.return_action in s.legal_actions()
    s = s.apply(s.return_action)
    assert s.vehicle is None and s.loc == 0

def test_cannot_return_to_depot_without_delivering():
    inst = make_instance([1.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(inst.n + 0)
    assert s.return_action not in s.legal_actions()

def test_terminal_only_after_all_delivered_and_back_at_depot():
    inst = make_instance([1.0], capacities=[10.0])
    s = VRPState.initial(inst).apply(1).apply(0)   # dispatch, deliver
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
        replay(inst, [0])                                 # visit before dispatch
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
    # the VRP space contains the legacy solution as a special case
    inst = generate_instance(6, load_factor=0.8, seed=4)
    legacy = solve_legacy_pipeline(inst, iterations=3000, seed=0)[1]
    _, cost, _ = solve_mcts_vrp(inst, iterations=3000, seed=0)
    assert cost <= legacy + 1e-9