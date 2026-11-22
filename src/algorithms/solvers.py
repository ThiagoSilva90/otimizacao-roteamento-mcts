import math
from typing import Optional, Tuple
from src.models.Instance import RoutingInstance
from src.models.Vehicle import Vehicle
from src.simulation.Solution import Solution, Trip
from src.algorithms.vrp_state import VRPState, replay
from src.algorithms.mcts import MCTS, SearchResult

def solve_mcts_vrp(inst: RoutingInstance, iterations: int = 5000, exploration: float = 0.7,
                   rollout_epsilon: float = 0.05, seed: Optional[int] = None,
                   time_limit: Optional[float] = None, stickiness: float = 0.9
                   ) -> Tuple[Solution, float, SearchResult]:
    mcts = MCTS(iterations, exploration, rollout_epsilon, seed, time_limit, stickiness)
    result = mcts.search(VRPState.initial(inst))
    state, solution = replay(inst, result.actions)
    return solution, state.cost, result

def solve_nearest_neighbor(inst: RoutingInstance) -> Tuple[Solution, float]:
    """
    Greedy baseline. For each vehicle type able to carry every order, build
    trips with the nearest-neighbour rule (go to the closest order that still
    fits; return to the depot when none fits) and keep the cheapest result.
    """
    best = None
    for v in range(inst.m):
        if inst.capacity[v] + 1e-9 < max(inst.volumes):
            continue
        state, actions = VRPState.initial(inst), []
        while not state.is_terminal():
            if state.vehicle is None:
                a = inst.n + v
            else:
                feasible = state.feasible_orders()
                row = inst.dist[state.loc]
                a = min(feasible, key=lambda i: row[i + 1]) if feasible else state.return_action
            actions.append(a)
            state = state.apply(a)
        if best is None or state.cost < best[0]:
            best = (state.cost, actions)
    cost, actions = best
    return replay(inst, actions)[1], cost


def solve_legacy_pipeline(inst: RoutingInstance, iterations: int = 5000, seed: Optional[int] = None,
                          exploration: float = 0.7, rollout_epsilon: float = 0.05
                          ) -> Optional[Tuple[Solution, float, float]]:
    """
    Previous formulation: (1) MCTS finds the visiting order of ALL orders as a
    plain TSP (capacity ignored); (2) the cheapest vehicle holding the total
    volume is chosen afterwards. Returns None when no single vehicle holds the
    total volume (the case that motivated the VRP formulation).

    :return: (solution, cost_in_R$, tsp_distance) or None.
    """
    total = inst.total_volume
    candidates = [v for v in inst.fleet if v.holds_load(total)]
    if not candidates:
        return None

    virtual = Vehicle(id=-1, name="TSP", volumetric_capacity=total + 1.0, consumption=1.0)
    tsp = RoutingInstance(inst.depot, inst.orders, [virtual], fuel_price=1.0, allow_early_return=False)
    result = MCTS(iterations, exploration, rollout_epsilon, seed, stickiness=0.0).search(VRPState.initial(tsp))
    _, tsp_solution = replay(tsp, result.actions)
    sequence = tsp_solution.trips[0].orders
    distance = result.cost

    best_vehicle = min(candidates,
                       key=lambda v: v.estimate_combustible_cost(distance, v.price_or(inst.fuel_price)) + v.fixed_cost)
    solution = Solution([Trip(vehicle=best_vehicle, orders=list(sequence))])
    cost = solution.total_cost(inst.depot, inst.fuel_price)
    return solution, cost, distance

def solve_exact(inst: RoutingInstance, upper_bound: float = math.inf) -> Tuple[Solution, float]:
    best = [upper_bound, None]

    def dfs(state: VRPState, actions: list):
        if state.cost >= best[0] - 1e-12:
            return
        if state.is_terminal():
            best[0], best[1] = state.cost, list(actions)
            return
        for a in state.legal_actions():
            actions.append(a)
            dfs(state.apply(a), actions)
            actions.pop()

    if upper_bound == math.inf:
        upper_bound = solve_nearest_neighbor(inst)[1] + 1e-6
        best[0] = upper_bound
    dfs(VRPState.initial(inst), [])
    state, solution = replay(inst, best[1])
    return solution, state.cost