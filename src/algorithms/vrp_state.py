import random
from typing import List, Optional, Tuple
from src.models.Instance import RoutingInstance
from src.simulation.Solution import Solution, Trip

EPS = 1e-9

class VRPState:
    __slots__ = ("inst", "loc", "remaining", "vehicle", "cap_left", "cost", "last_vehicle")

    def __init__(self, inst: RoutingInstance, loc: int, remaining: int,
                 vehicle: Optional[int], cap_left: float, cost: float,
                 last_vehicle: Optional[int] = None):
        self.inst = inst
        self.loc = loc
        self.remaining = remaining
        self.vehicle = vehicle
        self.cap_left = cap_left
        self.cost = cost
        self.last_vehicle = last_vehicle

    @classmethod
    def initial(cls, inst: RoutingInstance) -> "VRPState":
        return cls(inst, 0, (1 << inst.n) - 1, None, 0.0, 0.0)

    @property
    def return_action(self) -> int:
        return self.inst.n + self.inst.m

    def is_terminal(self) -> bool:
        return self.remaining == 0 and self.loc == 0 and self.vehicle is None

    def feasible_orders(self) -> List[int]:
        rem, vol, cap = self.remaining, self.inst.volumes, self.cap_left + EPS
        return [i for i in range(self.inst.n) if (rem >> i) & 1 and vol[i] <= cap]

    def legal_actions(self) -> List[int]:
        inst = self.inst
        if self.remaining == 0:
            return [] if self.loc == 0 else [self.return_action]

        if self.vehicle is None:  # at the depot: choose who leaves
            rem, vol = self.remaining, inst.volumes
            min_vol = min(vol[i] for i in range(inst.n) if (rem >> i) & 1)
            return [inst.n + v for v in range(inst.m) if inst.capacity[v] + EPS >= min_vol]

        acts = self.feasible_orders()
        if self.loc != 0 and (inst.allow_early_return or not acts):
            acts.append(self.return_action)
        return acts

    def apply(self, action: int) -> "VRPState":
        inst = self.inst
        n, m = inst.n, inst.m
        if action < n:  # visit order
            v = self.vehicle
            step = inst.dist[self.loc][action + 1] * inst.km_cost[v]
            return VRPState(inst, action + 1, self.remaining & ~(1 << action), v,
                            self.cap_left - inst.volumes[action], self.cost + step, self.last_vehicle)
        if action < n + m:  # dispatch vehicle
            v = action - n
            return VRPState(inst, 0, self.remaining, v, inst.capacity[v],
                            self.cost + inst.fixed_cost[v], self.last_vehicle)
        # return to depot
        step = inst.dist[self.loc][0] * inst.km_cost[self.vehicle]
        return VRPState(inst, 0, self.remaining, None, 0.0, self.cost + step, self.vehicle)

    def rollout_action(self, rng: random.Random, epsilon: float = 0.05, stickiness: float = 0.9) -> int:
        acts = self.legal_actions()
        if len(acts) == 1:
            return acts[0]
        if rng.random() < epsilon:
            return rng.choice(acts)
        if self.vehicle is None:
            if self.last_vehicle is not None and rng.random() < stickiness:
                a = self.inst.n + self.last_vehicle
                if a in acts:
                    return a
            return rng.choice(acts)
        feasible = self.feasible_orders()
        if not feasible:
            return self.return_action
        row = self.inst.dist[self.loc]
        return min(feasible, key=lambda i: row[i + 1])

def replay(inst: RoutingInstance, actions: List[int]) -> Tuple[VRPState, Solution]:
    state = VRPState.initial(inst)
    trips: List[Trip] = []
    current: Optional[Trip] = None
    for a in actions:
        if a not in state.legal_actions():
            raise ValueError(f"Illegal action {a} in replay.")
        if a < inst.n:
            current.orders.append(inst.orders[a])
        elif a < inst.n + inst.m:
            current = Trip(vehicle=inst.fleet[a - inst.n])
            trips.append(current)
        state = state.apply(a)
    if not state.is_terminal():
        raise ValueError("Action sequence does not reach a terminal state.")
    return state, Solution(trips=trips)