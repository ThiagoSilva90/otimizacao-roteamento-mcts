import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

class Node:
    __slots__ = ("state", "parent", "action", "children", "untried", "visits", "total")

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["Node"] = []
        self.untried = state.legal_actions()
        self.visits = 0
        self.total = 0.0

@dataclass
class SearchResult:
    actions: List[int]
    cost: float
    iterations: int
    seconds: float
    history: List[tuple] = field(default_factory=list)

class MCTS:
    def __init__(self, iterations: int = 5000, exploration: float = 0.7,
                 rollout_epsilon: float = 0.05, seed: Optional[int] = None,
                 time_limit: Optional[float] = None, stickiness: float = 0.9):
        """
            :param iterations: max number of simulations.
            :param exploration: UCT constant applied to rewards normalised to [0, 1].
            :param rollout_epsilon: probability of a random move in the default policy.
            :param seed: RNG seed (reproducible experiments).
            :param time_limit: optional wall-clock budget in seconds.
            :param stickiness: prob. that a simulation re-dispatches the vehicle of the previous trip.
        """
        self.iterations = iterations
        self.c = exploration
        self.epsilon = rollout_epsilon
        self.rng = random.Random(seed)
        self.time_limit = time_limit
        self.stickiness = stickiness

    def search(self, root_state) -> SearchResult:
        t0 = time.perf_counter()
        root = Node(root_state)
        best_cost, best_actions = math.inf, []
        r_min, r_max = math.inf, -math.inf
        history = []
        it = 0

        while it < self.iterations:
            if self.time_limit and time.perf_counter() - t0 >= self.time_limit:
                break
            it += 1

            node = root
            while not node.untried and node.children:
                node = self._uct_child(node, r_min, r_max)

            if node.untried:
                idx = self.rng.randrange(len(node.untried))
                node.untried[idx], node.untried[-1] = node.untried[-1], node.untried[idx]
                action = node.untried.pop()
                child = Node(node.state.apply(action), node, action)
                node.children.append(child)
                node = child

            prefix = []
            p = node
            while p.parent is not None:
                prefix.append(p.action)
                p = p.parent
            prefix.reverse()

            state, tail = node.state, []
            while not state.is_terminal():
                a = state.rollout_action(self.rng, self.epsilon, self.stickiness)
                tail.append(a)
                state = state.apply(a)
            cost = state.cost
            if cost < best_cost:
                best_cost, best_actions = cost, prefix + tail
                history.append((it, best_cost))

            reward = -cost
            r_min, r_max = min(r_min, reward), max(r_max, reward)
            while node is not None:
                node.visits += 1
                node.total += reward
                node = node.parent

        return SearchResult(best_actions, best_cost, it, time.perf_counter() - t0, history)

    def _uct_child(self, node: Node, r_min: float, r_max: float) -> Node:
        span = (r_max - r_min) or 1.0
        log_n = math.log(node.visits)
        best, best_score = None, -math.inf
        for ch in node.children:
            q = (ch.total / ch.visits - r_min) / span
            score = q + self.c * math.sqrt(log_n / ch.visits)
            if score > best_score:
                best, best_score = ch, score
        return best
