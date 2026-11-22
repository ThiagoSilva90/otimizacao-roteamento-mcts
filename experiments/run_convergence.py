
import argparse
import bisect
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.instances import generate_instance
from src.algorithms.solvers import solve_mcts_vrp, solve_nearest_neighbor
from src.simulation.Visualization import plot_convergence

OUT = os.path.join("data", "results")
SEED0 = 100
SCENARIOS = [("A (cabe em 1 veiculo)", 0.8, 20),
             ("B (varias viagens)", 2.5, 20),
             ("B (varias viagens)", 2.5, 30)]
MILESTONES = [100, 500, 1000, 5000]


def best_at(history, iteration):
    """Best cost found within the first `iteration` iterations (history is sorted by iteration)."""
    idx = bisect.bisect_right([h[0] for h in history], iteration) - 1
    return history[idx][1]


def quartiles(values):
    q = st.quantiles(values, n=4, method="inclusive")
    return q[0], st.median(values), q[2]


def checkpoints(max_iter, points=60):
    xs, x = set(), 1.0
    ratio = max_iter ** (1 / (points - 1))
    for _ in range(points):
        xs.add(max(1, min(max_iter, round(x))))
        x *= ratio
    xs.add(max_iter)
    return sorted(xs)


def run(n_instances, max_iter):
    os.makedirs(OUT, exist_ok=True)
    xs = checkpoints(max_iter)
    milestones = [m for m in MILESTONES if m < max_iter] + [max_iter]
    rows, series, lines = [], [], []

    for name, lf, n in SCENARIOS:
        ratios_per_x = [[] for _ in xs]
        beat_at = []
        for k in range(n_instances):
            seed = SEED0 + k
            inst = generate_instance(n, lf, seed=seed)
            _, cost_nn = solve_nearest_neighbor(inst)
            _, cost, res = solve_mcts_vrp(inst, iterations=max_iter, seed=seed)
            for i, x in enumerate(xs):
                ratios_per_x[i].append(best_at(res.history, x) / cost_nn)
            first = next((it for it, c in res.history if c < cost_nn - 1e-9), None)
            beat_at.append(first)
            row = dict(scenario=name, n=n, seed=seed, cost_nn=round(cost_nn, 4),
                       cost_final=round(cost, 4), iters_to_beat_nn=first if first else "")
            for m in milestones:
                row[f"ratio_{m}"] = round(best_at(res.history, m) / cost_nn, 4)
            rows.append(row)
            print(f"[{name[0]} n={n}] seed={seed} NN={cost_nn:7.1f} MCTS={cost:7.1f} "
                  f"bate o NN em: {first if first else 'nunca'}", flush=True)

        stats = [quartiles(r) for r in ratios_per_x]
        series.append((f"{name}, n={n}", [s[1] for s in stats], [s[0] for s in stats], [s[2] for s in stats]))

        reached = [b for b in beat_at if b is not None]
        lines.append(f"{name}, n={n} ({n_instances} instancias)")
        lines.append(f"   superaram o vizinho mais proximo: {len(reached)}/{n_instances}")
        if reached:
            lines.append(f"   iteracoes ate superar o NN (mediana entre as que superaram): {st.median(reached):.0f}  "
                         f"(maximo: {max(reached)})")
        for m in milestones:
            vals = [r[f"ratio_{m}"] for r in rows if r["scenario"] == name and r["n"] == n]
            lines.append(f"   custo / NN com {m:>6d} iteracoes: mediana={st.median(vals):.3f}  "
                         f"pior instancia={max(vals):.3f}")
        lines.append("")

    plot_convergence(xs, series, title=f"Convergência do MCTS-VRP ({n_instances} instâncias por cenário)",
                     save_path=os.path.join(OUT, "convergencia.png"))
    with open(os.path.join(OUT, "convergencia.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    text = "\n".join(lines)
    with open(os.path.join(OUT, "convergencia_resumo.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few instances/iterations, for a smoke test")
    ap.add_argument("--iterations", type=int, default=10000)
    args = ap.parse_args()
    if args.quick:
        run(4, 1000)
    else:
        run(20, args.iterations)