
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.instances import generate_instance
from src.algorithms.solvers import (solve_mcts_vrp, solve_nearest_neighbor,
                                    solve_legacy_pipeline, solve_exact)
from src.simulation.Visualization import plot_solution

OUT = os.path.join("data", "results")
SEED0 = 100


def run(n_instances: int, iterations: int):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    # (scenario, load_factor, sizes)
    scenarios = [("A: cabe em 1 veiculo", 0.8, [10, 15, 20]),
                 ("B: exige varias viagens", 2.5, [10, 20, 30])]
    for name, lf, sizes in scenarios:
        for n in sizes:
            for k in range(n_instances):
                seed = SEED0 + k
                inst = generate_instance(n, lf, seed=seed)
                nn_sol, nn_cost = solve_nearest_neighbor(inst)
                leg = solve_legacy_pipeline(inst, iterations=iterations, seed=seed)
                sol, cost, res = solve_mcts_vrp(inst, iterations=iterations, seed=seed)
                assert sol.is_feasible(inst)
                rows.append(dict(scenario=name, n=n, seed=seed, load_factor=lf,
                                 cost_nn=nn_cost, cost_legacy=leg[1] if leg else "",
                                 cost_mcts=cost, trips_nn=len(nn_sol.trips),
                                 trips_mcts=len(sol.trips), seconds_mcts=res.seconds))
                if k == 0 and n == sizes[-1]:  # one example figure per scenario
                    tag = name[0]
                    plot_solution(nn_sol, inst.depot, f"Vizinho mais próximo (n={n})", inst.fuel_price,
                                  os.path.join(OUT, f"rota_nn_{tag}.png"))
                    plot_solution(sol, inst.depot, f"MCTS-VRP (n={n})", inst.fuel_price,
                                  os.path.join(OUT, f"rota_mcts_{tag}.png"))
                print(f"[{name[0]}] n={n:2d} seed={seed}  NN={nn_cost:7.1f}  "
                      f"legacy={('%7.1f' % leg[1]) if leg else '  inviavel'}  MCTS={cost:7.1f}", flush=True)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def optimality_gap(n_instances: int, iterations: int, n: int = 7):
    gaps = []
    for k in range(n_instances):
        inst = generate_instance(n, 2.0, seed=SEED0 + k)
        _, opt = solve_exact(inst)
        _, cost, _ = solve_mcts_vrp(inst, iterations=iterations, seed=SEED0 + k)
        gaps.append(100 * (cost - opt) / opt)
    return gaps


def summarize(rows, gaps, iterations):
    lines = [f"MCTS iterations per run: {iterations}\n"]
    header = f"{'Cenario':26s}{'n':>4s}{'NN (R$)':>10s}{'Legacy':>10s}{'MCTS':>10s}{'vs NN':>9s}{'vs Legacy':>11s}{'vitorias/NN':>13s}{'viagens':>9s}{'tempo(s)':>10s}"
    lines.append(header)
    lines.append("-" * len(header))
    keys = sorted({(r["scenario"], r["n"]) for r in rows})
    for sc, n in keys:
        g = [r for r in rows if r["scenario"] == sc and r["n"] == n]
        nn = st.mean(r["cost_nn"] for r in g)
        mc = st.mean(r["cost_mcts"] for r in g)
        leg_vals = [r["cost_legacy"] for r in g if r["cost_legacy"] != ""]
        leg = st.mean(leg_vals) if leg_vals else None
        imp_nn = 100 * (nn - mc) / nn
        imp_leg = 100 * (leg - st.mean(r["cost_mcts"] for r in g if r["cost_legacy"] != "")) / leg if leg else None
        wins = sum(r["cost_mcts"] <= r["cost_nn"] + 1e-9 for r in g)
        lines.append(f"{sc:26s}{n:>4d}{nn:>10.1f}{(f'{leg:.1f}' if leg else 'inviavel'):>10s}{mc:>10.1f}"
                     f"{imp_nn:>8.1f}%{(f'{imp_leg:.1f}%' if imp_leg is not None else '-'):>11s}"
                     f"{wins:>9d}/{len(g):<3d}{st.mean(r['trips_mcts'] for r in g):>9.1f}"
                     f"{st.mean(r['seconds_mcts'] for r in g):>10.2f}")
    if gaps:
        lines.append(f"\nGap para o otimo exato (n=7, {len(gaps)} instancias): "
                     f"media={st.mean(gaps):.2f}%  max={max(gaps):.2f}%  "
                     f"otimo encontrado em {sum(g < 1e-6 for g in gaps)}/{len(gaps)}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few instances, for a smoke test")
    ap.add_argument("--iterations", type=int, default=5000)
    args = ap.parse_args()
    n_inst = 3 if args.quick else 20
    rows = run(n_inst, args.iterations)
    gaps = optimality_gap(3 if args.quick else 10, args.iterations)
    summarize(rows, gaps, args.iterations)