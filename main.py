from src.utils.instances import generate_instance
from src.algorithms.solvers import solve_mcts_vrp, solve_nearest_neighbor, solve_legacy_pipeline
from src.simulation.Visualization import plot_solution

if __name__ == "__main__":
    inst = generate_instance(n_orders=20, load_factor=2.5, seed=7)
    print(inst)

    nn_sol, nn_cost = solve_nearest_neighbor(inst)
    print(f"\nVizinho mais próximo: R$ {nn_cost:.2f} ({len(nn_sol.trips)} viagens)")

    legacy = solve_legacy_pipeline(inst, iterations=3000, seed=7)
    print("Pipeline anterior (TSP + escolha do veículo): "
          + ("inviável - nenhum veículo comporta o volume total" if legacy is None else f"R$ {legacy[1]:.2f}"))

    sol, cost, res = solve_mcts_vrp(inst, iterations=5000, seed=7)
    print(f"MCTS-VRP: R$ {cost:.2f} ({len(sol.trips)} viagens, {res.seconds:.2f}s)")
    print(sol)

    plot_solution(sol, inst.depot, "MCTS-VRP", inst.fuel_price)