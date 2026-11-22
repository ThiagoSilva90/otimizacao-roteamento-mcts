import matplotlib.pyplot as plt
from src.simulation.Route import Route


def plot_route(route: Route, title: str = "Rota", save_path: str = None):
    xs = [p.x for p in route.points]
    ys = [p.y for p in route.points]
    names = [p.name for p in route.points]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(xs, ys, "-o", color="steelblue", markersize=8, zorder=1)

    if route.points:
        ax.scatter(xs[0], ys[0], color="crimson", s=150, zorder=2, label="Início")

    for i, (x, y, name) in enumerate(zip(xs, ys, names)):
        ax.annotate(f"{i}: {name}", (x, y), textcoords="offset points", xytext=(8, 8))

    ax.set_title(f"{title}\nDistância total: {route.total_distance():.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_solution(solution, depot, title: str = "Solução", fuel_price: float = None,
                  save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")

    for k, trip in enumerate(solution.trips):
        pts = trip.route(depot).points
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        ax.plot(xs, ys, "-o", color=cmap(k % 10), markersize=6, zorder=1,
                label=f"Viagem {k+1}: {trip.vehicle.name} "
                      f"({trip.load:.1f}/{trip.vehicle.volumetric_capacity:g} m³)")
        for o in trip.orders:
            p = o.delivery_coordinate
            ax.annotate(str(o.id), (p.x, p.y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.scatter(depot.x, depot.y, color="black", marker="s", s=140, zorder=3, label="Depósito")

    subtitle = f"Distância total: {solution.total_distance(depot):.1f} km"
    if fuel_price is not None:
        subtitle += f" | Custo: R$ {solution.total_cost(depot, fuel_price):.2f}"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()