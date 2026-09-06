"""
Visualization utility for routes.

Useful during development to sanity-check that a route "makes sense"
visually, and later to compare the nearest neighbor baseline against
the MCTS solution side by side in the article's results section.
"""

import matplotlib.pyplot as plt
from src.simulation.Route import Route


def plot_route(route: Route, title: str = "Rota", save_path: str = None):
    """
    Plots a route: points as markers, connected by lines in visit order.
    The first point is highlighted differently (as the depot/start).

    :param route: the Route to plot.
    :param title: title shown on the chart.
    :param save_path: if provided, saves the figure to this path instead
        of (or in addition to) displaying it.
    """
    xs = [p.x for p in route.points]
    ys = [p.y for p in route.points]
    names = [p.name for p in route.points]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Linha conectando os pontos na ordem de visita
    ax.plot(xs, ys, "-o", color="steelblue", markersize=8, zorder=1)

    # Ponto inicial (depósito) destacado em outra cor
    if route.points:
        ax.scatter(xs[0], ys[0], color="crimson", s=150, zorder=2, label="Início")

    # Rótulo de cada ponto com a ordem de visita
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