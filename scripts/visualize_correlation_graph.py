"""CLI entry point: build the partner correlation graph (Step 6/7 of the
preprocessing pipeline, see canids.correlation) from a normal-only CSV and
render it as a network diagram.

The graph itself is not new here -- this reuses build_registry, align_to_grid,
build_joint_vector, and build_correlation_graph exactly as run_detector.py
does. It just adds a visual (PNG) view of the same edges run_detector.py
already prints as text, so the two can be cross-checked against each other.

Usage:
    python scripts/visualize_correlation_graph.py
    python scripts/visualize_correlation_graph.py --normal-csv data/raw/syncan_train_1.csv --output data/raw/correlation_graph_real.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from canids.config import GRID_STEP_SECONDS, SYNTHETIC_DATA_DIR
from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal
from canids.data.windowing import build_joint_vector
from canids.registry import build_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=SYNTHETIC_DATA_DIR / "normal.csv")
    parser.add_argument("--output", type=Path, default=None, help="defaults to <normal-csv's dir>/correlation_graph.png")
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    args = parser.parse_args()

    output = args.output or (args.normal_csv.parent / "correlation_graph.png")

    print(f"normal CSV: {args.normal_csv}")
    registry = build_registry([args.normal_csv])
    print(f"registry: {registry.n_signals} signals, vector_size={registry.vector_size}")

    normal_df = load_normal(args.normal_csv)
    alignment = align_to_grid(normal_df, registry, step=args.grid_step)
    joint = build_joint_vector(alignment, registry)
    graph = build_correlation_graph(joint, registry)

    print(f"\ncorrelation graph: {len(graph.edges)} edges, {graph.n_folds}-fold stable")
    for edge in graph.edges:
        a = registry.entries[edge.signal_a].name
        b = registry.entries[edge.signal_b].name
        print(f"  {a} <-> {b}   strength={edge.strength:.4f}  fold_agreement={edge.fold_agreement}/{graph.n_folds}")

    g = nx.Graph()
    for entry in registry.entries:
        g.add_node(entry.name)
    for edge in graph.edges:
        a = registry.entries[edge.signal_a].name
        b = registry.entries[edge.signal_b].name
        g.add_edge(a, b, strength=edge.strength)

    isolated = [n for n in g.nodes if g.degree(n) == 0]
    connected = [n for n in g.nodes if g.degree(n) > 0]
    if isolated:
        print(f"\nno stable partner (honest limitation -- can't be replay-attributed): {', '.join(isolated)}")

    pos = nx.spring_layout(g, seed=42, k=1.5, iterations=200)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, nodelist=connected, node_color="#4C72B0", node_size=1400, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=isolated, node_color="#C44E52", node_size=1400, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, font_color="white", ax=ax)

    strengths = [abs(g.edges[e]["strength"]) for e in g.edges]
    widths = [1 + 4 * s for s in strengths]
    nx.draw_networkx_edges(g, pos, width=widths, edge_color="#555555", ax=ax)
    edge_labels = {e: f"{g.edges[e]['strength']:.2f}" for e in g.edges}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title(f"Partner Correlation Graph\n{args.normal_csv.name} -- {len(graph.edges)} stable edges, {graph.n_folds}-fold")
    ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=12, label="has stable partner(s)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=12, label="no stable partner (isolated)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
