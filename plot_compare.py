import re
import sys
from pathlib import Path
import json
import argparse

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

def load_from_dot(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    edge_re = re.compile(r'^\s*(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]', re.MULTILINE)

    nodes = set()
    edges = []
    for m in edge_re.finditer(text):
        u = int(m.group(1))
        v = int(m.group(2))

        lab = m.group(3)
        lab = lab.replace("\\n", "").replace("\n", "").strip()
        
        parts = [x.strip() for x in lab.split(",") if x.strip()]
        for a in parts:
            edges.append((u, v, a))
        nodes.add(u); nodes.add(v)
    return nodes, edges

def load_from_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    nodes = set()
    edges = []
    for e in obj["delta"]:
        u = int(e["source"]); v = int(e["target"]); a = str(e["label"])
        edges.append((u, v, a))
        nodes.add(u); nodes.add(v)
    if "start" in obj:
        nodes.add(int(obj["start"]))
    return nodes, edges

def build_graph(nodes, edges):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)

    for u, v, a in edges:
        if G.has_edge(u, v):
            prev = G[u][v].get("label", "")
            labs = set([x for x in prev.split(",") if x] + [a])
            G[u][v]["label"] = ",".join(sorted(labs))
        else:
            G.add_edge(u, v, label=a)
    return G

def draw(ax, G, title, pos):
    ax.set_title(title)
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=650)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=22,
    width=1.8,
    min_source_margin=15,
    min_target_margin=15,
    connectionstyle="arc3,rad=0.08",
)
    labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=9)

def load_any(path: Path):
    if path.suffix.lower() == ".dot":
        return load_from_dot(path)
    if path.suffix.lower() == ".json":
        return load_from_json(path)
    raise ValueError("Need .dot or .json")

def bfs_layers(nodes, edges, start=0):
    adj = defaultdict(list)
    rev = defaultdict(list)

    for u, v, a in edges:
        adj[u].append((a, v))
        rev[v].append((a, u))

    for u in adj:
        adj[u].sort(key=lambda x: (x[0], x[1]))
    for v in rev:
        rev[v].sort(key=lambda x: (x[0], x[1]))

    dist = {}
    q = deque()

    if start in nodes:
        dist[start] = 0
        q.append(start)

    while q:
        u = q.popleft()
        for a, v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    maxd = max(dist.values()) if dist else 0
    for n in sorted(nodes):
        if n not in dist:
            maxd += 1
            dist[n] = maxd

    return dist, adj, rev


def node_signature(n, G, dist):
    out_edges = []
    for _, v, data in G.out_edges(n, data=True):
        out_edges.append((data.get("label", ""), dist.get(v, 999999)))
    out_edges.sort()

    in_edges = []
    for u, _, data in G.in_edges(n, data=True):
        in_edges.append((data.get("label", ""), dist.get(u, 999999)))
    in_edges.sort()

    sig = (
        dist.get(n, 999999),
        G.in_degree(n),
        G.out_degree(n),
        tuple(out_edges),
        tuple(in_edges),
        n,
    )
    return sig


def fixed_compare_layout(G, start=0, x_gap=1.7, y_gap=1.0):
    nodes = list(G.nodes())
    edges = [(u, v, d.get("label", "")) for u, v, d in G.edges(data=True)]

    dist, _, _ = bfs_layers(set(nodes), edges, start=start)

    layers = defaultdict(list)
    for n in nodes:
        layers[dist[n]].append(n)

    pos = {}
    all_depths = sorted(layers.keys())

    for depth in all_depths:
        layer_nodes = layers[depth]
        layer_nodes.sort(key=lambda n: node_signature(n, G, dist))

        k = len(layer_nodes)
        ys = [((k - 1) / 2.0 - i) * y_gap for i in range(k)]

        for n, y in zip(layer_nodes, ys):
            x = depth * x_gap
            pos[n] = (x, y)

    return pos

def should_save(argv):
    return len(argv) >= 2 and argv[-1].lower() == "save"

def output_png_path(p: Path) -> Path:
    return p.with_suffix(".png")

def output_compare_png_path(p1: Path, p2: Path) -> Path:
    return  f"{p1.stem}_VS_{p2.stem}.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="1 or 2 input files")
    parser.add_argument("--save", action="store_true", help="save figure as png")
    return parser.parse_args()

def output_four_compare_png_path(files):
    stems = [Path(f).stem for f in files]
    return "FOUR_COMPARE_" + "_VS_".join(stems) + ".png"

def main():
    args = parse_args()

    if len(args.files) not in (1, 2, 4):
        print("Usage:")
        print("  python plot_compare.py <DOT_OR_JSON> [--save]")
        print("  python plot_compare.py <REFERENCE> <LEARNT> [--save]")
        print("  python plot_compare.py <REFERENCE> <EDSM_BASELINE> <RANDOM_ADDED> <CONFIDENCE_ADDED> [--save]")
        raise SystemExit(2)

    if len(args.files) == 1:
        p = Path(args.files[0])
        nodes, edges = load_any(p)
        G = build_graph(nodes, edges)

        pos = fixed_compare_layout(G, start=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        draw(ax, G, p.name, pos)
        plt.tight_layout()

        if args.save:
            out_png = output_png_path(p)
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {out_png}")

        plt.show()
        return
    
    if len(args.files) == 2:
        p1 = Path(args.files[0])
        p2 = Path(args.files[1])

        n1, e1 = load_any(p1)
        n2, e2 = load_any(p2)

        G1 = build_graph(n1, e1)
        G2 = build_graph(n2, e2)

        pos1 = fixed_compare_layout(G1, start=0)
        pos2 = fixed_compare_layout(G2, start=0)

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        draw(ax1, G1, f"Reference: {p1.name}", pos1)
        draw(ax2, G2, f"Learnt: {p2.name}", pos2)

        plt.tight_layout()

        if args.save:
            out_png = output_compare_png_path(p1, p2)
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {out_png}")

        plt.show()
        return

    if len(args.files) == 4:
        paths = [Path(f) for f in args.files]

        titles = [
            "Reference Automaton",
            "EDSM Baseline",
            "Random traces Added",
            "Confidence-based Added",
        ]

        graphs = []
        positions = []

        for p in paths:
            nodes, edges = load_any(p)
            G = build_graph(nodes, edges)
            pos = fixed_compare_layout(G, start=0)

            graphs.append(G)
            positions.append(pos)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.flatten()

        for ax, G, pos, title, path in zip(axes, graphs, positions, titles, paths):
            draw(ax, G, f"{title}\n{path.name}", pos)

        plt.tight_layout()

        if args.save:
            out_png = output_four_compare_png_path(paths)
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved figure to: {out_png}")

        plt.show()
        return

    # p1 = Path(args.files[0])
    # p2 = Path(args.files[1])
    # n1, e1 = load_any(p1)
    # n2, e2 = load_any(p2)
    # G1 = build_graph(n1, e1)
    # G2 = build_graph(n2, e2)

    # pos1 = fixed_compare_layout(G1, start=0)
    # pos2 = fixed_compare_layout(G2, start=0)

    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # draw(ax1, G1, f"Reference: {p1.name}", pos1)
    # draw(ax2, G2, f"Learnt: {p2.name}", pos2)
    # plt.tight_layout()

    # if args.save:
    #     out_png = output_compare_png_path(p1, p2)
    #     fig.savefig(out_png, dpi=300, bbox_inches="tight")
    #     print(f"Saved figure to: {out_png}")

    # plt.show()

if __name__ == "__main__":
    main()