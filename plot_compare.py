import re
import sys
from pathlib import Path
import json

import networkx as nx
import matplotlib.pyplot as plt

def load_from_dot(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    edge_re = re.compile(r'^\s*(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]', re.MULTILINE)

    nodes = set()
    edges = []
    for m in edge_re.finditer(text):
        u = int(m.group(1)); v = int(m.group(2))
        lab = m.group(3).strip()
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

def draw(ax, G, title):
    ax.set_title(title)
    ax.axis("off")
    pos = nx.spring_layout(G, seed=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=650)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
    labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=9)

def load_any(path: Path):
    if path.suffix.lower() == ".dot":
        return load_from_dot(path)
    if path.suffix.lower() == ".json":
        return load_from_json(path)
    raise ValueError("Need .dot or .json")

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage:")
        print("  python plot_compare.py <DOT_OR_JSON>")
        print("  python plot_compare.py <REF.dot> <LEARN.dot>")
        raise SystemExit(2)

    if len(sys.argv) == 2:
        p = Path(sys.argv[1])
        nodes, edges = load_any(p)
        G = build_graph(nodes, edges)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        draw(ax, G, p.name)
        plt.tight_layout()
        plt.show()
        return

    p1 = Path(sys.argv[1])
    p2 = Path(sys.argv[2])
    n1, e1 = load_any(p1)
    n2, e2 = load_any(p2)
    G1 = build_graph(n1, e1)
    G2 = build_graph(n2, e2)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    draw(ax1, G1, f"Reference: {p1.name}")
    draw(ax2, G2, f"Learnt: {p2.name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()