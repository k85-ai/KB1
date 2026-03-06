import sys
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from common import Automaton, write_dot

def generate_connected_sparse_dfa_java_like(n_states: int, alphabet_size: int, graph_density: int, seed: int) -> Automaton:
    """
    Java-like: reachable + global sparse + deterministic, edge count fluctuates.
    """
    rnd = random.Random(seed)
    alphabet = [f"L{i}" for i in range(alphabet_size)]
    start = 0
    delta: Dict[Tuple[int, str], int] = {}

    # spanning tree
    for s in range(1, n_states):
        parent = rnd.randrange(0, s)
        labels = alphabet[:]
        rnd.shuffle(labels)
        chosen = None
        for a in labels:
            if (parent, a) not in delta:
                chosen = a
                break
        if chosen is None:
            chosen = rnd.choice(alphabet)
        delta[(parent, chosen)] = s

    # random target around density*n
    max_edges = n_states * alphabet_size
    min_edges = n_states - 1
    base = graph_density * n_states
    jitter = max(1, int(0.35 * base))
    target = int(base + rnd.randint(-jitter, jitter))
    target = max(min_edges, target)
    if alphabet_size <= 2:
        target = min(target, max_edges - 2)  # avoid complete for tiny alphabet
    else:
        target = min(target, max_edges)

    # fill random edges
    attempts = 0
    while len(delta) < target and attempts < 200000:
        attempts += 1
        u = rnd.randrange(0, n_states)
        used = {a for (uu, a) in delta.keys() if uu == u}
        avail = [a for a in alphabet if a not in used]
        if not avail:
            continue
        a = rnd.choice(avail)
        v = rnd.randrange(0, n_states)
        delta[(u, a)] = v

    return Automaton(n_states=n_states, alphabet=alphabet, start=start, delta=delta)


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_automata.py <OUT_DIR> <MAX_STATES>")
        print("Example: python generate_automata.py Data/automata 20")
        raise SystemExit(2)

    out_dir = Path(sys.argv[1])
    max_states = int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_density = 2  # same as Java examples
    n = 5
    per_size = 10

    while n <= max_states:
        alphabet_size = max(1, n // 2)
        for counter in range(per_size):
            name = f"automaton_{n}_{graph_density}_{alphabet_size}_{counter}"
            seed = (n * 1000) + counter
            A = generate_connected_sparse_dfa_java_like(n, alphabet_size, graph_density, seed)

            (out_dir / f"{name}.json").write_text(
                json.dumps(A.to_json_obj(graph_density=graph_density), indent=2),
                encoding="utf-8"
            )
            nodes = sorted(A.states())
            edges = [(u, v, a) for (u, a), v in A.delta.items()]
            write_dot(out_dir / f"{name}.dot", nodes, edges, title="dotMachine")
        n *= 2

    print(f"Done. Automata written to: {out_dir}")

if __name__ == "__main__":
    main()