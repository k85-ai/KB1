import sys
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from common import Automaton, write_dot

def reachable_states(start: int, delta: dict) -> set:
    """Return all states reachable from start."""
    seen = {start}
    stack = [start]
    while stack:
        s = stack.pop()
        for (u, a), v in delta.items():
            if u == s and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen
def generate_connected_sparse_dfa(
    n_states: int,
    alphabet_size: int,
    graph_density: int,
    seed: int,
) -> Automaton:
    """
    Generate automata suitable for random-walk trace generation.

    Design goals:
    - all states reachable from start
    - sparse but not tree-like
    - for tiny alphabets (especially size 2), do NOT fill every state with all labels
      so that negative traces can still be generated
    """
    rnd = random.Random(seed)
    alphabet = [f"L{i}" for i in range(alphabet_size)]
    start = 0
    delta: Dict[Tuple[int, str], int] = {}

    # -------------------------------------------------
    # Case 1: tiny alphabet (2 labels)
    # -------------------------------------------------
    if alphabet_size <= 2:
        # 1) build a cycle using only L0
        #    guarantees reachability and keeps one label free in many states
        base_label = alphabet[0]
        for u in range(n_states):
            v = (u + 1) % n_states
            delta[(u, base_label)] = v

        # 2) randomly choose only SOME states to receive a second outgoing edge (L1)
        #    this keeps some states missing L1, so negative walks can be generated
        extra_label = alphabet[1] if alphabet_size > 1 else alphabet[0]

        # choose how many states get the second label
        # not all states usually about half to three-quarters
        num_extra = max(1, int(0.6 * n_states))
        chosen_states = list(range(n_states))
        rnd.shuffle(chosen_states)
        chosen_states = chosen_states[:num_extra]

        for u in chosen_states:
            # prefer target different from self and different from L0 target
            l0_target = delta[(u, base_label)]
            candidates = [x for x in range(n_states) if x != u and x != l0_target]
            if not candidates:
                candidates = [x for x in range(n_states) if x != l0_target]
            if not candidates:
                candidates = list(range(n_states))
            v = rnd.choice(candidates)
            delta[(u, extra_label)] = v

        return Automaton(n_states=n_states, alphabet=alphabet, start=start,delta=delta)
    
    # -------------------------------------------------
    # Case 2: larger alphabet
    # -------------------------------------------------
    # build a cycle first
    for u in range(n_states):
        v = (u + 1) % n_states
        a = alphabet[u % alphabet_size]
        delta[(u, a)] = v

    # target total edges ~ density * states
    max_edges = n_states * alphabet_size
    target_edges = int(graph_density * n_states)
    jitter = max(1, int(0.25 * target_edges))
    target_edges += rnd.randint(-jitter, jitter)
    target_edges = max(n_states, target_edges)
    target_edges = min(target_edges, max_edges)

    attempts = 0
    while len(delta) < target_edges and attempts < 200000:
        attempts += 1
        u = rnd.randrange(0, n_states)

        used_labels = {a for (uu, a) in delta.keys() if uu == u}
        avail_labels = [a for a in alphabet if a not in used_labels]
        if not avail_labels:
            continue

        a = rnd.choice(avail_labels)

        existing_targets = {v for (uu, _), v in delta.items() if uu == u}
        candidates = [x for x in range(n_states) if x != u and x not in existing_targets]
        if not candidates:
            candidates = [x for x in range(n_states) if x != u]
        if not candidates:
            candidates = list(range(n_states))

        v = rnd.choice(candidates)
        delta[(u, a)] = v

    # ensure each state has at least 2 outgoing edges if possible
    min_out = min(2, alphabet_size)
    for u in range(n_states):
        while True:
            used_labels = {a for (uu, a) in delta.keys() if uu == u}
            if len(used_labels) >= min_out:
                break

            avail_labels = [a for a in alphabet if a not in used_labels]
            if not avail_labels:
                break

            a = rnd.choice(avail_labels)
            existing_targets = {v for (uu, _), v in delta.items() if uu == u}
            candidates = [x for x in range(n_states) if x != u and x not in existing_targets]
            if not candidates:
                candidates = [x for x in range(n_states) if x != u]
            if not candidates:
                candidates = list(range(n_states))

            v = rnd.choice(candidates)
            delta[(u, a)] = v

    return Automaton(n_states=n_states, alphabet=alphabet, start=start,delta=delta)


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_automata.py <OUT_DIR> <MAX_STATES>")
        print("Example: python generate_automata.py Data/automata 20")
        raise SystemExit(2)

    out_dir = Path(sys.argv[1])
    max_states = int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_density = 2 
    n = 5
    per_size = 10

    while n <= max_states:
        alphabet_size = max(1, n // 2)
        for counter in range(per_size):
            name = f"automaton_{n}_{graph_density}_{alphabet_size}_{counter}"
            seed = (n * 1000) + counter
            A = generate_connected_sparse_dfa(n, alphabet_size, graph_density, seed)

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
