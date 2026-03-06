from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

Symbol = str
Seq = List[Symbol]
State = int

@dataclass
class Automaton:
    n_states: int
    alphabet: List[Symbol]
    start: State
    delta: Dict[Tuple[State, Symbol], State]

    def step(self, s: State, a: Symbol) -> Optional[State]:
        return self.delta.get((s, a))

    def enabled(self, s: State) -> List[Symbol]:
        return [a for (u, a) in self.delta.keys() if u == s]

    def states(self) -> Set[int]:
        st = {self.start}
        for (u, _a), v in self.delta.items():
            st.add(u); st.add(v)
        return st

    def accepts_path(self, seq: Seq) -> bool:
        """Trace semantics: True iff the path exists for the whole sequence."""
        s = self.start
        for a in seq:
            s = self.delta.get((s, a))
            if s is None:
                return False
        return True

    def to_json_obj(self, graph_density: int = -1) -> dict:
        return {
            "n_states": int(self.n_states),
            "alphabet": list(self.alphabet),
            "start": int(self.start),
            "graph_density": int(graph_density),
            "delta": [{"source": int(u), "label": str(a), "target": int(v)}
                      for (u, a), v in self.delta.items()],
        }

    @staticmethod
    def from_json(path: Path) -> "Automaton":
        obj = json.loads(path.read_text(encoding="utf-8"))
        n_states = int(obj["n_states"])
        alphabet = list(obj.get("alphabet", []))
        start = int(obj.get("start", 0))
        delta: Dict[Tuple[int, str], int] = {}
        for e in obj["delta"]:
            delta[(int(e["source"]), str(e["label"]))] = int(e["target"])
        if not alphabet:
            alphabet = sorted({a for (_u, a) in delta.keys()})
        return Automaton(n_states=n_states, alphabet=alphabet, start=start, delta=delta)


def write_dot(path: Path, nodes: List[int], edges: List[Tuple[int, int, str]], title: str = "dotMachine") -> None:
    def esc(s: str) -> str:
        return s.replace('"', '\\"')

    lines = [f"digraph {title} {{", "  rankdir=LR;"]
    for n in nodes:
        lines.append(f'  {n} [shape=circle, label="{n}"];')
    for u, v, a in edges:
        lines.append(f'  {u} -> {v} [label="{esc(a)}"];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_traces_txt(path: Path):
    pos, neg = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sign, rest = line[0], line[1:].strip()
        seq = [x for x in rest.split(",") if x] if rest else []
        if sign == "+":
            pos.append(seq)
        elif sign == "-":
            neg.append(seq)
        else:
            raise ValueError(f"Bad line in {path}: {line}")
    return pos, neg


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1


def stable_salt_from_path(path_str: str) -> int:
    """Stable 32-bit FNV-1a hash for separating train/test streams without extra args."""
    s = path_str.replace("\\", "/").lower().encode("utf-8")
    h = 2166136261
    for b in s:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)