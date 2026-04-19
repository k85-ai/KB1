import sys
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import deque

from common import parse_traces_txt, Automaton, write_dot

Symbol = str
Seq = List[Symbol]
State = int

@dataclass
class DFA:
    start: int
    delta: Dict[Tuple[int, str], int]

    def accepts_path(self, seq: Seq) -> bool:
        s = self.start
        for a in seq:
            s = self.delta.get((s, a))
            if s is None:
                return False
        return True

    def states(self) -> Set[int]:
        st = {self.start}
        for (u, _), v in self.delta.items():
            st.add(u); st.add(v)
        return st

    def outgoing_labels(self, s: int) -> Set[str]:
        return {a for (u, a) in self.delta.keys() if u == s}

def compatible_by_negatives(dfa: DFA, neg: List[Seq]) -> bool:
    for seq in neg:
        if dfa.accepts_path(seq):
            return False
    return True

def successors_of(states: Set[int], dfa: DFA) -> Set[int]:
    succ = set()
    for (u, _a), v in dfa.delta.items():
        if u in states:
            succ.add(v)
    return succ

def build_pta_with_forbidden(pos: List[Seq], neg: List[Seq]) -> Tuple[DFA, Dict[int, Set[str]], List[str]]:
    """
    Build PTA from positives, and collect 'forbidden labels' from negatives.

    For a negative trace: prefix + [bad_label]
      - follow prefix in PTA (create states if needed)
      - record that bad_label must NOT be enabled at the reached state
    """
    delta: Dict[Tuple[int, str], int] = {}
    forbidden: Dict[int, Set[str]] = {}
    next_id = 0
    start = next_id
    next_id += 1
    forbidden[start] = set()

    # Insert positives fully
    for seq in pos:
        s = start
        for a in seq:
            key = (s, a)
            if key not in delta:
                delta[key] = next_id
                forbidden[next_id] = set()
                next_id += 1
            s = delta[key]

    # Insert negative prefixes and mark forbidden last label
    for seq in neg:
        if not seq:
            continue
        prefix, last = seq[:-1], seq[-1]
        s = start
        for a in prefix:
            key = (s, a)
            if key not in delta:
                delta[key] = next_id
                forbidden[next_id] = set()
                next_id += 1
            s = delta[key]
        forbidden.setdefault(s, set()).add(last)

    alphabet = sorted({a for (_u, a) in delta.keys()} | {a for fs in forbidden.values() for a in fs})
    return DFA(start=start, delta=delta), forbidden, alphabet


def compatible_by_negatives(dfa: DFA, neg: List[Seq]) -> bool:
    """
    Same as your current semantics:
    learnt automaton must NOT accept any negative trace (i.e., path must not exist).
    """
    for seq in neg:
        if dfa.accepts_path(seq):
            return False
    return True


def merge_states_with_forbidden(dfa: DFA, forbidden: Dict[int, Set[str]], p: int, q: int) -> Tuple[DFA, Dict[int, Set[str]]]:
    """
    Merge q into p in the DFA, and merge forbidden-label evidence as well.
    If merge introduces forbidden-enabled conflict, we keep it detectable by simulation (caller rejects).
    """
    if p == q:
        return dfa, forbidden

    new_delta = dict(dfa.delta)
    new_forbidden = {k: set(v) for k, v in forbidden.items()}
    start = dfa.start

    # redirect incoming edges v==q -> p
    for (u, a), v in list(new_delta.items()):
        if v == q:
            new_delta[(u, a)] = p
    if start == q:
        start = p

    # merge forbidden sets
    new_forbidden.setdefault(p, set())
    new_forbidden.setdefault(q, set())
    new_forbidden[p] |= new_forbidden[q]
    if q in new_forbidden:
        del new_forbidden[q]

    def outgoing(s: int) -> Dict[str, int]:
        return {a: t for (u, a), t in new_delta.items() if u == s}

    out_p = outgoing(p)
    out_q = outgoing(q)

    # remove outgoing from q
    for a in list(out_q.keys()):
        new_delta.pop((q, a), None)

    # merge outgoing; conflicts trigger recursive merges
    work: List[Tuple[int, int]] = []
    for a, tq in out_q.items():
        if a not in out_p:
            # keep edge unless it violates forbidden evidence; we still add it and let simulation reject merges
            new_delta[(p, a)] = tq
        else:
            tp = out_p[a]
            if tp != tq:
                work.append((tp, tq))

    merged = DFA(start=start, delta=new_delta)

    while work:
        x, y = work.pop()
        if x != y:
            merged, new_forbidden = merge_states_with_forbidden(merged, new_forbidden, x, y)

    return merged, new_forbidden


class DSU:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        return p

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra


def merge_score_simulate(dfa: DFA, forbidden: Dict[int, Set[str]], p: int, q: int) -> Tuple[bool, float]:
    """
    Evidence-driven scoring:
      - Simulate merging p and q with propagation:
        If both states have outgoing on the same label, their targets must also be merged.
      - Evidence score counts:
        +1 per successfully unified state-pair
        +1 per matched outgoing label that forces propagation
      - Negative evidence constraint:
        If any forbidden label becomes enabled in a merged component, reject.
    """
    dsu = DSU()
    work = deque([(p, q)])

    out_cache: Dict[int, Dict[str, int]] = {}
    def out(s: int) -> Dict[str, int]:
        if s not in out_cache:
            out_cache[s] = {a: t for (u, a), t in dfa.delta.items() if u == s}
        return out_cache[s]

    comp_forbidden: Dict[int, Set[str]] = {}
    score = 0.0

    while work:
        a, b = work.popleft()
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue

        fa = comp_forbidden.get(ra, set(forbidden.get(ra, set())))
        fb = comp_forbidden.get(rb, set(forbidden.get(rb, set())))
        f_union = fa | fb

        oa = out(ra)
        ob = out(rb)
        enabled_union = set(oa.keys()) | set(ob.keys())

        # forbidden label cannot be enabled
        if f_union & enabled_union:
            return (False, -1.0)

        rep = dsu.union(ra, rb)
        comp_forbidden[rep] = f_union
        score += 1.0  # evidence: unified one pair

        # propagation evidence: matched labels force successor merges
        common = set(oa.keys()) & set(ob.keys())
        if common:
            score += float(len(common))
            for lab in common:
                work.append((oa[lab], ob[lab]))

    return (True, score)


def successors_of(states: Set[int], dfa: DFA) -> Set[int]:
    succ = set()
    for (u, _a), v in dfa.delta.items():
        if u in states:
            succ.add(v)
    return succ


def learn_edsm_bluefringe(
    pos: List[Seq],
    neg: List[Seq],
    score_threshold: float = 1.0,
    log_every_seconds: float = 2.0,
    max_merges: int = 50000,
) -> Tuple[DFA, List[str], "ConfidenceData"]:
    """
    Blue-Fringe (RED/BLUE) EDSM with evidence-driven scoring, adapted to your trace semantics.
    Returns:
      (learnt_dfa, alphabet, confidence)
    """
    from confidence import _MergeRecord, compute_confidence, ConfidenceData
    dfa, forbidden, alphabet = build_pta_with_forbidden(pos, neg)
    merge_log: List = []

    # Consistency check
    if not compatible_by_negatives(dfa, neg):
        raise ValueError("Initial PTA accepts some negatives (inconsistent data).")

    RED: Set[int] = {dfa.start}
    BLUE: Set[int] = successors_of(RED, dfa) - RED

    merges = 0
    rounds = 0
    last_log = time.time()

    while merges < max_merges and BLUE:
        rounds += 1

        best_pair = None
        best_score = -1.0
        promote: Set[int] = set()

        for b in list(BLUE):
            best_for_b = None
            best_for_b_score = -1.0

            for r in RED:
                ok, sc = merge_score_simulate(dfa, forbidden, r, b)
                if not ok:
                    continue
                if sc < score_threshold:
                    continue

                merged, merged_forbidden = merge_states_with_forbidden(dfa, forbidden, r, b)

                if not compatible_by_negatives(merged, neg):
                    continue

                if sc > best_for_b_score:
                    best_for_b_score = sc
                    best_for_b = (r, b)

            if best_for_b is None:
                promote.add(b)
            else:
                if best_for_b_score > best_score:
                    best_score = best_for_b_score
                    best_pair = best_for_b

        if best_pair is None:
            # no merge found, expand frontier
            if not promote:
                promote.add(next(iter(BLUE)))
            RED |= promote
            BLUE = successors_of(RED, dfa) - RED
        else:
            r, b = best_pair
            dfa, forbidden = merge_states_with_forbidden(dfa, forbidden, r, b)
            merge_log.append(_MergeRecord(red=r, blue=b, score=best_score))
            merges += 1
            BLUE = successors_of(RED, dfa) - RED

        now = time.time()
        if now - last_log >= log_every_seconds:
            last_log = now
            print(f"[round={rounds} merges={merges}] states={len(dfa.states())} RED={len(RED)} BLUE={len(BLUE)} best_score={best_score:.3f}")

    print(f"[done] rounds={rounds} merges={merges} states={len(dfa.states())}")
    return dfa, alphabet, compute_confidence(merge_log)


@dataclass
class Metrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    def bcr(self) -> float:
        tpr = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        tnr = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
        return 0.5 * (tpr + tnr)

def eval_on_traces(model: DFA, pos: List[Seq], neg: List[Seq]) -> Metrics:
    m = Metrics()
    for seq in pos:
        if model.accepts_path(seq):
            m.tp += 1
        else:
            m.fn += 1
    for seq in neg:
        if model.accepts_path(seq):
            m.fp += 1
        else:
            m.tn += 1
    return m

def save_learnt(model: DFA, alphabet: List[str], out_json: Path, out_dot: Path, conf=None) -> None:
    st = sorted(model.states())
    n_states = (max(st) + 1) if st else 1
    obj = {
        "n_states": n_states,
        "alphabet": alphabet,
        "start": model.start,
        "graph_density": -1,
        "delta": [{"source": u, "label": a, "target": v} for (u, a), v in model.delta.items()],
    }
    if conf is not None:
        obj["confidence"] = {
            "global_conf": conf.global_conf,
            "state_scores": {str(k): v for k, v in conf.state_scores.items()},
            "n_merges": conf.n_merges,
        }
    out_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    nodes = sorted(set(range(n_states)) & set(st))
    edges = [(u, v, a) for (u, a), v in model.delta.items()]
    write_dot(out_dot, nodes, edges, title="learntMachine")

def main():
    if len(sys.argv) != 5:
        print("Usage: python edsm_train_eval.py <TRAIN_DIR> <EVAL_DIR> <LEARN_DIR> <OUT_CSV>")
        print("Example: python edsm_train_eval.py Data/train_data Data/test_data Data/learning_E0 Data/outcome_E0.csv")
        raise SystemExit(2)

    train_dir = Path(sys.argv[1])
    eval_dir = Path(sys.argv[2])
    learn_dir = Path(sys.argv[3])
    out_csv = Path(sys.argv[4])

    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(eval_dir)

    learn_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_dir.glob("*.txt"))
    if not train_files:
        raise FileNotFoundError(f"No .txt files in {train_dir}")

    headers = ["file","TP","TN","FP","FN","BCR","N","train_pos","train_neg","eval_pos","eval_neg","seconds","global_conf","n_merges"]
    rows = []

    start_all = time.time()
    for idx, train_file in enumerate(train_files, start=1):
        eval_file = eval_dir / train_file.name
        if not eval_file.exists():
            print(f"[SKIP {idx}/{len(train_files)}] no eval file for {train_file.name}")
            continue

        t0 = time.time()
        pos_tr, neg_tr = parse_traces_txt(train_file)
        pos_ev, neg_ev = parse_traces_txt(eval_file)

        # infer alphabet from training data labels (robust)
        # alphabet = sorted({a for seq in (pos_tr + neg_tr) for a in seq})

        print(f"\n[{idx}/{len(train_files)}] Learning: {train_file.name}")
        model, alphabet, conf = learn_edsm_bluefringe(pos_tr, neg_tr, score_threshold=1.0)

        stem = train_file.stem
        save_learnt(model, alphabet, learn_dir / f"learnt-{stem}.json", learn_dir / f"learnt-{stem}.dot", conf)

        m = eval_on_traces(model, pos_ev, neg_ev)
        elapsed = time.time() - t0
        N = m.tp + m.tn + m.fp + m.fn

        print(f"[DONE {idx}/{len(train_files)}] BCR={m.bcr():.3f} TP={m.tp} TN={m.tn} FP={m.fp} FN={m.fn} (N={N}, {elapsed:.1f}s) global_conf={conf.global_conf:.6f} merges={conf.n_merges}")

        rows.append([
            train_file.name, m.tp, m.tn, m.fp, m.fn, f"{m.bcr():.6f}", N,
            len(pos_tr), len(neg_tr), len(pos_ev), len(neg_ev), f"{elapsed:.3f}",
            f"{conf.global_conf:.6f}", conf.n_merges,
        ])

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"\nAll done. CSV: {out_csv}  Learnt dir: {learn_dir}  Total time: {time.time()-start_all:.1f}s")

if __name__ == "__main__":
    main()