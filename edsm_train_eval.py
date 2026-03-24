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
    More faithful Blue-Fringe/EDSM-style compatibility simulation.

    Key idea:
    - simulate component merges, not just single-state merges
    - after union, outgoing labels of a component are the UNION of all members' outgoing
    - if two merged components both have transition on same label, their targets must also merge
    - if a forbidden label becomes enabled anywhere in a merged component, reject
    - score prefers merges that propagate consistently through more of the PTA
    """
    dsu = DSU()
    work = deque([(p, q)])

    # members of each current component representative
    members: Dict[int, Set[int]] = {}

    def find(x: int) -> int:
        r = dsu.find(x)
        members.setdefault(r, {r})
        return r

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        new_rep = dsu.union(ra, rb)
        if new_rep == ra:
            members.setdefault(ra, {ra})
            members.setdefault(rb, {rb})
            members[ra] |= members[rb]
            del members[rb]
            return ra
        else:
            members.setdefault(ra, {ra})
            members.setdefault(rb, {rb})
            members[rb] |= members[ra]
            del members[ra]
            return rb

    # cache original outgoing per state
    out0: Dict[int, Dict[str, int]] = {}
    for (u, a), v in dfa.delta.items():
        out0.setdefault(u, {})[a] = v

    def comp_members(rep: int) -> Set[int]:
        rep = find(rep)
        return members.setdefault(rep, {rep})

    def comp_forbidden(rep: int) -> Set[str]:
        fs: Set[str] = set()
        for s in comp_members(rep):
            fs |= forbidden.get(s, set())
        return fs

    def comp_out(rep: int) -> Dict[str, Set[int]]:
        """
        merged component outgoing:
        label -> set of target component reps reachable by that label
        """
        rep = find(rep)
        out_map: Dict[str, Set[int]] = {}
        for s in comp_members(rep):
            for lab, t in out0.get(s, {}).items():
                out_map.setdefault(lab, set()).add(find(t))
        return out_map

    score = 0.0
    seen_pairs: Set[Tuple[int, int]] = set()

    while work:
        a, b = work.popleft()
        ra, rb = find(a), find(b)
        if ra == rb:
            continue

        pair = (min(ra, rb), max(ra, rb))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Check forbidden-vs-enabled on the merged component
        fa = comp_forbidden(ra)
        fb = comp_forbidden(rb)
        f_union = fa | fb

        oa = comp_out(ra)
        ob = comp_out(rb)
        enabled_union = set(oa.keys()) | set(ob.keys())

        if f_union & enabled_union:
            return (False, -1.0)

        # Labels present on both sides force target merges for determinism
        common = set(oa.keys()) & set(ob.keys())

        # reward: merging larger compatible structures
        size_a = len(comp_members(ra))
        size_b = len(comp_members(rb))
        # score += 1.0 + 0.2 * (size_a + size_b - 2)

        new_rep = union(ra, rb)

        for lab in common:
            ta = oa[lab]
            tb = ob[lab]

            # if multiple distinct target components remain on same label after merge, unify them pairwise
            la = sorted(ta)
            lb = sorted(tb)

            # reward matched labels
            score += 1.0

            for xa in la:
                for xb in lb:
                    rxa, rxb = find(xa), find(xb)
                    if rxa != rxb:
                        work.append((rxa, rxb))

        # after union, check determinism again:
        # same merged source cannot keep a forbidden label enabled
        o_new = comp_out(new_rep)
        if comp_forbidden(new_rep) & set(o_new.keys()):
            return (False, -1.0)

    return (True, score)


def learn_edsm_bluefringe(
    pos: List[Seq],
    neg: List[Seq],
    score_threshold: float = 0.0,
    log_every_seconds: float = 2.0,
    max_merges: int = 50000
) -> Tuple[DFA, List[str]]:
    dfa, forbidden, alphabet = build_pta_with_forbidden(pos, neg)

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
        best_score = None
        promote: Set[int] = set()

        for b in sorted(BLUE):
            best_for_b = None
            best_for_b_score = None

            for r in sorted(RED):
                ok, sc = merge_score_simulate(dfa, forbidden, r, b)
                if not ok:
                    continue
                if sc < score_threshold:
                    continue

                merged, merged_forbidden = merge_states_with_forbidden(dfa, forbidden, r, b)
                if not compatible_by_negatives(merged, neg):
                    continue

                shrink = len(dfa.states()) - len(merged.states())
                cand = (sc, shrink, -r, -b)

                if best_for_b is None or cand > best_for_b_score:
                    best_for_b_score = cand
                    best_for_b = (r, b, merged, merged_forbidden)

            if best_for_b is None:
                promote.add(b)
            else:
                if best_pair is None or best_for_b_score > best_score:
                    best_score = best_for_b_score
                    best_pair = best_for_b

        if best_pair is None:
            if not promote:
                promote.add(next(iter(BLUE)))
            RED |= promote
            BLUE = successors_of(RED, dfa) - RED
        else:
            r, b, merged, merged_forbidden = best_pair
            dfa, forbidden = merged, merged_forbidden
            merges += 1
            RED = {s for s in RED if s in dfa.states()}
            if r in dfa.states():
                RED.add(r)
            BLUE = successors_of(RED, dfa) - RED

        now = time.time()
        if now - last_log >= log_every_seconds:
            last_log = now
            best_sc_display = best_score[0] if best_score is not None else -1.0
            print(f"[round={rounds} merges={merges}] states={len(dfa.states())} RED={len(RED)} BLUE={len(BLUE)} best_score={best_sc_display:.3f}")

    print(f"[done] rounds={rounds} merges={merges} states={len(dfa.states())}")
    dfa, forbidden = greedy_post_merge(dfa, forbidden, neg)
    return dfa, alphabet

def greedy_post_merge(dfa: DFA, forbidden: Dict[int, Set[str]], neg: List[Seq]) -> Tuple[DFA, Dict[int, Set[str]]]:
    changed = True
    while changed:
        changed = False
        states = sorted(dfa.states())
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p, q = states[i], states[j]
                merged, merged_forbidden = merge_states_with_forbidden(dfa, forbidden, p, q)
                if compatible_by_negatives(merged, neg):
                    dfa, forbidden = merged, merged_forbidden
                    changed = True
                    break
            if changed:
                break
    return dfa, forbidden


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

def save_learnt(model: DFA, alphabet: List[str], out_json: Path, out_dot: Path) -> None:
    st = sorted(model.states())
    n_states = (max(st) + 1) if st else 1
    obj = {
        "n_states": n_states,
        "alphabet": alphabet,
        "start": model.start,
        "graph_density": -1,
        "delta": [{"source": u, "label": a, "target": v} for (u, a), v in model.delta.items()],
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

    headers = ["file","TP","TN","FP","FN","BCR","N","train_pos","train_neg","eval_pos","eval_neg","seconds"]
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
        # model = learn_edsm_redblue(pos_tr, neg_tr, score_threshold=0.0)
        model, alphabet = learn_edsm_bluefringe(pos_tr, neg_tr, score_threshold=0.0)

        stem = train_file.stem
        save_learnt(model, alphabet, learn_dir / f"learnt-{stem}.json", learn_dir / f"learnt-{stem}.dot")

        m = eval_on_traces(model, pos_ev, neg_ev)
        elapsed = time.time() - t0
        N = m.tp + m.tn + m.fp + m.fn

        print(f"[DONE {idx}/{len(train_files)}] BCR={m.bcr():.3f} TP={m.tp} TN={m.tn} FP={m.fp} FN={m.fn} (N={N}, {elapsed:.1f}s)")

        rows.append([
            train_file.name, m.tp, m.tn, m.fp, m.fn, f"{m.bcr():.6f}", N,
            len(pos_tr), len(neg_tr), len(pos_ev), len(neg_ev), f"{elapsed:.3f}"
        ])

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"\nAll done. CSV: {out_csv}  Learnt dir: {learn_dir}  Total time: {time.time()-start_all:.1f}s")

if __name__ == "__main__":
    main()