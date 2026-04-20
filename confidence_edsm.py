import sys
import time
import json
import math
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
    """
    Simple deterministic automaton used during learning and evaluation.

    Args:
        start: Start state id.
        delta: Transition map (state, label) -> next state.

    Methods:
        accepts_path(seq): Return True if the full sequence can be followed.
        states(): Return the set of reachable state ids appearing in delta/start.
        outgoing_labels(s): Return all labels enabled from state s.
    """
    start: int
    delta: Dict[Tuple[int, str], int]

    def accepts_path(self, seq: Seq) -> bool:
        """
        Check path-existence acceptance under the project trace semantics.

        Args:
            seq: Input label sequence.

        Returns:
            True if every symbol in seq can be followed from the start state,
            otherwise False.
        """
        s = self.start
        for a in seq:
            s = self.delta.get((s, a))
            if s is None:
                return False
        return True

    def states(self) -> Set[int]:
        """
        Collect all state ids present in the DFA.

        Returns:
            A set containing the start state and every source/target state that
            appears in the transition map.
        """
        st = {self.start}
        for (u, _), v in self.delta.items():
            st.add(u)
            st.add(v)
        return st

    def outgoing_labels(self, s: int) -> Set[str]:
        return {a for (u, a) in self.delta.keys() if u == s}


@dataclass
class LearnResult:
    """
    Bundle returned by the confidence-aware EDSM learner.

    Args:
        model: Learnt DFA after blue-fringe learning and greedy post-merge.
        alphabet: Alphabet inferred from training data / forbidden labels.
        state_conf: Confidence assigned to each final learnt state.
        merge_history: Log of accepted merge operations with scores/confidence.
    """
    model: DFA
    alphabet: List[str]
    state_conf: Dict[int, float]
    merge_history: List[dict]


@dataclass
class Metrics:
    """
    Confusion-matrix style evaluation summary.

    Args:
        tp: Positive traces accepted by the model.
        tn: Negative traces rejected by the model.
        fp: Negative traces incorrectly accepted by the model.
        fn: Positive traces incorrectly rejected by the model.
    """
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def bcr(self) -> float:
        tpr = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        tnr = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
        return 0.5 * (tpr + tnr)


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
    delta: Dict[Tuple[int, str], int] = {}
    forbidden: Dict[int, Set[str]] = {}
    next_id = 0
    start = next_id
    next_id += 1
    forbidden[start] = set()

    for seq in pos:
        s = start
        for a in seq:
            key = (s, a)
            if key not in delta:
                delta[key] = next_id
                forbidden[next_id] = set()
                next_id += 1
            s = delta[key]

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


def merge_confidence_from_score(score: float) -> float:
    """
    Convert a merge score into a confidence value.

    Args:
        score: Merge evidence score.

    Returns:
        0 for negative scores, otherwise 1 - 0.5**score.
    """
    if score < 0:
        return 0.0
    return 1.0 - (0.5 ** score)


def normalize_state_conf(dfa: DFA, state_conf: Dict[int, float]) -> Dict[int, float]:
    """
    Clamp stored state-confidence values to [0, 1] and fill missing states.

    Args:
        dfa: Final DFA whose states should be covered.
        state_conf: Partial or unnormalized confidence map.

    Returns:
        A normalized confidence map containing every DFA state.
    """
    out = {}
    for s in dfa.states():
        x = state_conf.get(s, 1.0)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        out[s] = x
    return out


def merge_states_with_forbidden(
    dfa: DFA,
    forbidden: Dict[int, Set[str]],
    p: int,
    q: int
) -> Tuple[DFA, Dict[int, Set[str]], Dict[int, int]]:
    """
    Merge q into p.
    Returns:
      merged_dfa,
      merged_forbidden,
      rep_map: old_state -> representative_state_after_merge
    """
    if p == q:
        rep_map = {s: s for s in dfa.states()}
        return dfa, forbidden, rep_map

    new_delta = dict(dfa.delta)
    new_forbidden = {k: set(v) for k, v in forbidden.items()}
    start = dfa.start

    old_states = set(dfa.states())
    rep_map: Dict[int, int] = {s: s for s in old_states}
    rep_map[q] = p

    for (u, a), v in list(new_delta.items()):
        if v == q:
            new_delta[(u, a)] = p
    if start == q:
        start = p

    new_forbidden.setdefault(p, set())
    new_forbidden.setdefault(q, set())
    new_forbidden[p] |= new_forbidden[q]
    if q in new_forbidden:
        del new_forbidden[q]

    def outgoing(s: int) -> Dict[str, int]:
        return {a: t for (u, a), t in new_delta.items() if u == s}

    out_p = outgoing(p)
    out_q = outgoing(q)

    for a in list(out_q.keys()):
        new_delta.pop((q, a), None)

    work: List[Tuple[int, int]] = []
    for a, tq in out_q.items():
        if a not in out_p:
            new_delta[(p, a)] = tq
        else:
            tp = out_p[a]
            if tp != tq:
                work.append((tp, tq))

    merged = DFA(start=start, delta=new_delta)

    while work:
        x, y = work.pop()
        if x == y:
            continue
        merged, new_forbidden, inner_rep = merge_states_with_forbidden(merged, new_forbidden, x, y)
        for k, v in list(rep_map.items()):
            v2 = inner_rep.get(v, v)
            rep_map[k] = v2

    current_states = merged.states()
    for s in list(rep_map.keys()):
        while rep_map[s] in rep_map and rep_map[rep_map[s]] != rep_map[s]:
            rep_map[s] = rep_map[rep_map[s]]
        if rep_map[s] not in current_states and p in current_states:
            rep_map[s] = p

    return merged, new_forbidden, rep_map


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
    Simulate a candidate merge and estimate its evidence score.

    The simulation merges components symbolically, propagates deterministic
    constraints along common labels, and rejects merges that would enable a
    forbidden label inside a merged component.

    Args:
        dfa: Current DFA.
        forbidden: Forbidden-label evidence map.
        p: First candidate state.
        q: Second candidate state.

    Returns:
        A pair (ok, score):
            - ok is False if the merge is incompatible
            - score is the simulated evidence score when compatible
    """
    dsu = DSU()
    work = deque([(p, q)])
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

        fa = comp_forbidden(ra)
        fb = comp_forbidden(rb)
        f_union = fa | fb

        oa = comp_out(ra)
        ob = comp_out(rb)
        enabled_union = set(oa.keys()) | set(ob.keys())

        if f_union & enabled_union:
            return (False, -1.0)

        common = set(oa.keys()) & set(ob.keys())
        new_rep = union(ra, rb)

        for lab in common:
            ta = oa[lab]
            tb = ob[lab]

            la = sorted(ta)
            lb = sorted(tb)

            score += 1.0

            for xa in la:
                for xb in lb:
                    rxa, rxb = find(xa), find(xb)
                    if rxa != rxb:
                        work.append((rxa, rxb))

        o_new = comp_out(new_rep)
        if comp_forbidden(new_rep) & set(o_new.keys()):
            return (False, -1.0)

    return (True, score)


def update_conf_after_merge(
    merged_dfa: DFA,
    old_state_conf: Dict[int, float],
    rep_map: Dict[int, int],
    merge_conf: float,
) -> Dict[int, float]:
    """
    Update state confidence after a successful merge.

    Confidence is propagated with a weakest-link rule: the new representative
    receives the minimum of all contributing old-state confidences and the
    current merge confidence.

    Args:
        merged_dfa: DFA after the merge.
        old_state_conf: Confidence map before the merge.
        rep_map: Mapping from pre-merge states to post-merge representatives.
        merge_conf: Confidence of the current merge.

    Returns:
        Updated normalized state-confidence map for the merged DFA.
    """
    new_conf: Dict[int, float] = {}

    groups: Dict[int, List[int]] = {}
    for old_s, rep in rep_map.items():
        if rep in merged_dfa.states():
            groups.setdefault(rep, []).append(old_s)

    for rep, olds in groups.items():
        vals = [old_state_conf.get(s, 1.0) for s in olds]
        vals.append(merge_conf)
        new_conf[rep] = min(vals)

    for s in merged_dfa.states():
        new_conf.setdefault(s, old_state_conf.get(s, 1.0))

    return normalize_state_conf(merged_dfa, new_conf)


def greedy_post_merge(
    dfa: DFA,
    forbidden: Dict[int, Set[str]],
    neg: List[Seq],
    state_conf: Optional[Dict[int, float]] = None,
    merge_history: Optional[List[dict]] = None,
) -> Tuple[DFA, Dict[int, Set[str]], Dict[int, float], List[dict]]:
    if state_conf is None:
        state_conf = {s: 1.0 for s in dfa.states()}
    if merge_history is None:
        merge_history = []

    changed = True
    while changed:
        changed = False
        states = sorted(dfa.states())
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p, q = states[i], states[j]

                ok, sc = merge_score_simulate(dfa, forbidden, p, q)
                if not ok:
                    continue

                merged, merged_forbidden, rep_map = merge_states_with_forbidden(dfa, forbidden, p, q)
                if compatible_by_negatives(merged, neg):
                    c_merge = merge_confidence_from_score(sc)
                    state_conf = update_conf_after_merge(merged, state_conf, rep_map, c_merge)

                    merge_history.append({
                        "phase": "greedy_post_merge",
                        "merge_to": p,
                        "merge_from": q,
                        "score": sc,
                        "confidence": c_merge,
                        "states_before": len(dfa.states()),
                        "states_after": len(merged.states()),
                    })

                    dfa, forbidden = merged, merged_forbidden
                    changed = True
                    break
            if changed:
                break

    return dfa, forbidden, normalize_state_conf(dfa, state_conf), merge_history


def learn_edsm_bluefringe(
    pos: List[Seq],
    neg: List[Seq],
    score_threshold: float = 0.0,
    log_every_seconds: float = 2.0,
    max_merges: int = 50000
) -> LearnResult:
    """
    Learn a DFA with confidence-aware blue-fringe EDSM plus greedy post-merge.

    The learner builds a PTA from positive traces, uses negatives as forbidden
    evidence, repeatedly chooses the best compatible RED/BLUE merge, records
    merge confidence, and finally applies a greedy cleanup pass.

    Args:
        pos: Positive training traces.
        neg: Negative training traces.
        score_threshold: Minimum simulated merge score to consider.
        log_every_seconds: Progress-log interval in seconds.
        max_merges: Hard cap on accepted blue-fringe merges.

    Returns:
        LearnResult containing the learnt DFA, inferred alphabet, state
        confidence map, and merge history.
    """
    dfa, forbidden, alphabet = build_pta_with_forbidden(pos, neg)

    if not compatible_by_negatives(dfa, neg):
        raise ValueError("Initial PTA accepts some negatives (inconsistent data).")

    state_conf: Dict[int, float] = {s: 1.0 for s in dfa.states()}
    merge_history: List[dict] = []

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

                merged, merged_forbidden, rep_map = merge_states_with_forbidden(dfa, forbidden, r, b)
                if not compatible_by_negatives(merged, neg):
                    continue

                shrink = len(dfa.states()) - len(merged.states())
                cand = (sc, shrink, -r, -b)

                if best_for_b is None or cand > best_for_b_score:
                    best_for_b_score = cand
                    best_for_b = (r, b, merged, merged_forbidden, rep_map, sc)

            if best_for_b is None:
                promote.add(b)
            else:
                if best_pair is None or best_for_b_score > best_score:
                    best_score = best_for_b_score
                    best_pair = best_for_b

        if best_pair is None:
            if not promote:
                promote.add(min(BLUE))
            RED |= promote
            BLUE = successors_of(RED, dfa) - RED
        else:
            r, b, merged, merged_forbidden, rep_map, sc = best_pair
            c_merge = merge_confidence_from_score(sc)
            state_conf = update_conf_after_merge(merged, state_conf, rep_map, c_merge)

            merge_history.append({
                "phase": "bluefringe",
                "merge_to": r,
                "merge_from": b,
                "score": sc,
                "confidence": c_merge,
                "states_before": len(dfa.states()),
                "states_after": len(merged.states()),
            })

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
            print(
                f"[round={rounds} merges={merges}] "
                f"states={len(dfa.states())} RED={len(RED)} BLUE={len(BLUE)} "
                f"best_score={best_sc_display:.3f}"
            )

    print(f"[done] rounds={rounds} merges={merges} states={len(dfa.states())}")

    dfa, forbidden, state_conf, merge_history = greedy_post_merge(
        dfa, forbidden, neg, state_conf=state_conf, merge_history=merge_history
    )

    return LearnResult(
        model=dfa,
        alphabet=alphabet,
        state_conf=normalize_state_conf(dfa, state_conf),
        merge_history=merge_history,
    )


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


def one_prefix_per_state(dfa: DFA) -> Dict[int, List[str]]:
    """
    Find one representative prefix from the start state to each DFA state.

    The prefixes are built by a BFS-style traversal and are later used for
    confidence ranking and local refinement.

    Args:
        dfa: Learnt DFA.

    Returns:
        Mapping state -> one input sequence that reaches that state.
    """
    prefix = {dfa.start: []}
    q = deque([dfa.start])

    while q:
        s = q.popleft()
        seq = prefix[s]
        outgoing = [(a, v) for (u, a), v in dfa.delta.items() if u == s]
        outgoing.sort(key=lambda x: (x[0], x[1]))

        for a, v in outgoing:
            if v not in prefix:
                prefix[v] = seq + [a]
                q.append(v)

    return prefix


def trace_confidence(dfa: DFA, state_conf: Dict[int, float], seq: List[str]) -> float:
    """
    Compute confidence of a trace/prefix using the weakest state on its path.

    Args:
        dfa: Learnt DFA.
        state_conf: Confidence assigned to final DFA states.
        seq: Input trace or prefix.

    Returns:
        0.0 if the trace cannot be followed in the DFA; otherwise the minimum
        confidence of all states visited along the path.
    """
    s = dfa.start
    confs = [state_conf.get(s, 1.0)]
    for a in seq:
        s = dfa.delta.get((s, a))
        if s is None:
            return 0.0
        confs.append(state_conf.get(s, 1.0))
    return min(confs) if confs else 0.0


def save_learnt(
    model: DFA,
    alphabet: List[str],
    out_json: Path,
    out_dot: Path,
    state_conf: Optional[Dict[int, float]] = None,
    merge_history: Optional[List[dict]] = None,
    confidence_json: Optional[Path] = None,
) -> None:
    """
    Save the learnt DFA and optional confidence diagnostics to disk.

    Args:
        model: Learnt DFA to save.
        alphabet: Alphabet used by the model.
        out_json: Output path for automaton JSON.
        out_dot: Output path for automaton DOT graph.
        state_conf: Optional state-confidence map.
        merge_history: Optional merge log.
        confidence_json: Optional output path for confidence diagnostics JSON.
    """
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

    if confidence_json is not None:
        prefixes = one_prefix_per_state(model)
        conf_obj = {
            "state_conf": {str(k): float(v) for k, v in sorted((state_conf or {}).items())},
            "prefix_conf": {
                str(s): {
                    "prefix": prefixes.get(s, []),
                    "confidence": trace_confidence(model, state_conf or {}, prefixes.get(s, []))
                }
                for s in sorted(model.states())
            },
            "merge_history": merge_history or [],
        }
        confidence_json.write_text(json.dumps(conf_obj, indent=2), encoding="utf-8")


# def main():
#     if len(sys.argv) != 5:
#         print("Usage: python confidence_edsm.py <TRAIN_DIR> <EVAL_DIR> <LEARN_DIR> <OUT_CSV>")
#         print("Example: python confidence_edsm.py ./train ./test ./learning_E0 ./outcome_E0.csv")
#         raise SystemExit(2)

#     train_dir = Path(sys.argv[1])
#     eval_dir = Path(sys.argv[2])
#     learn_dir = Path(sys.argv[3])
#     out_csv = Path(sys.argv[4])

#     if not train_dir.is_dir():
#         raise FileNotFoundError(train_dir)
#     if not eval_dir.is_dir():
#         raise FileNotFoundError(eval_dir)

#     learn_dir.mkdir(parents=True, exist_ok=True)
#     out_csv.parent.mkdir(parents=True, exist_ok=True)

#     train_files = sorted(train_dir.glob("*.txt"))
#     if not train_files:
#         raise FileNotFoundError(f"No .txt files in {train_dir}")

#     headers = [
#         "file", "TP", "TN", "FP", "FN", "BCR", "N",
#         "train_pos", "train_neg", "eval_pos", "eval_neg",
#         "seconds", "min_state_conf", "avg_state_conf"
#     ]
#     rows = []

#     start_all = time.time()
#     for idx, train_file in enumerate(train_files, start=1):
#         eval_file = eval_dir / train_file.name
#         if not eval_file.exists():
#             print(f"[SKIP {idx}/{len(train_files)}] no eval file for {train_file.name}")
#             continue

#         t0 = time.time()
#         pos_tr, neg_tr = parse_traces_txt(train_file)
#         pos_ev, neg_ev = parse_traces_txt(eval_file)

#         print(f"\n[{idx}/{len(train_files)}] Learning: {train_file.name}")
#         result = learn_edsm_bluefringe(pos_tr, neg_tr, score_threshold=0.0)

#         model = result.model
#         alphabet = result.alphabet
#         state_conf = result.state_conf
#         merge_history = result.merge_history

#         stem = train_file.stem
#         save_learnt(
#             model,
#             alphabet,
#             learn_dir / f"learnt-{stem}.json",
#             learn_dir / f"learnt-{stem}.dot",
#             state_conf=state_conf,
#             merge_history=merge_history,
#             # confidence_json=learn_dir / f"confidence-{stem}.json",
#         )

#         m = eval_on_traces(model, pos_ev, neg_ev)
#         elapsed = time.time() - t0
#         N = m.tp + m.tn + m.fp + m.fn

#         conf_vals = list(state_conf.values()) if state_conf else [1.0]
#         min_conf = min(conf_vals) if conf_vals else 1.0
#         avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 1.0

#         print(
#             f"[DONE {idx}/{len(train_files)}] "
#             f"BCR={m.bcr():.3f} TP={m.tp} TN={m.tn} FP={m.fp} FN={m.fn} "
#             f"(N={N}, {elapsed:.1f}s, min_conf={min_conf:.3f}, avg_conf={avg_conf:.3f})"
#         )

#         rows.append([
#             train_file.name, m.tp, m.tn, m.fp, m.fn, f"{m.bcr():.6f}", N,
#             len(pos_tr), len(neg_tr), len(pos_ev), len(neg_ev), f"{elapsed:.3f}",
#             f"{min_conf:.6f}", f"{avg_conf:.6f}"
#         ])

#     with out_csv.open("w", encoding="utf-8") as f:
#         f.write(",".join(headers) + "\n")
#         for r in rows:
#             f.write(",".join(map(str, r)) + "\n")

#     print(f"\nAll done. CSV: {out_csv}  Learnt dir: {learn_dir}  Total time: {time.time()-start_all:.1f}s")


# if __name__ == "__main__":
#     main()