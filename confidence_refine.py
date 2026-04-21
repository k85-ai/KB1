import sys,time
import shutil
import json
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict

from common import Automaton, parse_traces_txt
from confidence_edsm import (
    learn_edsm_bluefringe,
    eval_on_traces,
    one_prefix_per_state,
    trace_confidence,
    save_learnt
)


Seq = List[str]
import random

def run_prefix(A: Automaton, seq: Seq):
    """Run a prefix on the reference automaton and return the reached state, or None if invalid."""
    s = A.start
    for a in seq:
        s = A.step(s, a)
        if s is None:
            return None
    return s

def gen_positive_from_prefix(A: Automaton, prefix: Seq, rnd: random.Random, extra_steps: int = 1):
    """Generate a new positive trace by extending a valid prefix with legal transitions."""
    s = run_prefix(A, prefix)
    if s is None:
        return None

    seq = list(prefix)
    for _ in range(extra_steps):
        enabled = A.enabled(s)
        if not enabled:
            break
        a = rnd.choice(enabled)
        seq.append(a)
        s = A.step(s, a)
        if s is None:
            return None
    return seq

def gen_negative_from_prefix(A: Automaton, prefix: Seq, rnd: random.Random, extra_steps: int = 1):
    """Generate a new negative trace by extending a valid prefix and forcing an invalid final step."""
    s = run_prefix(A, prefix)
    if s is None:
        return None

    seq = list(prefix)
    for _ in range(extra_steps):
        enabled = A.enabled(s)
        if not enabled:
            break
        a = rnd.choice(enabled)
        seq.append(a)
        s = A.step(s, a)
        if s is None:
            return None

    enabled = set(A.enabled(s))
    missing = [a for a in A.alphabet if a not in enabled]
    if not missing:
        return None

    bad = rnd.choice(missing)
    return seq + [bad]


def append_traces_txt(path: Path, pos_traces: List[Seq], neg_traces: List[Seq]) -> None:
    """Append newly generated positive and negative traces to the working training file."""
    with path.open("a", encoding="utf-8") as f:
        for seq in pos_traces:
            f.write("+ " + ",".join(seq) + "\n")
        for seq in neg_traces:
            f.write("- " + ",".join(seq) + "\n")

def filter_candidate_traces(
    pos_new: List[Seq],
    neg_new: List[Seq],
    existing_pos: List[Seq],
    existing_neg: List[Seq],
) -> Tuple[List[Seq], List[Seq]]:
    """
    Remove duplicates against existing train data and remove pos/neg conflicts.
    """
    existing_pos_set = {tuple(x) for x in existing_pos}
    existing_neg_set = {tuple(x) for x in existing_neg}

    pos_new = [
        x for x in pos_new
        if tuple(x) not in existing_pos_set and tuple(x) not in existing_neg_set
    ]
    neg_new = [
        x for x in neg_new
        if tuple(x) not in existing_neg_set and tuple(x) not in existing_pos_set
    ]

    pos_new_set = {tuple(x) for x in pos_new}
    neg_new_set = {tuple(x) for x in neg_new}
    conflict_set = pos_new_set & neg_new_set

    if conflict_set:
        pos_new = [x for x in pos_new if tuple(x) not in conflict_set]
        neg_new = [x for x in neg_new if tuple(x) not in conflict_set]

    pos_new = [list(x) for x in sorted({tuple(x) for x in pos_new})]
    neg_new = [list(x) for x in sorted({tuple(x) for x in neg_new})]

    return pos_new, neg_new

# def load_reference_automaton(reference_dir: Path, train_filename: str) -> Automaton:
#     """
#     Load the reference automaton that matches a given training trace file.
#     train filename example:
#       automaton_10_2_5_0~1.txt
#     reference json:
#       automaton_10_2_5_0.json
#     """
#     stem = Path(train_filename).stem
#     base = stem.split("~")[0]
#     ref_json = reference_dir / f"{base}.json"
#     if not ref_json.exists():
#         raise FileNotFoundError(f"Reference automaton not found: {ref_json}")
#     return Automaton.from_json(ref_json)

def load_reference_automaton(reference_dir: Path, train_filename: str) -> Automaton:
    """
    Load the reference automaton that matches a given training trace file.

    train filename example:
      automaton_10_2_5_0~1.txt

    supported reference files:
      automaton_10_2_5_0.json
      automaton_10_2_5_0.dot
      automaton_10_2_5_0.xml
    """
    stem = Path(train_filename).stem
    base = stem.split("~")[0]

    candidates = [
        reference_dir / f"{base}.json",
        reference_dir / f"{base}.dot",
        reference_dir / f"{base}.xml",
    ]

    for path in candidates:
        if path.exists():
            return Automaton.from_file(path)

    raise FileNotFoundError(
        "Reference automaton not found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )

def propose_random_traces_match_counts(
    ref_A: Automaton,
    model,
    target_pos: int,
    target_neg: int,
    existing_pos: Optional[List[Seq]] = None,
    existing_neg: Optional[List[Seq]] = None,
    max_attempts: int = 5000,
    extra_steps: int = 1,
    seed: int = 0,
) -> Tuple[List[Seq], List[Seq], List[dict]]:
    """
    Random baseline that tries to add exactly target_pos positives and
    target_neg negatives, matching the actual counts from the confidence-guided
    method in a comparable round.

    Sampling is done from random valid prefixes of the current learnt model.
    """

    if target_pos <= 0 and target_neg <= 0:
        return [], [], []

    if existing_pos is None:
        existing_pos = []
    if existing_neg is None:
        existing_neg = []

    existing_pos_set = {tuple(x) for x in existing_pos}
    existing_neg_set = {tuple(x) for x in existing_neg}

    prefix_map = one_prefix_per_state(model)

    candidates = []
    for s, prefix in prefix_map.items():
        if run_prefix(ref_A, prefix) is not None:
            candidates.append({
                "state": s,
                "prefix": prefix,
                "confidence": None,
                "uncertainty": None,
            })

    if not candidates:
        return [], [], []

    rnd = random.Random(seed)

    pos_set: Set[Tuple[str, ...]] = set()
    neg_set: Set[Tuple[str, ...]] = set()
    chosen_meta: List[dict] = []

    attempts = 0
    while attempts < max_attempts and (len(pos_set) < target_pos or len(neg_set) < target_neg):
        attempts += 1
        item = rnd.choice(candidates)
        prefix = item["prefix"]
        chosen_meta.append(item)

        if len(pos_set) < target_pos:
            seq = gen_positive_from_prefix(ref_A, prefix, rnd, extra_steps=extra_steps)
            if seq is not None:
                t = tuple(seq)
                if t not in existing_pos_set and t not in existing_neg_set and t not in neg_set:
                    pos_set.add(t)

        if len(neg_set) < target_neg:
            seq = gen_negative_from_prefix(ref_A, prefix, rnd, extra_steps=extra_steps)
            if seq is not None:
                t = tuple(seq)
                if t not in existing_neg_set and t not in existing_pos_set and t not in pos_set:
                    neg_set.add(t)

    pos_new = [list(x) for x in sorted(pos_set)]
    neg_new = [list(x) for x in sorted(neg_set)]

    return pos_new, neg_new, chosen_meta


def propose_additional_traces(
    ref_A: Automaton,
    model,
    state_conf: Dict[int, float],
    top_k_states: int = 3,
    per_prefix_pos: int = 5,
    per_prefix_neg: int = 5,
) -> Tuple[List[Seq], List[Seq], List[dict]]:
    """
    Generate confidence-guided additional traces around uncertain learnt states.

    The current learnt model is used only to identify low-confidence prefixes.
    Actual new traces are generated from the reference automaton so that labels
    remain consistent with the original trace semantics.

    Args:
        ref_A: Reference automaton used as oracle.
        model: Current learnt DFA.
        state_conf: Confidence map for learnt states.
        top_k_states: Number of lowest-confidence prefixes to refine.
        per_prefix_pos: Number of positive samples to try per chosen prefix.
        per_prefix_neg: Number of negative samples to try per chosen prefix.

    Returns:
        A tuple of:
            - new positive traces
            - new negative traces
            - metadata describing the chosen uncertain prefixes
    """

    prefix_map = one_prefix_per_state(model)

    ranked = []
    for s, prefix in prefix_map.items():
        c = trace_confidence(model, state_conf, prefix)
        ranked.append({
            "state": s,
            "prefix": prefix,
            "confidence": c,
            "uncertainty": 1.0 - c,
        })

    ranked.sort(key=lambda x: (x["confidence"], len(x["prefix"]), x["state"]))
    chosen = ranked[:top_k_states]

    rnd = random.Random(0)
    pos_new: List[Seq] = []
    neg_new: List[Seq] = []

    for item in chosen:
        prefix = item["prefix"]

        if run_prefix(ref_A, prefix) is None:
            continue

        for _ in range(per_prefix_pos):
            seq = gen_positive_from_prefix(ref_A, prefix, rnd, extra_steps=1)
            if seq is not None:
                pos_new.append(seq)

        for _ in range(per_prefix_neg):
            seq = gen_negative_from_prefix(ref_A, prefix, rnd, extra_steps=1)
            if seq is not None:
                neg_new.append(seq)

    pos_new = [list(x) for x in sorted({tuple(x) for x in pos_new})]
    neg_new = [list(x) for x in sorted({tuple(x) for x in neg_new})]

    return pos_new, neg_new, chosen

def sanitize_trace_sets(pos_tr, neg_tr):
    """
    Remove exact label conflicts and negatives that are already accepted by the positive PTA.
    """
    pos_set = {tuple(x) for x in pos_tr}
    neg_set = {tuple(x) for x in neg_tr}

    exact_conflict = pos_set & neg_set
    if exact_conflict:
        print(f"[WARN] exact conflicts: {len(exact_conflict)}")
        neg_set -= exact_conflict

    neg_prefix_conflict = set()
    for seq in neg_set:
        for other in neg_set:
            if seq == other:
                continue
            if len(seq) < len(other) and other[:len(seq)] == seq:
                neg_prefix_conflict.add(seq)
                break

    if neg_prefix_conflict:
        print(f"[WARN] negative prefix conflicts: {len(neg_prefix_conflict)}")
        print(f"[WARN] examples: {[list(x) for x in sorted(neg_prefix_conflict)[:10]]}")
        neg_set -= neg_prefix_conflict

    pos_clean = [list(x) for x in sorted(pos_set)]
    neg_raw = [list(x) for x in sorted(neg_set)]

    neg_clean = []
    bad_neg = []
    for seq in neg_raw:
        if pta_accepts(seq, pos_clean):
            bad_neg.append(seq)
        else:
            neg_clean.append(seq)

    if bad_neg:
        print(f"[WARN] negatives accepted by PTA: {len(bad_neg)}")
        print(f"[WARN] examples: {bad_neg[:10]}")

    return pos_clean, neg_clean

def pta_accepts(seq, pos_traces):
    """
    Check whether a sequence is accepted by the PTA built from the current positive traces.
    """
    trie = {}
    END = "__end__"

    for tr in pos_traces:
        node = trie
        for a in tr:
            node = node.setdefault(a, {})
        node[END] = True

    node = trie
    for a in seq:
        if a not in node:
            return False
        node = node[a]
    return True

def run_refinement_for_one_file(
    train_file: Path,
    eval_file: Path,
    reference_dir: Path,
    work_dir: Path,
    rounds: int = 2,
    top_k_states: int = 3,
    max_new_traces_per_round: int = 20,
    proposal_mode: str = "confidence",
    target_added_per_round: Optional[Dict[int, Dict[str, int]]] = None,
):
    """
    Run the full confidence-guided refinement loop for one training file.

    The function creates a working copy of the training traces, repeatedly
    learns a confidence-aware DFA, optionally augments the data around low-
    confidence prefixes, and keeps the best round by BCR /
    confidence.

    Args:
        train_file: Original training trace file.
        eval_file: Evaluation trace file matched to train_file.
        reference_dir: Directory containing reference automata.
        work_dir: Per-file working directory used during refinement.
        rounds: Maximum number of refinement rounds after the initial learning.
        top_k_states: Number of uncertain prefixes to refine each round.
        max_new_traces_per_round: Approximate budget of added traces per round.

    Returns:
        Dictionary summarizing the best result and the per-round history.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    active_train = work_dir / train_file.name
    shutil.copyfile(train_file, active_train)

    ref_A = load_reference_automaton(reference_dir, train_file.name)

    summary_rows = []
    best_result = None

    for round_idx in range(rounds + 1):
        round_start = time.time()

        pos_tr, neg_tr = parse_traces_txt(active_train)
        pos_tr, neg_tr = sanitize_trace_sets(pos_tr, neg_tr)

        pos_ev, neg_ev = parse_traces_txt(eval_file)

        result = learn_edsm_bluefringe(pos_tr, neg_tr, score_threshold=0.0)
        m = eval_on_traces(result.model, pos_ev, neg_ev)

        round_time = time.time() - round_start
        print(f"[DONE round {round_idx}] "
            f"[ROUND TIME] time={round_time:.3f}s"
            f" BCR={m.bcr():.3f} TP={m.tp} TN={m.tn} FP={m.fp} FN={m.fn}")

        conf_vals = list(result.state_conf.values()) if result.state_conf else [1.0]
        min_conf = min(conf_vals)
        avg_conf = sum(conf_vals) / len(conf_vals)

        learnt_states = len(result.model.states())
        ref_states = ref_A.n_states
        state_gap = abs(learnt_states - ref_states)

        row = {
            "round": round_idx,
            "file": train_file.name,
            "train_pos": len(pos_tr),
            "train_neg": len(neg_tr),
            "TP": m.tp,
            "TN": m.tn,
            "FP": m.fp,
            "FN": m.fn,
            "BCR": m.bcr(),
            "states": learnt_states,
            "ref_states": ref_states,
            "state_gap": state_gap,
            "min_state_conf": min_conf,
            "avg_state_conf": avg_conf,
            "added_pos": 0,
            "added_neg": 0,
            "chosen_uncertain_prefixes": [],
            "proposal_mode": proposal_mode,
            "target_added_pos": 0,
            "target_added_neg": 0,
        }

        summary_rows.append(row)

        cand_key = (m.bcr(), -state_gap, avg_conf)
        if best_result is None or cand_key > best_result["key"]:
            best_result = {
                "key": cand_key,
                "round": round_idx,
                "learn_result": result,
                "metrics": m,
                "row": row.copy(),
            }

        if m.bcr() >= 0.999999:
            print(f"[STOP] {train_file.name} round={round_idx} reached BCR≈1.0, stop refinement.")
            break

        if round_idx == rounds:
            break

        existing_pos, existing_neg = parse_traces_txt(active_train)

        if proposal_mode == "confidence":
            per_prefix_pos = max_new_traces_per_round // (2 * top_k_states)
            per_prefix_neg = max_new_traces_per_round // (2 * top_k_states)

            pos_new, neg_new, chosen = propose_additional_traces(
                ref_A=ref_A,
                model=result.model,
                state_conf=result.state_conf,
                top_k_states=top_k_states,
                per_prefix_pos=per_prefix_pos,
                per_prefix_neg=per_prefix_neg,
            )

            pos_new, neg_new = filter_candidate_traces(
                pos_new=pos_new,
                neg_new=neg_new,
                existing_pos=existing_pos,
                existing_neg=existing_neg,
    )

            summary_rows[-1]["target_added_pos"] = len(pos_new)
            summary_rows[-1]["target_added_neg"] = len(neg_new)

        elif proposal_mode == "random":
            if target_added_per_round is None:
                raise ValueError("target_added_per_round is required for random mode.")

            target_pos = int(target_added_per_round.get(round_idx, {}).get("pos", 0))
            target_neg = int(target_added_per_round.get(round_idx, {}).get("neg", 0))

            summary_rows[-1]["target_added_pos"] = target_pos
            summary_rows[-1]["target_added_neg"] = target_neg

            pos_new, neg_new, chosen = propose_random_traces_match_counts(
                ref_A=ref_A,
                model=result.model,
                target_pos=target_pos,
                target_neg=target_neg,
                existing_pos=existing_pos,
                existing_neg=existing_neg,
                max_attempts=5000,
                extra_steps=1,
                seed=0,
            )

            pos_new, neg_new = filter_candidate_traces(
                pos_new=pos_new,
                neg_new=neg_new,
                existing_pos=existing_pos,
                existing_neg=existing_neg,
            )
            
            pos_new = pos_new[:target_pos]
            neg_new = neg_new[:target_neg]

        else:
            raise ValueError(f"Unknown proposal_mode: {proposal_mode}")

        append_traces_txt(active_train, pos_new, neg_new)

        summary_rows[-1]["added_pos"] = len(pos_new)
        summary_rows[-1]["added_neg"] = len(neg_new)
        summary_rows[-1]["chosen_uncertain_prefixes"] = chosen

        if len(pos_new) + len(neg_new) == 0:
            break

    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_lr = best_result["learn_result"]
    stem = train_file.stem

    save_learnt(
        best_lr.model,
        best_lr.alphabet,
        best_dir / f"best-learnt-{stem}.json",
        best_dir / f"best-learnt-{stem}.dot",
        state_conf=best_lr.state_conf,
        merge_history=best_lr.merge_history,
        confidence_json=best_dir / f"best-{proposal_mode}-{stem}.json",
    )

    best_result["row"]["saved_as_best"] = True
    best_result["row"]["best_round"] = best_result["round"]

    return {
        "file": train_file.name,
        "proposal_mode": proposal_mode,
        "best_round": best_result["round"],
        "best_bcr": best_result["metrics"].bcr(),
        "best_states": len(best_lr.model.states()),
        "reference_states": ref_A.n_states,
        "best_tp": best_result["metrics"].tp,
        "best_tn": best_result["metrics"].tn,
        "best_fp": best_result["metrics"].fp,
        "best_fn": best_result["metrics"].fn,
        "best_train_pos": best_result["row"]["train_pos"],
        "best_train_neg": best_result["row"]["train_neg"],
        "best_min_state_conf": best_result["row"]["min_state_conf"],
        "best_avg_state_conf": best_result["row"]["avg_state_conf"],
        "best_state_gap": best_result["row"]["state_gap"],
        "summary_rows": summary_rows,
        
    }

def load_targets_from_confidence_summary(conf_summary_path: Path) -> Dict[str, Dict[int, Dict[str, int]]]:
    """
    Read an existing confidence summary json and build:

        {
            "automaton_xxx.txt": {
                0: {"pos": 6, "neg": 5},
                1: {"pos": 0, "neg": 3},
                ...
            },
            ...
        }
    """
    data = json.loads(conf_summary_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, Dict[str, int]]] = {}

    for item in data:
        fname = item["file"]
        per_round: Dict[int, Dict[str, int]] = {}

        for row in item.get("summary_rows", []):
            per_round[int(row["round"])] = {
                "pos": int(row.get("added_pos", 0)),
                "neg": int(row.get("added_neg", 0)),
            }

        out[fname] = per_round

    return out

def main_random_from_summary():
    start_time = time.time()
    if len(sys.argv) != 9:
        print("Usage: python confidence_refine.py random <REFERENCE_AUTOMATA_DIR> <TRAIN_DIR> <EVAL_DIR> <WORK_DIR> <CONF_SUMMARY_JSON> <OUT_JSON> <OUT_CSV>")
        print("Example: python confidence_refine.py random ./automata ./train ./test ./refine_work ./refine_summary_confidence.json ./refine_summary_random.json ./best_results_random.csv")
        raise SystemExit(2)

    reference_dir = Path(sys.argv[2])
    train_dir = Path(sys.argv[3])
    eval_dir = Path(sys.argv[4])
    work_dir = Path(sys.argv[5])
    conf_summary_json = Path(sys.argv[6])
    out_json = Path(sys.argv[7])
    out_csv = Path(sys.argv[8])

    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(eval_dir)
    if not conf_summary_json.exists():
        raise FileNotFoundError(conf_summary_json)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_dir.glob("*.txt"))
    if not train_files:
        raise FileNotFoundError(f"No .txt files in {train_dir}")

    targets_by_file = load_targets_from_confidence_summary(conf_summary_json)

    all_rand_results = []

    for idx, train_file in enumerate(train_files, start=1):
        eval_file = eval_dir / train_file.name
        if not eval_file.exists():
            print(f"[SKIP] no eval file for {train_file.name}")
            continue

        if train_file.name not in targets_by_file:
            print(f"[SKIP] no confidence summary target for {train_file.name}")
            continue

        print(f"[RANDOM {idx}/{len(train_files)}] {train_file.name}")

        result_rand = run_refinement_for_one_file(
            train_file=train_file,
            eval_file=eval_file,
            reference_dir=reference_dir,
            work_dir=work_dir / "random" / train_file.stem,
            rounds=2,
            top_k_states=3,
            max_new_traces_per_round=20,
            proposal_mode="random",
            target_added_per_round=targets_by_file[train_file.name],
        )
        all_rand_results.append(result_rand)

    out_json.write_text(json.dumps(all_rand_results, indent=2), encoding="utf-8")

    headers = [
        "file",
        "best_round",
        "best_bcr",
        "best_tp",
        "best_tn",
        "best_fp",
        "best_fn",
        "best_states",
        "reference_states",
        "best_state_gap",
        "best_train_pos",
        "best_train_neg",
        "best_min_state_conf",
        "best_avg_state_conf",
    ]

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in all_rand_results:
            row = [
                r["file"],
                r["best_round"],
                f"{r['best_bcr']:.6f}",
                r["best_tp"],
                r["best_tn"],
                r["best_fp"],
                r["best_fn"],
                r["best_states"],
                r["reference_states"],
                r["best_state_gap"],
                r["best_train_pos"],
                r["best_train_neg"],
                f"{r['best_min_state_conf']:.6f}",
                f"{r['best_avg_state_conf']:.6f}",
            ]
            f.write(",".join(map(str, row)) + "\n")

    print(f"Done. Random summary written to: {out_json}")
    print(f"Done. Random CSV written to: {out_csv}")
    print(f"Total time: {time.time()-start_time:.1f}s")

def main():
    start_time = time.time()
    if len(sys.argv) != 7:
        print("Usage: python confidence_refine.py <REFERENCE_AUTOMATA_DIR> <TRAIN_DIR> <EVAL_DIR> <WORK_DIR> <OUT_JSON> <OUT_CSV>")
        print("Example: python confidence_refine.py ./automata ./train ./test ./refine_work ./refine_summary.json ./best_results.csv")
        raise SystemExit(2)

    reference_dir = Path(sys.argv[1])
    train_dir = Path(sys.argv[2])
    eval_dir = Path(sys.argv[3])
    work_dir = Path(sys.argv[4])
    out_json = Path(sys.argv[5])
    out_csv = Path(sys.argv[6])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(eval_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_dir.glob("*.txt"))
    if not train_files:
        raise FileNotFoundError(f"No .txt files in {train_dir}")

    all_conf_results = []
    for idx, train_file in enumerate(train_files, start=1):
        eval_file = eval_dir / train_file.name
        if not eval_file.exists():
            print(f"[SKIP] no eval file for {train_file.name}")
            continue

        print(f"[REFINE {idx}/{len(train_files)}] {train_file.name}")
        result_conf = run_refinement_for_one_file(
        train_file=train_file,
        eval_file=eval_file,
        reference_dir=reference_dir,
        work_dir=work_dir / "confidence" / train_file.stem,
        rounds=2,
        top_k_states=3,
        max_new_traces_per_round=20,
        proposal_mode="confidence"
    )
        all_conf_results.append(result_conf)

    out_json.write_text(json.dumps(all_conf_results, indent=2), encoding="utf-8")

    headers = [
        "file",
        "best_round",
        "best_bcr",
        "best_tp",
        "best_tn",
        "best_fp",
        "best_fn",
        "best_states",
        "reference_states",
        "best_state_gap",
        "best_train_pos",
        "best_train_neg",
        "best_min_state_conf",
        "best_avg_state_conf",
    ]

    # confidence csv
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in all_conf_results:
            row = [
                r["file"],
                r["best_round"],
                f"{r['best_bcr']:.6f}",
                r["best_tp"],
                r["best_tn"],
                r["best_fp"],
                r["best_fn"],
                r["best_states"],
                r["reference_states"],
                r["best_state_gap"],
                r["best_train_pos"],
                r["best_train_neg"],
                f"{r['best_min_state_conf']:.6f}",
                f"{r['best_avg_state_conf']:.6f}",
            ]
            f.write(",".join(map(str, row)) + "\n")


    print(f"Done. Confidence summary written to: {out_json}")
    print(f"Done. Confidence CSV written to: {out_csv}")
    print(f"Total time: {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "random":
        main_random_from_summary()
    else:
        main()