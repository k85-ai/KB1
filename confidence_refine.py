import sys
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Set, Dict

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
    s = A.start
    for a in seq:
        s = A.step(s, a)
        if s is None:
            return None
    return s

def gen_positive_from_prefix(A: Automaton, prefix: Seq, rnd: random.Random, extra_steps: int = 1):
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

def expand_prefix(prefix: Seq, alphabet: List[str], max_depth: int = 2) -> List[Seq]:
    results: List[Seq] = []

    def dfs(cur: Seq, depth: int):
        if depth == 0:
            results.append(cur[:])
            return
        for a in alphabet:
            cur.append(a)
            dfs(cur, depth - 1)
            cur.pop()

    for d in range(1, max_depth + 1):
        dfs(prefix[:], d)

    return results


def append_traces_txt(path: Path, pos_traces: List[Seq], neg_traces: List[Seq]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for seq in pos_traces:
            f.write("+ " + ",".join(seq) + "\n")
        for seq in neg_traces:
            f.write("- " + ",".join(seq) + "\n")


def load_reference_automaton(reference_dir: Path, train_filename: str) -> Automaton:
    """
    train filename example:
      automaton_10_2_5_0~1.txt
    reference json:
      automaton_10_2_5_0.json
    """
    stem = Path(train_filename).stem
    base = stem.split("~")[0]
    ref_json = reference_dir / f"{base}.json"
    if not ref_json.exists():
        raise FileNotFoundError(f"Reference automaton not found: {ref_json}")
    return Automaton.from_json(ref_json)


def propose_additional_traces(
    ref_A: Automaton,
    model,
    state_conf: Dict[int, float],
    top_k_states: int = 3,
    per_prefix_pos: int = 5,
    per_prefix_neg: int = 5,
) -> Tuple[List[Seq], List[Seq], List[dict]]:
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
    pos_set = {tuple(x) for x in pos_tr}
    neg_set = {tuple(x) for x in neg_tr}

    exact_conflict = pos_set & neg_set
    if exact_conflict:
        print(f"[WARN] exact conflicts: {len(exact_conflict)}")
        neg_set -= exact_conflict

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
    max_new_traces_per_round: int = 40,
):
    work_dir.mkdir(parents=True, exist_ok=True)

    active_train = work_dir / train_file.name
    shutil.copyfile(train_file, active_train)

    ref_A = load_reference_automaton(reference_dir, train_file.name)

    summary_rows = []
    best_result = None

    for round_idx in range(rounds + 1):
        pos_tr, neg_tr = parse_traces_txt(active_train)
        pos_tr, neg_tr = sanitize_trace_sets(pos_tr, neg_tr)

        pos_ev, neg_ev = parse_traces_txt(eval_file)

        result = learn_edsm_bluefringe(pos_tr, neg_tr, score_threshold=0.0)
        m = eval_on_traces(result.model, pos_ev, neg_ev)

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

        pos_new, neg_new, chosen = propose_additional_traces(
            ref_A=ref_A,
            model=result.model,
            state_conf=result.state_conf,
            top_k_states=top_k_states,
            per_prefix_pos=max_new_traces_per_round // (2 * top_k_states),
            per_prefix_neg=max_new_traces_per_round // (2 * top_k_states),
)

        existing_pos, existing_neg = parse_traces_txt(active_train)
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
        confidence_json=best_dir / f"best-confidence-{stem}.json",
    )

    best_result["row"]["saved_as_best"] = True
    best_result["row"]["best_round"] = best_result["round"]

    return {
        "file": train_file.name,
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


def main():
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

    all_rows = []
    for idx, train_file in enumerate(train_files, start=1):
        eval_file = eval_dir / train_file.name
        if not eval_file.exists():
            print(f"[SKIP] no eval file for {train_file.name}")
            continue

        print(f"[REFINE {idx}/{len(train_files)}] {train_file.name}")
        result = run_refinement_for_one_file(
        train_file=train_file,
        eval_file=eval_file,
        reference_dir=reference_dir,
        work_dir=work_dir / train_file.stem,
        rounds=2,
        top_k_states=3,
        max_new_traces_per_round=40,
    )
        all_rows.append(result)

    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

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
        for r in all_rows:
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

    print(f"Done. Summary written to: {out_json}")
    print(f"Done. Best-results CSV written to: {out_csv}")


if __name__ == "__main__":
    main()