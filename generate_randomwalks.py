import os
import sys
import random
from pathlib import Path
from typing import List, Optional, Tuple, Set
import argparse
from pathlib import Path

from common import Automaton, make_even, stable_salt_from_path

def rand_len(rnd: random.Random, max_len: int) -> int:
    return rnd.randint(1, max_len)  

def gen_positive(A: Automaton, rnd: random.Random, length_mult: int) -> Tuple[List[str], int]:
    max_len = max(1, length_mult * A.n_states)
    L = rand_len(rnd, max_len)
    s = A.start
    seq: List[str] = []
    for _ in range(L):
        enabled = A.enabled(s)
        if not enabled:
            break
        a = rnd.choice(enabled)
        seq.append(a)
        s2 = A.step(s, a)
        if s2 is None:
            break
        s = s2
    return seq, s

def gen_negative(A: Automaton, rnd: random.Random, length_mult: int) -> Optional[List[str]]:
    seq, s = gen_positive(A, rnd, length_mult)
    enabled = set(A.enabled(s))
    missing = [a for a in A.alphabet if a not in enabled]
    if not missing:
        return None
    bad = rnd.choice(missing)
    return seq + [bad]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("src_dir", type=Path)
    parser.add_argument("tgt_dir", type=Path)

    parser.add_argument("--seq-mult", type=int, help="number of traces per state multiplier (default: 30)", required=True)
    parser.add_argument("--length-mult", type=int, help="maximum walk length multiplier relative to number of states (default: 2)", required=True)
    parser.add_argument("--pos-only", type=str, default="false")
    parser.add_argument("--sets-per", type=int,  help="number of trace sets generated for each automaton (default: 10)", required=True)

    args = parser.parse_args()
    args.pos_only = args.pos_only.lower() == "true"
    return args

def main():
    args = parse_args()

    src_dir = args.src_dir
    tgt_dir = args.tgt_dir
    seq_mult = args.seq_mult
    length_mult = args.length_mult
    pos_only = args.pos_only
    sets_per = args.sets_per

    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    salt = stable_salt_from_path(str(tgt_dir))  # separates train vs test by directory path

    automata_files = sorted([p for p in src_dir.iterdir() if p.name.startswith("automaton_") and p.suffix == ".json"])
    if not automata_files:
        raise FileNotFoundError(f"No automaton_*.json found in {src_dir}")

    for ap in automata_files:
        A = Automaton.from_json(ap)
        base = ap.stem

        total = make_even(A.n_states * seq_mult)
        n_pos = total if pos_only else total // 2
        n_neg = 0 if pos_only else total // 2

        for k in range(sets_per):
            rnd = random.Random(salt + k)

            # Try to reduce duplicates: keep a set, but if cannot reach target, allow repeats to fill.
            pos_set: Set[Tuple[str, ...]] = set()
            neg_set: Set[Tuple[str, ...]] = set()

            max_attempts = 200000
            attempts = 0
            while (len(pos_set) < n_pos or len(neg_set) < n_neg) and attempts < max_attempts:
                attempts += 1
                if len(pos_set) < n_pos:
                    seq, _ = gen_positive(A, rnd, length_mult)
                    if seq:
                        pos_set.add(tuple(seq))
                if len(neg_set) < n_neg:
                    neg = gen_negative(A, rnd, length_mult)
                    if neg:
                        neg_set.add(tuple(neg))

            pos_list = [list(t) for t in sorted(pos_set)]
            neg_list = [list(t) for t in sorted(neg_set)]

            # If uniqueness space is small (especially for 5-state), fill with duplicates to reach required count.
            if not pos_list and n_pos > 0:
                pos_list = [["L0"]]
            if not neg_list and n_neg > 0:
                neg_list = [["L0"]]

            while len(pos_list) < n_pos:
                pos_list.append(rnd.choice(pos_list))
            while len(neg_list) < n_neg:
                neg_list.append(rnd.choice(neg_list))

            rnd.shuffle(pos_list)
            rnd.shuffle(neg_list)

            out_path = tgt_dir / f"{base}~{k}.txt"
            with out_path.open("w", encoding="utf-8") as out:
                for seq in pos_list[:n_pos]:
                    out.write("+ " + ",".join(seq) + "\n")
                for seq in neg_list[:n_neg]:
                    out.write("- " + ",".join(seq) + "\n")

    print(f"Done. Traces written to: {tgt_dir}")

if __name__ == "__main__":
    main()