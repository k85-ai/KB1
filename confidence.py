"""
Confidence scoring for EDSM-learned DFAs.

Based on: Lang, Pearlmutter, Price (1998) — Abbadingo One / EDSM.

Formulas:
  single merge probability:       c_i = 1 - 0.5^t_i
  global model confidence:        D   = prod_i c_i
  trace path confidence (weakest-link): C_trace = min_{q in path} c_q
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from edsm_train_eval import DFA

Seq = List[str]


@dataclass
class _MergeRecord:
    red: int
    blue: int
    score: float


@dataclass
class ConfidenceData:
    global_conf: float
    state_scores: Dict[int, float]  # state -> highest merge score involving that state
    n_merges: int


def _single_merge_prob(t: float) -> float:
    if t <= 0:
        return 0.0
    return 1.0 - 0.5 ** t


def compute_confidence(merge_log: List[_MergeRecord]) -> ConfidenceData:
    """Compute ConfidenceData from the internal merge log produced during EDSM."""
    global_conf = 1.0
    state_scores: Dict[int, float] = {}

    for rec in merge_log:
        global_conf *= _single_merge_prob(rec.score)
        state_scores[rec.red] = max(state_scores.get(rec.red, 0.0), rec.score)

    return ConfidenceData(
        global_conf=global_conf,
        state_scores=state_scores,
        n_merges=len(merge_log),
    )


def trace_confidence_weakest_link(
    trace: Seq,
    dfa: "DFA",
    conf: ConfidenceData,
) -> float:
    """
    C_trace = min_{q in path} (1 - 0.5^t_q).
    States with no merge record are skipped (treated as c=1.0 — direct PTA evidence).
    Returns 0.0 if trace leaves the DFA (undefined transition).
    """
    s = dfa.start
    min_c = 1.0
    t = conf.state_scores.get(s)
    if t is not None:
        min_c = min(min_c, _single_merge_prob(t))

    for sym in trace:
        s = dfa.delta.get((s, sym))
        if s is None:
            return 0.0
        t = conf.state_scores.get(s)
        if t is not None:
            min_c = min(min_c, _single_merge_prob(t))

    return min_c


@dataclass
class EvalResult:
    accepted: bool
    confidence: float  # weakest-link confidence along the traversed path


def evaluate(trace: Seq, dfa: "DFA", conf: ConfidenceData) -> EvalResult:
    """
    Walk trace on dfa and return acceptance + path confidence.

    accepted=True  → all transitions defined; confidence = weakest-link on full path.
    accepted=False → transition missing at some symbol; confidence = weakest-link
                     up to (and including) the last reachable state before failure.
    """
    s = dfa.start
    min_c = 1.0
    t = conf.state_scores.get(s)
    if t is not None:
        min_c = min(min_c, _single_merge_prob(t))

    for sym in trace:
        nxt = dfa.delta.get((s, sym))
        if nxt is None:
            return EvalResult(accepted=False, confidence=min_c)
        s = nxt
        t = conf.state_scores.get(s)
        if t is not None:
            min_c = min(min_c, _single_merge_prob(t))

    return EvalResult(accepted=True, confidence=min_c)


def load_conf_from_json(path: Path) -> ConfidenceData:
    """Load ConfidenceData saved inside a learnt-*.json file."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = obj.get("confidence", {})
    return ConfidenceData(
        global_conf=raw.get("global_conf", 1.0),
        state_scores={int(k): v for k, v in raw.get("state_scores", {}).items()},
        n_merges=raw.get("n_merges", 0),
    )
