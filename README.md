# KB1
# EDSM
# Python Automata Learning Pipeline (EDSM)

This project provides a Python workflow similar to the Statechum COM6911 experiments:

1. Generate **reference automata** (random state machines)  
2. Generate **random-walk traces** for **training** and **testing**  
3. Learn an automaton using **EDSM** (Blue-Fringe / Red-Blue state merging)  
4. Evaluate learnt models (TP/TN/FP/FN/BCR) and save results to CSV  
5. Visualise and compare reference vs learnt automata  

> **Trace semantics used in this pipeline**
>
> - **Positive trace (`+`)**: the full path exists in the automaton  
> - **Negative trace (`-`)**: the full path does **not** exist (typically the last symbol is a missing transition)

---

## Project Files

- `common.py`  
  Shared data structures and helper functions (Automaton, DOT writer, trace parsing).

- `generate_automata.py`  
  Generates random reference automata into `.json` and `.dot`.

- `generate_randomwalks.py`  
  Generates training/testing trace files (`.txt`) from reference automata.

- `edsm_train_eval.py`  
  Learns automata using EDSM and evaluates on test traces. Saves learnt `.json`, `.dot`, and a CSV report.

- `plot_compare.py`  
  Visualises a single automaton, or compares two automata side-by-side.

- `confidence.py`  
  Computes confidence scores for EDSM-learned DFAs, based on the evidence accumulated during state merging.

---

## Requirements

Install Python dependencies:

```
pip install networkx matplotlib
```
## Recommended Directory Layout
```text
automata/        # reference automata (.json + .dot)
train_data/      # training traces (.txt)
test_data/       # evaluation traces (.txt)
learning_E0/     # learnt automata (.json + .dot)
outcome_E0.csv   # evaluation summary
```
## Step 1 — Generate Reference Automata

```
python generate_automata.py <OUT_DIR> <MAX_STATES>
```

## Step 2 — Generate Training Traces

```
python generate_randomwalks.py ./automata ./train 30 2 false 10
```

## Step 3 — Generate Testing Traces

```
python generate_randomwalks.py ./automata ./test 40 4 false 10
```

## Step 4 — Learn with EDSM and Evaluate

```
python edsm_train_eval.py ./train ./test ./learning_E0 ./outcome_E0.csv
```

## Step 5 — Visualise / Compare Automata

```
python plot_compare.py ./automata/automaton_5_2_2_0.dot ./learning_E0/learnt-automaton_5_2_2_0~0.dot
```

---

## Confidence Scoring

Each learnt automaton now carries a confidence score derived from the evidence collected during EDSM state merging.

### Background

During EDSM learning, every state-merge operation produces an **evidence score** `t` — the number of state-pairs and shared labels that were unified in that merge. The probability that a single merge is correct is modelled as:

```
c = 1 - 0.5^t
```

Two aggregate scores are computed and saved into the learnt JSON:

| Score | Formula | Meaning |
|---|---|---|
| **Global model confidence** | `D = ∏ (1 - 0.5^t_i)` | Joint probability that every merge in the model is correct |
| **Trace path confidence** | `min_{q ∈ path} (1 - 0.5^t_q)` | Weakest-link confidence along the states visited by a specific trace |

> Reference: Lang, Pearlmutter & Price — *Results of the Abbadingo One DFA Learning Competition and a New Evidence-Driven State Merging Algorithm* (1998)

### Learnt JSON format

After running Step 4, each `learning_E0/learnt-*.json` contains a `confidence` block:

```json
"confidence": {
  "global_conf": 0.7031,
  "state_scores": {"0": 4.0, "3": 2.0},
  "n_merges": 12
}
```

- `global_conf` — overall model confidence `D`
- `state_scores` — maps state ID to the highest merge-evidence score of any merge that involved that state
- `n_merges` — total number of merge operations performed

### Querying a trace

**Option A — train and query in one session:**

```python
from edsm_train_eval import learn_edsm_bluefringe
from confidence import evaluate
from common import parse_traces_txt
from pathlib import Path

pos, neg = parse_traces_txt(Path("train/automaton_5_2_2_0~0.txt"))
dfa, alphabet, conf = learn_edsm_bluefringe(pos, neg)

result = evaluate(["L0", "L1", "L0"], dfa, conf)
print(result.accepted)     # True if all transitions are defined for the trace
print(result.confidence)   # weakest-link confidence along the traversed path
```

**Option B — load from saved JSON:**

```python
from common import Automaton
from confidence import evaluate, load_conf_from_json
from pathlib import Path

dfa  = Automaton.from_json(Path("learning_E0/learnt-automaton_5_2_2_0~0.json"))
conf = load_conf_from_json(Path("learning_E0/learnt-automaton_5_2_2_0~0.json"))

result = evaluate(["L0", "L1", "L0"], dfa, conf)
print(result.accepted, result.confidence)
```

### Interpreting `EvalResult`

| `accepted` | `confidence` | Interpretation |
|---|---|---|
| `True` | high (→ 1.0) | Trace accepted; path well-supported by training evidence |
| `True` | low (→ 0.0) | Trace accepted; but path passes through weakly-evidenced states — treat with caution |
| `False` | any | Trace rejected (missing transition); confidence reflects reliability of the path up to the point of failure |