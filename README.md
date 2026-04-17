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
python generate_randomwalks.py <SRC_AUTOMATA_DIR> <TGT_TRACES_DIR> <seq_mult> <length_mult> <pos_only> <sets_per>
python generate_randomwalks.py ./automata ./train --seq-mult 30 --length-mult 2 --sets-per 10
```

## Step 3 — Generate Testing Traces

```
python generate_randomwalks.py <SRC_AUTOMATA_DIR> <TGT_TRACES_DIR> <seq_mult> <length_mult> <pos_only> <sets_per>
python generate_randomwalks.py ./automata ./test --seq-mult 40 --length-mult 4 --sets-per 10
```

## Step 4 — Learn with EDSM and Evaluate

```
python edsm_train_eval.py  <TRAIN_DIR> <EVAL_DIR> <LEARN_DIR> <OUT_CSV>
python edsm_train_eval.py ./train ./test ./learning_E0 ./outcome_E0.csv
```

## Step 5 — Visualise / Compare Automata

```
python plot_compare.py ./automata/automaton_5_2_2_0.dot ./learning_E0/learnt-automaton_5_2_2_0~0.dot --save
```
Use save at the end of the command for saving figures.

Plot the figure and check the BCR euqal to 1's automata, it same structure for both 5-state and 10-state.

## Step 6 — Confidence Refinement (Active Trace Augmentation)

This step extends the original EDSM learner with a confidence-guided active refinement loop.

### Motivation

The original EDSM pipeline learns one automaton from training walks.
However, some merges may be supported by strong evidence, while others may be accepted with much weaker evidence.

To make this uncertainty explicit, we compute a confidence value for merges, states, and prefixes, and use it to guide where more traces should be added.

In this way, confidence is not only used as a diagnostic score, but also as a control signal for active trace augmentation.
### How confidence is defined
For each accepted merge with score t, merge confidence is defined as

$$
c\_merge = 1 - 0.5^t
$$ 

- a larger merge score gives confidence closer to `1`
- a smaller merge score gives lower confidence
- weakly supported merges therefore produce lower-confidence outputs

After that, confidence is propagated to the final learnt automaton:

- each learnt state is assigned a state confidence
- each representative prefix is assigned a prefix confidence
- prefix confidence is used as a proxy for how certain the learner is about that region of the automaton

How confidence is used

### Confidence is used in two roles:

1. Confidence as uncertainty indicator

Low-confidence states / prefixes are treated as regions where the learner is not sure.

That means confidence directly answers:

which part of the learnt automaton is weakly supported by evidence
where additional traces are most likely to help

2. Confidence as refinement guide

After one round of learning:

- compute state confidence and prefix confidence
- rank learnt states by confidence
- select the lowest-confidence prefixes
- generate additional traces around those prefixes
- add the new traces into the working training set
- retrain the learner
- compare the new model against the original one

So confidence is actively used to decide `where to collect more data.`

### Command

```
python confidence_refine.py <REFERENCE_AUTOMATA_DIR> <TRAIN_DIR> <EVAL_DIR> <WORK_DIR> <OUT_JSON> <OUT_CSV>

python confidence_refine.py ./automata ./train ./test ./refine_work ./refine_summary.json ./best_results.csv
```

### Outputs

For each training file, the best result is saved in:

```
<WORK_DIR>/<train_file_stem>/best/
```

This includes:

- best-learnt-<stem>.json
- best-learnt-<stem>.dot
- best-confidence-<stem>.json

The script also outputs:

- refine_summary.json — full multi-round refinement history
- best_results.csv — one-row summary of the best result for each training file

## Final result:

BCR before in average: 0.9769375
BCR after using confidence Refinement in average: 0.9910125