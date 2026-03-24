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

Plot the figure and check the BCR euqal to 1's automata, it same structure for both 5-state and 10-state.