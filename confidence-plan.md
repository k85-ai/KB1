# Confidence plan (v1)

- Input:
  - `trace`: The unknown sequence (a sequence of symbols) to be evaluated.
  - `DFA`: The Deterministic Finite Automaton generated through the EDSM learning algorithm.
  - `merge_scores`: The "evidence scores" (merge scores) recorded during each state merging operation in the EDSM learning process. If calculating trace-specific confidence, these scores need to be mapped to the corresponding states or paths in the DFA.

- Output:
  - `confidence_score`: A floating-point number between [0, 1] representing the model's confidence level in its prediction (accept/reject) for the given trace.

- Candidate formula:
  - **1. Probability of a correct single merge:**
    $$c_i = 1 - 0.5^{t_i}$$
    *(where $t_i$ is the merge score of the $i$-th merge, assuming a 0.5 probability of discovering an error per test)*
  - **2. Global Model Confidence:**
    $$D = \prod_{i=1}^{m} (1 - 0.5^{t_i})$$
    *(where $m$ is the total number of merge operations executed by the EDSM algorithm)*
  - **3. Trace-specific Path Confidence:**
    Assuming the `trace` traverses the state sequence $q_0, q_1, \dots, q_k$ on the DFA, and each state $q$ is associated with its corresponding merge score $t_q$ generated during the merge.
    - *Option A (Weakest link)*: Take the lowest probability along the path as the confidence upper bound.
      $$C_{trace} = \min_{q \in path} (1 - 0.5^{t_q})$$
    - *Option B (Joint probability)*: Alternatively, just use the global model confidence $D$ as the baseline.

- Needed reference/paper keyword:
  - **Paper:** "Results of the Abbadingo One DFA Learning Competition and a New Evidence Driven State Merging Algorithm" (Kevin J. Lang, Barak A. Pearlmutter, Rodney A. Price, 1998)

@inproceedings{Lang1998ResultsOT,
  title={Results of the Abbadingo One DFA Learning Competition and a New Evidence-Driven State Merging Algorithm},
  author={Kevin J. Lang and Barak A. Pearlmutter and Rodney A. Price},
  booktitle={International Conference on Graphics and Interaction},
  year={1998},
  url={https://api.semanticscholar.org/CorpusID:2132053}
}
