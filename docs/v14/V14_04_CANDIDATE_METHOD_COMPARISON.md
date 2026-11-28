# V14-04: Candidate Methodological Directions

| Candidate | Core Mechanism | Challenge & Verdict |
|---|---|---|
| **A: Conditional Rescue-Harm Policy** | Separate `R(s,a)` and `H(s,a)` estimators with asymmetric harm penalty `alpha > 1` | Under symmetric `alpha=1`, reduces to net gain `R - H`; requires `alpha > 1` or class-asymmetric risk AND sufficient positive transitions (`42-76` on `val` is tight) |
| **B: Joint Sequential-Parallel Option-Value Policy (PRIMARY)** | Finite-horizon Bellman/value-of-information policy over `{STOP, ACQ_A, ACQ_V, ACQ_AV_PAR}` balancing option value vs RTT deadline feasibility | Directly addresses V13.2's fixed `T->A->V` order and `OneShot` vs `Sequential` latency/byte trade-off; finite 3-modality horizon admits exact/tabular-style recursion without heavy RL |
| **C: Risk-Aware Conservative Stopping (BACKUP)** | Lower-confidence-bound (LCB) on net gain `G_hat(s,a) - kappa * sigma(s,a)` + deadline chance constraint | Standard robust optimization if uncoupled from sequential-parallel option value; serves well as a conservative risk layer inside Candidate B |
