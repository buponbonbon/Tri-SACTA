# Tri-SACTA V13.2 — Foundation Closure Report

- **Closure Branch:** `research/v13.2-phase3-network-aware-systems-evaluation`
- **Frozen Scientific Source:** `d59c6f5ba6834494078d9cece2241e7838723df6`
- **Authorized Phase II Freeze Commit:** `c9177414a2eadd45a6df99dd832b3512a460e3d2`
- **Phase II Confirmatory Results:** `f559b7d82973bcd2b1d5237da36a105d2ac0c5de` (`CONFIRMATORY_SUPPORTED`, `4,131` utterances / `617` videos, single execution verified in `holdout_access_ledger.json`)
- **Phase III Pilot (`pilot_v1/`):** Preserved untouched (`4,224` requests, post-hoc delay injection, historical `MAE = 4.74 ms`, F7 `29.49 s` OS/socket stall).
- **Phase III Rectified (`rectified/`):** `26,136` requests (`99` dev utterances / `12` videos, `seed=42`, in-acquisition delay injection + `CommunicationEvent` traces).

## Resolution of Remaining Technical Issues (A2)
1. **Critical-path round accounting:** Updated `scripts/phase3_network/run_rectified_benchmark.py` to derive `actual_rounds` from overlapping `CommunicationEvent` intervals rather than policy-name heuristics (verified for both `always_on_parallel` and `confidence_routing_055`).
2. **Network timing semantics:** Documented as controlled application-layer acquisition delay emulation over TCP loopback (not packet-level WAN simulation).
3. **Paired exogenous jitter:** Updated `RectifiedNetworkCoordinator.process_query_with_events` to seed jitter deterministically per `(base_jitter_seed, sample_idx, parent_request_id)` independent of policy call counts.
4. **Statistical scope:** `26,136` requests reflect `99` utterances / `12` video clusters (`seed=42`) — exploratory only, not independent samples.
5. **Phase II independent verifier:** `verify_phase2_integrity.py` verified exact `694,008` TCP + `1,388,016` offline rows, LF-normalized Git blob `391f1990789d5faa01d369881f13663f2c897310`, and `query_traces.json.gz` SHA-256 `38d852b72ec0768e64371654069b6228b2eb2412c42421286251bc745e86c8a1`.
6. **CI triggers:** Extended `.github/workflows/regression-v13-2.yml` to run on `research/v13.2-phase3-network-aware-systems-evaluation` and `research/v14-*` pushes.

## Foundation Closure Verdict
**`V13_2_FOUNDATION_USABLE_WITH_DOCUMENTED_LIMITATIONS`**
- **Safely Reusable in V14:** 21 frozen predictor checkpoints (`results/v13_utility/training/checkpoints/`), `val` (`N=250`) and `cal` (`N=250`) development caches, `DistributedCoordinator`/`ModalitySocketServer`/`RectifiedNetworkCoordinator`, and `paired_cluster_bootstrap_on_df`.
- **Not Reusable as Fresh V14 Holdout:** The `4,131` Phase II utterances (`eligible_holdout_manifest.json`) are permanently exposed and must NOT be reused for V14/V15 confirmation.
