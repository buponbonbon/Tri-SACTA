# Tri-SACTA — Independent V4 post-merge audit and V5 action plan

**Audit date:** 2026-11-22 (Vietnam time)  
**Repository:** https://github.com/buponbonbon/Tri-SACTA  
**Pinned main:** `52ad9b21f66bfd9667baffaf259fe56b8112cb41`  
**V4 implementation:** `6a36978f3aa09d75a8a61a4b9796c822a0d458ea`  
**Previous main:** `4777b64582206bb0e718fe800558ba96b073fedc`  
**Tree of both V4 and merge:** `9f161a0afabbf628d9066fa43fd9426b2805b55a`

## 1. Executive verdict

PR #4 was merged on 22 November 2026 at 09:57:24 ICT. The Git trees at the V4 implementation and merge are identical. No additional implementation accompanied the merge. V4 rightly quarantines pseudo-real V3 artifacts, reports missing media as BLOCKED, fixes duplicate prequential spectral history updates, and corrects a sufficient-condition statement in the theory. These are substantial integrity/engineering improvements, not evidence of real-data scientific superiority.

Readiness estimate: 50–55% of the research workflow, roughly 4.5/10 submission readiness (subjective, not a publication probability). Stage: **data acquisition and real pipeline validation**, before empirical validation. Full authentic multimodal benchmark: BLOCKED. Do not submit. No V4 result JSON demonstrating a new experimental run is visible. No independent reproduction of the full test suite or GPU runs was possible: Git clone/ls-remote failed DNS in this environment. Local annotation files, source checkpoints, and execution logs are not tracked in the reviewed snapshot. Do not infer they do not exist on the user's workstation. The static findings below are grounded in GitHub source; the controller counterexample is an independently executed minimal reproduction of the relevant update rule, not a run of the full repository.

Pinned file citations throughout this report can be opened with base URL `https://github.com/buponbonbon/Tri-SACTA/blob/52ad9b21f66bfd9667baffaf259fe56b8112cb41/` followed by the path. PR: https://github.com/buponbonbon/Tri-SACTA/pull/4.

## 2. Evidence inventory and status changes

| Area | Observed in V4 | Independent status |
|---|---|---|
| Quarantine | Old manifests and JSON files moved into `quarantined_v3_fixtures/`; notices present | IMPLEMENTED for tracked artifacts; local checkpoint/cache moves not independently verified; aggregation bypass remains |
| Acquisition | `scripts/acquire_authentic_data.py` downloads official MELD CSVs and third-party CMU-MOSEI text CSVs; reports counts/hashes | IMPLEMENTED for acquisition of annotations only; no actual media acquisition or dataset-to-feature join |
| Authenticated raw files | Provenance document reports 13,708 MELD rows and 22,788 CMU-MOSEI rows | REPORTED_LOCAL; raw CSVs and logs not visible in GitHub; hashes not compared to a trusted pinned reference in code |
| Extraction | `scripts/extract_features_sequential.py` unchanged from V3 | NOT IMPLEMENTED for actual multimodal extraction; fake generation still callable |
| Loader | rejects known fake ID prefixes, missing manifest; selected checksum verification | PARTIAL, not a full provenance policy |
| Checkpoint binding | legacy fallback accepted; code does not verify manifest hash | OPEN, contradicts R4-06 report |
| Spectral update | `precomputed_shift_scores` passed from runner into adapt_step | IMPLEMENTED and source-consistent; test present, not independently executed |
| Theory | sufficient conditions corrected, immediate scope clarified | PARTIAL; delayed theorem invalid as stated for clipped implementation |
| Result quarantine | artifact move but JSON retains `data_mode: real`; recursive aggregator does not exclude quarantine | OPEN; direct recontamination path |
| Unit tests | reported 24+9+4 = 37; no complete logs or CI | REPORTED, not independently reproduced; does not encompass every test module |
| Multi-seed, real benchmark, ablations | none new | BLOCKED / NOT STARTED |

## 3. Results: admissible vs retracted

The latest non-quarantined V3 synthetic run (`results/synthetic_mosei/synthetic/prequential/tri_sacta_online_aci_delay0_seed42.json`) has 3,600 predictions from 600 unique test samples replayed over 6 regimes, seed 42, Macro-F1 0.4091847, accuracy 0.4180556, ECE 0.2064447, NLL 1.5735023, Brier 0.7627288, coverage 0.9027778 (3,250/3,600), mean set size 3.3633333, peak allocated VRAM 32.0005 MB, latency 8.4175 ms/sample. V3 synthetic regime F1: .48394/.42960/.49165/.29863/.23675/.48445. R6-R1 = .0005094 (about 0.051 percentage points); not evidence of improved backward transfer against an unadapted baseline. Current V4 code has changed the spectral-velocity history; those archived metrics are not V4 metrics. V2 seed-42 synthetic F1 .42872, coverage .90, size 3.27 under different protocol and five regimes, not a controlled comparison.

The V3 pseudo-real CMU-MOSEI result (F1 .116739, accuracy .120833, coverage .891667, size 6.270833) is **retracted**. It may be cited only as evidence of the previous integrity failure, never as real experimental evidence. V3 manifests had 320 rows apiece, all 320 label sequences and 320 video IDs matched across the two datasets, and all 640 records had empty feature path/checksum maps. V4 quarantine preserves history.

No source-only genuine multimodal checkpoint, no same-protocol V4 baseline, no real run, no multi-seed uncertainty, no meaningful significance test, no ablation, no deployment-grade profile. Do not infer SOTA or coverage guarantees from V3 smoke data.

## 4. Priority defect register (V5 acceptance gates)

| ID | Priority | Source and failure | Required fix / acceptance |
|---|---|---|---|
| V5-01 | P0 | `scripts/05_aggregate_results.py:35–70` recursive glob excludes legacy/summary but not `quarantined_v3_fixtures`; archived JSON still declares real | Reject every quarantined path, missing provenance, unapproved schema/run ID. Add a regression where real-labelled quarantined file is placed under root; aggregator must not emit real summary. Use canonical run registry, do not modify historical raw JSON. |
| V5-02 | P0 | `scripts/extract_features_sequential.py` same blob as V3, `build_pilot_real_cache`, `--generate-pilot-cache`, Gaussian vectors/random labels in default real path; loading encoder then discarding without forward | Separate fixture command/module with `data_mode=synthetic_fixture`, never real directory. Real extractor must require actual assets and fail on missing inputs/dependencies; actual batched encoder forwards, pooling, alignment, checked serialization. `--dataset-root` must be used. Test missing assets, invalid media, no zero/random fallback. |
| V5-03 | P0 | `src/data/manifest.py:63–71,~179–199`: hash only includes dataset/version + selected IDs/label; file checksum check optional, first 50 rows only, missing expected digest ignored | Require complete manifest and trust anchor, source revision, every source/cache shard SHA-256, metadata/labels/masks/timestamps; canonical serialization and hash; hash each unique file once, verify all referenced files and indices; validate existence, shape/dtype/finiteness, splits, sample joins. Reject empty checksum, row 51 tamper, path traversal/outside approved root. |
| V5-04 | P0 | `src/data/dataset.py:65–120,235–279` manifest checked separately; loader uses config cache_dir independently; no row join, segment_id fallback, availability determined from file presence | Bind cache row to manifest by canonical utterance ID and row index. Verify order, labels, source keys and availability per sample. No fake IDs. Reject mismatch in dataset name/root/hash, missing splits, extra rows, all-missing depending on policy. |
| V5-05 | P0 | `scripts/01_train_source.py:169–189,225–230` checkpoint payload has no manifest_hash and writes raw legacy copy; `02_run_stream_benchmark.py:95–165` still accepts legacy bare dict and loads stats without dataset/hash checks | Reject legacy checkpoints for real runs. Require schema/dataset/mode/seed/config hash/manifest hash/feature hash/source training run ID and stats identity; compare all fields and actual hashes. Test cross-dataset/fixture/mismatched seed/manifest and altered checkpoint. |
| V5-06 | P0 | `docs/THEORY_V3.md:135–153` applies immediate theorem to delayed observed errors; `src/conformal/online_aci.py:238–245` clips fixed state at [-eta,1+eta] | Retract unsupported lag bound for current code. Derive bound including clipping residual and delayed state overshoot, or remove projection and prove boundedness for specified max pending feedback. See §5. |
| V5-07 | P0 | `02_run_stream_benchmark.py` transductive branch calls `shift_vel.values()`, `TriSACTA.get_shift_velocity()` returns float via detector | Unify interface to scalar nonnegative velocity (and optional per-stream map separate). Test both protocols, all methods, 0 and nonzero delay. |
| V5-08 | P1 | `scripts/acquire_authentic_data.py` uses mutable `main`/`master` links and checks locally calculated SHA only; doesn't cross-check expected hashes, official splits, or emit durable acquisition record; no executable raw media downloader despite docstring | Pin dataset revisions, record source URLs/license/citation, size/hash and join counts. Cross-check CMU text mirror against official SDK data; do not call third-party mirror official. Download tools with resume, disk precheck, atomic temporary files; verify archives from source. |
| V5-09 | P1 | `docs/CLAIM_EVIDENCE_MATRIX.md` Gate 4 and 6 checked PASS despite missing actual extraction and multi-seed; `README.md` retains old guaranteed validity interpretation and lacks explicit synthetic caveat near old table | Mark engineering vs scientific gates separately; replace PASS with BLOCKED/NOT_EVALUATED, rename old tables synthetic and specify protocol/seed; update all docs including local file:// links to relative links. |
| V5-10 | P1 | Fixed/dynamic CP default is dynamic; benchmark compares source/static vs TTA/dynamic; no controller identity, per-mask support, full/empty fraction in run JSON | Add controller selection, equal calibration and identical prediction log replay, issued vs observed metrics, velocity/disagreement and clipping residual telemetry. |
| V5-11 | P1 | `fusion_backbone.py:248–250` all-rejected samples restore physical mask; possible corrupted channel leakage in fallback | Explicit fallback/abstain policy, report count, test all physical/effective masks and NaN inputs, no unconditional isolation claim. |
| V5-12 | P1 | 6 regimes loop over same test_loader; filenames lack full immutable run provenance; old results with no protocol default to prequential in aggregator | Distinguish replay stress test from unique progressive stream. Require metadata (commit, dataset/manifest/checkpoint/config/stream hashes, protocol, controller, seeds). Reject ambiguous old runs; no duplicate silent overwrite. |
| V5-13 | P1 | 37 reported tests combine selected suites; `run_all_tests.py` excludes new tests and full pytest parity not demonstrated | One standard CI command; fixtures tagged synthetic, raw data tests explicitly skipped/blocked if input missing, no false success. Save exact commands, logs, counts and exit codes. Test all above failure cases. |
| V5-14 | P1 | SAR and READ approximations, no authentic baseline and no ablation | Use faithful ports or clearly named SAR-style, READ-style, TENT-LN. Compare equal predictor/CP and source budget, never claim SOTA from proxy. |
| V5-15 | P1 | Spectral null grid only median limited resamples/nearest sizes; reliability classification includes prevalence in masked mean; detector has no FPR/AUROC proof | Evaluate null distributions across n, missingness and severity, calibrate thresholds validation-only; compare mean-only, spectrum-only, simpler confidence/moment detectors; record coverage/false alarms. |

## 5. Independent delayed-feedback counterexample to the V4 claimed bound

For fixed step eta=0.01, nominal alpha=.1, one group, score window W=200, delay D=10, T=10,000. Use LAC score for the true class s_t=.2+.75*t/T, increasing and strictly less than 1. These are feasible classification probabilities (true-class probability 1-s_t, remaining mass allocated to six other classes). Emit full/empty sets at boundaries, otherwise use finite rank with cold start and alpha clamp exactly as the relevant code; on matured feedback apply projection [-eta,1+eta]. This minimal controller reproduction was executed independently in Python, not the full Tri-SACTA runner. It gives:

| Delay | Mean error | Gap vs .1 | Claimed upper bound | Cumulative signed clipping residual |
|---|---:|---:|---:|---:|
| 0 | .1010 | .0010 | .0101 | 0 |
| 10 | .3237 | .2237 | .0111 | 22.276 |
| 50 | .4609 | .3609 | .0151 | 35.991 |

The claimed bound is violated for delayed feedback. Immediate boundary emissions only ensure the current prediction's error; stale feedback may continue pushing the state outward after alpha crosses a boundary. Counting pending errors alone cannot correct the accumulated projection residual. Reproduce:

```python
from collections import deque
from math import ceil

T, eta, target = 10000, .01, .1
for delay in (0, 10, 50):
    a = target
    pending, window = deque(), deque(maxlen=200)
    misses, residual = 0, 0.
    for t in range(T):
        score = .2 + .75 * t / T
        if a <= 0:
            e = 0
        elif a >= 1:
            e = 1
        else:
            k = ceil((len(window)+1) * (1-min(1-1e-4, max(1e-4,a))))
            q = 1.0 if k > len(window) or not window else sorted(window)[max(0,k-1)]
            e = int(score > q)
        misses += e
        pending.append((e, score))
        if len(pending) > delay:
            old, old_score = pending.popleft()
            window.append(old_score)
            raw = a + eta*(target-old)
            a = min(1+eta, max(-eta,raw))
            residual += a-raw
    print(delay, misses/T, abs(misses/T-target), residual,
          (1+eta)/(eta*T)+delay/T)
```

A correct diagnostic bound for fixed eta, observed feedback order, N emitted and n observed, K=N-n pending, projection residual c_j defined as projected minus raw: `|mean_emitted_error-alpha| <= |a_after-a_initial|/(eta*N) + sum_j |c_j|/(eta*N) + K/N`. Signed residual identity is exact. The delay bound requires additional reasoning about actual state range and residual terms, not application of immediate theorem. Optional research direction: unprojected fixed-step state with finite maximum outstanding records; prove its delay-dependent state range using maximum number of stale updates and boundary emissions. Do not promise asymptotic validity for arbitrary label missingness or variable eta. Validate under adversarial streams and exact event timing; compare fixed and dynamic controllers using identical logits. Avoid paper novelty claims about basic ACI telescoping.

## 6. Data acquisition plan that works on 4 GB GPU

1. Freeze official sample identities and annotation revisions. MELD official: https://github.com/declare-lab/MELD (ACL 2019 https://aclanthology.org/P19-1050/). CMU-MOSEI original: https://aclanthology.org/P18-1208/ and https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK (verify dataset access). Third-party text mirror https://huggingface.co/datasets/vintp/CMU-Mosei-text must be checked against official IDs/timestamps/splits and licensed use. Do not assume 10–15 GB exact download size; verify actual files and disk.
2. Start with authentic small pilot, e.g. 100–300 utterances per dataset selected from official source partitions, no random replacement. Record selection manifest and class counts. Use official MELD dialogue grouping; recurring characters across splits allowed, never assert speaker-disjoint official MELD. Canonical dialogue IDs must disambiguate split-local IDs using source metadata. Split validation/calibration by groups from allowed source data; keep test frozen. CMU-MOSEI segment timestamp joins and official video folds; no invented speakers.
3. Text-only sanity path now: real tokenizer/forward (DeBERTa or small documented text encoder), attention-masked pooling; use text-only baseline, not Tri-SACTA evidence. If authentic aligned pre-extracted audio and video features are legitimately available, prefer that route for first real benchmark. Update dimensions/extractor names honestly. E.g. official SDK COVAREP/OpenFace features may support latent perturbations; do not call them WavLM/CLIP or report raw-media SNR/WER/blur from feature noise.
4. Genuine raw-media pipeline if accessible: one encoder at a time, batch size 1–8 as fits, inference_mode, fp16 CUDA / FP32 CPU, decode audio to 16kHz mono, record resampling and mask, video frame sampling and pooling, actual openSMILE extraction and transcript tokenizer. Commit versioned extraction code and reports, not restricted raw media. Verify group joins, finite nonzero dimensions and variation, per-item output correspondence, labels from source. Cache per shard, hashes, restart-safe atomic outputs. Verify VRAM allocated/reserved, CPU/host peak and wall-clock; no guarantee without measurements.
5. Source-only pilot on authentic train/val/cal/test, report class distribution, confusion, per-class F1, unimodal baselines, no test tuning. Majority/random baseline sanity check, label mapping and val selection. Thresholds chosen from validation, not an arbitrary fixed performance threshold across datasets. If pipeline is blocked, stop and report exact required archive/path and verified instructions, no synthetic substitutions.

## 7. Scientific contributions and full benchmark plan

C1 sample-size-aware spectral reliability: promising engineering improvement, novelty not established. Match clean sample-size/null distribution, compare simple distance, confidence/entropy, MMD or documented alternative; quantify clean FPR/AUROC/AUPRC and severity monotonicity. Measure CPU/GPU cost and calibration stability on authentic features.

C2 anchor-guided selective adaptation: plausible contribution but must isolate gating-only from adaptation. Compare frozen source, spectral-gated source, uniform adapter TTA, anchor-guided adapters, full adaptation, remove alignment, remove spectral signal, remove hard gates. Measure fraction of actual adaptation steps, selected param deltas, clean/recovery performance and negative transfer per regime.

C3 online mask-aware CP: highest scientific leverage if distinct from established ACI/online conformal works. Fixed vs dynamic, pooled vs per-mask, constant vs velocity-only vs disagreement-only, warm start, label delays, budget 100/25/10/0%, mask subgroup support and size/full/empty rates, worst-group gap. Guarantee must match code, assumptions and feedback. Guarantee of empirical tracking is not pointwise conditional coverage or finite-sample exchangeability validity. Dynamic controller remains empirical heuristic absent a valid theorem.

Fair factorial: evaluate same classifier logits under static split CP, pooled fixed ACI, mask fixed ACI, mask dynamic ACI; evaluate source, gated-only, TENT-LN, faithful SAR/READ or explicitly proxy versions, Tri-SACTA. Same train/val/cal splits, same pretrained source architecture/checkpoint and corruption schedule per seed. Retain replay for recovery only or label it clearly, use unique-sample primary stream grouped by dialogue/video. Latent perturbations named latent; physically grounded corruptions applied before encoding and severity measured. Pre-register test-free hyperparameter tuning and primary endpoints. Run one authentic pilot first; then full one-seed matrix; only after valid outputs run >=5 paired seeds (five exact two-sided Wilcoxon tests cannot attain p<.05; increase seeds if inference desired, report effect sizes and uncertainty, cluster/block bootstrap respecting video/dialogue and replay). Both datasets desirable; one credible benchmark is better than two mislabeled datasets. Literature check on READ, TENT, SAR, ACI, adaptive online CP, recent multimodal reliability methods: verify primary sources and novelty before manuscript. No new literature review claimed in this merge-only recheck.

## 8. Stage gates and owner instructions

| Gate | Condition | Status now |
|---|---|---|
| G0 | Quarantined artifacts inaccessible to real runner AND aggregation | PARTIAL/FAIL |
| G1 | P0 regression tests with exact commit, CI logs and negative controls | PARTIAL |
| G2 | Authentic annotation-to-feature manifest and verified source checkpoint | BLOCKED |
| G3 | Math corresponds to actual controller including delay | PARTIAL/FAIL |
| G4 | Same-input genuine source/tri/baselines, frozen config and metrics | NOT STARTED |
| G5 | Factorial and multi-seed; per-mask delay/budget metrics | NOT STARTED |
| G6 | Novelty and limitations substantiated, independent reproduction | NOT STARTED |
| G7 | Paper and artifact package approved by human PI | NOT STARTED |

**Branch next:** `research/v5-provenance-enforcement`. Agent 2 implements; Agent 3 independently challenges data chain and quarantined aggregation; Agent 4 checks theorem and counterexamples; Agent 5 reruns code from pinned SHA and saves logs, Agent 6 rejects unsupported claims. Agent 1 owns research design. Do not award `SCIENTIFICALLY VALIDATED` for implementation or local assertion alone. Human PI approves G0–G3 before full experiment. No unrestricted auto-merge; protect main with independent checks. Use `BLOCKED` with concrete remediation rather than creating replacement data.

## 9. Copy-ready prompt for Antigravity

Read this report in full. Work from main commit `52ad9b21f66bfd9667baffaf259fe56b8112cb41`. Create a new branch `research/v5-provenance-enforcement`. First fix V5-01 (quarantine bypass) and V5-02–05 (real provenance, actual extraction, cache/checkpoint binding), V5-06 (delayed-feedback theorem), V5-07 (transductive scalar interface). Implement negative-control tests, run actual tests, and commit logs with commands, exact SHA, dependencies, exit codes and explicit unresolved issues. Preserve historical files. Do not fabricate data or logs and do not move a quarantined run back into any publication table. Never accept a manifest merely because IDs lack known synthetic prefixes. Real data must have traceable upstream records and full verified checksums. Separate synthetic fixtures and authentic caches. No silent zeros or Gaussian values in real extraction. If raw assets are blocked, build a genuine text-only sanity run (not a multimodal result) and/or documented legitimate aligned feature import, and give the user concrete download/mount requirements. Only proceed to authentic source-only pilots once gate G0/G1 pass. No large runs until provenance and source validation pass. Update README, CLAIM_EVIDENCE_MATRIX, docs/AUDIT_STATUS_V5.md, docs/HANDOVER_V5.md, docs/DATA_PROVENANCE_AUDIT.md, docs/THEORY_V3.md and the experiment protocol. Distinguish code implemented, tested, independently reproduced, scientifically validated and blocked. Use real evidence; do not claim publication readiness. Submit a PR for human review; do not auto-merge. Report separately what you actually performed, what is blocked, what changed, exact evidence locations, and the next minimal runnable command.
