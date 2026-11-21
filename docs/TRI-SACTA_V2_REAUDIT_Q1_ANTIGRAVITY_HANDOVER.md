# TRI-SACTA V2 — RE-AUDIT, SCIENTIFIC VALIDITY REVIEW & V3 IMPLEMENTATION HANDOVER

**Ngày kiểm định:** 21/11/2026, 21:17 ICT  
**Repository:** https://github.com/buponbonbon/Tri-SACTA  
**Snapshot kiểm định:** `1e00431666a331529c461fce23ccda387a2570df` (`main`, merge PR #2, 21:03:19 ICT)  
**Snapshot cũ:** `17f1acbb777ff7f970210af4d75cd13bcc710798` (17:05:19 ICT)  
**Tài liệu kế thừa:** `docs/TRI-SACTA_Q1_AUDIT_AND_IMPLEMENTATION_ROADMAP.md`  
**Đối tượng:** Antigravity 3.8 / Senior ML Research Engineer / người hướng dẫn nghiên cứu / reviewer Q1.  
**Quyết định:** **NO-GO cho submission Q1; GO cho V3 engineering + scientific validation.**

> Đây là audit độc lập dựa trên mã nguồn, bảng kết quả JSON và tài liệu được commit, không phải chứng nhận đã tái lập toàn bộ huấn luyện. Git clone trong môi trường kiểm định thất bại do DNS; chưa chạy bộ test hoặc GPU benchmark độc lập. Các lỗi dữ liệu, giao thức và toán học được xác định qua đọc code; các giả thuyết hiệu năng phải được thử nghiệm. Không có căn cứ kết luận số liệu bị bịa; các số tổng hợp coverage/set size đã đối chiếu nhất quán với time series. Không đánh đồng số liệu nhất quán với chứng minh khoa học hợp lệ.

## 0. PROMPT THỰC THI CHO ANTIGRAVITY (copy nguyên khối)

```text
Bạn là Senior ML Research Scientist + Research Software Engineer + reviewer Q1. Đọc hết tài liệu TRI-SACTA_V2_REAUDIT_Q1_ANTIGRAVITY_HANDOVER.md và audit trước trong docs/. Kiểm tra lại HEAD thực tế, không ghi đè kết quả V1/V2. Tạo branch research/v3-scientific-validation, bảo lưu SHA và run artifacts. Không tự nhận Phases complete chỉ vì có skeleton, YAML, bảng Markdown, unit test yếu hoặc lời tuyên bố.

Mục tiêu: sửa lỗi khoa học và thực thi còn lại, đặc biệt R3-01..R3-12. Trước tiên loại bỏ theorem/guarantee sai khỏi README, METHOD_SPEC, THEORY và PROJECT_SPEC; bổ sung phản ví dụ variable-step ACI; triển khai fixed-step per-mask ACI tham chiếu đúng với boundary full/empty action, rồi đánh giá controller adaptive như biến thể thực nghiệm hoặc chứng minh chính xác với giả định/terms rõ ràng. Không tự tạo định lý mới nếu chưa chứng minh được. Sau đó hoàn thiện real dataset ingestion, manifest validator bắt buộc trong train/benchmark, checkpoint namespace + provenance, extractor thực sự xử lý từng mẫu, correct corruption semantics, strict prequential và transductive tách riêng, stream manifest dùng chung mọi method, và cổng synthetic/real nghiêm ngặt.

Chạy test thực tế, lưu command, stdout/stderr, git SHA, dependency versions, GPU/CPU, exit code và artifact hash. Phải có non-vacuous tests cho active adapter update, corrupted-token invariance, source sparse-mask spectral null, feedback delay 0/10/50, wrong checkpoint reject, empty manifests reject, stream protocol parity, matrix job paths, unique-seed aggregation. Nếu thiếu raw data/quyền tải/licensing/GPU/network, ghi BLOCKED với bằng chứng, không fallback synthetic trong real mode, không tạo fake files. Nếu chưa có raw media có quyền, triển khai feature-cache pilot được ghi nhãn rõ; không gọi latent noise là waveform SNR hoặc feature masking là WER.

Hoàn thiện thật sự experiment matrix: train source riêng mỗi dataset/seed, lưu checkpoint và stats cùng manifest hash, tạo frozen stream (ordered event_id/mask/corruption/payload hashes) trước khi chạy method; dùng same stream & same checkpoint cho đối chứng. Chạy smoke synthetic end-to-end rồi >=3 seed debug, pilot dữ liệu thật, sau đó 5 seeds x 2 datasets với paired design. Đối chiếu Source-only + cùng CP, reliability-only, entropy-only, selective random, aligned adaptation, constant ACI, dynamic ACI no-mask/with-mask, rare fallback and no fallback; chạy baseline chính thức/port được ghi rõ, không gọi SAR/READ-style là original SOTA. Thêm ablations và report per regime, per mask (n/hits/size/CI), R1 vs R6 recovery, delay and label-budget, calibration Pareto, efficiency, CIs and effect sizes. Không chọn seed/hyperparam bằng test.

Mỗi WP có PASS/FAIL/BLOCKED/NOT RUN; không đánh dấu PASS khi chỉ có code hoặc chưa có log. Ghi docs/AUDIT_STATUS_V3.md, docs/HANDOVER_V3.md, docs/CLAIM_EVIDENCE_MATRIX.md, docs/EXPERIMENT_PROTOCOL_V3.md và docs/THEORY_V3.md. Sửa các file được nêu trong issue register và bổ sung tests. Chỉ khi đủ evidence mới đóng issue. Không tự submit bài hoặc push main nếu chưa được yêu cầu. Khi kết thúc trả về: commit SHA, files changed, test report, verified real dataset counts, run manifests, exact results, remaining blockers, next executable command. Bắt đầu từ WP0, WP1, WP2; các blocker cần xử lý trước khi chạy full matrix.
```

## 1. Executive summary & quyết định đầu tư

Antigravity tiến bộ rất mạnh về software engineering trong đợt hardening (từ V1 sang V2): sửa attention một token, thay đổi scoring, xây dựng selective freeze, auxiliary heads, positive learning-rate bounds, ghi nhận prediction record, tách debug data, mở rộng test suite, tạo docs và ma trận benchmark. Model collapse không còn thể hiện trong kết quả synthetic mới. Tuy nhiên, scientific readiness vẫn thấp hơn engineering readiness.

**Kết luận trọng tâm:** V2 là engineering prototype có kết quả synthetic pilot đáng khích lệ, chưa phải validated research method. Báo cáo `HANDOVER.md` tự gọi Phase 1–3 complete và `AUDIT_STATUS.md` đánh dấu 26/26 resolved; không nên đồng nhất với hoàn thành nghiên cứu. Một số hạng mục resolved thực chất mới là partial/stub; có claim toán học không hợp lệ. Cần giữ định hướng đề tài nhưng giảm phạm vi claim, tập trung phần mới có thể chứng minh được, tránh thêm nhiều module chỉ để tăng complexity.

| Dimension | V1 (ước lượng) | V2 (ước lượng) | Điều kiện nâng mức |
|---|---:|---:|---|
| Tiềm năng đề tài | 8/10 | 8/10 | Xác lập gap với ACI 2024, MCCP 2026, multimodal TTA 2025–26 |
| Engineering | 4/10 | 7/10 | CI + integration tests + actual data pipeline |
| Tính đúng đắn mô hình | 3/10 | 5.5/10 | Non-vacuous adaptation, mask isolation, sample-size matched detector |
| Theory | 3/10 | 3/10 | Sửa theorem sai; chứng minh đúng code |
| Thực nghiệm khoa học | 1.5/10 | 3/10 | Hai real datasets, multiple seeds, parity, ablations |
| Reproducibility | 3/10 | 4.5/10 | Immutable provenance, real checkpoint/stream identity, actual rerun |
| Sẵn sàng submission Q1 | 3/10 | 4/10 | Các gates dưới đây |

Điểm chủ quan, không phải xác suất acceptance hay thang điểm của journal. Tiến độ tổng thể research ước lượng **40–45%** (V1 30–35%); cơ sở code khoảng 65–75%, evidence scientific khoảng 20–30%. Các phần không tỷ lệ tuyến tính. **Đang cuối giai đoạn prototype engineering và đầu giai đoạn scientific validation**, chưa bước vào full experiments trên dữ liệu thật. Không cần đổi hướng ngay; chỉ đổi/pivot nếu pilot thật và ablation không cho thấy lợi ích sau khi sửa protocol.

### 1.1. Lịch sử cập nhật

| Commit | ICT | Mô tả |
|---|---|---|
| `17f1acb` | 17:05 | V1 merge |
| `96d8d8e` | 20:04 | P0 hardening, synthetic restoration, 15 tests được báo cáo |
| `8a1ba91` | 20:24 | P1, data architecture, theory, matrix, 24 tests được báo cáo |
| `571e48f` | 20:34 | README audit update |
| `8fe100e` | 21:01 | README academic rewrite |
| `1e00431` | 21:03 | Merge PR #2 |

Nguồn: https://github.com/buponbonbon/Tri-SACTA/commits/main/ . Tổng 8 commits quan sát được, branch list chỉ `main` tại thời điểm audit. Checkpoints trong git chỉ có `.gitkeep`, không kết luận checkpoint local không tồn tại.

## 2. Audit kết quả thực nghiệm — đọc trực tiếp JSON

Tất cả 7 run V2 trong `results/*synthetic*` khai báo `data_mode=synthetic`, dataset `mosei` (tên cấu hình, **không phải CMU-MOSEI thật**), seed 42. Mỗi run có 95 batch steps (19 x 5) và 3.000 prediction events từ 600 test samples lặp lại 5 lần. Không có kết quả real/MELD trong tree. Có hai JSON V1 còn ở thư mục results gốc và bản sao `results/legacy/`; không trộn với V2. Bảng dưới làm tròn từ JSON, không phải chạy độc lập.

| Method | Accuracy % | Macro-F1 % | Coverage % | Gap pp | Set size | ECE % | NLL | Brier | Peak allocated MiB | ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source-only / static CP | 42.9667 | 42.2032 | 91.7333 | 1.7333 | 3.2967 | 18.9187 | 1.4670 | .7296 | 21.59 | .6838 |
| TENT-style / online ACI | 43.0333 | 42.1559 | 90.3000 | .3000 | 3.3267 | 19.4602 | 1.4922 | .7394 | 36.82 | 2.8442 |
| SAR-style / online ACI | 43.2333 | 42.4836 | 90.1333 | .1333 | 3.3060 | 19.1176 | 1.4844 | .7372 | 36.82 | 2.8018 |
| READ-style / online ACI | 40.7333 | 38.9452 | 90.3333 | .3333 | 3.4230 | 21.0622 | 1.5244 | .7590 | 49.19 | 4.8900 |
| Tri-SACTA / online ACI | 43.5333 | 42.8723 | 90.0000 | .0000 | 3.2700 | 18.6214 | 1.4716 | .7343 | 31.79 | 5.5372 |

**Conclusions:** +0.6691 pp Macro-F1 over source-only, +0.3887 pp over SAR-style, +0.7164 pp over TENT-style, +3.9271 pp over READ-style. Accuracy improvement vs source-only 0.5667 pp = 17 correct events / 3,000 (not 17 independent samples). Improvement vs SAR-style 0.3000 pp = 9 events. Not sufficient to claim statistically significant superiority. Source-only has best NLL and Brier, slightly larger set size but higher coverage, and much faster latency. Tri-SACTA latency ~8.1x source-only, ~2x SAR-style. Coverage equal target is calibration tracking, not accuracy, not proof of guarantee. Gap absolute penalizes overcoverage and undercoverage equally but risk implications differ. Compare set size at matched coverage, include undercoverage separately and coverage-efficiency frontier. Do not cherry-pick metrics.

### 2.1. Regime evidence

| Regime | Source F1 % | TENT F1 % | SAR F1 % | READ-style F1 % | Ours F1 % | Source coverage % | Ours coverage % | Ours set size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 clean | 48.87 | 49.14 | 49.01 | 47.21 | 48.45 | 93.33 | 90.17 | 2.94 |
| R2 audio | 44.73 | 46.64 | 46.78 | 42.51 | 47.49 | 94.33 | 90.00 | 2.83 |
| R3 text | 44.83 | 44.84 | 45.77 | 40.87 | 46.72 | 94.50 | 91.00 | 2.95 |
| R4 visual/missing | 33.14 | 33.04 | 33.49 | 30.06 | 34.39 | 88.67 | 91.50 | 4.17 |
| R5 compound + recovery | 38.53 | 36.01 | 36.45 | 33.24 | 36.51 | 87.83 | 87.33 | 3.46 |

R2–R4 gains are encouraging, R5 **ours loses ~2.02 pp F1 vs source-only and ~0.50 pp coverage**. R5 is mixed severe + clean recovery; pure severe and pure recovery are not isolated. Cannot claim prevention of forgetting. Overall 90% hides R5 undercoverage. R1 also slightly worse than Source/TENT/SAR. Evaluate R6 with matched clean reference and disjoint held-out events. No per-mask output or support counts to verify worst group. In particular, rare-mask fallback not validated by this aggregate.

### 2.2. Delay evidence

| Delay (sample prediction steps, not batch) | Coverage % | hits / 3000 | Set size | F1 % | R2 coverage % | R5 coverage % |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 90.00 | 2700 | 3.2700 | 42.8723 | 90.00 | 87.33 |
| 10 | 89.43 | 2683 | 3.2063 | 42.8723 | 89.50 | 86.67 |
| 50 | 88.30 | 2649 | 3.2923 | 42.8723 | 84.33 | 84.50 |

Coverage drops 1.7 pp at 50, especially R2/R5. F1 unchanged is expected because CP labels are not used for model adaptation. This validates consistency of design, not robustness guarantee. JSON confirms `coverage_hits` mean and set length means match aggregate across all 7 runs. Accuracy, F1, ECE cannot be independently reconstructed from saved aggregate logs; no per-event label/logits/confusion matrix. The delay result does not verify correctness of timing until event ordering and feedback replay are tested. 95 shift velocity values, 3,000 alphas, 3,000 set sizes/hits; different units must be explicit. Tri-SACTA velocity min -85.0756/max 46.4746; disagreement mean 0.03839, max 0.07275. This verifies disagreement no longer identically zero, not that pairwise valid masks/definition are correct.

### 2.3. Legacy comparison

V1 ours F1 3.7985%, acc 15.3333%, coverage 79.70%, set size 5.5873. V2 ours F1 42.8723%, acc 43.5333%, coverage 90%, set size 3.27. Strong sign engineering collapse removed in stored outputs. But source model, synthetic seed generator, metric definitions, noise and architecture changed; this is NOT controlled causal attribution. Keep V1 unchanged; rerun controlled module ablations using V2/V3 frozen data stream for claims. Do not claim `11x F1` as research gain.

### 2.4. Statistical status

`results/summary/multi_seed_statistical_summary.md` reports Seeds=1 for every row. No empirical seed variance estimate; `±0.0000` is display fallback, should be `NA (n=1)`. No p-values. With five genuinely paired independent seed units, exact two-sided Wilcoxon minimum attainable p-value without ties is 2/32 = 0.0625; don't promise p<.05 with only five. Choose preregistered effect sizes and intervals; if formal exact seed-level tests needed, increase independent runs (>=6 theoretical minimum and more for power). Do not treat 3,000 repeated events as iid. Use cluster/block bootstrap by original video/dialogue/sample identity with temporal awareness and report seed-level paired differences. If multiple hypotheses, adjust (e.g. Holm), don't selectively report. Define primary endpoint before real test.

## 3. Original AUD-01..26 status reclassification

Labels: `IMPLEMENTED` = static source evidence, not independently executed; `PARTIAL` = incomplete design/verification; `REOPEN` = concrete issue conflicts with reported resolved status. Prior test counts are agent-reported. Full status is not 26/26 closed.

| Old ID | Re-audit | Evidence / caveat |
|---|---|---|
| AUD-01 | PARTIAL | normalized scoring, floor, no collapse in JSON; sample-size bias persists |
| AUD-02 | IMPLEMENTED | multi-token Transformer replaces singleton softmax |
| AUD-03 | REOPEN | effective corrupted mask not applied inside transformer; physical missing handled better |
| AUD-04 | PARTIAL | target valid rows filtered; source anchor collection includes zero-filled rows |
| AUD-05 | PARTIAL | selective requires_grad implemented, tests may pass without active update; no-anchor promotion conflicts spec |
| AUD-06 | IMPLEMENTED | auxiliary supervised heads added; independent head metrics absent |
| AUD-07 | PARTIAL | bounded eta and nonzero JS implemented; signed average cancellation, missing rows in JS; theorem invalid |
| AUD-08 | PARTIAL | stored emitted sets and feedback queue; explicit timing/IDs and deep immutability missing |
| AUD-09 | PARTIAL | exact rank implemented; finite-sample guarantee assumptions/boundaries still require audit |
| AUD-10 | PARTIAL | explicit real/synthetic; real mode still allows optional metadata and legacy checkpoint fallback |
| AUD-11 | REOPEN | manifest overlap helper not enforced in actual loader; no checksum/order validation |
| AUD-12 | IMPLEMENTED | per-sample collator; cache missingness real metadata and all-missing policy unresolved |
| AUD-13 | IMPLEMENTED | deepcopy best state |
| AUD-14 | IMPLEMENTED | stable MD5 for synthetic split; broader seed provenance still needed |
| AUD-15 | IMPLEMENTED | safe method interface; baseline fidelity separate issue |
| AUD-16 | REOPEN | latent perturbations mislabeled SNR/WER/blur; generator V2 not connected to runner |
| AUD-17 | PARTIAL | post adaptation eval but comparable training mode must be explicitly reset; baseline fidelity unresolved |
| AUD-18 | REOPEN | hard global fallback, not hierarchical shrinkage theorem; conditional coverage unsupported |
| AUD-19 | PARTIAL | coverage_hits naming fixed; filename collision across config/dataset in checkpoint/matrix; missing provenance |
| AUD-20 | PARTIAL | improved metrics source; some saved NLL values FP16-like, commit provenance missing; rerun |
| AUD-21 | REOPEN | incorrect variable-step theorem, delayed theorem unsupported, specification/code mismatches |
| AUD-22 | PARTIAL | 24/24 claimed; no independent logs, non-vacuous & integration tests needed |
| AUD-23 | PARTIAL | profiler extended, output JSON only peak allocated/latency |
| AUD-24 | REOPEN | checkpoint names mode+seed only; cross-dataset collision; legacy fallback |
| AUD-25 | IMPLEMENTED | post-update eval observed; strict prequential not implemented |
| AUD-26 | REOPEN | extractor returns empty dict; CLI just instantiates |

## 4. Issue register V3 (priorities and acceptance tests)

Critical P0: integrity, theory and correctness before large experiments. P1: scientific evaluation and reproducibility. P2: manuscript/polish. Every closure requires evidence path + command + run SHA. Lines refer to pinned HEAD; may shift after fix. Use links `https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/<path>#Lx-Ly`.

### R3-01 (P0) — Invalid variable-step ACI convergence theorem
**Evidence:** `docs/THEORY.md`, Theorem 2 and Proposition 1; `src/conformal/online_aci.py:190–199`, clipping to [0.01,0.99]. Bounded positive adaptive eta does not imply unweighted calibration. Proof drops total variation, clipping residual. Current code clips away full/empty boundary state, so persistent errors can saturate. README promises asymptotic coverage under non-exchangeability.

**Mathematical counterexample:** alpha target a=0.1; initialize a_1=0.5. e_(odd)=1, eta_(odd)=0.01; e_(even)=0, eta_(even)=0.09. Odd update = -0.009, even = +0.009. a_t alternates .5,.491, entirely interior; e mean=.5 while target=.1. No clipping involved. This refutes the stated algebraic theorem under only its given step bounds/interior assumptions; not a claim this is the particular natural stream. The proof therefore cannot support current state-dependent algorithm without extra assumptions. See independent arithmetic check `verify_aci_counterexample.py` equivalent test below.

**Correct identity:** Define clipping residual c_t = a_(t+1) - [a_t + eta_t(a-e_t)]. Then

$$\sum_{t=1}^T(a-e_t)=\sum_{t=1}^T\frac{a_{t+1}-a_t}{\eta_t}-\sum_{t=1}^T\frac{c_t}{\eta_t}.$$

Let b_t=1/eta_t, |a_t|<=A. Summation by parts gives

$$\sum_{t=1}^T b_t(a_{t+1}-a_t)=b_T a_{T+1}-b_1 a_1+\sum_{t=2}^T(b_{t-1}-b_t)a_t.$$

Hence a valid general diagnostic bound is

$$\left|\bar e_T-a\right|\le\frac{2A/\eta_{min}+ A\sum_{t=2}^T|1/\eta_t-1/\eta_{t-1}|+\sum_{t=1}^T|c_t|/\eta_t}{T}.$$

This is not a useful vanishing guarantee if variation or clipping accumulates linearly. Weighted identity (without clipping): $\sum \eta_t(e_t-a)=a_1-a_{T+1}$; weighted calibration does not imply unweighted. Bounded step sizes alone do not make variation sublinear. If using sufficient conditions (sublinear variation and residual terms), justify why implemented algorithm satisfies them, otherwise state conditional result only. Do not advertise bound as novel without prior-art review.

**Recommended reference controller:** fixed learning rate per mask in local update index, unprojected alpha state with explicit all-label set when alpha<=0 and empty set when alpha>=1, using order statistic only inside interval, zero forced-top1 for boundary empty sets. Under immediate feedback and error e in {0,1}, initial alpha in [0,1], overshoot bounded in [-eta(1-a),1+eta*a]. For group g after n_g observed updates, telescoping gives |mean error-a| <= (1+eta)/(eta n_g). State exact assumptions, including label feedback available, non-empty/empty policy and fixed eta; no per-event conditional probability guarantee, rare group may have no useful bound. Implementation may choose empirical controller if empty sets unacceptable, but then no general pathwise theorem. Report abstention/empty rates. Adaptive recommendation: implement official ACI/FACI/online conformal reference and compare fixed/heuristic adaptive; only pursue a theorem compatible with actual code. Do not force new theorem to fit desired results. A validated empirical paper without an invalid theorem is preferable.

**Tests:** 100, 10k-step alternating counterexample; constant eta telescoping incl boundary; all misses until full set; all hits until empty set; clipping residual accounting; independent masks, rare group with n tiny; ties and score normalization. Documentation must explicitly distinguish exchangeable split CP vs time-average tracking under arbitrary streams. No claim "90% guarantee" unless proved for exact variant. **Acceptance:** theorem reviewed independently, proof + code correspondence table + test logs, old claims removed. If no proof, mark theoretical extension OPEN and report empirical only.

### R3-02 (P0) — Unsupported delay theorem and timing semantics
**Evidence:** `docs/THEORY.md` Theorem 3 claims a bound involving delay, eta_max and spectral L_shift, without derivation of loss/set dependence. Feature spectral velocity alone cannot control label/concept shift: X stationary, labels change. `online_aci.py:116–235` uses per-prediction `_global_t`, buffers labels immediately in simulator, matures after prediction, `observe` pops record id and silently returns for missing ID. Frozen dataclass with list/tensor contents is not deeply immutable; CPU detach can alias memory. `predict_set` returns same mutable list. `update` implicitly uses latest id ignoring logits/mask args. Per-batch labels can become available sequentially even though logits computed batchwise. JSON lacks delayed pending/observed counts.

**Fix:** explicit event simulator separate from predictor: predict(event_id,time,mask,features)-> immutable tuple of label IDs and cloned FP32 scores; schedule feedback in external queue; process due labels before next prediction if defined, or document after prediction; call observe(id,label,arrival_time) only when due. Record issuance time, maturity, actual label feedback order, steps in sample-vs-batch units. Define behavior for missing feedback, duplicate/unknown id (raise or audited rejection), out-of-order events, final pending labels. Distinguish emitted-event coverage computed by offline evaluator from observed-feedback coverage (do not flush early and feed future scores). Don’t silently re-compute sets or use future labels. For fixed-step delayed controller, prove bounded outstanding feedback model separately, including local mask clocks and pending terms; do not use L_shift-only claim unless justified. Test shift of labels without feature change, variable delays, delay 0/10/50, all pending for short stream, late feedback, mutation and duplicate record. **Acceptance:** event log replay yields exactly the same emitted sets and outcomes; delay semantics stated; theorem either rigorously proved or withdrawn.

### R3-03 (P0) — Real feature extraction is a stub
**Evidence:** `scripts/extract_features_sequential.py:46–147` functions initialize models, set `extracted_files={}`, return empty dict; visual `# Feature extraction loop...`; audio/prosody/text no extraction, CLI at bottom only instantiates and logs Ready. No raw path / metadata mapping input, no output tensors or manifests. Cannot claim peak extraction VRAM or successful Stage 1.

**Fix:** implement actual CMU-MOSEI/MELD data loaders and aligned sample IDs, licensing-aware access instructions (do not fabricate raw data). Alternative fast path: officially released pre-extracted features with explicit provenance/dims; change config to actual features, no CLIP/WavLM claim unless used. For raw: video segment decoding sampled frames, CLIP processor+mean pool; 16kHz audio preprocessing/chunking, WavLM pooling; openSMILE eGeMAPS functionals; DeBERTa masked token mean pooling; explicit corrupt-asset/missing policy. Cache by dataset version/sample ID/encoder revision/config, atomic writes, FP32 computation or safe FP16 storage, CPU fallback (avoid unsupported fp16 CPU ops). Requirements lock or optional extra dependencies (transformers, decoders, opensmile, etc). Implement resume, hash verification, non-empty outputs, memory barriers plus measured peak. Add CLI `--manifest --dataset-root --output-dir --modalities --device --resume --max-samples`. Only claim supported flags after implementation. Pilot >=20–100 samples across modalities, inspect actual files shapes and counts, then train pilot with real data. **Acceptance:** command logs, real features and hash manifest, nonzero valid data, verified train/val/cal/test cache, no placeholder success. If no access, status BLOCKED and explain exact assets needed.

### R3-04 (P0) — Real mode provenance and split leakage not enforced
**Evidence:** `src/data/dataset.py:65–103` `meta.pt` optional; if missing, invents segment/speaker IDs in real mode. Available feature file -> marks every row present irrespective true metadata. `src/data/manifest.py` only compares speaker/video sets across supplied splits, empty manifest returns valid, no duplicate IDs, label/dim/count/checksum/path verification. `scripts/00_validate_data.py` real branch only checks manifest existence/overlaps; no cross-check of actual tensors. `get_dataloaders` does not enforce manifest validator; train/benchmark can bypass. `configs/meld_real.yaml` claims dialogue_speaker_disjoint; MELD recurring characters across official partitions makes universal speaker-disjoint potentially infeasible (must check actual manifests); dialogue_id available but not validated.

**Fix:** one schema validator called automatically before all real jobs. Enforce non-empty allowed splits, unique sample/event ID, dataset/version, required original metadata, verified label mapping within range, finite tensors/shapes/dtype/lengths, true per-sample masks, SHA-256 of cache and ordered ID index, no duplicates or cross-split overlap at dataset-specific granularity. For MOSEI choose standard official video splits and inspect actual speaker identities; separate calibration from train/validation/test via grouped partition. For MELD maintain official dialogue splits, verify no dialogue overlap; report speaker overlap (or construct genuinely disjoint separate protocol with class support and explicit comparability consequences). Never fabricate IDs. Reject all-missing samples or implement documented no-input abstention and report count, never silent bias logits. Build tests on tiny fixture with manifest intentionally wrong, duplicate ID, wrong SHA, count mismatch, invented metadata absent, missing dialogue ID, no valid split, unknown class, NaN, row permutation, zero mask. **Acceptance:** actual per-split counts/classes/groups and hashes; zero forbidden overlap; verified provenance at train/benchmark entry. No data -> blocked.

### R3-05 (P0) — Checkpoint and source stats collision/mode contamination
**Evidence:** `scripts/01_train_source.py:165–195` uses `source_model_{data_mode}_seed{seed}.pt` and stats analog; no dataset/config identity. `scripts/02_run_stream_benchmark.py:95–151` searches both mode-only and legacy seed-only checkpoint/stats, permits real mode to load legacy/no matching meta. Both datasets seven classes same architecture, silent load possible. Debug may use random model. Source stats missing recomputed from train; source training and extraction flavors need consistency.

**Fix:** namespace `checkpoints/{dataset}/{data_mode}/{feature_manifest_hash}/{source_config_hash}/seed{seed}/...`. Metadata embedded in checkpoint (state_dict + schema + hashes + split ID + architecture + seed + best epoch/metric + commit + package versions) and stats. Require exact match in scientific mode. No untrained/legacy fallback except explicit `--allow-untrained-debug` stamped debug and excluded by aggregator. Real fail-closed unless verified manifest and checkpoint. If no existing checkpoint, matrix trains once per dataset/seed/config. Identical checkpoint hash across baseline pair; distinct per dataset. **Acceptance:** cross-dataset same-shape checkpoint rejected, synthetic passed to real rejected, wrong config/stats rejected, missing trained source fails for scientific runs. Record model hashes in JSON.

### R3-06 (P0) — Protocol mismatch, repeated samples, fairness parity
**Evidence:** `docs/EXPERIMENT_PROTOCOL.md` defines strict prequential and six regimes; `scripts/02_run_stream_benchmark.py:202–288` instantiates `ShiftStreamGenerator(config,seed=...)` (default `use_protocol_v2=False`), loops full test_loader each regime and calls adapt_step before predict. `configs/shift_regimes_v2.yaml` exists but not consumed by the shown runner. R5 combines severe and return-to-clean. Same sample across regimes okay as stress replay if explicitly reported; cannot count all as independent prospective events.

**Fix:** explicit `--protocol prequential|transductive`, `--stream-spec` parsed validated YAML, `--stream-id` frozen ordered event file. Prequential emit with theta_t then adapt current batch; transductive adapt then emit; choose default according to research question, label both, same method protocol. Compute shift only once (avoid tracker update twice). Design 6 independent phases with recovery R6; pilot strict chronological/dialogue order and separate repeated controlled evaluation for paired resilience. Use source-disjoint heldout clean reference for forgetting. For each event store ID, original sample, regime, mask, corruption param/seed/hash, step, group/time; separate stream RNG from model RNG, materialize offline or stateless by event. All methods must read identical perturbations; checksum event stream, including seeds. Include abrupt/gradual/recurrent/drift and label-shift stress (explicitly separate). Add unit tests config honored, 6 regimes actual, no duplicate original IDs in live stream except documented replay, predict-before-update vs adapt-before-predict, same stream digest. **Acceptance:** actual event logs, R6 separate, no protocol mixing, pair comparability verified. Reject output if metadata missing.

### R3-07 (P0) — Corrupted modalities still leak into fusion and disagreement
**Evidence:** `src/adaptation/tri_sacta.py:111–200` all model calls use original `modality_mask`; `src/models/fusion_backbone.py:198–274` passes tokens through attention using presence mask only, reliability weighting applied after transformer, high shift reliability floor >0. Parameter freeze only prevents updates, not influence. JS at `tri_sacta.py:315–337` computes over whole batch for available modality names without intersecting valid sample masks. `classify_modalities:216–248` uses mean of mask*r (missingness biases r); promotes adaptable modality to anchor, contradicting "skip when no clean anchor" spec. README says strictly isolates corrupt channels, too strong.

**Fix:** decide and document hard gating vs soft reliability; if hard use effective reliability mask before embedding attention (including query/key/value and pooling), remove corruption content from anchor teacher. If no clean anchor, skip unless explicitly defined weak-anchor variant. If all corrupted and some present, controlled fallback/abstain with reason, do not relabel as clean silently. Separate `quality_r` among observed rows from `availability_fraction`, per-sample masks vs batch partition. JS only valid pair intersection per sample, handle single stream no pairs and empty validity. Define pairwise vs Jensen-Shannon to pooled mean consistently. In AMP, `where` or masked_fill before arithmetic to avoid NaN*0. Use consistent KL teacher->student (PyTorch KL(log student, teacher) computes KL(teacher||student), doc currently flipped), align averaging, evaluate teacher confidence/margin. **Acceptance:** change corrupted token to arbitrary finite large values (with same gate) yields identical clean token/logit and no gradient influence under hard gate; physically missing as well; no-anchor skip; valid-pair disagreement test. Record fallback counts, gate histograms, actual per-modality parameter deltas. If soft gate chosen, no claim strict isolation; demonstrate sensitivity bounds empirically.

### R3-08 (P0) — Spectral score confounds missingness, sample size and shift
**Evidence:** `src/models/spectral_detector.py:71–169`. Null median from five random subsamples of size 32, **training** features, not held-out validation as docs say; k=min(32,D) not sample-size matched. Covariance rank <=n_valid-1, logs of floored zero eigenvalues can heavily inflate small batches (n=8,16 or missing rows) vs n=32. n<2 sets d=10 instead of unknown. Source stats collection in `01_train_source.py:169–177` and benchmark fallback includes zero-filled missing rows. No mask-aware heldout quantile, no FPR evidence (median subtraction is not calibrated 5% threshold). Uses diagonal standardized mean, not full Mahalanobis. Eig spectrum loses orientation, cannot detect all distribution shifts.

**Fix:** source train stats on valid rows only; source validation null by modality x valid-count (and possibly missingness regime), with enough resamples, explicit floor/rank/shrinkage and confidence/uncertainty; compare fixed k=min(k,n-1), matched reference spectra, regularized covariance, random projections or Gram/SVD for laptop. For rank n-1 compare corresponding rank source only and calibrate same procedure. n=0 absent; n=1 mean-only or uncertain/no adaptation instead of forced corruption. Explicit handling label-shift (can change mean not indicate sensor corruption), common covariance orientation rotation example. Select thresholds on validation, test detector ROC/PR/FPR by modality severity and batch sizes 1/2/4/8/16/32 including identical clean distribution partial missingness. `avg(abs(delta))` not `abs(avg(delta))` to avoid cancellation; either per event or batch documented. **Acceptance:** fixed clean validation FPR and confidence bounds (target FPR <=5% only if evidence), clean small batch does not spuriously drop, monotone severity trend tested, compare cheap mean-only/variance-only/MMD or CORAL; report computational cost, detector ablation. Don't claim exact distribution detection.

### R3-09 (P0) — Latent corruption mislabeled physically grounded + all missing on real input
**Evidence:** `src/data/shift_generator.py:149–195` noise power on embeddings, text modifies coordinates with random dimension mask, visual adds Gaussian feature noise and zeros random dims. Not waveform SNR, word edit rate or image blur. Dropout at `204–238` ensures not all *chosen* modalities removed, but not necessarily an already available modality remains when original input has partial missingness. Prosody derived from audio; independently corrupted prosody features not same as extraction from noisy waveform.

**Fix:** two benchmark tiers. Tier A `latent_snr`, `feature_mask_rate`, `feature_noise_std` with no physical label; exact measured feature power or expected SNR distinction. Tier B raw signal pipeline with waveform SNR (normalize measured noise power, compute realized dB), transcript edit operations/actual ASR WER (distance on words), frame blur/occlusion before encoding, recompute prosody from same corrupted audio, missingness mechanisms MCAR/MAR/MNAR or structured sensor outage described. Validate dropout on preexisting partial masks and all-missing policies. Make feature-space tier fast optional, actual raw corruption pilot across subset before claiming sensor robustness. **Acceptance:** raw artifact evidence and metric verification OR correct latent-only claim; exact event masks recorded.

### R3-10 (P0) — Experiment matrix runner cannot guarantee trained/immutable jobs
**Evidence:** `scripts/04_run_experiment_matrix.py` computes `method_conformal_data_mode_delay0_seed42.json` while benchmark writes `method_conformal_dataset_data_mode_seed42.json` omitting delay0; nonzero delay paths also differ by dataset. Skip checks wrong filename, returned target path may not exist. `run_single_experiment` only benchmarks, never source-trains for new seeds, no artifact existence/hash guard, default 3 seeds conflicting with advertised 5. `02` writes same path overwriting previous jobs. Pair fairness unverified.

**Fix:** shared run identity schema + helper; matrix schedules source train and validated stats/checkpoint per seed; run path e.g. `results/runs/{dataset}/{mode}/{protocol}/{stream_hash}/{source_hash}/{method}/{cp}/{config_hash}/seed42/delay0/run.json`. Include timestamp/UUID or reject overwrite if different identity; atomic outputs and status files, logs stdout/stderr, resume only when run identity and schema and artifact hashes valid. Benchmark explicit output path returned in machine-readable record. Matrix asserts output exists, expected metadata matches, exits nonzero on failed jobs. Dry-run prints actual valid commands. Test two seeds x two methods integration CPU tiny dataset; rerun skip correct; changed dataset/delay/protocol not skip. **Acceptance:** complete run matrix with verified trained checkpoint and stream per job; zero missing output paths; actual saved multi-seed runs.

### R3-11 (P0) — Aggregator mixes distinct experiment identities and misreports uncertainty
**Evidence:** `scripts/05_aggregate_results.py:28–50` groups only method/conformal/delay, not dataset, mode (when all), protocol, config, stream, checkpoint, seed uniqueness. `format_mean_std` sets n=1 std=0; `n_seeds=len(runs)`, repeated seed can double count; matched by seed only. Summary includes seed=1 yet filename multiseed. Results old and new can blend if pattern filtering brittle.

**Fix:** aggregate only schema-valid runs; group by dataset/version/mode/protocol/stream family/config/preprocessing/checkpoint family/label budget/delay etc; pair by same seed + stream hash + exact source checkpoint. Reject duplicate seeds or explicitly distinguish repeated stream replicate, never count as unique seed. N=1 -> variance NA; n<2 omit tests; no fabricated p-value. Real and synthetic tables physically separate, publication table only real verified. Add provenance links and per-seed raw values, effect sizes, uncertainty, worst regime/mask, signed gaps. Test duplicate, mixed modes/datasets, unequal seeds, double job, old JSON lacking schema, missing metric, NaN. **Acceptance:** aggregation program refuses incompatible runs, output reconstructable from full logs, valid unique seed counts, statistics preregistered.

### R3-12 (P0) — Test suite may pass vacuously; reported 24/24 not independent
**Evidence:** `tests/test_selective_adaptation.py:25–71` checks frozen params only; no assertion adaptable_set non-empty, adapted True, or expected adapter changes. B=8 vs null n=32 can cause all corrupt. `test_adaptation_skip` conditional only runs checks if set empty. `tests/run_all_tests.py` custom runner detects `_is_fixture` shim; installed real pytest fixtures use native semantics, potential fixture discovery/handling mismatch (verify with pytest installed). No CI workflow in tree. Source test claims Val F1 50.01% not backed by checkpoint/log committed.

**Fix:** use standard pytest as primary, remove brittle shim or prove both pathways. Deterministic mock detector to create exact A/U/D; assert nonempty active set and gradients, at least one expected active param changes, other params bitwise equal before/after, optimizer momentum not updating inactive params, skip deterministic. Separate actual spectral tests from gradient gating tests. Add integration tests P0 above, CI CPU and GPU manual smoke; record test report JUnit, environment, commit. Test finite score and no gradients from masked rows. Add coverage of source training end-to-end tiny real fixture. Do not claim 24 independently verified. **Acceptance:** actual command and complete logs, non-vacuous tests with failure injection (mutate mask or disable optimizer and test must fail).

### R3-13 (P1) — Original baseline identity and training-mode parity
**Evidence:** `src/adaptation/baselines_tta.py` SAR-style uses filtered single-step entropy + periodic restoration (not demonstrated original sharpness-aware two-pass optimizer). READ-style is generic cross-modal consistency, not official READ implementation. TENT LayerNorm port may be reasonable for this backbone but not original BN configuration. End of adapt_step sets eval; verify next call explicitly configures intended train/norm settings. Do not label official SOTA without faithful implementation and parity. **Fix:** retain diagnostic styles under names `tent_ln`, `sar_style`, `consistency_tta`, add official or carefully validated ports referencing algorithm/commit, disclose deviations. Include robust recent MM-TTA baselines (READ official, ICLR 2025 Smoothing the Shift, relevant 2026 methods if applicable). Match backbone, data, model access, adaptation steps, compute budgets; if original needs a different backbone, separately label faithful original setting and controlled adaptation port. Source-only + each CP, reliability-only + each CP; compare ACI and dynamic CP on identical predictor outputs. **Acceptance:** baseline implementation notes, original links, matched hyperparameter-search budget and reproducible config; avoid claims unsupported by port.

### R3-14 (P1) — Confounding online CP and TTA contribution
**Evidence:** source-only uses Static CP while TTA methods use Online ACI; gains in coverage cannot be causally attributed to model adaptation. CP dynamic signals are zero for baseline methods while ours receives shift/JS. **Fix:** build factorial predictor x controller design (Section 7). Record cached logits for each model run before CP variants; if CP feedback not used for model update, compare controller variants offline replay of same immutable logits, dramatically reducing GPU cost. Give all controllers the same feedback schedule and data available at prediction time, compare at equal nominal alpha and set sizes. **Acceptance:** isolated effects of spectral weighting, adaptation, mask conditioning, adaptive step size and fallback; no claim of synergy before interaction analysis.

### R3-15 (P1) — Missing per-mask evidence and invalid rare-group terminology
**Evidence:** `src/conformal/online_aci.py:238–267` hard switch from local to global window, not true hierarchical shrinkage (no partial pooling or mixture). `get_metrics()` available but benchmark not serializing per-mask stats. Global pooled score distribution can differ from rare mask distribution; mask-wise finite-sample validity not inherited. **Fix:** rename fallback `pooled cold-start` unless actually implementing hierarchical model. Compare local conservative, global, pooled shrinkage, group Mondrian/mask-conditional CP, fixed-step group risk controller. Measure support n_g, hit count, size, worst-group/worst-regime, intervals, alpha saturation, unsupported masks, all-label and empty predictions. Appropriate minimum-support policy chosen before test. Use group local index for theoretical claims; report small-n uncertainty, not hide rare groups. **Acceptance:** per-mask table (15 theoretical non-empty patterns, observed support), rare worst-case simulation and delayed feedback test, explicit conditional vs marginal distinction.

### R3-16 (P1) — Output telemetry insufficient for independent verification
**Evidence:** current JSON aggregate-only classification plus `coverage_hits`, `prediction_set_sizes`. Cannot reconstruct F1, label confusion, subset identity, entropy or actual adapted steps; `peak_reserved_mb` profiler exists but not saved in final results. No checkpoint/log hashes, run commit, protocol fingerprint, actual shift parameters. **Fix:** versioned run schema with metadata and event-level JSONL/Parquet (sample ID/hash, group, time, mask, label for offline evaluation, prediction/logits/scores, prediction set, alpha/eta/q, matured feedback ID, detector stats, A/U/D, adapted flag, parameter delta, finite status, regime); avoid disclosing copyrighted raw data. Log batch metrics, CPU+GPU memory allocated/reserved/driver if available, peak process RSS, CPU eig time, per-batch p50/p95 latency, throughput, warmup and sync. Clearly delineate feature extraction vs fusion/CTTA memory budgets, precision and hardware. **Acceptance:** recompute all summaries independently from artifact, schema and hash validated, output no NaN. No end-to-end <50 MB claim from feature-only allocated memory.

### R3-17 (P1) — Synthetic generator insufficient to prove robustness mechanism
**Evidence:** `src/data/dataset.py` synthetic samples generated from random per-sample center plus label scalar offset; directional class semantics largely encoded in scalar offsets, not realistic multimodal class evidence. Stable hash improves reproducibility but does not create realistic joint dependency. **Fix:** use synthetic only engineering, add controlled toy regimes with stable shared class prototypes across splits, modality-specific strengths, one reliable modality, covariance rotation, cross-modal correlation, label shift, group rarity and no clean anchor scenarios. Make deterministic exact tests for mechanism. Real dataset indispensable. **Acceptance:** debug correctness under known ground truth, no synthetic results in real scientific tables.

### R3-18 (P1) — Source-model and detector calibration evidence missing
Source validation F1 50.01% appears in docs, not committed training logs or checkpoint provenance. Validate source-only across classes, unimodal heads, mask dropout training if used (same pretraining all methods), class imbalance, early stop on validation not calibration/test. For detector, fit source stats from train, null from validation; CP calibration use separate held-out data. For MOSEI report common classification protocols (7-class and optionally binary/multiclass conventions; label mapping documented), for MELD macro and weighted F1. **Acceptance:** train logs, class support/confusion, source calibration status and fixed checkpoint IDs; no hard-coded universal F1 threshold for heterogeneous tasks.

### R3-19 (P1) — Method and paper claims stale/mismatched
`PROJECT_SPEC.md` remains V1 kernel MMD/pseudo-label formulation; README/METHOD_SPEC_V2/THEORY do not match code exactly. Four feature streams versus name Tri-Modal: explain three sensor/information modalities (audio, visual, text) with prosody derived from audio, or rename to multi-stream explicitly. Model consumes precomputed embeddings, not foundation-model adaptation. KL direction and averaging, null source train vs heldout, pairwise JS vs JS to mean, anchor promotion, signed velocity, actual stream timing, mask conditional guarantee all align. Define sample and batch symbols consistently. **Acceptance:** one canonical versioned method spec, deprecated document banner, equation-to-function mapping, claim-evidence matrix. No `guaranteed`, `perfect`, `eliminates`, `SOTA`, `100%` without evidence and scope.

### R3-20 (P1) — Lack of realistic label availability protocol
ACI requires labels at test time (delayed or immediate), unlike label-free TTA. Label feedback is assumed to arrive for every event in current synthetic pilot. Distinguish adaptation label-free vs controller supervised feedback. Include 100%, 25%, 10%, 0% feedback, delayed/random/missing labels, selection mechanism; no-label static/exchangeable calibration is control, guarantee only when assumptions hold. Do not assume selective feedback is unbiased; log label rate and selection. **Acceptance:** fixed label budgets across methods; show controller behavior and limits with sparse/no labels; note deployment implications.

### R3-21 (P2) — Manuscript readiness / open science
No full paper in tree. Build paper outline only after scientific claims evidenced. Provide limitations, licensing/citation of datasets/encoders, release scripts and manifests rather than unauthorized raw media, significance/negative results, compute cost and ablation. A code badge is not publication quality. **Acceptance:** complete reproducibility appendix + claim-evidence table, abstract not overclaiming, reviewer readiness assessment based on real results.

## 5. Novelty review and research positioning (limited targeted literature scan)

Prior art verified against public publication pages on audit date; not a systematic review or novelty clearance. Search/update before submission. No copied abstracts and no assumed results. Key direct overlap:

| Prior research | What it covers | Impact on proposed novelty |
|---|---|---|
| Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021 | Online adaptive miscoverage control with changing distributions | Baseline; ACI itself not novel |
| Gibbs & Candès, Conformal Inference for Online Prediction with Arbitrary Distribution Shifts, JMLR 2024 | Online adjustment of ACI learning rates, local regret | Variable eta itself not novel; compare principled adaptive controller |
| READ, Test-Time Adaptation against Multi-modal Reliability Bias, ICLR 2024 | Reliability-aware fusion, multimodal TTA | Reliability weights and channel noise motivation not enough |
| Smoothing the Shift: Towards Stable Test-Time Adaptation under Complex Multimodal Noises, ICLR 2025 | Simultaneous modality corruptions/missingness, stable multimodal TTA | Directly overlaps compound shift motivation; must compare |
| Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation, AAAI 2026 | Modality realignment at test time | Alignment alone insufficient novelty |
| Mask-Conditional Conformal Prediction: Valid Uncertainty For All Missing Data Mechanisms, AISTATS 2026 | Missing-covariate mask-conditional prediction | Mask conditioning is already researched; distinguish online setting and assumptions |
| Adapt-then-fuse: Instance feature adaptation for multi-modal test-time adaptation, KBS 2026 | Reliability strata and adaptive fusion | Explicit selective adaptation novelty risk |
| STA: Spatio-temporal alignment for multimodal test-time adaptation, KBS 2026 | Robust alignment and reliability-aware inference | Compare relevance, not just early baselines |
| Conformal Calibration for Multi-Modal Regression with Missing Modalities, COPA 2026 | Disagreement and mask-aware calibration for regression | Close intersection; task and online feedback distinguish but not proof of novelty |
| Guarantees That Survive a Missing Scan, MICCAI satellite 2026 | Modality-conditional CP via stratification | Another example of existing modality-pattern calibration |

### 5.1. Claim matrix and recommendation

- C1 **sample-size-aware spectral reliability**: potentially meaningful method contribution if proven detector discrimination under missingness and dimension imbalance; normalized diagonal mean + log spectrum is combination of existing statistics, not inherently novel. Compare simpler alternatives and detection failures.
- C2 **anchor-guided parameter-selective continual adaptation**: plausible practical contribution if clean anchor reliability correlates with teacher accuracy and selective updates beat equal-compute alternatives, especially recovery and severe shifts. Present as conditional mechanism, not mathematically eliminates forgetting. Current gains modest and R5 negative.
- C3 **mask-aware online uncertainty with drift/disagreement and delayed feedback**: strongest candidate research focus if valid controller, separate group and time guarantees or strong controlled empirical results vs fixed/adaptive CP. Mask-conditioning and adaptive ACI are established separately. Specific coupling plus robust validation may be novel, but no 'first' claim without systematic review. Current theorem invalid and pooled fallback heuristic.
- C4 **benchmark and resource efficiency**: potentially supportive contribution, not independent novelty if synthetic only. Realistic raw or honestly limited latent shifts; public recipe & fair evaluation. 4GB GPU is valuable constraint; not enough to guarantee Q1 alone.

Recommended paper story: **Reliable multimodal continual adaptation with mask-aware online uncertainty under compound shifts and delayed labels**. Keep 2–3 claims, avoid adding unrelated advanced modules. If classification gains negligible but calibration robust and efficient at matched coverage, narrow contribution to uncertainty control with predictor-agnostic method and compare multiple predictors. Conversely if CP lacks theoretical or empirical edge but selective adaptation convincingly improves robust F1, make adaptation primary and CP ancillary. If both fail on pilot real data, revise mechanism not merely hyperparameter hunt. Do not promise SOTA or significance as mandatory numeric outcomes; maintain negative results.

## 6. Proposed V3 design

### 6.1. Data and experiment architecture

```text
licensed raw assets / official feature cache
  -> provenance manifest + canonical ID alignment + split validation
  -> source-only train (train, val early stop)
  -> checkpoint + stats (train-valid rows; null from val)
  -> CP calibration (independent cal; mask-matched if needed)
  -> frozen event stream generator (test-only, no labels exposed to adapter)
  -> predict/adapt protocol runner (prequential OR transductive)
  -> frozen event logits and gate telemetry
  -> CP controller replay on same logits / feedback queue
  -> offline evaluation (all labels for scoring only)
  -> immutable run manifest + event records + aggregate
```

Avoid unnecessary expensive reruns: CP variants can reuse saved logits as long as CP controls do not feed adaptation. Cache features and corruption stream by hash. CPU cheap test first; do not generate all raw corruptions until pilot proves feasibility. GPU: preserve target laptop RTX2050 4GB; measure actual rather than promise. Use sequential encoders on GPU/CPU, microbatch, chunk features; covariance computations use SVD/Gram where rank small rather than full D^2 matrix eig if equivalent. Train adapters on FP32/AMP where stable; CP scores and detector stats FP32 minimum, exact rank robust against floating-point boundaries. Report runtime for source and deployment separately.

### 6.2. Algorithm pseudocode (strict prequential candidate)

```python
for event_batch in frozen_stream:
    feedback.deliver_due_before(event_batch.time, controller.observe)
    # Evaluate scores on observed rows only, one detector state update.
    d, uncertainty = detector.score(event_batch.features, event_batch.present_mask)
    r = quality(d, uncertainty)
    gates = partition(r, present=event_batch.present_mask)
    effective_mask = choose_policy(gates, event_batch.present_mask)
    with inference_context():
        logits, unimodal = model.predict(event_batch.x, effective_mask, r)
    velocity = mean_abs_valid_delta(d, previous_d)
    disagreement = js_on_pairwise_valid_rows(unimodal, effective_mask)
    # Issue records before adaptation and before current labels arrive.
    for event in event_batch.ordered_events:
        record = cp.predict(event_id=event.id, logits=logits[event.row],
                            mask_key=event.observed_mask_key,
                            signals=(velocity, disagreement[event.row]))
        evaluator.record_prediction(record)
    # Adapt using unlabeled batch only, real no-anchor skip or explicit fallback.
    if gates.has_trusted_anchor and gates.has_adaptable:
        model.adapt(event_batch.x, gates, effective_mask)
    feedback.enqueue_current_events(event_batch.ids) # evaluator schedules truth separately
```

Careful: batch-time CP issuing all records before feedback is a different protocol from sample-time immediate feedback within batch. Choose exact timing, document both if supported and use identical schedule for all baselines. No future samples when strict per-sample causality claimed; batch processing allowed only if deployment observes batch together. Dropout deterministic / eval emission, train-time adapter controllable. The data mask used to stratify CP is observed sensor mask, not a silently changed reliability gate, unless study explicitly defines group from effective mask and logs both.

### 6.3. Minimum run identity / artifacts

```yaml
schema_version: 3
run_id: <stable unique hash>
git_sha: <actual>
dirty_tree: false
dataset: {name: cmu_mosei, version: <verified>, mode: real, manifest_sha256: <hash>}
source: {seed: 42, checkpoint_sha256: <hash>, stats_sha256: <hash>, config_sha256: <hash>, trained: true}
stream: {protocol: prequential, config_sha256: <hash>, stream_sha256: <hash>, seed: <int>, n_events: <int>, event_ids_hash: <hash>}
method: {name: tri_sacta_v3, implementation_version: <sha>, config_sha256: <hash>}
conformal: {controller: fixed_mask_aci, alpha: 0.1, score: aps, delay_unit: sample, delay: 0, feedback_budget: 1.0}
environment: {python: <actual>, torch: <actual>, cuda: <actual>, gpu: <actual>, precision: <actual>}
status: PASS|FAIL|BLOCKED|NOT_RUN
artifacts: {events: <path/hash>, logs: <path/hash>, metrics: <path/hash>}
```

Actual schema must validate e.g. enums, nonfinite values, required keys. Example paths/placeholders are specification, not existing commands/artifacts. Keep datasets private/licensed as required; no secrets in repo.

## 7. Experimental design and ablations: staged, resource-aware

### Stage A — engineering smoke (no publication claim)
- CPU tests and tiny training, one batch each regime, deterministic mock detector. Reproduce V2 synthetic results if exact config/checkpoint available; otherwise explicitly mark unavailable and run V3 debug anew. Freeze same source & stream for all tests. Run 3 seeds synthetic to expose fragility; not paper evidence.
- Validate actual six-regime runner and prequential order, corruption identity, feedback timing, real-mode rejection, all P0 regressions. Run counterexample and CP boundary test. Matrix 2x2 end-to-end and aggregation rejects duplicate.

### Stage B — real pilot (go/pivot decision)
- Choose dataset access path; ideally start with CMU-MOSEI official precomputed features or raw pilot, then MELD. Do not claim named encoders unless actually extracted. Check original split counts and label mapping; log license/version. Pilot 1 seed and limited slice using grouped split and real features. Train strong source; sanity-check source-only and unimodal heads before any adaptation. Run strict six-regime protocol; report per-mask/regime, R6 and actual adapter update fraction. Include at least source-only, reliability-only, TENT-LN, fixed ACI vs mask/dynamic and full V3. If real feature access blocked, do not mark stage done.
- Go/no-go: does C1 detector beat simple means, C2 adaptation outperform reliability-only under relevant shifts, C3 controller improve efficiency at matched coverage without unacceptable worst-mask decline? If pilot negative, diagnose and preregister changes using validation; no test-set tuning. Need not force positive result; report honestly.

### Stage C — core real experiments (two datasets, five seeds)

Primary predictor methods: frozen source, uniform mask fusion; source with reliability-only; TENT-LN; faithful SAR port or clear SAR-style; READ official/port; ICLR2025 complex-noise TTA port where feasible; Tri-SACTA full. Add CoTTA/EATA where appropriate; compute matched baselines. Primary CP controllers: Static CP, fixed global ACI, fixed mask ACI, official adaptive/FACI reference if feasible, proposed dynamic mask CP with explicit assumptions, global fallback comparator. Don't require full Cartesian product if expensive: core factorial pair for frozen source + full method across CP; compare common fixed CP for model-only contributions; replay CP on saved logits.

| Experiment family | What to keep fixed | Main quantity |
|---|---|---|
| Detector comparisons | Same raw/latent shifts, valid count, seeds | AUROC/AUPRC, FPR@threshold, corruption severity and cost |
| Fusion effect | Same source model, stream, no parameter update | F1, robustness, mask invariance |
| Adaptation effect | Same detector/weights, CP fixed or none | F1, drift/forgetting, error recovery, parameter count |
| CP effect | Exact same logits & feedback events | coverage gap/undercoverage, matched set size, worst mask/regime |
| Synergy | 2x2 reliability/selective adaptation x CP variant | joint effect; no claim without evidence |
| Robustness | Same model, new pre-registered streams | abrupt/gradual/recurrent, label shift, dropout pattern, R6 |
| Delays and labels | Same logits, same feedback schedule | delay 0/10/50, label feedback fraction, pending events |

Ablations minimum (each isolate change): A0 full; A1 no spectral (uniform); A2 mean-only; A3 spectrum-only; A4 no null/matched-count test (diagnostic); A5 reliability-only no updates; A6 all adapter update; A7 no anchor KL; A8 no param reg; A9 no hard mask (if hard used); A10 constant eta; A11 no mask group; A12 no drift feature; A13 no JS; A14 no pooled fallback; A15 fixed global/rare conservative. Hold model config same unless intentionally changing. Include simple alternatives rather than multiple weak variants. Avoid combinatorial explosion: run selected 1-seed pilot, then key ablations in full seeds/datasets. Report adaptation percent and compute overhead.

Metrics: accuracy/Macro-F1 and weighted F1 with class counts; per-regime and R1 vs R6 recovery; per-mask coverage n/hits/size/CI; overall and local-window coverage, signed gap, undercoverage, absolute gap, max worst supported mask; set size distribution incl full/empty, efficiency vs target coverage; NLL, Brier, ECE and reliability diagrams; latency p50/p95 per sample/batch, throughput, peak allocated/reserved/driver footprint, CPU stats. Profile same hardware/precision/batch and synchronized boundaries. Unit-test detector FPR, false anchor and adaptation delta. Store full event identities.

### Stage D — statistical evidence
Use paired seed designs with fixed splits; distinguish source randomness, stream randomness and repeat event IDs. Report per-seed mean±sample SD only for n>=2, paired mean difference and appropriately justified CI, block/group bootstrap by videos/dialogues and sequence blocks, not naive 3,000 iid events. For a primary claim define meaningful improvement margin prior to test (e.g. practical delta F1/coverage-efficiency trade-off from validation), not universal mandatory threshold. Report negative regime findings; multiple testing adjustment. Five seeds are a minimum pragmatic Q1 target not guarantee; if uncertain, expand with power analysis. Explain inference scope. Include benchmark/analysis code and reproducible tables. For theoretical claims present proof respecting implementation and test counterexamples, with explicit assumptions and boundary behavior.

## 8. Actual research gates (independent re-audit)

| Gate | Re-audit status | Evidence required to close |
|---|---|---|
| G0 Legacy preservation | PASS (static) | V1 results in legacy confirmed; record SHA |
| G1 Engineering correctness | PARTIAL | 24/24 self-reported, nonvacuous tests/full integration needed |
| G2 Source-model integrity | PARTIAL | synthetic validation score reported, actual trained checkpoint provenance absent |
| G3 Collapse remediation | PASS for stored synthetic output only | no collapse in V2 JSON; independent same-seed rerun, real validation pending |
| G4 Conformal validity | FAIL | empirical average 90% only; invalid theorem, R5 undercoverage, rare-mask/feedback correctness |
| G5 Real pipeline | BLOCKED/NOT RUN | extractor stub, real manifest+features+trained pilot not yet evidenced |
| G6 Theory/method spec | FAIL/PARTIAL | theory docs exist, key theorem wrong and protocol mismatch |
| G7 Publication evidence | NOT RUN | real x2 datasets, multi-seed, baseline/ablation, verified scientific claims |

Do not equate document creation with passed gate. Status on date based on visible repo; local artifacts not ruled out. Recommended progress bookkeeping: engineering ~70%; scientific validation ~25%; overall ~40–45%; subjective estimates. Actual paper readiness ~4/10. No guarantee Q1 outcome; venue requirements and quartile independently check when submission-ready.

## 9. Work packages with dependency order

| WP | Priority | Work and deliverable | Exit criterion |
|---|---|---|---|
| WP0 | P0 | Branch, tag V2, remove overclaims, claim-evidence matrix, save environment | SHA and honest status |
| WP1 | P0 | R3-01/02 fixed-step reference, counterexample, feedback event semantics | mathematical + executable regression tests |
| WP2 | P0 | R3-03/04/05 real ingestion/validator/checkpoint; fail-closed | valid tiny fixture & actual data pilot or BLOCKED |
| WP3 | P0 | R3-07/08/09 gate mask, valid-count detector, latent/raw corruption semantics | deterministic tests, detector diagnostics |
| WP4 | P0 | R3-06/10/11/12 protocol runner, matrix, aggregation, integration | 2x2 matrix, 6-regime stream hash parity, logs |
| WP5 | P1 | real source pilot and 1-seed controlled study | actual real results & go/pivot decision |
| WP6 | P1 | baseline parity, factorial CP replay and full datasets/seeds | verified run manifests and stats |
| WP7 | P1 | ablations, recovery, worst-mask, delay/label budget, profiling | claims linked to evidence, negatives included |
| WP8 | P2 | theoretical review, related work, paper and reproducibility appendix | independent reviewer check, no unsupported claims |

WP1 and WP2 can parallelize; all P0 gates before large matrix. Do not start manuscript as substitute for data. If time/compute limited, ship correct WP1–WP4, then real pilot. Avoid writing speculative results or blocking entire project on downloading all datasets; implement verified feature-cache route if appropriate.

## 10. Mandatory regression tests (proposed, not claimed executed)

Use pytest standard; provide machine-readable outputs. Suggested IDs:

| ID | Test and expected outcome |
|---|---|
| V3-T01 | alternating eta algebra counterexample mean miss .5 and alpha bounded |
| V3-T02 | constant eta pathwise telescope matches identity; boundary full/empty |
| V3-T03 | clipping residual accounting; dynamic eta no unsupported guarantee |
| V3-T04 | CP quantile rank for n=1,..,100, ties, APS max rounding, empty/full |
| V3-T05 | group states independent; rare fallback not advertised valid |
| V3-T06 | emitted record deep-immutable even caller mutates logits/set |
| V3-T07 | delayed feedback exact record/arrival order 0/10/50 |
| V3-T08 | missing/duplicate/out-of-order feedback and pending end |
| V3-T09 | missing real metadata fails closed |
| V3-T10 | empty manifest/wrong split/duplicate ID/checksum/count/NaN rejected |
| V3-T11 | same-shape cross-dataset or synthetic checkpoint rejected |
| V3-T12 | actual extraction tiny asset outputs nonempty shapes, resume hashes |
| V3-T13 | source stats exclude missing rows; null matched valid count |
| V3-T14 | clean distribution varying batch sizes with partial absence |
| V3-T15 | no-anchor skip, active adapters updated nonzero, inactive unchanged |
| V3-T16 | quality vs availability separation; mask invariance corrupt token |
| V3-T17 | valid-pair JS independent of missing values, no pairs safe |
| V3-T18 | velocity abs before mean and finite extreme values |
| V3-T19 | all-missing/NaN masks explicit reject or abstain |
| V3-T20 | real waveform/text/image shift semantics verified or latent named |
| V3-T21 | partial preexisting missingness dropout cannot produce undocumented all-zero |
| V3-T22 | 6 actual regimes, config values honored, explicit R6 |
| V3-T23 | prequential versus transductive order instrumented |
| V3-T24 | same checkpoint and frozen stream across methods; different RNG consumption irrelevant |
| V3-T25 | matrix train unique seeds once, correct files and resume |
| V3-T26 | aggregator rejects duplicate/mixed dataset/mode/protocol and n=1 variance NA |
| V3-T27 | per-event metrics reproduce aggregate and mask/regime counts |
| V3-T28 | label budget feedback no future information; no-label run |
| V3-T29 | no nonfinite probabilities, gradients, logits under AMP |
| V3-T30 | memory/latency scopes recorded and hardware verified |

For each: command, expected, actual, PASS/FAIL/SKIP with reason; verify tests fail when intentional regression introduced. Not all tests imply theory; proof independent.

## 11. Execution commands and handover contract

Existing commands (verify flags with `--help`; current code has deficiencies documented above):

```bash
git rev-parse HEAD
git status --short
python -m pytest -q
python tests/run_all_tests.py
python scripts/00_validate_data.py --config configs/debug_synthetic.yaml --data-mode synthetic
python scripts/01_train_source.py --config configs/debug_synthetic.yaml --data-mode synthetic --seed 42
python scripts/02_run_stream_benchmark.py --config configs/debug_synthetic.yaml --data-mode synthetic --method tri_sacta --conformal online_aci --seed 42
python scripts/04_run_experiment_matrix.py --config configs/debug_synthetic.yaml --data-mode synthetic --seeds 42 100 2026 --dry-run
```

Before scientific runs, fix matrix runner, source namespaces, and protocol flags. After implementation, use actual CLI help to compose commands; proposed flags in this report are NOT currently available. Capture exit codes, logs, SHA, hardware and input hashes. Run real validator for selected config before training, not synthetic fallback. Record output paths and no overwrites. If laptop or dependencies block, report what is run and exact next command, not pretend completion.

Deliverables: `docs/AUDIT_STATUS_V3.md`, `docs/HANDOVER_V3.md`, `docs/CLAIM_EVIDENCE_MATRIX.md`, `docs/THEORY_V3.md`, `docs/EXPERIMENT_PROTOCOL_V3.md`; versioned schema; tests; explicit run artifacts; validated manifests; source and stream hashes; real results only when available; reproducibility README. Suggested status table:

```markdown
| Issue | Status | Commit | Test command | Evidence path | Limitations |
|---|---|---|---|---|---|
| R3-01 | FAIL/PASS/... | ... | ... | ... | ... |
```

Final handover form: (1) HEAD/base SHA, (2) changed issue IDs, (3) test totals actual/failed/skipped with logs, (4) current data mode/verified counts, (5) experiment table with n seeds, (6) claim changes and any proofs with assumptions, (7) artifact index/hash, (8) blockers and actionable next step, (9) research gate decision. No auto-close or fabricated numbers. Do not submit externally without user instruction.

## 12. References and primary evidence

### Repository pinned evidence
- Commit: https://github.com/buponbonbon/Tri-SACTA/commit/1e00431666a331529c461fce23ccda387a2570df
- V1: https://github.com/buponbonbon/Tri-SACTA/commit/17f1acbb777ff7f970210af4d75cd13bcc710798
- Audit status: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/docs/AUDIT_STATUS.md
- Theory: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/docs/THEORY.md
- Handover: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/docs/HANDOVER.md
- Method: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/docs/METHOD_SPEC_V2.md
- Protocol: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/docs/EXPERIMENT_PROTOCOL.md
- Results: https://github.com/buponbonbon/Tri-SACTA/tree/1e00431666a331529c461fce23ccda387a2570df/results
- Summary: https://github.com/buponbonbon/Tri-SACTA/blob/1e00431666a331529c461fce23ccda387a2570df/results/summary/multi_seed_statistical_summary.md
- Key source: https://github.com/buponbonbon/Tri-SACTA/tree/1e00431666a331529c461fce23ccda387a2570df/src
- Scripts: https://github.com/buponbonbon/Tri-SACTA/tree/1e00431666a331529c461fce23ccda387a2570df/scripts
- Tests: https://github.com/buponbonbon/Tri-SACTA/tree/1e00431666a331529c461fce23ccda387a2570df/tests

### Targeted related work (publication pages)
1. Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift (NeurIPS 2021): https://papers.nips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
2. Gibbs & Candès, Conformal Inference for Online Prediction with Arbitrary Distribution Shifts (JMLR 2024): https://jmlr.org/beta/papers/v25/22-1218.html
3. Test-Time Adaptation against Multi-modal Reliability Bias (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/f25d75fc760aec0a6174f9f5d9da59b8-Abstract-Conference.html ; code https://github.com/XLearning-SCU/2024-ICLR-READ
4. Smoothing the Shift: Towards Stable Test-Time Adaptation under Complex Multimodal Noises (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/065e259a1d2d955e63b99aac6a3a3081-Abstract-Conference.html
5. Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/39457
6. Fan et al., Mask-Conditional Conformal Prediction: Valid Uncertainty For All Missing Data Mechanisms (AISTATS 2026): https://proceedings.mlr.press/v300/fan26a.html
7. Adapt-then-fuse: Instance feature adaptation for multi-modal test-time adaptation (2026): https://www.sciencedirect.com/science/article/pii/S0950705126011536
8. STA: Spatio-temporal alignment for multimodal test-time adaptation (2026): https://www.sciencedirect.com/science/article/pii/S0950705126012694
9. Azizi, Conformal Calibration for Multi-Modal Regression with Missing Modalities (COPA 2026): https://proceedings.mlr.press/v329/azizi26a.html
10. Guarantees That Survive a Missing Scan: Modality-Conditional Conformal Prediction for Multimodal Medical Diagnosis (MICCAI satellite 2026): https://papers.miccai.org/miccai-2026-sat/MLMI_024.html

Review complete literature in full before manuscript; verify exact bibliographic author/year, assumptions and baseline code. No guarantee Q1 tier by these citations alone. The numerical results here derive from repository JSON, not external papers.

## 13. Final decision

**Invest further, not submit yet.** Antigravity has made tangible progress in modular code and synthetic stability. Main remaining blockers are mathematical validity, genuine real data, protocol alignment and fair measurements, not simply more code. Priority is to fix theorem and provenance before running expensive experiments. Re-audit after commits with verified P0 tests and actual real pilot artifacts. Current progress ~40–45% toward Q1-ready manuscript; not equivalent to the agent's Phase 3 complete statement. If future agent reports all issues resolved, require exact run IDs and independently check the evidence, not a prose assertion.
