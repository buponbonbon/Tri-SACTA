# TRI-SACTA — Báo cáo audit nghiên cứu, kế hoạch khắc phục và đặc tả triển khai hướng tới Q1

**Tài liệu bàn giao cho Antigravity 3.8 / coding agent**  
**Ngày audit:** 21/11/2026 (UTC+7)  
**Repository:** https://github.com/buponbonbon/Tri-SACTA  
**Snapshot tham chiếu:** `main@17f1acbb777ff7f970210af4d75cd13bcc710798`  
**Trạng thái:** MVP có mã nguồn và kết quả sơ bộ, **NO-GO cho Q1 submission**.  
**Ràng buộc đề xuất:** ưu tiên laptop GPU 4 GB, nghiên cứu tái lập, không bịa kết quả.  
**Phạm vi:** static audit mã nguồn, cấu hình và JSON hiện có; rà soát tài liệu khoa học có chọn lọc. Chưa chạy lại training, GPU benchmark hoặc test độc lập. Không có quyền suy luận rằng dữ liệu hoặc checkpoint local của chủ repo không tồn tại chỉ vì chưa được commit.

> **Đọc trước khi thực hiện:** Đây là tài liệu thực thi, không phải manuscript và không phải bằng chứng phương pháp mới đã thành công. Các công thức thiết kế V2, ngưỡng nghiệm thu, cách bố trí experiment và cấu trúc file trong tài liệu là đề xuất cần kiểm chứng. Không thay đổi hoặc đánh bóng số liệu cũ. Nếu phương pháp thất bại sau khi đã kiểm tra công bằng, giữ kết quả âm và báo cáo nguyên nhân.

## Mục lục

1. [Chỉ thị cho coding agent](#1-chỉ-thị-cho-coding-agent)
2. [Executive summary và tiến độ](#2-executive-summary-và-tiến-độ)
3. [Bài toán và phạm vi nghiên cứu](#3-bài-toán-và-phạm-vi-nghiên-cứu)
4. [Snapshot repository và kết quả](#4-snapshot-repository-và-kết-quả)
5. [Issue register](#5-issue-register)
6. [Thiết kế kỹ thuật V2](#6-thiết-kế-kỹ-thuật-v2)
7. [Dữ liệu thật và benchmark](#7-dữ-liệu-thật-và-benchmark)
8. [Conformal theory và giới hạn claim](#8-conformal-theory-và-giới-hạn-claim)
9. [Baselines, ablation và thống kê](#9-baselines-ablation-và-thống-kê)
10. [Tài nguyên, hiệu năng, reproducibility](#10-tài-nguyên-hiệu-năng-reproducibility)
11. [Test plan và acceptance gates](#11-test-plan-và-acceptance-gates)
12. [Roadmap giao việc và deliverables](#12-roadmap-giao-việc-và-deliverables)
13. [Novelty, reviewer và manuscript Q1](#13-novelty-reviewer-và-manuscript-q1)
14. [Tài liệu tham khảo và liên kết audit](#14-tài-liệu-tham-khảo-và-liên-kết-audit)
15. [Prompt thực thi và mẫu handover](#15-prompt-thực-thi-và-mẫu-handover)

## 1. Chỉ thị cho coding agent

### 1.1 Mission

Biến Tri-SACTA từ prototype thành một research artifact có thể tái lập và kiểm chứng; trước hết khắc phục suy biến, đảm bảo đúng protocol, sau đó chứng minh hoặc bác bỏ các giả thuyết nghiên cứu bằng thí nghiệm. Không tối ưu để đạt một con số Q1 đã định trước.

### 1.2 Nguyên tắc không được vi phạm

- [ ] Kiểm tra `git status`, `git log`, `git rev-parse HEAD`, môi trường, dữ liệu và checkpoint thực tế trước khi sửa. Nếu HEAD mới hơn snapshot, đối chiếu từng issue, đánh dấu `already fixed` khi có test chứng minh.
- [ ] Không ghi đè thay đổi local của người dùng; không force-push, reset hard, xóa checkpoint/results, hoặc tự ý public dữ liệu. Tạo nhánh riêng nếu worktree cho phép, ví dụ `research/q1-hardening`.
- [ ] Bảo tồn kết quả lịch sử bằng commit hash/manifest; kết quả mới dùng run ID riêng. Không chỉnh JSON cũ để làm đẹp bảng điểm.
- [ ] Không chạy benchmark công bố bằng synthetic data hoặc random weights mà không gắn nhãn rõ ràng. Real mode phải fail closed.
- [ ] Không dùng nhãn test để cập nhật TTA backbone, chọn hyperparameter, chọn seed, checkpoint, phương pháp hay kịch bản thuận lợi. Nhãn có thể dùng riêng cho conformal controller theo protocol feedback được công bố.
- [ ] Không tuyên bố SOTA, finite-sample coverage, mask-conditional guarantee, GPU 4 GB end-to-end hay paper-ready nếu chưa có bằng chứng tương ứng.
- [ ] Không dùng số liệu bịa đặt, paper/citation bịa đặt, output giả, hoặc tests chỉ in PASS. Mỗi claim phải nối được đến code, config, log và kết quả.
- [ ] Không tự tải dataset/model nhiều GB hay thực hiện job tốn phí vượt ngân sách người dùng; ghi rõ dependency bị chặn và cung cấp lệnh có thể chạy khi tài nguyên sẵn sàng.
- [ ] Dùng ít nhất một baseline đơn giản và đối chứng matched-policy trước khi thêm kiến trúc; không sửa nhiều thành phần cùng lúc mà không có isolate test.
- [ ] Chỉ báo hoàn thành những gì đã chạy, ghi `NOT RUN` hoặc `BLOCKED` cho phần chưa thực hiện. Không giả định có GPU/network.

### 1.3 Nhãn bằng chứng và độ ưu tiên

| Nhãn | Ý nghĩa |
|---|---|
| CONFIRMED | Hành vi/lỗi đọc được trực tiếp từ source hoặc output JSON tại snapshot. |
| HYPOTHESIS | Giải thích khả dĩ, phải kiểm chứng bằng test/log. |
| MISSING EVIDENCE | Không thấy artifact cần thiết tại snapshot; không có nghĩa local không có. |
| PROPOSAL | Phương án thiết kế hoặc ngưỡng nghiệm thu, không phải kết quả thực nghiệm. |
| P0 | Blocker về tính đúng, data integrity hoặc science validity; làm trước. |
| P1 | Blocker của nghiên cứu hoàn chỉnh; làm sau sanity gate. |
| P2 | Mở rộng, paper polish, tối ưu khi có evidence. |

Theo dõi issue bằng `docs/AUDIT_STATUS.md` với cột ID, evidence, status, commit sửa, lệnh test, kết quả và đường dẫn artifact. Status hợp lệ: `OPEN`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`, `DEFERRED`.

## 2. Executive summary và tiến độ

**Khuyến nghị:** giữ đề tài, thu hẹp đóng góp và sửa phương pháp trước khi đầu tư full benchmark. Đề tài có tính ứng dụng cao nhưng novelty chưa được xác lập. Phần lớn rủi ro hiện tại có thể phát hiện ở unit test/constrained experiment, không cần GPU lớn.

| Khía cạnh | Đánh giá chủ quan /10 | Căn cứ |
|---|---:|---|
| Ý nghĩa bài toán | 8.5 | CTTA đa phương thức dưới nhiễu và missingness, uncertainty ở deployment. |
| Khả năng tổ chức mã nguồn | 7.5 | Module tách rõ, có entry points và config. |
| Tiềm năng đóng góp | 7.5 | Giao điểm reliability-aware CTTA và online group coverage; cần đối chiếu prior art. |
| Độ đúng hiện tại | 4 | Nhiều lỗi P0 về masking, shift scaling và feedback. |
| Bằng chứng dữ liệu thật | 2 | Thiếu manifests, checkpoint, logs ở snapshot. |
| Thực nghiệm | 1.5 | Hai kết quả single-seed, phương pháp đề xuất suy giảm mạnh. |
| Lý thuyết | 3.5 | Spec/code lệch nhau, chưa có chứng minh cho biến thể thực tế. |
| Reproducibility | 4 | Có cấu trúc, nhưng thiếu provenance và regression tests. |
| Sẵn sàng gửi Q1 | 3 | Không đạt gate về dữ liệu, kết quả, novelty, theory. |

Ước lượng định tính: hoàn thành khoảng 30–35% hành trình hướng tới một paper mạnh (không phải đo % code, không phải xác suất accept). Source model, phương pháp, conformal, metrics, plotting đã có skeleton/implementation; phần quan trọng còn lại là sửa logic, kiểm chứng, thực nghiệm và paper.

**Ưu điểm:** mục tiêu có ý nghĩa thực tế; source-free feature-level TTA có thể nhẹ; thiết kế module dễ ablate; có ý tưởng anchor/modality reliability; kết hợp prediction set với missingness; benchmark lưu JSON cả kết quả xấu.

**Nhược điểm:** kết quả hiện tại không chứng minh hiệu quả; code và đặc tả mâu thuẫn; data pipeline và protocol chưa chặt; novelty có nguy cơ là ghép thuật toán; lời hứa coverage quá mạnh; baseline còn là bản giản lược; chưa có nghiên cứu thống kê đa seed.

**Chiến lược:** (1) reproduce + repair, (2) data thật + kiểm tra công bằng, (3) bằng chứng novelty, (4) paper. Không mở rộng sang thêm encoder/modalities/dataset trước khi qua gate tối thiểu.

## 3. Bài toán và phạm vi nghiên cứu

### 3.1 Problem formulation

Nguồn gồm các mẫu đa phương thức có nhãn; train source backbone và lưu feature statistics, không cần giữ raw source lúc deployment nếu claim source-free. Test stream gồm các mẫu `(x_t, m_t)` với mask quan sát được, các distribution shift thay đổi theo thời gian. Nhãn `y_t` không được biết khi dự đoán; có thể đến muộn để đánh giá và cập nhật conformal. Nếu không có nhãn phản hồi, không được khẳng định online ACI tự cập nhật có giám sát.

- `M`: số feature streams, hiện tại 4: visual, audio, text, prosody. Prosody là một nhánh của audio, không mặc nhiên là một sensor độc lập. Chốt tên gọi: **tri-modal with four feature streams** nếu gộp audio+prosody ở mức modality; nếu mask thực sự có 4 nhánh độc lập thì ghi rõ four-stream và 15 mask khác rỗng. Nếu hệ thống chỉ cho mất đồng thời audio/prosody, giới hạn tập mask thực tế và không tuyên bố 15 mask khả thi.
- `h_t^m`: biểu diễn modality đã project; `d_t^m`: shift score; `r_t^m`: reliability proxy chứ không phải xác suất đúng nếu chưa calibrate; `C_t`: tập nhãn prediction set; `alpha*=0.1`: target miscoverage.
- Mục tiêu: giữ chất lượng phân loại khi dữ liệu nhiễu/thiếu, kiểm soát prediction-set miscoverage, giữ set nhỏ và compute vừa phải.
- Hai protocol cần phân biệt: **strict prequential** (predict rồi adapt cho batch kế tiếp) và **transductive batch TTA** (được nhìn batch không nhãn và thích ứng trước khi phát hành prediction của batch đó). Chọn protocol chính, dùng giống nhau cho mọi method. Không trộn số liệu hai protocol.
- Phân biệt source-free TTA với hệ có phản hồi nhãn trực tuyến ở conformal layer; tên phù hợp là `label-free backbone adaptation + feedback-based online calibration`. Không gọi toàn bộ hệ thống là hoàn toàn unsupervised nếu ACI dùng nhãn test.

### 3.2 Research questions và falsifiable hypotheses

| ID | Research question | Kết quả có thể bác bỏ |
|---|---|---|
| RQ1 | Spectral detector phân biệt shift theo modality tốt hơn confidence/mean/covariance proxy đơn giản không? | Detector AUROC thấp, không monotonic theo severity hoặc fail khi clean. |
| RQ2 | Anchor-guided selective adaptation cải thiện robust F1 / hạn chế forgetting hơn gated source hoặc uniform adaptation? | Gating-only ngang hoặc tốt hơn, adaptation ít khi chạy hoặc giảm clean F1 nghiêm trọng. |
| RQ3 | Mask-aware feedback controller cải thiện coverage–efficiency trên mask đủ support so với pooled ACI / strong online CP? | Coverage gap không giảm hoặc set rất lớn; nhóm hiếm không đủ evidence. |
| RQ4 | Shift/disagreement-conditioned step size có ích hơn constant hoặc adaptive baseline sẵn có? | So sánh matched calibration budget không có lợi ích ổn định; adaptive control mất ổn định. |
| RQ5 | Hệ thống có lợi thế về resource/latency với feature cache và 4 GB GPU không? | Cost CPU eigendecomposition hoặc preprocessing quá lớn; peak end-to-end không phù hợp. |

Không bắt buộc tất cả hypotheses đúng. Nếu RQ2 thất bại còn RQ3 thành công, thu hẹp bài báo về online conformal. Nếu RQ3 thất bại, tập trung spectral selective CTTA nếu có novelty/evidence. Nếu tất cả thất bại, không submit chỉ để hy vọng may mắn.

### 3.3 Định hướng contribution

1. Modality-normalized spectral detector có hiệu chỉnh clean null, missing mask và size-aware estimation.
2. Selective adaptation có anchor thực sự đáng tin cậy, update được chứng minh là chọn lọc; có cơ chế không cập nhật khi không đủ bằng chứng.
3. Mask-aware online conformal controller dưới compound shift với replayable feedback and explicit assumptions; nếu có định lý mới, chứng minh riêng. Chỉ ghép module có sẵn không mặc nhiên là novelty.

## 4. Snapshot repository và kết quả

### 4.1 Lịch sử và cấu trúc

Ba commit ở snapshot: `6c473ea` (initial, 15:43:58), `1a0ab33` (framework, 16:44:39), `17f1acb` (merge PR #1, 17:05:19), giờ Việt Nam ngày 21/11/2026. Cây main gồm README, PROJECT_SPEC, 3 config YAML, source model/detector, adaptation, conformal, data, metrics, statistical helpers, 3 scripts, smoke test, 2 JSON và 2 hình kết quả. `checkpoints/` chỉ có `.gitkeep` ở snapshot; không kết luận về local. Không thấy manuscript hoặc data extraction pipeline hoàn chỉnh.

```text
Tri-SACTA/
  README.md, PROJECT_SPEC.md, requirements.txt
  configs/{mosei_mvp, meld, shift_regimes}.yaml
  src/data/{dataset,shift_generator}.py
  src/models/{fusion_backbone,spectral_detector}.py
  src/adaptation/{tri_sacta,baselines_tta}.py
  src/conformal/{online_aci,static_cp}.py
  src/utils/{metrics,stats_test}.py
  scripts/{01_train_source,02_run_stream_benchmark,03_plot_rolling_curves}.py
  tests/test_smoke.py
  checkpoints/.gitkeep
  results/{source_only_static_cp_mosei_seed42,tri_sacta_online_aci_mosei_seed42}.json
  results/figures/{macro_f1_regime_bars,rolling_coverage_comparison}.png
```

### 4.2 Số liệu lịch sử (không chỉnh sửa)

Cả hai run dùng `dataset=mosei`, `seed=42`, `delay=0` trong JSON. Chưa xác minh provenance của dữ liệu/weights, do đó không gọi đây là kết quả CMU-MOSEI thật.

| Metric | Source-only + Static CP | Tri-SACTA + Online ACI | Nhận xét |
|---|---:|---:|---|
| Accuracy | 41.1667% | 15.3333% | giảm 25.83 điểm % |
| Macro-F1 | 39.2106% | 3.7985% | giảm 35.41 điểm % |
| Weighted F1 | 38.8017% | 4.0771% | suy giảm |
| Empirical coverage | 74.4667% | 79.7000% | tăng 5.23 điểm %, vẫn thiếu 10.30 điểm % so với 90% |
| Coverage gap (absolute) | 15.5333% | 10.3000% | vẫn lớn |
| Avg set size | 2.7667 | 5.5873 | tăng ~2.02 lần; 5.59/7 nhãn |
| ECE | 0.2908498 | 0.0003792 | không đủ để kết luận calibration tốt |
| Brier (theo code) | 0.8573798 | 0.8571558 | gần tương đương |
| NLL | 2.9143479 | 1.9443359 | gần log(7) đối với Tri-SACTA |
| Peak allocated VRAM (MB được báo) | 20.8433 | 28.3467 | đo tensor allocation, không phải full GPU footprint |
| Avg latency (ms/sample) | 1.5775 | 7.8665 | ~4.99 lần chậm hơn, phạm vi đo cần xác nhận |

| Regime | Source Acc | Tri Acc | Source Macro-F1 | Tri Macro-F1 | Static coverage | Online coverage | Static size | Online size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 clean | 48.00% | 15.33% | 47.00% | 3.80% | 89.67% | 91.00% | 3.027 | 6.403 |
| R2 audio | 48.83% | 15.33% | 47.70% | 3.80% | 91.67% | 69.83% | 3.052 | 4.945 |
| R3 text | 50.50% | 15.33% | 49.63% | 3.80% | 93.00% | 91.50% | 3.090 | 6.458 |
| R4 visual+missing | 24.67% | 15.33% | 20.56% | 3.80% | 41.50% | 70.33% | 2.282 | 4.833 |
| R5 compound+recovery | 33.83% | 15.33% | 31.59% | 3.80% | 56.50% | 75.83% | 2.383 | 5.297 |

### 4.3 Diễn giải và việc cần xác nhận

- CONFIRMED: Tri-SACTA có cùng Acc/Macro-F1 ở tất cả regimes, NLL gần `ln(7)=1.9459`; phù hợp giả thuyết logits gần như đồng đều và argmax gần như cố định. Macro-F1 kỳ vọng nếu luôn đoán lớp có prevalence 0.153333 là `2*p/(1+p)/7 ≈ 0.037985` (khớp số liệu). Đây là bằng chứng mạnh về collapse; vẫn cần kiểm tra prediction histogram, logits variance và confusion matrix để xác nhận runtime.
- CONFIRMED: JSON lưu 3.000 lượt hit/set-size và 95 bước batch (19 batches x 5), phù hợp `600 test samples` của synthetic fallback. Nhưng không xác định được dữ liệu thật chỉ từ số mẫu. Không dùng cụm “đã chạy MOSEI thật”.
- CONFIRMED: hai phương pháp thay đổi đồng thời classifier adaptation và conformal layer. Không thể quy toàn bộ chênh lệch coverage cho ACI hoặc TTA riêng lẻ. Cần factorial comparison.
- CONFIRMED: shift velocity ban đầu có giá trị `-21.268057...`; `cross_modal_disagreement` trong run hiện có các giá trị đầu bằng 0, code không trả trường này ở TriSACTA nên benchmark luôn dùng default 0.
- ECE thấp không đồng nghĩa mô hình useful; cần dự đoán chính xác và uncertainty có khả năng phân biệt. Full set 7 nhãn cho coverage 100% một cách tầm thường, không phải thành công khoa học.
- Baseline cũng yếu/không rõ nguồn dữ liệu. Phải sửa scaffold chung và retrain source trước khi so sánh V2. Không so repaired backbone với unrepaired baseline làm main result.

## 5. Issue register

Line numbers dưới đây tham chiếu snapshot. Khi mã đã đổi, tìm theo symbol và kiểm tra thực tế. Các biện pháp sửa là đề xuất, không phải yêu cầu phải giữ nguyên thuật toán nếu có giải pháp đơn giản hơn đạt nghiệm thu.

| ID | Priority | Evidence | Module / vị trí | Vấn đề |
|---|---|---|---|---|
| AUD-01 | P0 | CONFIRMED/HYPOTHESIS | `spectral_detector.py:174–252`, `tri_sacta.py:435–502` | Raw shift không chuẩn hóa, exp underflow, ngưỡng không tương thích. |
| AUD-02 | P0 | CONFIRMED | `fusion_backbone.py:165–188` | Attention length=1, mask trên softmax một phần tử vô hiệu. |
| AUD-03 | P0 | CONFIRMED | `tri_sacta.py:208–299`, `fusion_backbone.py:408–435` | Effective mask và reliability weights không đồng nhất; anchor fallback không bảo đảm per-sample. |
| AUD-04 | P0 | CONFIRMED | `spectral_detector.py:174–208`, `tri_sacta.py:394–414` | Detector tính thống kê cả hàng missing/zero padding. |
| AUD-05 | P0 | CONFIRMED | `tri_sacta.py:125–144,245–287,571–588` | Optimizer chứa mọi adapter/LN; freeze và selective update chưa enforce. |
| AUD-06 | P0 | CONFIRMED | `01_train_source.py:47–70`, `tri_sacta.py:525–569` | Auxiliary classifiers không được huấn luyện trực tiếp, alignment target đáng nghi. |
| AUD-07 | P0 | CONFIRMED | `online_aci.py:291–317`, `02_run_stream_benchmark.py:223–225` | Signed velocity làm step size âm; disagreement không được nối. |
| AUD-08 | P0 | CONFIRMED | `online_aci.py:225–346` | Delayed feedback tính lại set thay vì set đã phát hành. |
| AUD-09 | P0 | CONFIRMED | `online_aci.py:198–218,355–381`, `static_cp.py:157–166,339–362` | Quantile rank/interpolation, cold start, alpha boundary thiếu kiểm chứng; claim coverage quá mạnh. |
| AUD-10 | P0 | CONFIRMED | `dataset.py:_load_or_generate_data`, `02_run_stream_benchmark.py:99–106` | Fallback silent từ dữ liệu lỗi/missing hoặc checkpoint thiếu. |
| AUD-11 | P0 | CONFIRMED | `dataset.py:create_speaker_disjoint_splits,get_dataloaders` | Giả lập speaker IDs, split thực tế không được verify. |
| AUD-12 | P0 | CONFIRMED | `dataset.py:MultimodalCollator` | Zero-filled modality vẫn gắn present; partial missing làm mất toàn bộ batch modality. |
| AUD-13 | P0 | CONFIRMED | `01_train_source.py:134–157` | Best state_dict không deepcopy, checkpoint có thể bị thay đổi đến epoch cuối. |
| AUD-14 | P0 | CONFIRMED | `dataset.py:synthetic generator`, seed setup scripts | Python hash ngẫu nhiên giữa tiến trình, RNG coupling. |
| AUD-15 | P0 | CONFIRMED | `02_run_stream_benchmark.py:223–225`, `baselines_tta.py` | Gọi get_shift_velocity vô điều kiện với baseline không định nghĩa method. |
| AUD-16 | P1 | CONFIRMED | `shift_generator.py:35–145`, benchmark loops | Feature noise bị gọi SNR/WER/blur; hardcoded regimes, test replay. |
| AUD-17 | P1 | CONFIRMED | `baselines_tta.py:308–379,413+` | SAR và READ là simplified approximations; CoTTA chưa có. |
| AUD-18 | P1 | CONFIRMED | `02_run_stream_benchmark.py:150–183` | Mask calibration trên clean masks, online không warm-start; tính công bằng thiếu. |
| AUD-19 | P1 | CONFIRMED | `02_run_stream_benchmark.py:289–306` | Hit stream bị đặt tên rolling; output thiếu delay/checkpoint/run ID trong filename. |
| AUD-20 | P1 | CONFIRMED | `src/utils/metrics.py:34–49,149–172` | Macro-F1 không dùng danh sách labels cố định, NLL cần FP32; cần counts/mask metrics. |
| AUD-21 | P1 | MISSING EVIDENCE | `PROJECT_SPEC.md`, README, source | Claims lệch code, không có theorem/derivation đúng biến thể. |
| AUD-22 | P1 | MISSING EVIDENCE | `tests/test_smoke.py`, results | Chưa có regression tests, run manifest, multiple seeds, ablation, paper. |
| AUD-23 | P1 | CONFIRMED | `metrics.py:182–221`, README | Peak allocated CUDA không phải end-to-end VRAM; latency scope thiếu. |
| AUD-24 | P1 | CONFIRMED | `mosei_mvp.yaml`, `meld.yaml`, training script | Checkpoint chỉ theo seed, có nguy cơ cross-dataset collision; cache directory không rõ namespace. |
| AUD-25 | P1 | CONFIRMED | `tri_sacta.py:245–299`, baseline adapt_step | Prediction từ train-mode trước optimizer step; protocol inference/adaptation chưa thống nhất. |
| AUD-26 | P1 | MISSING EVIDENCE | toàn repo | Provenance feature extractors, real source checkpoint, source-free protocol/information budget. |

### 5.1 P0: detector, fusion và collapse (AUD-01–04)

**Root cause khả dĩ:** với B=32 và D=512/768, covariance target rank tối đa 31 (còn thấp hơn khi missing), eigenvalues khác covariance source nhiều mẫu ngay cả khi cùng phân phối. `||mu_t-mu_s||²` tăng theo D; cộng khoảng cách eigenvalues không chuẩn hóa làm score lớn. `exp(-tau*d)` có thể về mức cực nhỏ/zero khi chuyển float32. Fusion chia tổng trọng số bị clamp min `1e-8`; nếu tổng thực thấp hơn clamp thì normalized weights không còn tổng 1. Classifier nhận near-zero hidden; dropout/mask không cứu được. Ngưỡng `0.3, 0.7` hiện được dùng trên raw score, mặc dù YAML mô tả confidence thresholds. Nếu tất cả corrupted, promote một anchor nhưng giữ reliability weight gần 0. Đây là giả thuyết cần tái hiện trước khi patch.

**Sửa:** đo null distribution trên source holdout windows matched B/D; standardize per modality và số quan sát; robust/shrinkage covariance; tính relative weights bằng masked log-softmax FP32 (không exponentiate raw score rồi divide by clamp). Dùng đủ mask và fallback an toàn, không drop hết evidence chỉ vì score vượt ngưỡng. Dùng mask hợp nhất ở mọi layer/fusion và exclude missing rows khi tính stats. Thiết kế attention nhiều token như §6. Log top scores, normalized weights, sum per sample, tiny/all-zero weights, mask và fallback.

**Nghiệm thu:** T01–T07 (§11); trong clean synthetic windows, detector không đánh dấu gần như mọi batch bị severe; test large distances vẫn normalize; chỉnh feature của missing stream không đổi logits; all missing có policy explicit. So sánh (old → fixed weighting → fixed attention → both) cùng checkpoint tương thích hoặc retrain theo từng kiến trúc; phân biệt hiệu ứng sửa code và hiệu ứng phương pháp.

### 5.2 P0: selective adaptation và auxiliary heads (AUD-05–06)

`get_adaptable_params` thu thập mọi LayerNorm và adapter, không tự freeze; optimizer chỉ update subset nhưng regularizer duyệt `requires_grad` và backward vẫn có thể tính gradient cho frozen-intended params. Anchor/degraded streams không thực sự được bảo vệ. KL target của anchor lấy từ per-modality logits **sau cross-fusion**, có thể đã nhiễm thông tin modality khác. Training chỉ dùng fused CE, không có loss riêng cho auxiliary heads nên không thể coi heads đó là teacher đáng tin cậy.

**Sửa:** explicit named parameter groups; freeze toàn bộ trước rồi mở whitelist; tách stream-local adapters/LN khỏi shared norm; chỉ update nhóm U khi policy yêu cầu. Nếu freeze theo batch, đảm bảo `grad=None` và optimizer momentum/weight-decay không làm thay đổi inactive params, bao gồm param đã active ở batch trước. Chỉ tính regularizer đúng param đang update. Dùng teacher/head đã được train + detach, pre-fusion và mask-aware. Train auxiliary CE có trọng số trên available modality, validate quality per modality; thử gated source-only và no-alignment controls trước. Nếu không có reliable anchor, skip adaptation hoặc dùng teacher source frozen theo rule đã pre-register; không promote modality hỏng thành “clean” chỉ để chạy.

**Nghiệm thu:** hash/compare tensors trước/sau nhiều bước, chỉ whitelist và selected groups thay đổi; anchor logits không bị gradient từ KL teacher path; no valid anchor / empty U -> no-op; aux heads có chất lượng > majority/random phù hợp trên validation; dropout policy rõ. Nếu adaptation không tốt hơn gating-only, không tuyên bố adaptation là nguyên nhân cải thiện.

### 5.3 P0: conformal controller (AUD-07–09)

Signed mean velocity có thể âm và cancellation giữa modalities; eta phải hữu hạn, dương, bounded. Kết quả TriSACTA không chứa disagreement nên dùng `0.0`. Khi label feedback trễ, update dùng `predict_set(logits, mask)` ở thời điểm label đến, thay vì `C_t` lúc phát hành. Quantile dùng interpolation, alpha bị clip, forced nonempty và cold-start có thể làm mất cơ chế boundary cần cho ACI theorem. Spec update trực tiếp quantile `q_{t+1}=q_t+eta(alpha-error)` còn sai chiều nếu q lớn làm set rộng hơn; code update alpha khác với update q.

**Sửa:** tách `predict()` và `observe()` / feedback queue; prediction immutable record lưu mask, logits/scores FP32, q, alpha, eta, set, timestamp và randomization seed; controller cập nhật miss đúng set. Không enqueue nhãn vào component có quyền truy cập trước khi đến hạn; simulator tách evaluator. Nếu cùng batch có feedback delay 0, định nghĩa rõ release timing: mẫu tuần tự hoặc batch barrier; không dùng feedback trong batch cho previous prediction đã phát hành. Delay theo sample index hoặc batch index rõ ràng. Warm-start với calibration data; strict finite sample order-statistic; handle alpha boundary và exact coverage claims theo §8. Add constant eta control, pooled ACI, mask-specific fixed control, adaptive controller only after stable.

**Nghiệm thu:** T12–T18; deterministic delay regression mà threshold đổi giữa prediction và feedback, updater dùng miss cũ; eta finite/bounded với negative/extreme velocity; coverage tests trên known exchangeable/nonstationary synthetic toy; log every alpha/q/mask/feedback event.

### 5.4 P0: provenance, data mask, determinism (AUD-10–14)

- Load real cache lỗi/thiếu phải raise, kể cả missing checkpoint; synthetic chỉ khi `--data-mode synthetic` và `--allow-random-weights` (debug) được yêu cầu rõ. Synthetic kết quả đánh dấu `SYNTHETIC_DEBUG`; không đi vào bảng paper.
- Loader dùng actual identifiers; hàm split fake `spk_000` không phải kiểm tra dataset. Verify các cặp split train/val/cal/test theo utterance/video/dialogue/group hợp lệ; MELD có recurring speakers, không áp đặt speaker-disjoint nếu official split khác. Report speaker overlap thay vì claim zero leakage sai.
- Collator phải giữ `available_mask` từng sample độc lập với feature value. Zero feature hợp lệ không đồng nghĩa missing; absent file/stream cần explicit mask và null reason. Kiểm tra shape, lengths, finite values và unique IDs.
- `best_weights = copy.deepcopy(model.state_dict())` hoặc clone CPU tensors ngay; kiểm tra val metric của loaded best model. Không dùng current state bị reference mutation.
- Không dùng built-in `hash()` làm seed; dùng fixed split offsets hoặc stable digest, separate RNG streams cho split/model/train/stream/corruption/CP randomization; set random, numpy, torch, CUDA, worker seeds. Record deterministic mode và nondeterministic ops. Generate corruption manifest độc lập method để mọi baseline gặp input giống nhau.
- Synthetic generator nên có reproducible class centers theo class/modality, noise và class imbalance có kiểm soát; không chỉ random center theo từng sample + label mean. Nhưng không sửa synthetic generator rồi dùng để claim hiệu quả trên real data.

### 5.5 P0: baseline interface (AUD-15)

Benchmark gọi `tta_method.get_shift_velocity()` với mọi TTA; các class TENT/SAR/MultimodalTTABaseline không thấy method tương ứng. Có nguy cơ AttributeError ngay khi chạy. Chuẩn hóa `AdaptationResult` (logits, adapted, losses, optional telemetry), hoặc tính shift/disagreement external trên inputs của tất cả methods cần ACI. Không silently bật adaptive step size chỉ cho ours trong matched comparison. TTA interface không nhận labels. Test integration cho từng combination CLI công bố trong README.

### 5.6 P1: những vấn đề nghiên cứu còn lại (AUD-16–26)

- **AUD-16:** shift YAML và code hardcoded lệch nhau; config phải là nguồn sự thật. Feature perturbation là latent corruption, không phải measured SNR/WER/blur. Test loader lặp 5 lần một test set; dùng protocol replay như diagnostic với ID rõ, main benchmark ưu tiên stream no-replacement/chunks có group-aware schedule; nếu thiếu số mẫu, thêm multi-domain stream hoặc báo cáo hạn chế.
- **AUD-17:** SAR hiện entropy filtering + periodic restore, chưa thể coi là faithful sharpness-aware method; generic multimodal entropy+consistency không phải READ chính thức. Gắn `SAR-inspired`, `MM-EntropyConsistency` cho bản đơn giản; port upstream hoặc ghi deviations đầy đủ. CoTTA phải implement nếu công bố so sánh; không đếm baseline chỉ có tên.
- **AUD-18:** source clean cal mask đa số `1111`; MCCF mask-specific thường fallback; ACI không init từ cal. Chạy cùng calibration budget, score, mask augmentation policy (source cal views, không target test) và cùng feedback delays; phân biệt no-feedback baseline với feedback-based method.
- **AUD-19:** JSON `rolling_coverage` thực ra binary hit stream; dùng `coverage_hits` và `rolling_coverage_w100` riêng. Run path dataset+seed+method+CP+delay+protocol+config hash+commit+run ID để không overwrite. Log run status, missing labels và group support.
- **AUD-20:** cố định class list `labels=list(range(C))` khi Macro-F1 và confusion matrix; report support, weighted F1, class distribution; NLL/softmax FP32; check empty dataset/invalid total. Coverage gap tuyệt đối và signed gap, set sizes, class/group coverage, rolling metrics; tránh chọn ECE làm metric chính.
- **AUD-21:** PROJECT_SPEC nói kernel MMD, pseudo-label, spectral alignment, randomized APS, finite sample guarantee; code mean/eig, entropy, KL, deterministic APS. Viết `METHOD_SPEC_V2.md` dựa trên implementation; hoặc triển khai variant đã định nhưng không tuyên bố MMD nếu chưa làm. GPU constraints và tỷ lệ param cũng cần thống nhất.
- **AUD-22:** smoke test chỉ imports, forward, detector, ACI; không test TTA backward/full stream, không test mask invariance hoặc checkpoint; thêm unit/integration/regression và CI CPU. Không dùng output PASS để thay cho số liệu nghiên cứu.
- **AUD-23:** đo device memory allocated vs reserved, process peak/RSS, model loading + cached feature vs end-to-end encoder. CPU eigendecomp có full D×D cost; ghi wall-clock, batch size, warmup, p50/p95, hardware, versions. Không so GPU claim với phần bị loại khỏi phạm vi đo.
- **AUD-24:** checkpoint `source_model_seed42.pt` và stats theo seed không chứa dataset/config; có nguy cơ load nhầm MOSEI sang MELD. Namespace dataset/task/encoder/config/seed và validate metadata, feature dimensions, source checkpoint hash. MOSEI config cache `features_cache/` chung trong khi MELD `features_cache/meld`; sửa đường dẫn rõ.
- **AUD-25:** TriSACTA/TENT/SAR trả logits tính trước step ở train mode khi adaptation, nhưng source-only eval mode. Chọn protocol explicit (predict eval mode trước adapt cho strict; nếu adapt-then-predict thì re-forward eval mode sau step); không coi logits train mode trước step là kết quả sau adaptation. Freeze running states/dropout phù hợp method.
- **AUD-26:** chưa có bằng chứng end-to-end feature extraction CLIP/WavLM/DeBERTa/eGeMAPS, metadata checkpoint và thông tin licenses; cung cấp artifact manifest, code extraction hoặc official feature pipeline. Raw video/audio có thể không tải được; ghi blocker và không giả lập thay thế trong real mode.

## 6. Thiết kế kỹ thuật V2

Tất cả thiết kế trong mục này là PROPOSAL. Tạo branch và commit nhỏ; không triển khai hết cùng lúc. Mỗi biến thể có config riêng, unit test và matched baseline. Ưu tiên sửa correctness trước, sau đó mới chọn formulation dựa trên validation.

### 6.1 Chuẩn hóa spectral shift detection

Vấn đề thống kê chính: cùng phân phối nhưng sample covariance kích thước n nhỏ có phổ khác population covariance. Do đó không so raw L2 eigenvalues với ngưỡng toàn cục. Một thiết kế khởi đầu:

1. Fit feature preprocessing từ source train, không target test: scale robust hoặc standardization với variance floor; optional fixed projection `P_m` (PCA source-only hoặc projection seeded không train) về `r=32/64/128` chiều.
2. Tại deployment, chọn các hàng modality thực sự available. Với `n_m < n_min`, dùng cached/rolling statistics đủ quan sát hoặc đánh dấu `insufficient_support`. Không coi không đủ dữ liệu là severe shift; nếu không dùng lịch sử thì chỉ dựa mean/confidence proxy có định nghĩa.
3. Ước lượng covariance shrinkage: `S=(1-rho)*sample_cov + rho*trace(sample_cov)/r * I`. Cân nhắc diagonal covariance/Gram trick nếu GPU/CPU hạn chế. Dùng FP32 hoặc FP64 cho decomposition, không FP16.
4. Tính vector discrepancy, ví dụ standardized mean difference và spectral discrepancy log-eigenvalues; lưu epsilon, rho, r. Không gọi đây là kernel MMD nếu chưa dùng kernel feature map.
5. Fit clean reference distribution của từng discrepancy từ source holdout windows, stratify/bucket theo `n_m` và mask phù hợp. Không tune theo test corruption labels. Robust z-score hoặc clean-null percentile; kết hợp các score với coefficients cố định/tune trên validation.
6. Calibrate ngưỡng anchor/moderate/severe theo source false alarm rate và validation corrupted streams. Xem `d_m` là severity proxy, không khẳng định đo corruption chính xác khi có pure label shift, semantic drift hoặc covariance rotation.

Ví dụ cụ thể để thử nghiệm:

```math
D_\mu^m=\frac{1}{r}\sum_{j=1}^{r}\frac{(\mu_{t,j}^m-\mu_{s,j}^m)^2}{\sigma_{s,j}^{2}+\epsilon}
```

```math
D_\lambda^m=\frac{1}{r}\sum_{j=1}^{r}\left(\log(\lambda_{t,j}^m+\epsilon)-\log(\lambda_{s,j}^m+\epsilon)\right)^2
```

```math
z_m=\max\{0,\operatorname{RobustScale}_{m,n_m}(D_\mu^m+\gamma D_\lambda^m)\}
```

Nếu dùng sample covariance low-rank comparison, so với clean windows cùng rank/sample size (không so trực tiếp full source covariance không matched). Median/MAD có floor để tránh chia cho 0; record clean validation FPR, coverage intervals. PCA có thể bỏ corruptions ở subspace bị loại: ablate r và raw/diagonal alternative. Eigenvalues không phát hiện mọi rotation of covariance; cần counterexample test và limitation.

**Tối ưu:** với `X in R^{n x D}`, nonzero eigenvalues của `X^T X` và `X X^T` giống nhau (cùng scaling) nên có thể dùng Gram matrix nếu n≪D; có tính đến centering, zeros/rank và shrinkage. Không nhầm việc tối ưu compute với novelty. Compare complexity (`O(D^3)` full symmetric eig vs `O(n²D+n³)` Gram route), đo thực nghiệm CPU time.

### 6.2 Stable reliability gating và missing mask

Tính trọng số trên log-space:

```python
# Illustrative pseudocode, implement with real tests.
# z: [B, M] broadcast batch-level scores or per-example scores.
# available and allowed are boolean [B, M].
valid = available & allowed
# If policy filtered everything but sensors exist, fall back using available.
valid = torch.where(valid.any(-1, keepdim=True), valid, available)
# All-missing rows handled outside softmax; explicit abstention/full label set.
log_r = -z.float() / temperature
masked_logits = log_r.masked_fill(~valid, -torch.inf)
weights = masked_softmax_safe(masked_logits, valid)
```

- `temperature>0`, dtype FP32. Stable subtract-max / logsumexp; no tiny exponent denominator clamp as normalizer.
- `weights.sum(-1)=1` trên mỗi valid sample, `weight=0` khi missing/disallowed, finite. All-missing trả sentinel abstain/no-input theo policy; classifier không được tự coi bias-only prediction là tin cậy.
- Hard gating phải dùng đúng effective mask ở attention, fusion, alignment, disagreement, detector; distinguish physical availability mask (group conditioning) và effective reliability mask (model decision). Không đổi group definition sau khi thấy label.
- Nếu tất cả available streams bị nghi nhiễu, dùng least-unreliable weighted inference hoặc source fallback; ghi `all_corrupt` và không update. Không gọi đó là reliable teacher.
- So sánh `uniform`, confidence, inverse entropy, normalized mean-only, normalized spectral, squared/raw legacy. Một tập weights learned ở source có thể là baseline; không fit target labels.

### 6.3 Cross-modal fusion đúng nghĩa

Thay pairwise length-1 attention bằng modality token sequence `[B, M, D]` với positional/modality embedding và attention key padding mask, hoặc dùng weighted pooling/DeepSets làm baseline mạnh đơn giản.

```text
cached visual/audio/text/prosody vectors
    -> local projection + local adapter (M tokens)
    -> unimodal heads (pre-cross-fusion, supervised at source)
    -> masked multi-token transformer / robust weighted pooling
    -> reliability-weighted masked aggregation
    -> fused classifier -> probabilities
```

- Với masked attention, logits/content của missing token không ảnh hưởng observed output; zero/mask query rows sau each residual/FFN nếu pooling hoặc các layer sau có thể dùng các rows đó. Handle all-masked row trước MHA để không NaN. Test inference `model.eval()` và train mode deterministic no dropout riêng.
- Nếu sử dụng reliability trong attention, additive log reliability prior có thể áp dụng trên key attention logits hoặc pooling. Ablate attention-only, weights-only, both để tránh double-count reliability.
- Nếu dữ liệu chỉ có một vector mỗi modality, không gọi đây là temporal attention. Để có spatiotemporal novelty phải giữ temporal features và thiết kế benchmark mới; không bắt buộc trong V2.
- Khi sửa backbone, checkpoint legacy không tương thích có thể cần retrain, ghi kiến trúc hash, không load silently `strict=False`.

### 6.4 Anchor-guided adaptation có kiểm SOÁT

Chọn theo source-calibrated `z`, per-sample availability và confidence: A clean+confident, U moderate, C severe/missing. Batch-level score chỉ là proxy; if sample-level anchor selection chưa có, nói rõ và exclude invalid samples trong loss. Cần tránh teacher generated from shifted post-fusion heads.

Đề xuất loss:

```math
L=L_{\mathrm{confident}}+\lambda_{\mathrm{align}}L_{\mathrm{anchor}}+\lambda_{\mathrm{reg}}\|\theta_U-\theta^0_U\|^2
```

- Source training: `L_src = CE(fused,y) + lambda_aux * mean_available CE(unimodal_head_m,y)`; optional train-time modality dropout/missing augmentation chỉ dùng source. Tất cả baselines dùng cùng source checkpoint trong within-backbone comparison.
- `L_confident`: chọn một cơ chế entropy minimization có filter hoặc confident pseudo labels từ detached teacher; không thêm cả hai mà không ablate. Pseudo-label confidence threshold được validation tuning.
- `L_anchor`: KL/JS hoặc distillation từ reliability-weighted pre-fusion teacher A đến U trên rows mà cả teacher và student present; loss normalize by valid pairs, no empty mean. Ghi rõ KL direction và temperature; teacher detach. Nếu claim spectral alignment, phải có differentiable loss trên features thực sự được update, không dùng detector `.detach().item()` làm regularizer có gradient; otherwise gọi anchor consistency.
- `L_reg`: chỉ trên selected adapter/LN parameters. Shared norm/heads freeze mặc định; nếu có shared update thì coi là riêng variant.
- No anchor, no U, too few confident samples, nonfinite gradient -> skip with reason. Limit per-batch update/gradient norm; optional restore/EMA teacher chỉ nếu evidence ủng hộ.
- Dùng optimizer có state quản lý theo nhóm, assert unchanged frozen tensors. Tránh gọi `.train()` gây dropout khác nhau giữa methods; choose deterministic eval-mode forward có grad khi cần, ghi policy rõ.

### 6.5 Interface / config đề xuất

```python
# API contract; names can change if backward-compatibility is documented.
@dataclass
class AdaptationResult:
    logits: torch.Tensor
    adapted: bool
    losses: dict
    shift_scores: dict
    reliability_weights: torch.Tensor | None
    disagreement: float | None
    selected_groups: list[str]
    skip_reason: str | None

class TTAMethod:
    def predict(self, features, availability, context) -> AdaptationResult: ...
    def adapt(self, features, availability, context) -> AdaptationResult: ...
    def reset(self, source_checkpoint): ...

class ConformalController:
    def predict(self, logits, mask, context) -> PredictionRecord: ...
    def observe(self, record_id, y, arrival_time): ...
```

Ví dụ config (đây là target schema, không phải tham số tối ưu hay lệnh chạy được ngay):

```yaml
experiment:
  protocol: strict_prequential
  data_mode: real
  seed: 42
  shift_seed: 1042
  feedback_delay_unit: sample
  device: auto
  output_root: results/runs
  allow_random_weights: false
  deterministic: true
model:
  fusion_dim: 128
  adapter_rank: 4
  num_heads: 4
  architecture: masked_modality_transformer
  auxiliary_loss_weight: 0.2
spectral:
  projection_dim: 64
  covariance: shrinkage
  min_observations: 8
  score: standardized_mean_log_spectrum
  thresholds_source_calibrated: true
  temperature: 1.0
adaptation:
  policy: selective
  lr: 0.0001
  max_steps: 1
  update_shared_norm: false
  skip_if_no_anchor: true
conformal:
  target_alpha: 0.1
  score: aps
  grouping: observed_mask
  window_size: 200
  eta_mode: constant
  eta_base: 0.01
  eta_min: 0.001
  eta_max: 0.05
  delay: 0
  warm_start: source_calibration
```

Các con số trên là defaults để debug, không phải settings chọn từ test. Tách `configs/debug_synthetic.yaml`, `configs/mosei_real.yaml`, `configs/meld_real.yaml`, `configs/protocols/`, `configs/ablations/` và schema validation. Lưu resolved config cùng run.

## 7. Dữ liệu thật và benchmark

### 7.1 Datasets

| Dataset | Nguồn chính thức | Task | Evaluation |
|---|---|---|---|
| CMU-MOSEI | [ACL 2018](https://aclanthology.org/P18-1208/), [CMU SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) | Sentiment 7-class theo label mapping explicit; optional Acc-2/MAE/correlation chỉ nếu đúng protocol | Official video-level splits, source train/val/cal và held-out target test; validate actual speaker overlap. |
| MELD | [ACL 2019](https://aclanthology.org/P19-1050/), [official repo](https://github.com/declare-lab/MELD) | Seven-class emotion | Official train/dev/test theo dialogue/utterance; dialogue/episode leakage audit, report speaker reuse; Macro-F1 + weighted F1 + Acc. |

MOSEI sentiment annotation có thể continuous; document cách discretize (rounding/cut points, clipping, neutral convention), không tự thay đổi mapping để tăng kết quả. MELD có imbalance lớn; report support per class, per-class F1, majority baseline. Hai tasks không share label semantics dù cùng C=7.

### 7.2 Data provenance và cache

Tạo `data_manifest.json`/CSV/Parquet với schema: `dataset_version`, `split`, `sample_id`, `video_id`, `speaker_id`, `dialogue_id`, `episode_id` (nếu có), `timestamp`, `raw_label`, `mapped_label`, `available_visual`, `available_audio`, `available_text`, `available_prosody`, `feature_source`, `encoder_id/revision`, `preprocess_hash`, `feature_dim`, `checksum`, `license/source_url`. Cho phép metadata unknown có lý do; không invent IDs từ số thứ tự để claim group split.

- Nếu dùng pre-extracted benchmark features thay vì CLIP/WavLM/DeBERTa/eGeMAPS thì đổi feature descriptions, dims, configs và manuscript. Không gọi features khác là embeddings của model chưa chạy.
- Nếu raw assets khả dụng, extract từng modality riêng, `no_grad`, encoder eval, batch small, unload giữa encoders, CPU cache; record frame rate, face crop, audio resampling, transcript tokenization, aggregation, prosody toolkit/version và failure modes. Chú ý redistribution license và dữ liệu cá nhân.
- Nếu chỉ official cached features khả dụng, chạy **latent shift benchmark** thật trên features (vẫn là real benchmark dataset), ghi rõ không phải raw corruption. Chuẩn hóa feature statistics bằng train split. Có thêm raw corruption subset nếu muốn claim physical corruption.
- Chia source cal từ train/dev theo actual group nếu official split không có cal. Không dùng test để fit clean null/source anchor/threshold. Separate validation for hparams and calibration for CP; avoid duplicate augmentations crossing splits.
- Validate all lengths/dtypes/dimensions, duplicate sample IDs, nonfinite, label range, zero vectors, missing flags, split intersections; fail closed nếu thiếu required metadata cho claim.

### 7.3 Shift protocol V2

Thiết kế hai benchmark tiers: `latent_controlled` (nhanh, tái lập) và `raw_realistic` (nếu có nguyên liệu). Không gộp bảng như cùng một loại nhiễu.

| Regime | Latent controlled | Raw realistic | Kỳ vọng đo |
|---|---|---|---|
| R1 clean | Identity transform | Original media | Clean retention, clean detector FPR |
| R2 audio | Standardized additive latent Gaussian by severity | Noise/reverb; measured SNR dB | Audio detection score, adaptation |
| R3 text | Latent dimension masking/replacement | ASR transcription or token edit; measured WER | Text reliability and F1 |
| R4 visual + missing | Latent masking/noise + modality missing | Actual blur/occlusion/compression + mask | Robustness, per-mask coverage |
| R5 compound | Multiple shifts, graded severity, missing bursts | Multi-channel physical corruptions | Worst-regime F1, coverage |
| R6 recovery | Return-to-clean on held-out future IDs | Clean new samples | Forgetting/recovery and stability |

V2 có thể tách R5 recovery thành R6 để đo rõ forgetting; giữ legacy 5-regime replay riêng cho regression, không đổi nhãn lịch sử.

**Thiết kế stream:** manifest gồm `stream_id`, `step`, sample_id, regime, observed_mask, severity vector, corruption seed, feedback schedule. Method không biết regime_id/severity khi inference (chỉ evaluator dùng). Dùng no-replacement main stream nếu có thể, order theo chronological/video/dialogue blocks hoặc reproducible group permutation có rationale. Đảm bảo shift regimes đủ support; nếu nhóm hiếm thì báo cáo confidence interval lớn, không giữ cùng mẫu lặp rồi coi là độc lập. Nếu replay cần cho repeated corruption comparison, mọi methods dùng cùng replay order và phân tích clustered by original sample/video.

**Missingness:** đánh giá MCAR độc lập sensor; burst/dropout có tương quan thời gian; structured missingness (audio/prosody cùng nguồn), mask severity-dependent. Nếu dùng label-dependent missingness trong stress test, chỉ evaluator tạo schedule và ghi MNAR assumption; không đưa true label vào model. Có explicit all-missing policy riêng; nếu benchmark loại `0000`, báo cáo tỷ lệ bị loại và lý do. Báo cáo support từng mask.

**Feedback regimes:** no-feedback (static CP, no ACI update) vs immediate (`d=0`) vs sample-delay `d=10,50` và delay random/bursty nếu phù hợp. Partial labels chỉ là extension sau fixed delay. Cùng mask/corruption/feedback schedule giữa baselines. Ghi cuối stream có bao nhiêu feedback chưa đến; coverage trên predictions có label đánh giá offline khác với controller online nhận feedback.

### 7.4 Protocol prequential cụ thể

```text
for time t / batch b:
    release feedback events whose arrival < current prediction boundary
    build unlabeled features, physical mask from manifest
    compute detector / model predictions under fixed protocol
    emit immutable prediction records + sets before labels are visible
    update TTA on current unlabeled batch (strict protocol: next batch only)
    queue/receive feedback according to explicit schedule
    evaluate offline using true labels, separate from adaptation state
    observe matured records in conformal controller exactly once
```

Nếu label delay 0 theo sample, có thể predict-observe tuần tự. Nếu batch protocol, dùng barrier và release sau khi batch predictions đã phát hành; không so sequential CP và batch CP như một setting. Event order phải có test. Target label never passed into TTA, including confidence filtering, threshold calibration and early stopping. Model reset only at declared boundary; continual mode giữ state qua regimes.

## 8. Conformal theory và giới hạn claim

### 8.1 Phân biệt ba loại bảo đảm

1. **Split CP exchangeable finite-sample marginal:** calibration và test score exchangeable với model/pipeline cố định (hay assumptions tương ứng). Rank correction chuẩn có thể đạt marginal coverage ≥ 1−alpha. Không áp dụng tự động cho continual adaptive model dưới target shift.
2. **Online long-run empirical coverage:** với update/controller assumptions thích hợp, trung bình lỗi theo thời gian tiến về target hoặc bị chặn bởi một bound. Không có nghĩa P(y_t∈C_t | x_t) ≥ 0.9 ở mọi thời điểm.
3. **Group/mask empirical coverage:** cân bằng sequence error trên subsequence mask; cần sample support và prediction/feedback protocol đúng. Mask group coverage không đồng nghĩa arbitrary covariate-conditional coverage. Rare masks có thể không có đủ data để tuyên bố mạnh. Đọc Gibbs/Candès 2021, 2024; Bastani et al. 2022; Gibbs et al. 2025 theo refs §14.

### 8.2 Score và quantile chuẩn

Dùng LAC (1−p_y) làm test oracle đơn giản hoặc APS với tie handling được document và seed. Với randomized APS, sample random variate trước prediction, lưu đúng biến random để tính true-label score sau feedback. Không redraw randomness lúc update. APS deterministic là variant riêng.

Calibration scores `s_1,...,s_n`, rank `k=ceil((n+1)*(1-alpha))`. Nếu `k>n`, threshold `+infinity` (full label set) cho split CP one-sided validity; nếu `1<=k<=n`, dùng `kth` order statistic, không linear interpolation thông thường. Với controller alpha biên, define alpha≤0 => full set; alpha≥1 => empty set (nếu proof cần). Forced top1/nonempty là policy riêng, có thể làm thay đổi exact long-run target; report separately. Với alpha ngoài [0,1], sử dụng full/empty convention nếu algorithm/proof yêu cầu, không tự clip làm mất telescoping identity.

Warm-start mask-specific state bằng calibration scores từ source calibration mask protocol; nếu support quá ít thì pooled fallback, full set hoặc hierarchical shrinkage (heuristic). Không claim mask-level guarantee cho fallback pooled nếu chưa chứng minh.

### 8.3 Fixed-step ACI: sanity proof, không phải đóng góp mới

Xét riêng mask `g`, đánh số các feedback event đã nhận cho mask đó bằng `i=1,...,n_g`. Prediction record gắn với mask lúc phát hành. Nếu update dùng lỗi thật của set đã phát hành `e_i in {0,1}` và **không projection/clipping**, với constant eta>0:

```math
a_{i+1}=a_i+\eta(\alpha^*-e_i).
```

Telescoping:

```math
\frac{1}{n_g}\sum_{i=1}^{n_g}(e_i-\alpha^*)=\frac{a_1-a_{n_g+1}}{\eta n_g}.
```

Nếu chứng minh được state `a_i` uniformly bounded thì empirical miscoverage gap trên **feedback đã quan sát** giảm O(1/n_g). Bound state cần boundary behavior hợp lệ; delayed feedback có thể tạo overshoot và cần phân tích bounded delay, số outstanding feedback và event order. Không tự áp dụng boundedness của immediate feedback vào delayed case. Với missing feedback phụ thuộc label, bound trên feedback subset không suy ra bound cho toàn bộ predictions; phải đưa assumptions hoặc đánh giá riêng. Đây là sanity derivation để kiểm tra implementation, không phải novelty.

### 8.4 Adaptive step size: điều cần chứng minh

Đề xuất telemetry `v_t=mean_m |z_{t,m}-z_{t-1,m}|` trên available common modalities (handle missing separately), `kappa_t` = mean JS disagreement normalized and masked trên confident/pre-fusion modality predictions. `v_t` và kappa phải được tính trước y, không dùng test labels. Candidate `eta_t=clip(eta0*(1+b1*v_norm+b2*kappa_norm),eta_min,eta_max)` với `eta_min>0`; normalize theo source/rolling past-only stats. Log raw vs clipped eta.

Với biến thiên eta, update không clip cho:

```math
\sum_{i=1}^{n_g}\eta_i(e_i-\alpha^*)=a_1-a_{n_g+1}.
```

Đây chỉ là **weighted** error control, không tự suy ra unweighted target. Một identity khác dùng summation-by-parts:

```math
\sum_{i=1}^{n_g}(e_i-\alpha^*)=
\frac{a_1}{\eta_1}-\frac{a_{n_g+1}}{\eta_{n_g}}+
\sum_{i=2}^{n_g}a_i\left(\frac{1}{\eta_i}-\frac{1}{\eta_{i-1}}\right).
```

Nếu |a_i|≤A thì absolute bias bound ≤ `A*(1/eta_1+1/eta_n+TV(1/eta))/n_g`. Muốn bound giảm cần control inverse-step total variation và state boundedness. Chỉ `eta_min>0` là chưa đủ khi TV có thể tuyến tính theo n. Với projection `a_{i+1}=clip(...)`, phải thêm projection residual vào identity; không được bỏ. Cần verify index/delay ordering khi định nghĩa eta từ prediction time hay feedback time. Đừng đưa identity này vào paper với tên theorem mới nếu chỉ là algebra cơ bản.

**Research directions:** constant eta per group với shift-aware base score/quantile là control có thể dễ phân tích; adaptive meta-controller có comparator/regret guarantee (so sánh AgACI/FACI/multivalid work); mask grouping/hierarchical pooling với explicit guarantee vs heuristic; bounded delay controller. Prioritize một variant có proof thật, không đổi algorithm chỉ để tạo công thức nhìn phức tạp.

**Acceptance:** viết `docs/THEORY.md` gồm assumptions, filtrations, variables, event order, theorem statement, proof, counterexamples, boundary behavior, correspondence to code và claim limitations. Chạy simulation kiểm tra bound với adversarial errors, varying eta, rare masks, delayed feedback. Simulation không thay proof. Nếu không có theorem mới, paper phải mô tả đây là empirical engineering contribution, không claim formal guarantee mới; novelty Q1 cần được đánh giá lại.

### 8.5 Disagreement và feedback record

Ví dụ `PredictionRecord`: `id`, `sample_id`, `emitted_at`, `available_mask`, `effective_mask`, `score_vector`, `random_u_or_rng_key`, `label_set`, `alpha_at_emit`, `q_at_emit`, `eta_at_emit`, `shift_score`, `disagreement`, `model_revision`, `feedback_due_at`. `observe(id,y,arrival)` reject duplicate/wrong ID/early label; compute `miss = y not in record.label_set`; append score for y; update alpha state của observed mask. Không recompute set từ live model/controller. Keep bounded buffer and log dropped/pending feedback with explicit policy.

## 9. Baselines, ablation và thống kê

### 9.1 Minimum comparison matrix

Các baseline phải được chọn vì giúp trả lời RQ, không phải vì cần đạt số lượng arbitrary. Mọi baseline dùng cùng source training/checkpoint per architecture, same samples/corruptions/masks, hparam budget và prediction protocol. Khi method đòi backbone khác, làm native reproduction separately và phân biệt cross-backbone comparison.

| ID | TTA/fusion | Conformal | Mục tiêu |
|---|---|---|---|
| B0 | Majority/random, uniform logits | None/full set diagnostic | sanity, trivial coverage |
| B1 | Source-only uniform fusion | None / static CP / pooled ACI | source reference + CP effect |
| B2 | Source-only + reliability weighting | Matched CP | isolate detector/gating |
| B3 | TENT (LayerNorm adaptation variant) | Matched CP | entropy baseline; document deviation from BN original |
| B4 | Faithful SAR where applicable | Matched CP | reliable sharpness-aware adaptation |
| B5 | CoTTA / faithful compatible variant | Matched CP | continual adaptation/failure recovery |
| B6 | READ faithful or documented port | Matched CP | strong reliability multimodal baseline |
| B7 | Recent relevant multimodal TTA (e.g., Adapt-then-fuse/STA, scope permitting) | Matched CP | compare closest approach; if incompatible explain |
| B8 | Uniform adapter+LN adaptation | Matched CP | same parameter budget control |
| B9 | Tri-SACTA V2 | None / static CP / pooled ACI / mask ACI / proposed | isolate complete contribution |
| C1 | Locked classifier | Split CP vs mask-conditional split CP | group calibration |
| C2 | Locked classifier | Fixed eta pooled ACI vs fixed eta mask ACI | mask effect |
| C3 | Locked classifier | strong adaptive online CP baseline vs proposed mask+adaptive | step size / novelty |

B4/B5/B6/B7 are planned, not claimed available or implemented. `READ-style` approximation must not be reported under original method name without qualifier; use upstream repo/license or documented adaptation; log source commit and hyperparameter selection. CoTTA raw augmentations might be incompatible with feature-only setting; define feature augmentation carefully and label method `CoTTA-style (feature-level)` if needed. Do not claim direct raw benchmark SOTA from latent comparison.

### 9.2 Ablation plan

Prioritize compact controlled matrix on one real dataset, then repeat key variants on second dataset if possible:

| A | Difference from full/controlled model | Expected interpretation |
|---|---|---|
| A0 | Uniform mask-aware pooling, no TTA | repair scaffold reference |
| A1 | Mean-only detector | spectral term contribution |
| A2 | Confidence/reliability prior instead of spectral | against simple proxy |
| A3 | Raw/unscaled vs normalized detector | correctness and clean FPR, diagnostic only |
| A4 | Gating-only, no parameter updates | isolates adaptation |
| A5 | Uniform adapters vs selective adapters at comparable budget | selectivity |
| A6 | No anchor KL / confidence filter / regularizer, one factor at time | component impact |
| A7 | Pooled vs observed-mask CP | group granularity |
| A8 | Constant eta vs shift only vs disagreement only vs combined | step-size components |
| A9 | Immediate vs delayed feedback | operational sensitivity |
| A10 | Mask-aware attention vs simple reliability-weighted pooling | architecture complexity |
| A11 | Different sample count/projection/rank/window/threshold | sensitivity/stability |

Không so full model với variants intentionally broken để chứng minh novelty. Báo cáo repaired baselines chính; legacy faulty code chỉ là engineering regression.

### 9.3 Metrics

- Classification: Acc, Macro-F1 (fixed classes), weighted F1, per-class recall/F1, confusion matrix, label histogram vs prediction histogram, NLL, Brier, ECE. Per-regime and severity, mean/worst-regime; recovery clean F1 and gap vs source clean; measure forgetting using held-out clean windows without resetting/adapting on them unless declared.
- Detector: AUROC/AUPRC on synthetic/known corruption labels held only by evaluator, false alarms on clean, rank correlation with severity, score distribution shift across clean vs corrupt, reliability/accuracy association, per-mask sample count.
- Conformal: empirical coverage, signed under-coverage/absolute gap, mean/median set size, p90, full-set rate, singleton rate, empty-set rate, conditional per-mask/per-class/rare masks, rolling min/mean and time to recovery. Coverage at 80/90/95% nominal as sensitivity; compare size at comparable realized coverage (coverage-size Pareto). Report counts e.g. `hit/n`, not just percentages. Global coverage near 90% can hide subgroup undercoverage.
- Feedback: delay, observed/pending/expired/duplicate, fraction labels released, hit at emission vs update record. Keep offline true-label evaluation separate from deployment observations.
- Resource: total params, trainable params before/after freeze, optimizer param count, number/ratio actually changed and average updated params per step, CPU/GPU memory, latency p50/p95, throughput, warmup, cache-only vs end-to-end.

### 9.4 Statistical inference

- Start seed 42 debug, then minimum 5 pre-specified seeds `[42,100,2026,777,999]` as config; more independent stream seeds if effect sizes uncertain. Separate seed for training and corruption/schedule; paired methods use identical random scenarios. Avoid selecting best seed or dropping failed runs silently.
- Report mean±SD and paired differences/CI; five seeds give weak power, do not infer absence of effect from p>0.05. Wilcoxon exact two-sided with n=5 nonzero pairs cannot achieve p<0.05 at most extreme case (minimum 2/32=0.0625). If hypothesis testing essential, plan more independent pairs before running.
- Use group-aware or moving/block bootstrap (video/dialogue/stream/time blocks) appropriate to dependence, not IID bootstrap over 3,000 replayed predictions. Preserve pair structure. State resampling parameters and check sensitivity. Per-mask Wilson/other binomial intervals are descriptive only under correlation; prefer cluster-aware methods when possible.
- Use effect size paired (mean difference / SD of differences for dz if meaningful), report assumptions; multiple testing Holm or pre-specified primary outcome. If reporting accuracy and coverage as co-primary, specify criterion and multiplicity plan. No cherry-picking metrics.
- Pre-register target metrics and hparam budget on validation; freeze config before full test. Save `docs/EXPERIMENT_PROTOCOL.md` commit hash prior to final benchmark. All run failures and missing results documented.

### 9.5 Efficient experiment order

1. CPU toy unit tests + synthetic debug 1 seed, small batches (all 5–6 regimes).
2. Real data audit + source-only clean training one dataset, validate baseline against majority and expected reference for given features/split (not universal target).
3. Fixed source checkpoint: B1, B2, B8, V2; fixed CP or no CP; inspect whether detector/adaptation actually works.
4. Locked classifier: static, pooled ACI, mask ACI, adaptive candidate; feedback delays. Confirm correctness before end-to-end joint run.
5. Pilot one real test protocol only after validation tune; do not iterate on final test. Then freeze methodology and run seeds, datasets and ablation matrix. If data/GPU blocked, leave explicit commands and statuses, not fabricated results.

## 10. Tài nguyên, hiệu năng, reproducibility

### 10.1 Hardware budget

Mục tiêu laptop 4GB là ràng buộc thực dụng đề xuất, dựa trên README; cần xác nhận GPU, RAM, CPU, disk và OS thực tế trước job. Không mặc định có A100/3090. Không coi cache-only <50 MB CUDA allocated là proof full CLIP/WavLM/DeBERTa end-to-end <4GB. Phân biệt ba stage:

| Stage | Chiến lược |
|---|---|
| Feature extraction | CPU hoặc single frozen encoder GPU, inference-only, small batches, unload encoder, cache to disk. Không tải tất cả encoder lên GPU cùng lúc. |
| Source training | fusion dim 128/256, AMP khi CUDA hỗ trợ, small batch/grad accumulation cho source; early stopping validation; monitor allocated/reserved/RSS. |
| Online TTA + CP | one-pass / one update per batch, FP32 shift statistics/CP, low-rank method; no unnecessary giant deep copy each step; bounded windows/buffer. |

GPU OOM không được silently giảm batch size ở riêng ours: batch size ảnh hưởng spectral statistics và online TTA, phải tạo experiment setting mới và chạy đối chứng. CPU fallback chỉ cho debug nếu quá chậm; separate hardware results.

### 10.2 Efficiency protocol

- Ghi model architecture, GPU/CPU name, software/driver versions, dtype, batch size, power setting nếu biết, cache/storage location; warmup riêng; CUDA synchronize khi đo latency; không trộn wall-clock của loading/compute nếu không định nghĩa.
- Report GPU peak allocated + peak reserved (PyTorch), process GPU usage nếu có NVIDIA tool, CPU RSS, disk cache, peak preprocessing memory; report unavailable nếu platform không đo được.
- Latency breakdown: feature reading/decoding, host-to-device, spectral stats CPU/eig, network forward, backward/update, conformal set build, feedback update; aggregate p50/p95 per batch and per sample, throughput. Thời gian chờ label không tính compute latency nhưng report end-to-end operational delay riêng.
- So sánh source, gating-only, full model với cùng batch/device. Nếu yêu cầu real-time, xét protocol khi stream đến nhanh hơn adaptation (Alfarra et al., 2024 [R13]).
- Parameter budget: báo `total`, `requires_grad`, `optimizer_params`, `active_params`, `changed_params` và tỷ lệ. Chọn ràng buộc budget dưới 5% hoặc 2% sau khi đo; README/PROJECT_SPEC phải thống nhất. Source model total params không tính frozen external encoders cần khai báo rõ.

### 10.3 Run manifest đề xuất

```json
{
  "run_id": "unique-id",
  "git_sha": "actual-commit-sha",
  "git_dirty": false,
  "status": "COMPLETED|FAILED|BLOCKED",
  "data_mode": "real|synthetic_debug",
  "dataset": "mosei",
  "dataset_version": "verified-version",
  "dataset_manifest_sha256": "actual-hash",
  "split_manifest_sha256": "actual-hash",
  "feature_manifest_sha256": "actual-hash",
  "checkpoint_sha256": "actual-hash",
  "source_stats_sha256": "actual-hash",
  "resolved_config_sha256": "actual-hash",
  "stream_manifest_sha256": "actual-hash",
  "method": "tri_sacta_v2",
  "conformal": "mask_aci_constant",
  "seed": 42,
  "shift_seed": 1042,
  "protocol": "strict_prequential",
  "feedback_delay": 0,
  "delay_unit": "sample",
  "calibration_budget": "record actual counts",
  "device": "actual-device",
  "env_versions": {},
  "start_time": "actual-timestamp",
  "end_time": "actual-timestamp",
  "command": "actual-command",
  "artifacts": {},
  "notes": []
}
```

Đây chỉ là schema mẫu, không phải run thật. Run folder lưu resolved YAML, manifest, source stdout/stderr, `metrics.json`, per-sample prediction records hoặc privacy-preserving summary đủ kiểm chứng, `telemetry.parquet/jsonl`, figures, environment. File output không được self-report `COMPLETED` nếu exception hoặc thiếu final manifest; atomic write/temp rename nếu cần. Dùng SHA-256 checksums; verify checkpoint metadata trước khi load; tránh public raw labels/identifiable metadata nếu license không cho phép.

## 11. Test plan và acceptance gates

### 11.1 Regression/unit tests bắt buộc

Các test dưới đây nên dùng `pytest`, CPU mặc định; CUDA tests skip có lý do nếu không có GPU. Mỗi lỗi P0 có test fail hoặc explicit reproduction trước fix nếu khả thi, rồi pass sau fix. Tests phải assert invariants, không chỉ print. Tên file là đề xuất.

| Test ID | Test / module | Assertions / tiêu chí pass | Issue |
|---|---|---|---|
| T01 | `test_spectral_clean_null` | score distribution clean windows matched dimension/size; validation FPR gần target thiết kế, report tolerance cố định trước test | 01 |
| T02 | `test_spectral_corruptions` | toy mean/variance shift tăng expected discrepancy, finite; rotation counterexample documented | 01 |
| T03 | `test_stable_weights_extreme` | z cỡ 1e3/1e6, no NaN/Inf, available weights sum≈1, masked weights exactly 0 | 01/03 |
| T04 | `test_detector_excludes_missing` | thay missing-filled features bằng random lớn không đổi stats/valid logits; n_m=0/1 handled | 04/12 |
| T05 | `test_mask_invariance_attention` | thay missing modality content không đổi valid outputs/logits trong eval mode; tolerance CPU 1e-5 hoặc documented dtype | 02/03 |
| T06 | `test_effective_mask_fallback` | all-filtered but at least one observed -> valid fallback; all-missing -> explicit policy; no NaN | 03 |
| T07 | `test_attention_token_axis` | weights across M>1 tokens respond to Q/K change, key mask effective; or declare pooling-only variant | 02 |
| T08 | `test_selective_parameter_delta` | inactive/corrupted/anchor local params và shared frozen params unchanged; active update nonzero khi valid loss | 05 |
| T09 | `test_optimizer_inactive_state` | param active at t then inactive at t+1 không bị momentum/weight decay update | 05 |
| T10 | `test_anchor_detach_and_aux_grad` | source auxiliary heads get gradients; TTA teacher detached; mask-invalid pairs contribute zero | 06 |
| T11 | `test_adaptation_skip` | no U/no valid anchor/all-missing/nonfinite -> no optimizer step; reason logged | 05/06 |
| T12 | `test_step_size_bounds` | negative/extreme/non-finite velocity; defined fallback and eta_min≤eta≤eta_max; nonzero disagreement affects candidate eta | 07 |
| T13 | `test_prediction_feedback_immutable` | emit set, change controller/model, observe original; miss equals original set; duplicate/early feedback rejected | 08 |
| T14 | `test_delay_ordering` | d=0/1/10, event counts/release order, pending tail, sample vs batch semantics; no early labels | 08 |
| T15 | `test_cp_order_statistic` | small n with exact rank, ties, n=0, k>n -> full set per policy; no interpolated under-coverage threshold | 09 |
| T16 | `test_aps_randomization_replay` | same random draw/seed for prediction and realized score, deterministic replay | 09 |
| T17 | `test_aci_boundary_and_identity` | fixed-eta update direction, full/empty alpha boundary if selected theorem; telescoping identity exact to tolerance | 07–09 |
| T18 | `test_mask_group_isolation` | observations for mask A do not silently mutate mask B unless explicitly pooled; support/fallback recorded | 09/18 |
| T19 | `test_real_data_fail_closed` | missing/corrupt file, wrong feature dims, invalid label -> explicit error; synthetic debug only opt-in | 10 |
| T20 | `test_split_group_leakage` | actual duplicate ID/group injected -> fail, official recurring-speaker exception documented and checked | 11 |
| T21 | `test_collator_per_sample_missing` | mixed present/missing rows preserve others, true zero-valued available row remains available | 12 |
| T22 | `test_best_state_immutable` | improve epoch then mutate weights, saved best state remains original, restores val best | 13 |
| T23 | `test_seed_cross_process` | same seeded synthetic data/manifest hashes in independent processes; corruption same across methods | 14 |
| T24 | `test_baseline_interface` | all implemented benchmark methods run tiny stream without missing methods/labels/NaN | 15/17 |
| T25 | `test_stream_no_leakage` | adaptation function never receives y/regime oracle; matching stream IDs/masks for all paired methods | 16/25 |
| T26 | `test_metric_known_values` | hand-calculated F1 including absent class, NLL, coverage, rolling vs binary hit sequence, empty guards | 19/20 |
| T27 | `test_run_id_and_checkpoint_compat` | differing dataset/config/delay cannot collide, wrong metadata/checkpoint rejected, reruns preserve existing output | 19/24 |
| T28 | `test_checkpoint_roundtrip` | saved/loaded model/detector outputs match same batch, reset continual state as specified | 24 |
| T29 | `test_cpu_gpu_precision` | if CUDA available, AMP forward/backward finite and dtype-tolerant, FP32 CP/detector checks; CPU pass | 23 |
| T30 | `test_prequential_protocol` | predictions use correct model version, strict method cannot update on current labels, no future info | 25 |
| T31 | `test_manifest_completeness` | completed real run contains provenance hashes, source checkpoint, data mode, config and status | 22/26 |

**Recommended numeric checks:** toy fixture with separable data should enable source learning above majority/random; when no-op detector/weights uniform and no adaptation, Tri-SACTA output must equal repaired source model output on same inputs within dtype tolerance. For high-D clean null, test multiple windows, not single random draw. FPR/coverage simulation thresholds should be statistically justified and fixed, not retrofitted after failure. Unit tests can verify guarantees of specific toy settings; they do not prove real-world accuracy or Q1 quality.

### 11.2 Integration scripts

Current CLI known from README: `python tests/test_smoke.py`, `python scripts/01_train_source.py --config configs/mosei_mvp.yaml --epochs 10 --seed 42`, `python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --seed 42`. These are legacy commands, may fall back to synthetic or fail; do not run as public evidence without guard. Preserve or document breaking CLI changes.

Target commands for implementation (not yet runnable until agent implements flags/files):

```bash
python -m pytest -q
python scripts/00_validate_data.py --config configs/mosei_real.yaml --data-mode real
python scripts/01_train_source.py --config configs/debug_synthetic.yaml --data-mode synthetic --epochs 2 --seed 42
python scripts/02_run_stream_benchmark.py --config configs/debug_synthetic.yaml --data-mode synthetic --method source_only --conformal static_cp --seed 42
python scripts/02_run_stream_benchmark.py --config configs/debug_synthetic.yaml --data-mode synthetic --method tri_sacta --conformal online_aci --seed 42
python scripts/01_train_source.py --config configs/mosei_real.yaml --data-mode real --seed 42
python scripts/04_run_experiment_matrix.py --matrix configs/experiments/pilot.yaml --dry-run
python scripts/04_run_experiment_matrix.py --matrix configs/experiments/pilot.yaml
python scripts/05_aggregate_results.py --input results/runs --only-real --output results/summary
```

Support PowerShell via same Python CLI; no bash-specific dependency for primary workflow. Record real commands with actual errors. Do not require 50 epochs or full data download to pass CI. Integration should check every supported `--method` and `--conformal` combo at tiny scale; reject unsupported combinations explicitly.

### 11.3 Stage gates (go/no-go)

| Gate | Necessary evidence | Decision if not passed |
|---|---|---|
| G0 Audit preserved | snapshot and current SHA, legacy JSON preserved, environment/data inventory, issue matrix | stop modifying blindly; record blockers |
| G1 Engineering correctness | all applicable P0 tests pass; no silent fallbacks; stable logits/weights; no leakage; tiny stream works | fix before growing experiments |
| G2 Data and source | real manifest, verified split/checkpoint, source above appropriate majority/simple baseline on validation, aux accuracy diagnostic | improve data/source training; do not run full target matrix |
| G3 Method sanity | validation clean detector FPR, severity separation, non-collapse, actual parameter updates/skip stats; fair B1/B2/B8 comparisons | isolate failure, simplify or drop method component |
| G4 Conformal correctness | immutable feedback, CP quantiles + boundary tests, fixed eta baseline, per-mask support, delayed settings; theory aligns with implementation | use simpler controller and narrow claims |
| G5 Research evidence | frozen protocol, multiple seeds, appropriate strong baselines, ablations, two real datasets or justified narrower claim, CI and resource metrics; advantage on primary metric or meaningful Pareto tradeoff | report negative results; no Q1-ready claim |
| G6 Theory/novelty | documented literature gap, actual new method/analysis beyond composition; theorem proof if claimed, limitations and feedback assumptions | narrow contribution/rework; don't invent theorem |
| G7 Manuscript | tables generated from completed manifests; no synthetic contamination; verified refs; journal scope and current quartile verified | do not submit |

Suggested scientific targets, **not fixed acceptance thresholds for results**: source model not collapsed on real clean validation; Tri-SACTA's robust macro-F1 or worst-group performance consistently better than strongest matched controls or demonstrates a quantified accuracy/coverage/efficiency tradeoff; calibrated target coverage close to 90% within pre-specified tolerance with useful set sizes; improvement in delayed/missing settings without unacceptable clean forgetting. Avoid demanding that all regimes have exactly 90% empirical coverage or that every seed wins; such targets invite overfitting and are not a universal Q1 requirement.

## 12. Roadmap giao việc và deliverables

### 12.1 Work packages

| WP | Dependencies | Output | Definition of done |
|---|---|---|---|
| WP0 Inventory/reproduce | None | `docs/AUDIT_STATUS.md`, legacy manifest, environment report | current HEAD confirmed; code/log evidence mapped; attempt legacy reproduction with explicit data mode, or blocker |
| WP1 Data integrity | WP0 | loader, split validator, explicit debug/real, checkpoint guards/namespace, reproducible RNG | T19–T23/T27/T31 pass or justified block |
| WP2 Fusion + spectral correctness | WP0/WP1 debug fixtures | stable weights, mask-correct fusion, null-calibrated detector, telemetry | T01–T07 pass; collapse diagnostic; old vs fixed results labeled |
| WP3 Selective adaptation | WP2 | explicit groups, aux CE, pre-fusion teacher, skip policy, common baseline interface | T08–T11/T24 pass; no-op parity and actual parameter delta |
| WP4 Conformal correctness | WP0/WP1 | immutable record/observe, rank quantile, constant-eta, delayed queue, validation | T12–T18/T30 pass; delay correctness; coverage sanity simulation |
| WP5 Real source pipeline | WP1/WP2 | CMU-MOSEI + MELD data/extraction manifests, source training and checkpoints | real data verified or blocked; G2 satisfied for first dataset |
| WP6 Method candidate / theory | WP3/WP4/WP5 | detector/calibration sensitivity, validated method spec and theory note | G3/G4; supported candidate selected on val only |
| WP7 Benchmarks & ablation | WP6 | official strong or documented baseline ports, paired run matrix, metrics aggregation | G5; real multi-seed results with provenance |
| WP8 Paper and reproducibility | WP7 + literature | manuscript draft, plots/tables/appendix, released code/config | G6/G7; human review; no auto submission |

WP2 and WP4 can proceed in parallel with toy fixtures, but independently validate. WP5 may be blocked by data/license/disk access; continue engineering tests and return a blocker. Estimate previous analysis 1–2 weeks correctness, 2–3 weeks data, 2–4 weeks experiments, several weeks theory/writing is only rough human workload; not an execution promise. Actual time depends on availability, implementation quality and negative results. Do not run unbounded loops or launch all seeds until preflight passes.

### 12.2 Suggested tree additions

```text
Tri-SACTA/
  docs/
    AUDIT_STATUS.md
    METHOD_SPEC_V2.md
    DATA_PROTOCOL.md
    EXPERIMENT_PROTOCOL.md
    THEORY.md
    RELATED_WORK_MATRIX.md
    REPRODUCIBILITY.md
    HANDOVER.md
  configs/
    debug_synthetic.yaml
    mosei_real.yaml
    meld_real.yaml
    protocols/
    experiments/{pilot,main,ablation}.yaml
  src/
    data/{manifest,validate,stream}.py          # proposed
    models/{masked_fusion,normalized_spectral}.py
    adaptation/{interfaces,parameter_policy}.py
    conformal/{records,feedback,quantiles}.py
    utils/{seeding,provenance,run_logging}.py
  scripts/
    00_validate_data.py
    01_train_source.py
    02_run_stream_benchmark.py
    03_plot_rolling_curves.py
    04_run_experiment_matrix.py
    05_aggregate_results.py
  tests/
    test_data_integrity.py
    test_spectral.py
    test_masked_fusion.py
    test_selective_adaptation.py
    test_conformal.py
    test_feedback.py
    test_reproducibility.py
    test_stream_integration.py
  results/
    legacy/          # optional copy with manifest; keep existing files
    debug/           # never included in paper aggregation
    runs/            # unique run IDs, immutable results
    summary/         # generated tables, plots and statistical reports
  paper/             # later, after method/experiments mature
```

Agent may adapt names to existing package to avoid unnecessary rewrite; requirement is separations and auditable outputs, not exact file count. Prefer incremental changes and coherent interfaces. Add pinned/lockable requirements or environment file, `.gitignore` for licensed data/checkpoints and logs too large, CI with CPU tests. Commit generated summary/result samples only if safe and policy permits; no raw licensed dataset commit.

### 12.3 First iteration handoff priorities

Do not attempt paper writing as first output. First iteration should complete WP0 and as much P0 correctness as resources permit, with a short report:

1. Confirm snapshot/current worktree and preserve history.
2. Reproduce `weight_sum`, raw `d`, `active_modality_count`, logits distribution and class collapse on available debug fixture; if unable, show precise error and assert minimal reproduction.
3. Fix stable masked weights and attention mask with tested minimal patch, validate no-op parity. If architecture changes, retrain source in debug and document checkpoint incompatibility.
4. Guard source/data provenance, fix seed and best-weight cloning, baseline interface.
5. Fix conformal update/eta/delay and quantile baseline independent of TTA.
6. Run CPU pytest + minimal end-to-end benchmark, report actual results. Do not conclude from synthetic F1 that real research succeeded.
7. Write next steps, blockers, commits and exact test outputs in `docs/HANDOVER.md`. Real data/novelty experiments are next only after G1.

## 13. Novelty, reviewer và manuscript Q1

### 13.1 Prior art matrix (verified bibliographic sources, preliminary comparison)

| Prior work | Already addresses | Implication for Tri-SACTA |
|---|---|---|
| TENT, ICLR 2021 [R1] | Entropy-minimization TTA | Entropy loss alone not novel. |
| CoTTA, CVPR 2022 [R2] | Continual adaptation and restoration | Continuous updating/forgetting mitigation alone not novel. |
| SAR, ICLR 2023 [R3] | Reliable sharpness-aware TTA under dynamic shifts | Reliability-based selection and collapse mitigation not new generally. |
| READ, ICLR 2024 [R4] | Reliability-aware multimodal fusion and robust adaptation | Strong direct counterpart; do not claim first reliability-aware MM-TTA. |
| APS, NeurIPS 2020 [R5] | Adaptive prediction sets for classification | APS score not new. |
| ACI, NeurIPS 2021 [R6] | Online coverage under distribution shift | Online alpha update not new. |
| Gibbs & Candès, JMLR 2024 [R7] | Online conformal with adaptive step-size tuning | Changing eta over time not enough novelty. |
| Bastani et al., NeurIPS 2022 [R8] | Adversarial multivalid online conformal | Group/threshold calibrated online coverage already studied. |
| Conformal prediction with conditional guarantees, JRSSB 2025 [R9] | Subgroup coverage formulations | Mask is a subgroup; compare assumptions and novelty. |
| Adapt-then-fuse, 2026 [R10] | Instance feature adaptation and reliability-aware fusion | Recent overlap with selective feature adaptation; study actual algorithm. |
| STA, 2026 [R11] | Spatio-temporal alignment and reliability-aware multimodal TTA | Recent overlap with alignment + weighting. |
| Delayed ACI, arXiv 2026 [R12] | Delayed feedback coverage analysis | Preprint, not verified peer-reviewed; investigate before claiming delayed-feedback novelty. |
| Alfarra et al., ICML 2024 [R13] | TTA under compute/time budgets | Low latency evaluation needs fair protocol. |

This is a targeted, non-exhaustive literature check through 21/11/2026, not a systematic review or proof no equivalent work exists. Papers [R10–R12] need full-text method/theorem comparison and final bibliographic verification; no fabricated descriptions of unavailable figures or appendices. Prefer peer-reviewed venue versions, label preprints. Search Semantic Scholar/Google Scholar/DBLP/OpenReview/ACL/IEEE/ScienceDirect for 2024–2026 and forward/backward citations of READ, ACI, multivalid approaches. Record date, query, inclusion/exclusion and comparison dimensions. Confirm if 2026 items are published online vs issue date. Do not treat earlier chat's paper recommendations as authoritative without source.

### 13.2 Candidate research gap (hypothesis to investigate)

> Can modality-wise, source-calibrated spectral evidence improve the efficiency and stability of group-aware online conformal calibration while enabling selective test-time adaptation in streams with compound, temporally evolving missing modalities under delayed label feedback and a resource-constrained feature-level setting?

Novelty must be in precise algorithmic coupling or theory+evaluation, not just applying ACI after MM-TTA. Candidate framing A: **shift-aware selective adaptation + feedback calibration** with new reliability estimator, robust empirical evidence. Framing B: **mask-conditional online conformal control** with formal analysis and strong CP baselines, TTA as application. Framing C: **resource-constrained benchmark/system** only if measured budget and benchmark contribution substantial. Prefer one main story and others supportive.

**Decision rule:** if prior art already solves B with same groups/delay/step adaptation, do not rename it as new. Derive a genuinely distinct algorithm or choose a narrower application contribution. Compare mathematically which assumptions and guarantees differ; test against actual strongest counterpart, not only source-only.

### 13.3 Simulated reviewer questions and response artifacts

| Reviewer objection | Evidence needed |
|---|---|
| Why is the combination novel, not READ + ACI? | prior-art comparison table, explicit algorithm difference, matched composite baseline READ+ACI, ablation and theoretical contribution if claimed |
| Is detector mostly measuring batch size, not corruption? | matched-size clean-null calibration, sample-size sensitivity, AUROC/FPR by modality, rotation/label-shift limitations |
| Could gating explain all gains? | gated source vs selective adaptation at same backbone/checkpoint and confidence policy |
| Are anchor heads trustworthy and really clean? | pre-fusion aux source accuracy, reliability ranking, anchor error vs shift, parameter-delta test |
| Does missing modality leak through attention? | masking invariance and all-missing tests, modality availability logs |
| Does online method use labels during test? | event ordering, adaptation API without labels, distinct no-feedback and delayed regimes |
| Why is 90% guaranteed? | exact definition, theorem with conditions or weaker empirical language; finite-sample vs long-run separation |
| Are rare mask results meaningful? | actual group counts, CIs, fallback support and assumptions, coverage-size tradeoff |
| Is dataset truly MOSEI/MELD? | provenance, licenses, official split, mapping, checkpoints, feature metadata |
| Are comparisons fair? | fixed stream manifests, paired seeds, equal feedback budgets, source model hash, faithful baseline notes |
| Are gains only on synthetic feature noise? | real dataset + latent/raw distinction; real corruption subset when feasible; external validity discussion |
| Is 4GB claim credible? | phase-specific measured device specs, allocated/reserved/system memory and preprocessing cost |
| Can other researchers reproduce? | exact commits, environment, manifests, commands, seeds, generated results and artifact limitations |

### 13.4 Manuscript skeleton

- **Title:** avoid overstating guarantees and modalities. Suggested working title: “Spectral Reliability-Guided Continual Multimodal Test-Time Adaptation with Mask-Aware Online Conformal Calibration”. Keep Tri-SACTA acronym if authors wish, define streams honestly. Final title depends on actual main contribution.
- **Abstract:** problem, precise gap, proposed method, datasets/protocol, key real numbers and limitations. Do not fill numbers until generated from completed runs.
- **Introduction:** deployment challenge, why separate reliability and uncertainty, three validated contributions at most.
- **Related Work:** multimodal TTA/reliability, CTTA/forgetting, missing modality, conformal/online/group/delay, efficiency.
- **Problem formulation:** inputs, masks, labels/feedback access, chronology, source-free assumptions, notation and objective.
- **Method:** detector, weighting, selective update, CP controller, algorithm pseudocode, complexity; equations identical to implementation.
- **Theory:** assumptions, theorem/lemma/proof or clearly labeled empirical analysis; explain alpha boundary/delay and rare masks. No unsupported guarantee.
- **Experiments:** data/provenance, stream construction, baseline reproduction/config, metrics, primary hypotheses, feedback and resources.
- **Results:** all primary metrics and CIs, per-regime and per-mask, ablation, recovery, sensitivity and failure cases. Do not conceal negative results.
- **Limitations/Ethics:** feature cache constraints, dataset demographics/context, sentiment/emotion model uncertainty, label feedback assumption, rare masks, correlation, source stats privacy and measurement limits. Do not claim emotion inference always accurate in real humans.
- **Reproducibility:** code, required licenses, source feature accessibility, exact commands, seeds, run hashes, environment, supplementary data.

Tables to generate, not fabricate: data split counts; main results mean±SD for 2 datasets; per-regime metrics; mask coverage/support; delay curves; ablation; detector discrimination; efficiency; comparison to related work. Figures: pipeline; stream regime/severity; scatter reliability vs correctness; rolling coverage w=100 annotated with regimes; coverage-size Pareto; confusion matrices; clean recovery; parameter/VRAM/latency tradeoff. Use vector PDF/SVG where allowed, accessible line styles, explicit uncertainty bands, correct axis and captions. Any real result in paper must be reproducible from run manifest.

### 13.5 Candidate venues (scope, not automatic Q1 or acceptance)

| Venue | Fit | Requirement for this project |
|---|---|---|
| Information Fusion | high if reliability-aware information fusion is central | strong fusion novelty, real multimodal datasets, comparison with related multimodal methods |
| IEEE Transactions on Multimedia | high if audio/visual/text multimedia analysis is central | multimedia-specific evidence, robust method, reliable reproduction |
| Knowledge-Based Systems | reasonable if adaptive decision + robust reasoning focus | novel method beyond combination, convincing benchmark |
| Expert Systems with Applications | possible if strong application/system validation | meaningful practical contribution and fair evaluation |
| Pattern Recognition | possible if mathematically strong adaptation/calibration contribution | rigorous methodological novelty and broad evaluation |

Before targeting, verify current JCR quartile vs SCImago/SJR/Scopus category and year requested by institution, indexing, scope, submission format, APC/options and ethical policy. This audit does not independently establish 2026 Q1 rank or acceptance probability of these journals. Never pay an agent promising guaranteed acceptance. Venue choice after G5/G6, not now.

## 14. Tài liệu tham khảo và liên kết audit

### 14.1 Repository references (commit-pinned)

- [Snapshot tree](https://github.com/buponbonbon/Tri-SACTA/tree/17f1acbb777ff7f970210af4d75cd13bcc710798)
- [Commit history](https://github.com/buponbonbon/Tri-SACTA/commits/main/)
- [README](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/README.md)
- [PROJECT_SPEC](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/PROJECT_SPEC.md)
- [Dataset loader](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/data/dataset.py), [shift generator](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/data/shift_generator.py)
- [Spectral detector](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/models/spectral_detector.py#L174-L252), [fusion attention](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/models/fusion_backbone.py#L165-L188), [fusion weighting](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/models/fusion_backbone.py#L408-L435)
- [TriSACTA adaptation](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/adaptation/tri_sacta.py), [baseline code](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/adaptation/baselines_tta.py)
- [Online ACI](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/conformal/online_aci.py), [static CP](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/conformal/static_cp.py)
- [Source training](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/scripts/01_train_source.py), [stream benchmark](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/scripts/02_run_stream_benchmark.py)
- [MOSEI config](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/configs/mosei_mvp.yaml), [MELD config](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/configs/meld.yaml)
- [Metrics](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/utils/metrics.py), [statistical helpers](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/src/utils/stats_test.py), [smoke test](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/tests/test_smoke.py)
- [Legacy source-only JSON](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/results/source_only_static_cp_mosei_seed42.json), [legacy Tri-SACTA JSON](https://github.com/buponbonbon/Tri-SACTA/blob/17f1acbb777ff7f970210af4d75cd13bcc710798/results/tri_sacta_online_aci_mosei_seed42.json)

Paths and line ranges obtained from repository audit; references pinned to known SHA for traceability rather than relying on moving `main`.

### 14.2 Research references

- **[R1]** Wang et al. *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR 2021. [Official code](https://github.com/DequanWang/tent), [bibliographic record](https://mlanthology.org/iclr/2021/wang2021iclr-tent/).
- **[R2]** Wang et al. *Continual Test-Time Domain Adaptation*. CVPR 2022. [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html).
- **[R3]** Niu et al. *Towards Stable Test-Time Adaptation in Dynamic Wild World*. ICLR 2023. [arXiv](https://arxiv.org/abs/2302.12400), [official SAR code](https://github.com/mr-eggplant/SAR).
- **[R4]** *Test-time Adaptation against Multi-modal Reliability Bias*. ICLR 2024. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f25d75fc760aec0a6174f9f5d9da59b8-Abstract-Conference.html), [READ official code](https://github.com/XLearning-SCU/2024-ICLR-READ).
- **[R5]** Romano, Sesia, Candès. *Classification with Valid and Adaptive Coverage*. NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html).
- **[R6]** Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS 2021. [Proceedings](https://papers.nips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
- **[R7]** Gibbs, Candès. *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*. JMLR 25, 2024. [Article](https://jmlr.org/beta/papers/v25/22-1218.html).
- **[R8]** Bastani et al. *Practical Adversarial Multivalid Conformal Prediction*. NeurIPS 2022. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bcdaaa1aec3ae2aa39542acefdec4e4b-Abstract-Conference.html).
- **[R9]** *Conformal prediction with conditional guarantees*. JRSS Series B, 87(4), 2025. [DOI](https://doi.org/10.1093/jrsssb/qkaf008).
- **[R10]** *Adapt-then-fuse: Instance feature adaptation for multi-modal test-time adaptation*. 2026. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0950705126011536). Confirm final metadata and full method before manuscript citation.
- **[R11]** *STA: Spatio-temporal alignment for multimodal test-time adaptation*. 2026. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0950705126012694). Confirm final metadata and full method before manuscript citation.
- **[R12]** *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*. arXiv preprint, September 2026. [arXiv:2609.07251](https://arxiv.org/abs/2609.07251). Do not present as peer-reviewed without verification.
- **[R13]** Alfarra et al. *Evaluation of Test-Time Adaptation Under Computational Time Constraints*. ICML 2024. [PMLR](https://proceedings.mlr.press/v235/alfarra24a.html).
- **[R14]** Bagher Zadeh et al. *Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph*. ACL 2018. [ACL Anthology](https://aclanthology.org/P18-1208/), [official dataset page](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
- **[R15]** Poria et al. *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*. ACL 2019. [ACL Anthology](https://aclanthology.org/P19-1050/), [official site](https://affective-meld.github.io/).

The bibliographic references and abstract-level relevance were checked with web sources; the document does not claim to have completed full-text mathematical comparison for all papers. For final submission retrieve original manuscripts and verify authors, DOI, page numbers, year and algorithm details. Do not copy this preliminary bibliography mechanically into paper.

## 15. Prompt thực thi và mẫu handover

### 15.1 Prompt giao trực tiếp cho Antigravity 3.8

Copy file Markdown này vào repository (gợi ý `docs/TRI-SACTA_Q1_AUDIT_AND_IMPLEMENTATION_ROADMAP.md`) và dùng prompt dưới đây. Đây là instruction từ người dùng dành cho coding agent; scope tùy quyền và tài nguyên local của agent.

```text
Bạn là Senior ML Research Engineer kiêm reviewer cho Tri-SACTA.
Đọc toàn bộ docs/TRI-SACTA_Q1_AUDIT_AND_IMPLEMENTATION_ROADMAP.md.
Mục tiêu: sửa research prototype thành artifact đúng, tái lập, đủ evidence
để đánh giá khả năng công bố Q1; không phải chỉ làm đẹp metrics.

1. Kiểm tra current git SHA/status, môi trường, dữ liệu/checkpoint/VRAM.
   Đối chiếu snapshot 17f1acb với code hiện tại. Không ghi đè thay đổi local.
2. Tạo/ cập nhật docs/AUDIT_STATUS.md với AUD-01 đến AUD-26 và P0/P1.
   Bảo tồn các JSON lịch sử; không chỉnh sửa số liệu cũ.
3. Ưu tiên WP0, WP1, WP2, WP3, WP4. Reproduce model collapse bằng
   minimal fixture/logs trước khi sửa nếu khả thi. Sửa stable spectral
   weights, attention masking, missing rows, selective parameter groups,
   auxiliary heads, seed, checkpoint, baseline interface, conformal
   quantiles/eta/disagreement/delayed feedback. Không mở rộng vô ích.
4. Implement các regression tests T01–T31 theo phần tương ứng. Chạy CPU
   pytest và tiny end-to-end stream. Ghi exact commands và output thật.
5. Real mode phải fail closed. Synthetic chỉ là debug. Không tự tải dữ liệu
   nhiều GB khi chưa có ngân sách/đồng ý; nếu thiếu, báo BLOCKED.
6. Dùng cùng backbone/checkpoint, manifest, seed, split, feedback protocol
   cho matched comparisons. Tuyệt đối không đưa nhãn test vào TTA hoặc
   dùng test để tune. Không tạo baseline giả dưới tên paper khác.
7. Sau G1 mới chuyển sang real data/source (G2), method sanity (G3),
   conformal (G4), multi-seed experiments (G5), novelty/theory (G6), paper (G7).
   Không chạy full matrix trước preflight. Không claim đảm bảo 90% nếu
   theorem không khớp implementation. Chấp nhận và báo cáo kết quả âm.
8. Sau mỗi work package ghi docs/HANDOVER.md: thay đổi, test, metrics thật,
   issues closed/open, blockers, artifacts, exact next commands, commit SHA.
   Chỉ đánh dấu PASS khi có evidence. Không tự push/submit nếu không được yêu cầu.

Hãy bắt đầu xử lý ngay các P0 có thể thực hiện trong môi trường hiện tại.
Nếu cần chọn thiết kế, ưu tiên phương án đơn giản, có test, giữ GPU 4GB
cho feature-level pipeline; ghi lý do và tradeoffs. Không dừng ở việc
lặp lại audit hoặc chỉ đề xuất kế hoạch: thực hiện các patch khả thi.
```

### 15.2 Mẫu `docs/HANDOVER.md`

```markdown
# Tri-SACTA handover — YYYY-MM-DD
## 1. Snapshot
- Base SHA / current SHA:
- Branch / git dirty status:
- Environment / GPU / RAM / disk:
- Data mode / dataset manifest / checkpoint hash:
## 2. Scope completed
- Work packages:
- Files modified:
- Decision and tradeoffs:
## 3. Issue status
| ID | Before evidence | Patch | Test | Status | Artifact/commit |
| --- | --- | --- | --- | --- | --- |
## 4. Actual commands and outputs
- Command:
- Exit code:
- Passed / failed / skipped:
- Log path:
## 5. Actual experiments
| Method | Dataset + mode | Seed | Protocol | Acc | Macro-F1 | Coverage | Set size | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
## 6. New risks / limitations / blockers
- What is not verified:
- Resources needed:
## 7. Gate status
- G0–G7: PASS/FAIL/BLOCKED/NOT RUN with evidence
## 8. Next actions
- Exact next command(s):
- Dependencies / expected artifacts:
```

### 15.3 Definition of research-ready vs Q1-ready

**Research-ready (engineering):** P0 tests pass, reproducible real data pipeline and source model, no collapse in valid sanity conditions, benchmark works under fair protocol, no false claims. **Q1-ready (submission candidate):** all scientific gates satisfied, verified meaningful contribution relative to closest prior art, strong fair experiments and uncertainty/efficiency analysis, correct claims and paper, human review. Q1 acceptance remains a peer-review decision, never guaranteed by any checklist or agent.

**Kết luận:** ở snapshot được audit, Tri-SACTA chưa có bằng chứng đủ để gửi Q1. Nên sửa lỗi trọng yếu và xác nhận dữ liệu thật, tiếp đến đánh giá tính mới và hiệu quả bằng thí nghiệm có kiểm soát. Không cần bỏ hướng nghiên cứu chỉ vì MVP thất bại, nhưng cũng không mặc định architecture V2 sẽ thắng. Kết quả thật, kể cả kết quả âm, phải quyết định bước tiếp theo.
