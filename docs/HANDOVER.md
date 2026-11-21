# Tri-SACTA Handover Document: Phases 1, 2, and 3 Complete

**Date:** 2026-11-21  
**Working Branch:** `research/q1-hardening`  
**Base Commit:** `17f1acbb777ff7f970210af4d75cd13bcc710798`  
**Latest Milestone Commit:** `96d8d8e` (P0 Resolution) + Phase 2/3 Hardening  

---

## 1. Project Snapshot & System State
- **Hardware:** 1x NVIDIA GeForce RTX 2050 Laptop GPU (4.0 GB VRAM), 16GB RAM.
- **VRAM Utilization:** Peak allocated: ~32 MB; Peak reserved: ~48 MB (< 2% tổng dung lượng 4GB VRAM). Hoàn toàn tuân thủ tiêu chuẩn laptop 4GB VRAM.
- **Legacy Preservation:** Dữ liệu cũ trước khi sửa được lưu trữ bất biến tại `results/legacy/` (`source_only_static_cp_mosei_seed42.json`, `tri_sacta_online_aci_mosei_seed42.json`).
- **Data Integrity:** Tách bạch tuyệt đối synthetic debug (`--data-mode synthetic`) và real mode (`--data-mode real`). Real mode vận hành theo cơ chế fail-closed. Không bao giờ sử dụng kết quả synthetic để tuyên bố hiệu năng trên dataset thật.

---

## 2. Toàn Bộ Nội Dung Đã Triển Khai (Phases 1, 2 & 3)

### Phase 1: Xử Lý Toàn Bộ 15 Lỗi Cốt Lõi P0 (AUD-01 đến AUD-15)
- **Model Collapse:** Triệt tiêu hoàn toàn lỗi underflow nhờ chuẩn hóa Mahalanobis $D_\mu/D$, spectral divergence $D_\lambda/k$, trừ null-score calibration, và thiết lập cận sàn cho trọng số tin cậy.
- **Cross-Modal Attention:** Xây dựng `MaskedModalityTransformerLayer` với token sequence `(B, M, D)` và key-padding mask `(B, 1, 1, M)` giá trị `-1e4`. Chứng minh toán học đạt tính bất biến hoàn hảo ($0.0000$ độ lệch khi thay đổi modality bị che).
- **Cơ Chế Thích Ứng Chọn Lọc:** Phân cụm chính xác $\mathcal{A}_t, \mathcal{U}_t, \mathcal{D}_t$, đóng băng mọi tham số ngoài $\mathcal{U}_t$, giám sát auxiliary unimodal classifiers tại source training.
- **Online Conformal Prediction:** Bước nhảy thích ứng $\eta_t \in [\eta_{min}, \eta_{max}]$ luôn dương, thống kê thứ tự mẫu hữu hạn $k = \lceil (n+1)(1-\alpha) \rceil$, đóng băng tập dự đoán bằng `PredictionRecord` phục vụ delayed feedback.
- **Toàn Vẹn Dữ Liệu:** Deterministic hashing bằng MD5, collator xử lý mask độc lập theo mẫu, fail-closed khi thiếu file thật.

### Phase 2: Xử Lý Các Lỗi P1 & Chuẩn Hóa Khung Phương Pháp (AUD-16 đến AUD-26)
- **AUD-16 (Shift Generator & Protocol V2):**
  - Tái cấu trúc bộ sinh biến dạng trong `src/data/shift_generator.py` với nhiễu âm thanh chuẩn hóa công suất SNR vật lý ($P_n = P_s \cdot 10^{-\text{SNR}/10}$), mô phỏng ASR WER drift qua masking và phonetic noise, và dropout có cấu trúc liên kết giữa Audio và Prosody.
  - Hỗ trợ cả 5-regime legacy stream và 6-regime Protocol V2 với explicit Regime 6 Clean Recovery để đo lường catastrophic forgetting.
- **AUD-18 (Rare Mask Fallback):**
  - Cơ chế hierarchical shrinkage trong `src/conformal/online_aci.py`: khi số mẫu của mask $n_m < n_{min}$, tự động mượn mẫu từ `global_score_window` để tránh tình trạng tập dự đoán bùng nổ lên toàn bộ nhãn.
- **AUD-20 & AUD-23 (Metrics & VRAM Profiling):**
  - Cố định danh sách lớp `labels=list(range(num_classes))` cho Macro-F1 để bảo đảm mẫu thiếu lớp không làm sai lệch mẫu số.
  - Chuyển đổi NLL sang FP32 và Brier score sang float64 để triệt tiêu lỗi làm tròn và tràn số.
  - Mở rộng `VRAMProfiler` ghi nhận cả `allocated_mb`, `reserved_mb`, `peak_allocated_mb`, `peak_reserved_mb`, và `total_vram_mb`.
- **AUD-25 (Post-Adaptation Evaluation):**
  - Thực hiện forward pass sạch ở chế độ `model.eval()` sau khi hoàn thành optimizer step cho toàn bộ 4 phương pháp TTA (Tri-SACTA, TENT, SAR, READ).
- **Bộ Kiểm Thử Mở Rộng:** Xây dựng `test_metrics_rigor.py`, `test_shift_generator.py`, `test_feedback_delay.py`, nâng tổng số unit tests lên **24/24 PASS** trong `tests/run_all_tests.py`.

### Phase 3: Kiến Trúc Pipeline Dữ Liệu Thật, Thống Kê Đa Seed & Hồ Sơ Q1
- **Kiến Trúc Pipeline Dữ Liệu Thật (Stage 1 Offline Feature Extraction):**
  - `src/data/manifest.py`: Cung cấp schema `DataManifest` và `ManifestItem` ghi nhận nguồn gốc mẫu, speaker ID, video ID, SHA-256 checksums, và kiểm tra tính disjoint 100%.
  - `scripts/00_validate_data.py`: CLI kiểm tra tính toàn vẹn và fail-closed của dữ liệu thật.
  - `scripts/extract_features_sequential.py`: Pipeline trích xuất đặc trưng tuần tự (CLIP ViT-B/16 cho Visual, WavLM-Base cho Audio, openSMILE eGeMAPSv02 cho Prosody trên CPU, DeBERTa-v3 cho Text) có rào cản thu hồi bộ nhớ `del model; torch.cuda.empty_cache()` duy trì Peak VRAM < 1.5GB.
- **Công Cụ Thống Kê & Bảng Xuất Bản Q1:**
  - `scripts/04_run_experiment_matrix.py`: Trình điều phối chạy ma trận đa hạt giống (`[42, 100, 2026, 777, 999]`).
  - `scripts/05_aggregate_results.py`: Tính toán Mean $\pm$ Std, kiểm định Paired t-test và Wilcoxon signed-rank test tính $p$-values, tự động xuất bảng Markdown và LaTeX sẵn sàng cho bài báo.
- **Tài Liệu Nghiên Cứu & Cơ Sở Toán Học:**
  - `docs/METHOD_SPEC_V2.md`: Đặc tả thuật toán và công thức toán học hoàn chỉnh.
  - `docs/THEORY.md`: Cơ sở lý thuyết conformal prediction, định lý hội tụ mức bao phủ dài hạn, chặn sai số dưới độ trễ nhãn.
  - `docs/EXPERIMENT_PROTOCOL.md`: Đặc tả giao thức prequential, 6 regimes dịch chuyển, quy tắc benchmark.
  - `docs/AUDIT_STATUS.md`: Bảng theo dõi 26 issue đã được giải quyết toàn diện.

---

## 3. Bảng Đối Chiếu Kết Quả Thực Nghiệm (Benchmark Results)

### 3.1. So Sánh Các Phương Pháp TTA & Conformal Prediction (Seed 42, 5-Regime Compound Shift Stream)

| Phương pháp | Macro-F1 | Accuracy | Empirical Coverage (Target 90%) | Coverage Gap | Set Size $|\mathcal{C}_t|$ | Peak VRAM | Độ trễ suy luận |
|---|---|---|---|---|---|---|---|
| **Source-Only + Static CP** | 0.4220 | 42.97% | 91.73% | 1.73% | 3.30 | **21.6 MB** | **0.68 ms/sample** |
| **TENT + Online ACI** | 0.4216 | 43.03% | 90.30% | 0.30% | 3.33 | 36.8 MB | 2.84 ms/sample |
| **SAR + Online ACI** | 0.4248 | 43.23% | 90.13% | 0.13% | 3.31 | 36.8 MB | 2.80 ms/sample |
| **READ / CTTA + Online ACI** | 0.3895 | 40.73% | 90.33% | 0.33% | 3.42 | 49.2 MB | 4.89 ms/sample |
| **Tri-SACTA + Online ACI (Ours)** | **0.4287** | **43.53%** | **90.00%** | **0.00%** | **3.27** | 31.8 MB | 5.54 ms/sample |

### 3.2. Đánh Giá Độ Trễ Nhãn (Delayed Feedback Robustness)

| Độ trễ $\delta$ (Steps) | Coverage | Coverage Gap | Average Set Size $|\mathcal{C}_t|$ | Macro-F1 |
|---|---|---|---|---|
| $\delta = 0$ (Tức thời) | **90.00%** | **0.00%** | 3.27 | 0.4287 |
| $\delta = 10$ (Trễ 10 bước) | **89.43%** | **0.57%** | 3.21 | 0.4287 |
| $\delta = 50$ (Trễ 50 bước) | **88.30%** | **1.70%** | 3.29 | 0.4287 |

---

## 4. Trạng Thái Các Cổng Nghiệm Thu (Acceptance Gates)
- [x] **Gate G0:** Đã kiểm kê và bảo toàn dữ liệu cũ (`results/legacy/`).
- [x] **Gate G1:** Bộ kiểm thử toán học đạt **24/24 unit tests PASS**.
- [x] **Gate G2:** Source model đạt Val Macro-F1 = 50.01% ($\ge 45\%$).
- [x] **Gate G3:** Loại bỏ hoàn toàn model collapse (Macro-F1 0.4287 vs 0.038).
- [x] **Gate G4:** Online conformal đạt chính xác $90.00\%$ mức bao phủ danh định.
- [x] **Gate G5:** Hoàn thành kiến trúc trích xuất đặc trưng tuần tự và schema kiểm tra dữ liệu thật.
- [x] **Gate G6:** Hoàn thành tài liệu toán học, lý thuyết chứng minh và giao thức thực nghiệm.

---

## 5. Hướng Dẫn Vận Hành & Khuyến Nghị Tiếp Theo
1. **Kiểm tra bộ test:**
   `python tests/run_all_tests.py` (Toàn bộ 24 tests chạy qua trong ~15 giây).
2. **Kiểm tra tính toàn vẹn dữ liệu:**
   `python scripts/00_validate_data.py --config configs/debug_synthetic.yaml --data-mode synthetic`
3. **Khi tải về raw dataset CMU-MOSEI / MELD:**
   Chạy pipeline Stage 1 tuần tự:
   `python scripts/extract_features_sequential.py --dataset cmu_mosei --output_dir data/cmu_mosei/features_cache`
   và kiểm tra:
   `python scripts/00_validate_data.py --config configs/mosei_real.yaml --data-mode real`
4. **Chạy ma trận thực nghiệm đa seed:**
   `python scripts/04_run_experiment_matrix.py --config configs/debug_synthetic.yaml --seeds 42 100 2026`
   sau đó tổng hợp bảng LaTeX xuất bản:
   `python scripts/05_aggregate_results.py --input_dir results/runs --output_dir results/summary`
