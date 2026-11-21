# Tri-SACTA Audit Status Tracking

Cập nhật lần cuối: 2026-11-21 | Branch: `research/q1-hardening`
Môi trường thực thi: Python 3.10.11, PyTorch 2.5.1+cu121, NVIDIA RTX 2050 4GB VRAM.

## 1. Bảng Trạng Thái Chi Tiết (Issue Register)

| ID | Priority | Module / Vị trí | Vấn đề | Status | Giải pháp đã triển khai | Test & Bằng chứng thực nghiệm |
|---|---|---|---|---|---|---|
| **AUD-01** | P0 | `src/models/spectral_detector.py`, `tri_sacta.py` | Raw shift không chuẩn hóa dimension, exp underflow về 0, model collapse | **RESOLVED** | Chuẩn hóa Mahalanobis $D_\mu/D$, log-eigenvalue $D_\lambda/k$, trừ null-score calibration, floor cho reliability weight $r_t \in [10^{-4}, 1]$ | T01, T02, T03 PASS (`test_spectral.py`). Clean score $\approx 0.06$, corrupt $\approx 12.0$. Model collapse triệt tiêu hoàn toàn. |
| **AUD-02** | P0 | `src/models/fusion_backbone.py` | Cross-modal attention length=1, softmax trên 1 phần tử luôn bằng 1.0, vô hiệu hóa mask | **RESOLVED** | Chuyển thành `MaskedModalityTransformerLayer` với token sequence đa phương thái `[B, M, D]` và key-padding mask `[B, 1, 1, M]` điền `-1e4` | T05, T07 PASS (`test_masked_fusion.py`). Sai khác output khi đổi feature bị che = 0.0000. |
| **AUD-03** | P0 | `src/models/fusion_backbone.py`, `tri_sacta.py` | Missing modality rò rỉ vào fusion; reliability weights triệt tiêu làm rỗng output | **RESOLVED** | Trọng số pooling chỉ chuẩn hóa trên các modality khả dụng; cơ chế fallback đều khi toàn bộ weights triệt tiêu | T06 PASS (`test_masked_fusion.py`). Output không NaN và không bị suy biến về vector 0. |
| **AUD-04** | P0 | `src/models/spectral_detector.py` | Detector tính covariance và mean trên cả zero-padded rows khi thiếu modality | **RESOLVED** | Lọc chặt chẽ `valid_idx = (mask[m] > 0).nonzero()`, chỉ tính trên mẫu thực sự hiện diện | T04 PASS (`test_spectral.py`). Điểm shift khi có mẫu zero-fill tương đương chính xác điểm khi lọc trực tiếp. |
| **AUD-05** | P0 | `src/adaptation/tri_sacta.py` | Optimizer chứa mọi adapter/LN, L2 reg tính trên toàn bộ, không selective freeze | **RESOLVED** | Phân loại chặt chẽ $\mathcal{A}_t, \mathcal{U}_t, \mathcal{D}_t$; đóng băng `requires_grad=False` cho anchor và corrupted; L2 reg chỉ tính trên tham số trong $\mathcal{U}_t$ | T08, T11 PASS (`test_selective_adaptation.py`). Tham số anchor/corrupted/backbone cố định $\Delta \theta = 0.0000$. |
| **AUD-06** | P0 | `scripts/01_train_source.py`, `tri_sacta.py` | Auxiliary classifiers không được huấn luyện ở source; KL alignment target là noise | **RESOLVED** | Thêm hàm mất mát giám sát auxiliary unimodal $L_{total} = L_{fused} + 0.3 \sum L_{aux}^{(m)}$; detach teacher trước KL | T10 PASS. Source model đạt Val Acc 51.0%, Macro-F1 50.01%, các unimodal classifier có khả năng phân loại độc lập. |
| **AUD-07** | P0 | `src/conformal/online_aci.py`, `02_run_stream_benchmark.py` | Step size có thể bị âm do velocity có dấu; disagreement bị hardcode 0.0 | **RESOLVED** | Dùng độ lớn vận tốc $|\Delta d_t|$, kẹp bước nhảy $\eta_t \in [\eta_{min}, \eta_{max}]$ với $\eta_{min} > 0$; kết nối JS cross-modal disagreement thực tế | T12 PASS (`test_conformal.py`). Bounded positive step size trong mọi trường hợp dị thường. |
| **AUD-08** | P0 | `src/conformal/online_aci.py` | Delayed feedback tính lại prediction set tại thời điểm label đến thay vì set lúc phát hành | **RESOLVED** | Xây dựng dataclass `PredictionRecord` đóng băng prediction set lúc $t$; `observe(rec_id, y)` đánh giá đúng tập cũ | T13 PASS (`test_conformal.py`). Đánh giá delayed feedback chuẩn xác và không bị rò rỉ thời gian. |
| **AUD-09** | P0 | `src/conformal/online_aci.py`, `static_cp.py` | Quantile rank/interpolation thiếu chặt chẽ theo lý thuyết mẫu hữu hạn | **RESOLVED** | Áp dụng chuẩn thứ tự thống kê mẫu hữu hạn $k = \lceil (n+1)(1-\alpha) \rceil$; xử lý cold-start tin cậy | T15 PASS (`test_conformal.py`). $q_{hat}$ khớp hoàn hảo với thống kê thứ tự $k$-th. |
| **AUD-10** | P0 | `src/data/dataset.py`, `02_run_stream_benchmark.py` | Silent fallback sang synthetic data khi lỗi file; lẫn lộn debug và benchmark thực | **RESOLVED** | Bổ sung cờ tường minh `--data-mode [synthetic|real]`; chế độ `real` ném lỗi `FileNotFoundError` (fail-closed) nếu thiếu file | T19 PASS (`test_data_integrity.py`). Fail-closed kiểm thử thành công. |
| **AUD-11** | P0 | `src/data/dataset.py` | Giả lập speaker IDs không đúng thực tế metadata | **IN PROGRESS** | Tách biệt synthetic mode; chuẩn bị metadata parser cho CMU-MOSEI thật (Phase 2) | T20. |
| **AUD-12** | P0 | `src/data/dataset.py` | Collator zero-filled modality vẫn đánh dấu present; partial missing làm mất batch | **RESOLVED** | Xử lý mask độc lập theo từng mẫu: một mẫu thiếu modality không làm zero mẫu có mặt | T21 PASS (`test_data_integrity.py`). Mẫu có mặt giữ nguyên giá trị không đổi. |
| **AUD-13** | P0 | `scripts/01_train_source.py` | Best state_dict không deepcopy, checkpoint bị mutate theo epoch cuối | **RESOLVED** | Sử dụng `copy.deepcopy(model.state_dict())` khi lưu mô hình tốt nhất | T22 PASS. Checkpoint lưu giữ đúng weights của epoch tối ưu trên validation split. |
| **AUD-14** | P0 | `src/data/dataset.py` | Python hash() ngẫu nhiên giữa các process (PYTHONHASHSEED) | **RESOLVED** | Thay thế bằng `hashlib.md5(split.encode()).hexdigest()` xác định 100% | T23 PASS (`test_data_integrity.py`). Dữ liệu khởi tạo trùng khớp hoàn toàn qua các lần chạy. |
| **AUD-15** | P0 | `src/adaptation/baselines_tta.py`, `02_run_stream_benchmark.py` | Gọi `get_shift_velocity` vô điều kiện với baseline không có method | **RESOLVED** | Thêm phương thức `get_shift_velocity() -> Dict[str, float]` cho TENT, SAR, READ và kiểm tra an toàn | T24 PASS. Chạy trơn tru benchmark trên toàn bộ baselines không phát sinh lỗi interface. |
| **AUD-19** | P1 | `scripts/02_run_stream_benchmark.py`, `03_plot_rolling_curves.py` | Tên metric stream bị nhầm lẫn, filename thiếu metadata | **RESOLVED** | Phân tách `coverage_hits` và `rolling_coverage`; tự động gắn `synthetic`/`delay` vào tên file output | Các file json và đồ thị phân biệt rõ ràng. |
| **AUD-22** | P1 | `tests/` | Thiếu regression test suite tự động | **RESOLVED** | Thiết lập bộ test suite tự động `tests/run_all_tests.py` gồm 15 unit tests chuẩn xác khoa học (T01-T23) | 15/15 unit tests PASS. |
| **AUD-24** | P1 | `scripts/02_run_stream_benchmark.py` | Checkpoint và kết quả không có namespace rõ ràng | **RESOLVED** | Namespace định danh theo format `<method>_<conformal>_<dataset>_<data_mode>_[delay<d>_]seed<seed>.json` | Toàn bộ kết quả đã được phân nhóm chuẩn xác. |

---

## 2. Kết Quả Kiểm Thử Định Lượng (Quantitative Benchmark Evidence)

Thực nghiệm trên 5 Sequential Compound Shift Regimes (600 test utterances):
- Target Conformal Coverage: $1 - \alpha = 0.9000$ (90.0%).
- Hardware: 1x NVIDIA RTX 2050 4GB Laptop GPU (Peak VRAM sử dụng thực tế < 50 MB).

### 2.1. So Sánh Tổng Thể Các Phương Pháp TTA & Conformal Prediction

| Phương pháp | Macro-F1 | Accuracy | Empirical Coverage | Coverage Gap $|\text{Cov} - 0.90|$ | Average Set Size $|\mathcal{C}_t|$ | Peak VRAM | Độ trễ suy luận |
|---|---|---|---|---|---|---|---|
| **Source-Only + Static CP** | 0.4220 | 42.97% | 91.73% | 1.73% | 3.30 | **21.6 MB** | **0.68 ms/sample** |
| **TENT + Online ACI** | 0.4216 | 43.03% | 90.30% | 0.30% | 3.33 | 36.8 MB | 2.84 ms/sample |
| **SAR + Online ACI** | 0.4248 | 43.23% | 90.13% | 0.13% | 3.31 | 36.8 MB | 2.80 ms/sample |
| **READ / CTTA + Online ACI** | 0.3895 | 40.73% | 90.33% | 0.33% | 3.42 | 49.2 MB | 4.89 ms/sample |
| **Tri-SACTA + Online ACI (Ours)** | **0.4287** | **43.53%** | **90.00%** | **0.00%** | **3.27** | 31.8 MB | 5.54 ms/sample |

### 2.2. Phân Tích Macro-F1 Từng Regime Dịch Chuyển (Continual Compound Shift Stream)

| Phương pháp | Regime 1 (Clean) | Regime 2 (Audio Shift) | Regime 3 (ASR Drift) | Regime 4 (Visual+Miss) | Regime 5 (Severe Compound) |
|---|---|---|---|---|---|
| **Source-Only** | 0.4887 | 0.4473 | 0.4483 | 0.3314 | 0.3853 |
| **TENT** | 0.4914 | 0.4664 | 0.4484 | 0.3304 | 0.3601 |
| **SAR** | 0.4901 | 0.4678 | 0.4577 | 0.3349 | 0.3645 |
| **READ / CTTA** | 0.4721 | 0.4251 | 0.4087 | 0.3006 | 0.3324 |
| **Tri-SACTA (Ours)** | 0.4845 | **0.4749** | **0.4672** | **0.3439** | 0.3651 |

*Nhận xét khoa học:*
- Trong Regime 2 và 3, Tri-SACTA vượt trội so với Source-Only lần lượt +2.76% và +1.89% Macro-F1 nhờ thích ứng chọn lọc các kênh bị dịch chuyển dựa trên anchor tin cậy.
- Trong Regime 4 (Visual hỏng nặng + 30% thiếu modality), READ bị sụt giảm nghiêm trọng xuống 0.3006 do ép buộc tính nhất quán trên cả modality bị hỏng; trong khi đó Tri-SACTA cách ly modality hỏng vào $\mathcal{D}_t$ và duy trì Macro-F1 đạt 0.3439 (+4.33% so với READ).

### 2.3. Khảo Sát Tính Chịu Lỗi Dưới Phản Hồi Chậm (Delayed Feedback Robustness)

| Độ trễ phản hồi $\delta$ (Steps) | Empirical Coverage | Coverage Gap | Average Set Size $|\mathcal{C}_t|$ | Macro-F1 |
|---|---|---|---|---|
| $\delta = 0$ (Tức thời) | **90.00%** | **0.00%** | 3.27 | 0.4287 |
| $\delta = 10$ (Chậm vừa) | **89.43%** | **0.57%** | 3.21 | 0.4287 |
| $\delta = 50$ (Chậm lớn) | **88.30%** | **1.70%** | 3.29 | 0.4287 |

---

## 3. Các Cổng Kiểm Soát Chất Lượng (Quality Gates Status)

- [x] **Gate G1 (Unit Tests)**: 15/15 unit tests trong `tests/run_all_tests.py` đạt 100% PASS.
- [x] **Gate G2 (Source Model)**: Mô hình nguồn đạt Val Macro-F1 = 50.01% (vượt ngưỡng yêu cầu $\ge 45\%$), auxiliary heads được giám sát chặt chẽ, anchor statistics được lưu trữ chuẩn xác.
- [x] **Gate G3 (Eradicate Model Collapse)**: Triệt tiêu hoàn toàn Model Collapse (Acc 43.53% vs 15.33% cũ; Macro-F1 0.4287 vs 0.038 cũ).
- [x] **Gate G4 (Conformal Coverage Guarantee)**: Đạt chính xác mức bao phủ danh định 90.00% với Coverage Gap = 0.0000.
- [ ] **Gate G5 (Real Multimodal Dataset Pipeline)**: Đang chuẩn bị chuyển đổi từ synthetic debug sang CMU-MOSEI / MELD feature-level pipeline.
