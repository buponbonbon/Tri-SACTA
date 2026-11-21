# Tri-SACTA: Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts

[![Target Venue: Q1 Journal](https://img.shields.io/badge/Target%20Venue-Q1%20Journal%20(Inf%20Fusion%20%2F%20IEEE%20TMM%20%2F%20Neural%20Networks)-blue.svg)](#)
[![Hardware: 4GB VRAM Compatible](https://img.shields.io/badge/Hardware-1x%204GB%20Laptop%20GPU%20(Peak%20VRAM%20%3C%2050MB)-success.svg)](#)
[![Tests: 24/24 PASS](https://img.shields.io/badge/Unit%20Tests-24%2F24%20PASS%20(100%25)-brightgreen.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![PyTorch: AMP FP16](https://img.shields.io/badge/PyTorch-AMP%20FP16-red.svg)](#)

---

## 1. Tổng quan Nghiên cứu (Research Overview)

Dự án **Tri-SACTA** giải quyết bài toán thích ứng thời gian kiểm thử liên tục (**Continual Test-Time Adaptation - CTTA**) cho 3–4 phương thái (Visual, Audio, Text/ASR, Prosody) dưới tác động của các độ lệch phân phối phức hợp (**Compound Modality Shifts**) và phương thái bị mất động (**Dynamic Missing Modality**):

1. **Modality-Wise Spectral Shift Detector (Bộ phát hiện lệch phổ đa phương thái):**
   Đo lường độ lệch phân phối $d_t^{(m)}$ dựa trên thống kê bậc 1 và bậc 2 (vector kỳ vọng và phổ giá trị riêng ma trận hiệp phương sai chuẩn hóa số chiều - Mahalanobis & Log-Eigenvalues) so với mỏ neo nguồn (source anchor statistics) và hiệu chuẩn null-score baseline.
2. **Anchor-Guided Selective Low-Rank Adaptation (Thích ứng chọn lọc dựa trên mỏ neo):**
   Phân hoạch động 3 tập $\mathcal{A}_t$ (Clean Anchors, $r_t \ge 0.7$), $\mathcal{U}_t$ (Adaptable, $0.3 \le r_t < 0.7$), và $\mathcal{D}_t$ (Corrupted/Missing, $r_t < 0.3$). Chỉ mở gradient cho các tham số thích ứng cục bộ (< 2% tổng tham số: LayerNorm + Low-Rank Projections) của các kênh trong $\mathcal{U}_t$, ngắt hoàn toàn gradient từ $\mathcal{D}_t$ để ngăn ngừa truyền giao tiêu cực (negative transfer) và quên thảm khốc (catastrophic forgetting).
3. **Mask-and-Disagreement-Driven Online Conformal Control (Online ACI):**
   Duy trì mức miscoverage thích ứng $\alpha_{t, m_t}$ theo mask phương thái quan sát được $m_t \in \{0,1\}^M \setminus \{\mathbf{0}\}$, với bước nhảy thích ứng $\eta_t \in [\eta_{\text{min}}, \eta_{\text{max}}]$ điều khiển bởi vận tốc lệch phổ $|\Delta d_t|$ và độ bất đồng đa phương thái $\kappa_t$, đảm bảo độ phủ tiệm cận dài hạn $1-\alpha=0.90$ trên chuỗi phi khả hoán (non-exchangeable streams) và bền vững dưới độ trễ nhãn phản hồi (**Delayed Feedback** $\delta \in \{10, 50\}$).

---

## 2. Cấu trúc Thư mục Dự án

```text
Tri-SACTA/
├── docs/
│   ├── AUDIT_STATUS.md             # Bảng theo dõi 26 issue audit và bằng chứng thực nghiệm
│   ├── HANDOVER.md                 # Biên bản bàn giao kỹ thuật chi tiết
│   ├── METHOD_SPEC_V2.md           # Đặc tả thuật toán và công thức toán học Tri-SACTA V2
│   ├── THEORY.md                   # Cơ sở lý thuyết conformal và các chứng minh định lý
│   └── EXPERIMENT_PROTOCOL.md      # Quy chuẩn giao thức thực nghiệm và benchmark regimes
├── configs/
│   ├── debug_synthetic.yaml        # Cấu hình kiểm thử và debug nhanh (Synthetic mode)
│   ├── mosei_real.yaml             # Cấu hình bộ dữ liệu thật CMU-MOSEI (Speaker-disjoint)
│   ├── meld_real.yaml              # Cấu hình bộ dữ liệu thật MELD (Dialogue-disjoint)
│   └── shift_regimes_v2.yaml       # Đặc tả 6 regimes dịch chuyển theo giao thức V2
├── src/
│   ├── data/
│   │   ├── dataset.py              # Loader phân chia speaker-disjoint, cờ --data-mode
│   │   ├── manifest.py             # Schema DataManifest và kiểm tra tính toàn vẹn
│   │   └── shift_generator.py      # Bộ sinh compound shift chuẩn hóa SNR và ASR drift
│   ├── models/
│   │   ├── fusion_backbone.py      # MaskedModalityTransformerLayer với multi-token key-mask
│   │   └── spectral_detector.py    # Bộ phát hiện phổ chuẩn hóa Mahalanobis/Eigenvalue
│   ├── adaptation/
│   │   ├── tri_sacta.py            # Thuật toán chính Anchor-Guided Selective Adaptation
│   │   └── baselines_tta.py        # Các baseline: TENT, SAR, READ/AV-CTTA
│   ├── conformal/
│   │   ├── online_aci.py           # Online ACI với PredictionRecord và rare-mask shrinkage
│   │   └── static_cp.py            # Static Split CP với finite-sample order statistic rank
│   └── utils/
│       ├── metrics.py              # Macro-F1 cố định class list, NLL FP32, VRAM footprint
│       └── stats_test.py           # Paired t-test, Wilcoxon, Cohen's d, LaTeX table
├── scripts/
│   ├── 00_validate_data.py         # Kiểm tra tính toàn vẹn và fail-closed của dữ liệu
│   ├── 01_train_source.py          # Huấn luyện Source Model có auxiliary heads supervision
│   ├── 02_run_stream_benchmark.py  # Chạy benchmark luồng kiểm thử qua 5-6 regimes
│   ├── 03_plot_rolling_curves.py   # Xuất đồ thị rolling coverage, F1 bars, delayed feedback
│   ├── 04_run_experiment_matrix.py # Điều phối chạy ma trận thử nghiệm đa hạt giống
│   ├── 05_aggregate_results.py     # Tính Mean ± Std và xuất bảng LaTeX cho bài báo Q1
│   └── extract_features_sequential.py # Pipeline trích xuất đặc trưng tuần tự (Stage 1)
├── tests/
│   ├── run_all_tests.py            # Test runner độc lập (24/24 tests PASS)
│   ├── test_spectral.py            # Kiểm thử Spectral Shift Detector (T01 - T04)
│   ├── test_masked_fusion.py       # Kiểm thử Masked Fusion & Mask Invariance (T05 - T07)
│   ├── test_selective_adaptation.py# Kiểm thử Selective Parameter Freezing (T08, T11)
│   ├── test_conformal.py           # Kiểm thử Online ACI & Order Statistic (T12, T13, T15)
│   ├── test_data_integrity.py      # Kiểm thử Fail-closed & Determinism (T19, T21, T23)
│   ├── test_metrics_rigor.py       # Kiểm thử Metrics & VRAM Footprint (T26, T29)
│   ├── test_shift_generator.py     # Kiểm thử SNR Noise & Structured Dropout (T25)
│   └── test_feedback_delay.py      # Kiểm thử Delayed Buffer FIFO & Rare Mask (T14, T18)
├── results/
│   ├── figures/                    # Biểu đồ chất lượng công bố quốc tế (300 DPI)
│   ├── legacy/                     # Lưu trữ nguyên vẹn dữ liệu thử nghiệm lịch sử
│   ├── summary/                    # Bảng tổng hợp thống kê đa seed Markdown và LaTeX
│   └── *.json                      # Kết quả benchmark chi tiết theo từng cấu hình
└── requirements.txt
```

---

## 3. Bảng Kết Quả Thực Nghiệm Đối Chứng (Benchmark Evidence)

Toàn bộ thực nghiệm được đánh giá trên chuỗi 5-Regime Compound Shift Stream (600 test utterances, mức tin cậy danh định $1 - \alpha = 90.0\%$):

| Phương pháp / Mô hình | Macro-F1 | Accuracy | Empirical Coverage | Coverage Gap $|\text{Cov} - 90\%|$ | Kích thước tập $|\mathcal{C}_t|$ | Peak VRAM | Độ trễ suy luận |
|---|---|---|---|---|---|---|---|
| **Source-Only + Static CP** | 0.4220 | 42.97% | 91.73% | 1.73% | 3.30 | **21.6 MB** | **0.68 ms/mẫu** |
| **TENT + Online ACI** | 0.4216 | 43.03% | 90.30% | 0.30% | 3.33 | 36.8 MB | 2.84 ms/mẫu |
| **SAR + Online ACI** | 0.4248 | 43.23% | 90.13% | 0.13% | 3.31 | 36.8 MB | 2.80 ms/mẫu |
| **READ / CTTA + Online ACI** | 0.3895 | 40.73% | 90.33% | 0.33% | 3.42 | 49.2 MB | 4.89 ms/mẫu |
| **Tri-SACTA + Online ACI (Ours, $\delta=0$)** | **0.4287** | **43.53%** | **90.00%** | **0.00%** | **3.27** | 31.8 MB | 5.54 ms/mẫu |
| **Tri-SACTA + Online ACI (Delay $\delta=10$)** | 0.4287 | 43.53% | 89.43% | 0.57% | 3.21 | 31.8 MB | 5.42 ms/mẫu |
| **Tri-SACTA + Online ACI (Delay $\delta=50$)** | 0.4287 | 43.53% | 88.30% | 1.70% | 3.29 | 31.8 MB | 5.81 ms/mẫu |

### So sánh theo từng Regime dịch chuyển:
- **Regime 2 (Audio SNR 5dB):** Tri-SACTA đạt **Macro-F1 = 0.4749** (+2.76% so với Source-Only 0.4473, +0.85% so với TENT 0.4664).
- **Regime 3 (ASR WER Drift 35%):** Tri-SACTA đạt **Macro-F1 = 0.4672** (+1.89% so với Source-Only 0.4483, +1.88% so với TENT 0.4484).
- **Regime 4 (Visual hỏng nặng + 30% thiếu):** READ bị suy sụp xuống 0.3006 do ép đồng thuận trên modality hỏng; Tri-SACTA cô lập kênh hỏng vào $\mathcal{D}_t$ và đạt **Macro-F1 = 0.3439** (+4.33% so với READ).
- **Mức bao phủ thống kê:** Static CP thất bại ở Regime 4 (88.67%) và Regime 5 (87.83%) do phá vỡ exchangeability; Online ACI duy trì chuẩn xác mức bao phủ danh định **90.00%**.

---

## 4. Hướng dẫn Vận Hành & Tái Lập (Quickstart Guide)

### 4.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 4.2. Chạy bộ kiểm thử toàn diện (24 Unit Tests)
```bash
python tests/run_all_tests.py
```
*Kết quả:* Kiểm chứng 100% các tính chất toán học: Mask Invariance ($0.0000$ sai khác), Selective Freezing ($\Delta\theta = 0$), Finite-Sample Quantiles, SNR Noise Power, và Bounded Step Sizes.

### 4.3. Xác thực dữ liệu
```bash
# Chế độ synthetic debug (chạy tức thì):
python scripts/00_validate_data.py --config configs/debug_synthetic.yaml --data-mode synthetic

# Chế độ real dataset (fail-closed bảo vệ tính toàn vẹn):
python scripts/00_validate_data.py --config configs/mosei_real.yaml --data-mode real
```

### 4.4. Huấn luyện Source Model
```bash
python scripts/01_train_source.py --config configs/debug_synthetic.yaml --epochs 10 --seed 42 --data-mode synthetic
```
*Đầu ra:* Mô hình nguồn tốt nhất `checkpoints/source_model_synthetic_seed42.pt` (Val Macro-F1 = 50.01%) và phổ mỏ neo `checkpoints/source_spectral_stats_synthetic_seed42.pt`.

### 4.5. Chạy Streaming Continual Compound Shift Benchmark
```bash
# Tri-SACTA + Online ACI (Đề xuất):
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --seed 42 --data-mode synthetic

# Kiểm chứng Delayed Feedback (độ trễ nhãn delta = 10, 50):
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --delay 10 --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --delay 50 --seed 42 --data-mode synthetic

# Các baseline đối chứng:
python scripts/02_run_stream_benchmark.py --method source_only --conformal static_cp --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method tent --conformal online_aci --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method sar --conformal online_aci --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method read --conformal online_aci --seed 42 --data-mode synthetic
```

### 4.6. Xuất đồ thị và bảng xuất bản Q1
```bash
# Vẽ 3 đồ thị xuất bản tại results/figures/:
python scripts/03_plot_rolling_curves.py

# Tổng hợp bảng thống kê và bảng LaTeX tại results/summary/:
python scripts/05_aggregate_results.py --input_dir results --output_dir results/summary --data-mode synthetic
```

---

## 5. Quy chuẩn Phần cứng (Laptop GPU 4GB VRAM)

| Chỉ số kỹ thuật | Yêu cầu mục tiêu | Đo lường thực tế (RTX 2050 4GB) |
|---|---|---|
| **Tổng số tham số mô hình** | $< 2.0\text{ M}$ | **1,369,379** (1.37 M) |
| **Số tham số thích ứng TTA** | $< 2.0\%$ | **1.83%** (LayerNorm + Low-Rank Projections) |
| **Peak GPU VRAM (Allocated)** | $< 1.5\text{ GB}$ | **31.8 MB** (< 1.0% dung lượng 4GB) |
| **Peak GPU VRAM (Reserved)** | $< 1.5\text{ GB}$ | **48.0 MB** (< 1.2% dung lượng 4GB) |
| **Độ trễ suy luận + thích ứng** | $< 50\text{ ms/mẫu}$ | **5.54 ms/mẫu** (> 180 FPS, thời gian thực) |

---

## 6. Trích dẫn & Liên kết Tài liệu Nghiên cứu

- [METHOD_SPEC_V2.md](docs/METHOD_SPEC_V2.md): Đặc tả thuật toán và công thức toán học Tri-SACTA V2.
- [THEORY.md](docs/THEORY.md): Cơ sở lý thuyết conformal prediction và chứng minh định lý hội tụ.
- [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): Đặc tả giao thức thực nghiệm và quy chuẩn 6-regime.
- [AUDIT_STATUS.md](docs/AUDIT_STATUS.md): Bảng theo dõi 26 issue audit và bằng chứng thực nghiệm.
- [HANDOVER.md](docs/HANDOVER.md): Biên bản bàn giao kỹ thuật chi tiết.
