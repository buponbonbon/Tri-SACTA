# Tri-SACTA: Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts

[![Target Venue: Q1 Journal](https://img.shields.io/badge/Target%20Venue-Q1%20Journal%20(Inf%20Fusion%20%2F%20IEEE%20TMM)-blue.svg)](#)
[![Hardware: 4GB VRAM Compatible](https://img.shields.io/badge/Hardware-1x%204GB%20Laptop%20GPU%20(Peak%20VRAM%20%3C%2050MB)-success.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![PyTorch: AMP FP16](https://img.shields.io/badge/PyTorch-AMP%20FP16-red.svg)](#)

---

## 1. Tổng quan Nghiên cứu
Dự án **Tri-SACTA** giải quyết bài toán thích ứng thời gian kiểm thử liên tục (**Continual Test-Time Adaptation - CTTA**) cho 3–4 phương thái (Thị giác, Âm thanh, Văn bản/ASR, Prosody) dưới tác động của các độ lệch phân phối phức hợp (**Compound Modality Shifts**) và phương thái bị mất động (**Dynamic Missing Modality**):

1. **Modality-Wise Spectral Shift Detector:** Đo lường độ lệch phân phối $d_t^{(m)}$ dựa trên thống kê bậc 1 và bậc 2 (vector kỳ vọng và phổ giá trị riêng ma trận hiệp phương sai - Top-$k$ PCA eigenvalues) so với mỏ neo nguồn (source anchor statistics) mà không cần lưu trữ dữ liệu nguồn thô.
2. **Anchor-Guided Selective Low-Rank Adaptation:** Chỉ cập nhật các tham số thích ứng cục bộ (< 2% tổng tham số: LayerNorm + Low-Rank Adapters) cho các phương thái lệch vừa phải, lấy phương thái sạch làm mỏ neo độ tin cậy ($r_t^{(m)}$) và ngắt gradient từ các phương thái nhiễu nặng hoặc bị khuyết thiếu để ngăn ngừa hiện tượng quên thảm họa (catastrophic forgetting).
3. **Mask-and-Disagreement-Driven Online Conformal Control (Online ACI):** Duy trì mức miscoverage thích ứng $\alpha_{t, m_t}$ theo mask phương thái quan sát được $m_t \in \{0,1\}^M \setminus \{\mathbf{0}\}$, với bước nhảy $\eta_t$ thích ứng theo vận tốc lệch phổ $\Delta d_t$ và độ bất đồng đa phương thái $\kappa_t$, đảm bảo độ phủ tiệm cận dài hạn $1-\alpha=0.90$ dưới chuỗi phi khả hoán (non-exchangeable stream).

---

## 2. Cấu trúc Repository
```text
Tri-SACTA/
├── PROJECT_SPEC.md                 # Đặc tả toán học & yêu cầu kỹ thuật chi tiết
├── requirements.txt                # Danh sách thư viện Python
├── configs/
│   ├── mosei_mvp.yaml              # Cấu hình CMU-MOSEI MVP
│   ├── meld.yaml                   # Cấu hình MELD emotion
│   └── shift_regimes.yaml          # Định nghĩa 5 chuỗi phân phối dịch chuyển
├── src/
│   ├── data/
│   │   ├── dataset.py              # Loader phân chia speaker-disjoint / video-disjoint
│   │   └── shift_generator.py      # Bộ sinh 5 chế độ compound shift liên tục
│   ├── models/
│   │   ├── fusion_backbone.py      # Mô hình Lightweight Multimodal Fusion (<2M tham số)
│   │   └── spectral_detector.py    # Bộ phát hiện phổ & tính toán độ tin cậy r_t^(m)
│   ├── adaptation/
│   │   ├── tri_sacta.py            # Thuật toán chính Tri-SACTA Selective Adaptation
│   │   └── baselines_tta.py        # Các baseline: TENT, SAR, READ/AV-CTTA
│   ├── conformal/
│   │   ├── online_aci.py           # Online ACI thích ứng theo mask và disagreement
│   │   └── static_cp.py            # Static Split CP & Mask-Conditional CP (MCCF)
│   └── utils/
│       ├── metrics.py              # Macro-F1, ECE, Coverage, Set Size, VRAM/Latency Profiler
│       └── stats_test.py           # Paired t-test, Wilcoxon, Cohen's d, LaTeX table
├── scripts/
│   ├── 01_train_source.py          # Huấn luyện Source Model & Trích xuất Anchor Stats
│   ├── 02_run_stream_benchmark.py  # Chạy benchmark luồng kiểm thử qua 5 Regimes
│   └── 03_plot_rolling_curves.py   # Vẽ biểu đồ rolling coverage & so sánh các phương pháp
├── tests/
│   └── test_smoke.py               # Bộ smoke test kiểm tra toàn bộ pipeline
└── results/
    ├── figures/                    # Biểu đồ chất lượng bài báo (DPI 300)
    └── *.json                      # Dữ liệu kết quả benchmark
```

---

## 3. Hướng dẫn Chạy (Quickstart)

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 1: Chạy Smoke Test kiểm tra hệ thống
```bash
python tests/test_smoke.py
```
*Kết quả:* Xác nhận tham số mô hình đạt 1.37M (< 2M), forward/backward AMP FP16 hoạt động chính xác.

### Bước 2: Huấn luyện Source Model & Trích xuất Anchor Statistics
```bash
python scripts/01_train_source.py --config configs/mosei_mvp.yaml --epochs 10 --seed 42
```
Checkpoint sẽ được lưu tại `checkpoints/source_model_seed42.pt` và phổ mỏ neo tại `checkpoints/source_spectral_stats_seed42.pt`.

### Bước 3: Chạy Streaming Continual Compound Shift Benchmark
Chạy phương pháp **Tri-SACTA + Online ACI**:
```bash
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --seed 42
```

Chạy baseline **Source-Only + Static CP**:
```bash
python scripts/02_run_stream_benchmark.py --method source_only --conformal static_cp --seed 42
```

Chạy baseline **TENT / SAR / READ**:
```bash
python scripts/02_run_stream_benchmark.py --method tent --conformal online_aci --seed 42
python scripts/02_run_stream_benchmark.py --method sar --conformal online_aci --seed 42
python scripts/02_run_stream_benchmark.py --method read --conformal online_aci --seed 42
```

### Bước 4: Vẽ biểu đồ chất lượng công bố quốc tế
```bash
python scripts/03_plot_rolling_curves.py
```
Biểu đồ sẽ xuất tại `results/figures/`:
- `rolling_coverage_comparison.png`: Độ phủ động qua 5 sequential regimes.
- `macro_f1_regime_bars.png`: So sánh Macro-F1 giữa các phương pháp.

---

## 4. Báo cáo Tối ưu Phần cứng (Laptop GPU 4GB VRAM)
- **Trainable Parameters:** **1.37M** (< 2.0M budget).
- **Peak VRAM thực tế:** **28.3 MB – 42.7 MB** (vượt xa chỉ tiêu < 1.5GB, hoàn toàn vận hành mượt mà trên laptop GPU 4GB VRAM).
- **Inference + TTA Latency:** **~7.87 ms/sample**.
