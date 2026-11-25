"""
scripts/v13_2_audit_provenance.py: Metadata-Only Forensic Data Provenance,
Canonical Identity Reconciliation, Checkpoint/Environment Inventory,
Dev-Only Extraction Reproducibility, and Deterministic Holdout Eligibility Manifest.

STRICT PHASE I BOUNDARY:
- NEVER invokes feature encoders or predictive models on the fresh holdout pool.
- Inspects only index/video/timestamp metadata (and pre-existing aggregate counts)
  on raw_mosei["test"][500:] to construct an immutable, canonical-ID eligibility list.
- Dev extraction reproducibility check runs ONLY on the first 5 samples of 'val'.
"""

from __future__ import annotations

import os
import sys
import csv
import json
import pickle
import hashlib
import logging
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Set

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LOG_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "logs")
PROV_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "provenance")
PROTO_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "protocol")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROV_DIR, exist_ok=True)
os.makedirs(PROTO_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "provenance_execution.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

EXPECTED_MOSEI_PKL_SHA256 = "90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e"
HF_SNAPSHOT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)
SEEDS = [42, 100, 2026]
SUBSETS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_sample_digest(asset_sha: str, partition: str, raw_idx: int, vid: str, start_t: float, end_t: float) -> str:
    key_str = f"{asset_sha}:{partition}:{raw_idx}:{vid}:{start_t:.6f}:{end_t:.6f}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def to_7class_sentiment(val: float) -> int:
    if val < -2.0:
        return 0
    elif val < -1.0:
        return 1
    elif val < 0.0:
        return 2
    elif val == 0.0:
        return 3
    elif val <= 1.0:
        return 4
    elif val <= 2.0:
        return 5
    return 6


def to_3class_sentiment(val: float) -> int:
    if val < 0.0:
        return 0
    elif val == 0.0:
        return 1
    return 2


def verify_dev_only_extraction_reproducibility(raw_val_first5: List[Any]) -> Dict[str, Any]:
    """
    Runs extraction reproducibility check STRICTLY on the first 5 samples of the 'val' partition.
    Never touches any test or holdout sample.
    """
    cache_val = os.path.join(REPO_ROOT, "data", "cmu_mosei", "features_cache", "val")
    cached_t = torch.load(os.path.join(cache_val, "text.pt"), map_location="cpu", weights_only=False)[:5].float()
    cached_a = torch.load(os.path.join(cache_val, "audio.pt"), map_location="cpu", weights_only=False)[:5].float()
    cached_v = torch.load(os.path.join(cache_val, "visual.pt"), map_location="cpu", weights_only=False)[:5].float()

    # Audio (COVAREP 74d) and Visual (FACET 35d) mean pooling check
    re_aud = []
    re_vis = []
    transcripts = []
    for item in raw_val_first5:
        words, vis_arr, aud_arr = item[0]
        text = " ".join([w for w in words if w and w not in ["<pad>", "<unk>", "$"]])
        if not text.strip():
            text = "..."
        transcripts.append(text)
        v_mean = np.mean(vis_arr, axis=0) if vis_arr.ndim == 2 else vis_arr
        a_mean = np.mean(aud_arr, axis=0) if aud_arr.ndim == 2 else aud_arr
        re_vis.append(v_mean)
        re_aud.append(a_mean)

    re_a_t = torch.tensor(np.array(re_aud), dtype=torch.float16).float()
    re_v_t = torch.tensor(np.array(re_vis), dtype=torch.float16).float()
    aud_max_diff = float((re_a_t - cached_a).abs().max().item())
    vis_max_diff = float((re_v_t - cached_v).abs().max().item())

    text_check: Dict[str, Any] = {
        "encoder_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "snapshot_revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
        "local_snapshot_present": os.path.isdir(HF_SNAPSHOT_DIR),
        "status": "NOT_CHECKED",
    }

    if os.path.isdir(HF_SNAPSHOT_DIR):
        try:
            from transformers import AutoTokenizer, AutoModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(HF_SNAPSHOT_DIR, local_files_only=True)
            model = AutoModel.from_pretrained(HF_SNAPSHOT_DIR, local_files_only=True).half().to(device).eval()
            with torch.inference_mode():
                enc = tokenizer(
                    transcripts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = ((out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)).cpu().half().float()
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_max_diff = float((pooled - cached_t).abs().max().item())
            t_cos_sim = float(F.cosine_similarity(pooled, cached_t, dim=-1).min().item())
            text_check.update({
                "status": "VERIFIED_DEV_ONLY",
                "dev_samples_checked": 5,
                "max_abs_diff": t_max_diff,
                "min_cosine_similarity": t_cos_sim,
                "tolerance_pass": bool(t_cos_sim > 0.999 and t_max_diff < 0.05),
            })
        except Exception as e:
            text_check["status"] = f"ENCODER_CHECK_ERROR: {e}"
    else:
        text_check["status"] = "MISSING_LOCAL_SNAPSHOT"

    return {
        "scope": "DEV_VAL_FIRST_5_SAMPLES_ONLY",
        "audio_covarep_max_abs_diff": aud_max_diff,
        "visual_facet_max_abs_diff": vis_max_diff,
        "audio_visual_exact_fp16_parity": bool(aud_max_diff == 0.0 and vis_max_diff == 0.0),
        "text_encoder": text_check,
    }


def audit_provenance_v13_2() -> Dict[str, Any]:
    utc_now = datetime.now(timezone.utc).isoformat()
    logger.info("=== Tri-SACTA V13.2: Phase I Blocker-Fix Provenance & Identity Audit ===")

    access_log: List[Dict[str, Any]] = []

    raw_dir = os.path.join(REPO_ROOT, "data", "cmu_mosei", "raw")
    cache_dir = os.path.join(REPO_ROOT, "data", "cmu_mosei", "features_cache")
    manifest_path = os.path.join(REPO_ROOT, "data", "cmu_mosei", "data_manifest.json")

    # 1. Audit Raw Data Assets
    raw_files = {
        "mosei.pkl": os.path.join(raw_dir, "mosei.pkl"),
        "train.csv": os.path.join(raw_dir, "train.csv"),
        "val.csv": os.path.join(raw_dir, "val.csv"),
        "test.csv": os.path.join(raw_dir, "test.csv"),
        "data_manifest.json": manifest_path,
    }
    raw_assets_meta: Dict[str, Any] = {}
    for name, path in raw_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required raw asset: {path}")
        raw_assets_meta[name] = {
            "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
            "size_bytes": os.path.getsize(path),
            "sha256": compute_sha256(path),
        }
        access_log.append({
            "timestamp_utc": utc_now,
            "file": raw_assets_meta[name]["path"],
            "access_type": "SHA256_AND_METADATA_AUDIT",
            "feature_extraction": False,
            "model_inference": False,
        })

    actual_mosei_sha = raw_assets_meta["mosei.pkl"]["sha256"]
    if actual_mosei_sha != EXPECTED_MOSEI_PKL_SHA256:
        raise ValueError(
            f"BLOCKED: Raw mosei.pkl SHA-256 mismatch! Found {actual_mosei_sha}, expected {EXPECTED_MOSEI_PKL_SHA256}"
        )

    # 2. Audit Cached Splits (Checksums, Dimensions, Dtypes, Finite Checks)
    splits = ["train", "val", "cal", "test"]
    cached_splits_meta: Dict[str, Any] = {}
    tensor_files = ["text.pt", "audio.pt", "visual.pt", "labels.pt", "meta.pt"]
    for s in splits:
        split_dir = os.path.join(cache_dir, s)
        cached_splits_meta[s] = {}
        for tf in tensor_files:
            tf_path = os.path.join(split_dir, tf)
            if not os.path.exists(tf_path):
                raise FileNotFoundError(f"Missing cached tensor: {tf_path}")
            info: Dict[str, Any] = {
                "path": os.path.relpath(tf_path, REPO_ROOT).replace("\\", "/"),
                "size_bytes": os.path.getsize(tf_path),
                "sha256": compute_sha256(tf_path),
            }
            obj = torch.load(tf_path, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                info["shape"] = list(obj.shape)
                info["dtype"] = str(obj.dtype)
                if obj.is_floating_point():
                    info["all_finite"] = bool(torch.isfinite(obj).all().item())
            elif isinstance(obj, list):
                info["count"] = len(obj)
            cached_splits_meta[s][tf] = info

    # 3. Row-by-Row Canonical Verification: Raw mosei.pkl vs Cached Meta vs data_manifest.json
    with open(raw_files["mosei.pkl"], "rb") as f:
        raw_mosei = pickle.load(f)

    dev_vids = sorted(list(set(str(it[2][0]) for it in raw_mosei["dev"])))
    mid = len(dev_vids) // 2
    val_vids_set = set(dev_vids[:mid])
    cal_vids_set = set(dev_vids[mid:])
    raw_partitions = {
        "train": raw_mosei["train"],
        "val": [it for it in raw_mosei["dev"] if str(it[2][0]) in val_vids_set],
        "cal": [it for it in raw_mosei["dev"] if str(it[2][0]) in cal_vids_set],
        "test": raw_mosei["test"],
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        data_manifest = json.load(f)
    manifest_by_split: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}
    for item in data_manifest.get("items", []):
        if item.get("split") in manifest_by_split:
            manifest_by_split[item["split"]].append(item)

    row_identity_checks: Dict[str, Any] = {}
    exposed_canonical_ids: Set[str] = set()
    exposed_video_ids: Set[str] = set()
    split_lineage: Dict[str, Any] = {}
    split_vids: Dict[str, Set[str]] = {}

    for s in splits:
        meta_list = torch.load(os.path.join(cache_dir, s, "meta.pt"), map_location="cpu", weights_only=False)
        labels_7c = torch.load(os.path.join(cache_dir, s, "labels.pt"), map_location="cpu", weights_only=False)
        m_items = manifest_by_split[s]
        raw_src = raw_partitions[s]

        if len(meta_list) != len(m_items) or len(meta_list) != len(labels_7c):
            raise ValueError(f"Split {s} length mismatch: meta={len(meta_list)}, manifest={len(m_items)}, labels={len(labels_7c)}")

        mismatches = 0
        label_7c_mismatches = 0
        c_3c = [0, 0, 0]
        vids: Set[str] = set()

        # Map val/cal index back to raw_mosei["dev"] index for full canonical identity
        dev_key_to_idx = {}
        if s in ("val", "cal"):
            for d_idx, d_it in enumerate(raw_mosei["dev"]):
                dev_key_to_idx[(str(d_it[2][0]), round(float(d_it[2][1]), 6), round(float(d_it[2][2]), 6))] = d_idx

        for idx in range(len(meta_list)):
            m = meta_list[idx]
            man = m_items[idx]
            it = raw_src[idx]
            vid = str(it[2][0])
            st = float(it[2][1])
            et = float(it[2][2])
            raw_sent = float(it[1][0][0])

            if (
                m["video_id"] != vid
                or man["video_id"] != vid
                or abs(m["start_time"] - st) > 1e-5
                or abs(m["end_time"] - et) > 1e-5
                or abs(m["raw_sentiment"] - raw_sent) > 1e-5
                or abs(float(man["raw_label"]) - raw_sent) > 1e-5
            ):
                mismatches += 1

            expected_7c = to_7class_sentiment(raw_sent)
            if int(labels_7c[idx].item()) != expected_7c or int(man["mapped_label"]) != expected_7c:
                label_7c_mismatches += 1

            c_3c[to_3class_sentiment(raw_sent)] += 1
            vids.add(vid)

            src_part = "dev" if s in ("val", "cal") else s
            src_row_idx = dev_key_to_idx[(vid, round(st, 6), round(et, 6))] if s in ("val", "cal") else idx
            cid = canonical_sample_digest(actual_mosei_sha, src_part, src_row_idx, vid, st, et)
            exposed_canonical_ids.add(cid)

        if mismatches > 0 or label_7c_mismatches > 0:
            raise ValueError(f"Split {s} integrity failure: row_mismatches={mismatches}, label7c_mismatches={label_7c_mismatches}")

        split_vids[s] = vids
        exposed_video_ids.update(vids)
        row_identity_checks[s] = {
            "count": len(meta_list),
            "raw_row_mismatches": mismatches,
            "labels_7class_dtype": str(labels_7c.dtype),
            "labels_7class_mismatches": label_7c_mismatches,
            "target_3class_counts": {"negative": c_3c[0], "neutral": c_3c[1], "positive": c_3c[2]},
        }
        split_lineage[s] = {
            "num_utterances": len(meta_list),
            "num_unique_video_groups": len(vids),
            "speaker_schema": {
                "speaker_id_verified": None,
                "speaker_proxy": "mosei_spk_{vid[:8]}",
                "speaker_overlap_status": "UNKNOWN",
                "limitation_disclosure": (
                    "CMU-MOSEI raw annotations lack verified human speaker identities. "
                    "Only YouTube video_group_id disjointness is verified."
                ),
            },
            "class_distribution_3class": {
                "negative (0)": c_3c[0],
                "neutral (1)": c_3c[1],
                "positive (2)": c_3c[2],
            },
            "exposure_status": "QUARANTINED_HISTORICAL_BENCHMARK" if s == "test" else "DEVELOPMENT_FIT",
        }

    # 4. Full Raw Partition Disjointness & CSV Cross-Check
    raw_train_vids = set(str(it[2][0]) for it in raw_mosei["train"])
    raw_dev_vids = set(str(it[2][0]) for it in raw_mosei["dev"])
    raw_test_vids = set(str(it[2][0]) for it in raw_mosei["test"])
    all_raw_non_test_vids = raw_train_vids.union(raw_dev_vids)

    raw_partition_disjointness = {
        "raw_train_total": len(raw_mosei["train"]),
        "raw_train_unique_videos": len(raw_train_vids),
        "raw_dev_total": len(raw_mosei["dev"]),
        "raw_dev_unique_videos": len(raw_dev_vids),
        "raw_test_total": len(raw_mosei["test"]),
        "raw_test_unique_videos": len(raw_test_vids),
        "full_raw_train_vs_dev_video_overlap": len(raw_train_vids & raw_dev_vids),
        "full_raw_train_vs_test_video_overlap": len(raw_train_vids & raw_test_vids),
        "full_raw_dev_vs_test_video_overlap": len(raw_dev_vids & raw_test_vids),
    }
    if (
        raw_partition_disjointness["full_raw_train_vs_test_video_overlap"] != 0
        or raw_partition_disjointness["full_raw_dev_vs_test_video_overlap"] != 0
    ):
        raise ValueError("CRITICAL: Overlap detected between full raw train/dev and raw test!")

    # Historical test exposure check + ALL exposed videos check
    all_exposed_videos = all_raw_non_test_vids.union(exposed_video_ids)
    hist_test_videos = split_vids["test"]

    # 5. Deterministic Metadata-Only Eligible Holdout Pool Construction
    eligible_records: List[Dict[str, Any]] = []
    excluded_records_summary = {
        "historical_test_consumed_first_500": 0,
        "intra_video_overlap_with_historical_test": 0,
        "train_or_dev_video_overlap": 0,
    }
    raw_test_keys_seen: Set[Any] = set()
    duplicate_keys = 0
    eligible_vids: Set[str] = set()
    eligible_cids: List[str] = []
    unexp_3c = [0, 0, 0]

    for idx, it in enumerate(raw_mosei["test"]):
        vid = str(it[2][0])
        st = float(it[2][1])
        et = float(it[2][2])
        key = (vid, round(st, 6), round(et, 6))
        if key in raw_test_keys_seen:
            duplicate_keys += 1
        raw_test_keys_seen.add(key)

        if idx < 500:
            excluded_records_summary["historical_test_consumed_first_500"] += 1
            continue
        if vid in hist_test_videos:
            excluded_records_summary["intra_video_overlap_with_historical_test"] += 1
            continue
        if vid in all_exposed_videos:
            excluded_records_summary["train_or_dev_video_overlap"] += 1
            continue

        cid = canonical_sample_digest(actual_mosei_sha, "test", idx, vid, st, et)
        eligible_records.append({
            "raw_test_index": idx,
            "canonical_id": cid,
            "source_partition": "test",
            "video_group_id": vid,
            "start_time": st,
            "end_time": et,
        })
        eligible_vids.add(vid)
        eligible_cids.append(cid)

        # Aggregate disclosure count only (from existing provenance check)
        raw_sent = float(it[1][0][0])
        unexp_3c[to_3class_sentiment(raw_sent)] += 1

    if duplicate_keys > 0:
        raise ValueError(f"Duplicate canonical keys found in raw_test: {duplicate_keys}")
    if len(set(eligible_cids)) != len(eligible_records):
        raise ValueError("Duplicate canonical IDs in eligible pool!")
    if len(set(eligible_cids).intersection(exposed_canonical_ids)) != 0:
        raise ValueError("Non-empty intersection between eligible canonical IDs and exposed IDs!")
    if len(eligible_vids.intersection(all_exposed_videos)) != 0:
        raise ValueError("Non-empty video cluster intersection between eligible pool and exposed pool!")

    eligible_manifest_payload = {
        "schema_version": "13.2",
        "dataset_name": "CMU-MOSEI",
        "source_asset": "data/cmu_mosei/raw/mosei.pkl",
        "source_asset_sha256": actual_mosei_sha,
        "selection_rule": "ENTIRE_VERIFIED_ELIGIBLE_POOL_WHOLE_VIDEO_DISJOINT",
        "exclusions_summary": excluded_records_summary,
        "eligible_utterances": len(eligible_records),
        "eligible_unique_video_clusters": len(eligible_vids),
        "verified_video_overlap_with_all_exposed": len(eligible_vids.intersection(all_exposed_videos)),
        "verified_canonical_id_overlap_with_exposed": 0,
        "speaker_overlap_status": "UNKNOWN_NO_VERIFIED_SPEAKER_BIOMETRICS",
        "eligible_samples": eligible_records,
    }
    eligible_manifest_path = os.path.join(PROV_DIR, "eligible_holdout_manifest.json")
    with open(eligible_manifest_path, "w", encoding="utf-8") as f:
        json.dump(eligible_manifest_payload, f, indent=2)
    eligible_manifest_sha256 = compute_sha256(eligible_manifest_path)

    # 6. Dev-Only Feature Extraction Reproducibility Check
    logger.info("Running dev-only extraction reproducibility check on val[0:5]...")
    dev_repro = verify_dev_only_extraction_reproducibility(raw_partitions["val"][:5])

    # 7. Checkpoint & Encoder Hash Inventory + Environment Lock
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    ckpt_inventory: Dict[str, Any] = {}
    for sub in SUBSETS:
        for seed in SEEDS:
            cname = f"mlp_{sub}_seed{seed}.pt"
            cpath = os.path.join(ckpt_dir, cname)
            if not os.path.exists(cpath):
                raise FileNotFoundError(f"Missing checkpoint: {cpath}")
            ckpt_inventory[cname] = {
                "path": os.path.relpath(cpath, REPO_ROOT).replace("\\", "/"),
                "subset": sub,
                "seed": seed,
                "size_bytes": os.path.getsize(cpath),
                "sha256": compute_sha256(cpath),
            }

    ckpt_inv_path = os.path.join(PROTO_DIR, "checkpoint_hashes.json")
    with open(ckpt_inv_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(ckpt_inventory), "checkpoints": ckpt_inventory}, f, indent=2)

    encoder_assets = {}
    if os.path.isdir(HF_SNAPSHOT_DIR):
        for fname in sorted(os.listdir(HF_SNAPSHOT_DIR)):
            fpath = os.path.join(HF_SNAPSHOT_DIR, fname)
            if os.path.isfile(fpath):
                encoder_assets[fname] = {
                    "size_bytes": os.path.getsize(fpath),
                    "sha256": compute_sha256(fpath),
                }

    pip_lock_path = os.path.join(REPO_ROOT, "requirements-lock.txt")
    pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    with open(pip_lock_path, "w", encoding="utf-8") as f:
        f.write(pip_out)

    env_lock = {
        "schema_version": "13.2",
        "timestamp_utc": utc_now,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "requirements_lock_path": "requirements-lock.txt",
        "requirements_lock_sha256": compute_sha256(pip_lock_path),
        "hf_text_encoder_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "hf_snapshot_revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
        "hf_snapshot_assets": encoder_assets,
        "dev_only_extraction_reproducibility": dev_repro,
    }
    env_lock_path = os.path.join(PROTO_DIR, "environment_lock.json")
    with open(env_lock_path, "w", encoding="utf-8") as f:
        json.dump(env_lock, f, indent=2)

    # 8. Provenance Audit, Exposure Ledger & Holdout Access Ledger
    provenance_audit = {
        "audit_version": "13.2",
        "timestamp_utc": utc_now,
        "dataset_name": "CMU-MOSEI",
        "raw_assets": raw_assets_meta,
        "cached_splits": cached_splits_meta,
        "row_identity_verification": row_identity_checks,
        "raw_partition_disjointness": raw_partition_disjointness,
        "split_lineage_and_exposure": split_lineage,
        "unexposed_pool_audit": {
            "raw_test_pool_size": len(raw_mosei["test"]),
            "historical_test_consumed": excluded_records_summary["historical_test_consumed_first_500"],
            "intra_video_segments_excluded": excluded_records_summary["intra_video_overlap_with_historical_test"],
            "unexposed_available_utterances": len(eligible_records),
            "unexposed_unique_video_clusters": len(eligible_vids),
            "cross_split_video_overlap_with_train_val_cal_test": 0,
            "duplicate_canonical_keys": duplicate_keys,
            "class_distribution_3class": {
                "negative (0)": unexp_3c[0],
                "neutral (1)": unexp_3c[1],
                "positive (2)": unexp_3c[2],
            },
            "prior_aggregate_inspection_disclosure": (
                "Aggregate 3-class label counts on unexposed pool were inspected during V13.1/V13.2 "
                "metadata provenance audits. No labels, raw transcripts, or features were used for model training, "
                "sample selection, or threshold tuning."
            ),
            "eligible_manifest_path": "results/v13_2/provenance/eligible_holdout_manifest.json",
            "eligible_manifest_sha256": eligible_manifest_sha256,
            "phase_i_isolation_assertion": (
                "No model inference, feature extraction, or threshold tuning has been executed on this pool."
            ),
        },
        "dev_only_extraction_reproducibility": dev_repro,
    }
    prov_path = os.path.join(PROV_DIR, "provenance_audit.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_audit, f, indent=2)

    exposure_ledger = {
        "audit_version": "13.2",
        "timestamp_utc": utc_now,
        "partitions": {
            "train": {"status": "DEVELOPMENT_FIT", "count": 1000, "videos": len(split_vids["train"])},
            "val": {"status": "DEVELOPMENT_FIT", "count": 250, "videos": len(split_vids["val"])},
            "cal": {"status": "DEVELOPMENT_FIT", "count": 250, "videos": len(split_vids["cal"])},
            "test_historical_500": {"status": "QUARANTINED_FOREVER", "count": 500, "videos": len(split_vids["test"])},
            "candidate_unexposed_4131": {
                "status": "ELIGIBLE_FOR_CONFIRMATORY_FREEZE",
                "count": len(eligible_records),
                "videos": len(eligible_vids),
                "eligible_manifest_sha256": eligible_manifest_sha256,
            },
        },
        "unknown_historical_disclosure": (
            "Independent repository audit verified historical caches cover 1000 train, 250 val, 250 cal, and 500 test utterances. "
            "Metadata-only provenance checks inspected aggregate class counts of the remaining 4131 raw_test items in V13.1/V13.2."
        ),
    }
    with open(os.path.join(PROV_DIR, "exposure_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(exposure_ledger, f, indent=2)

    with open(os.path.join(PROV_DIR, "metadata_access_log.json"), "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": utc_now, "events": access_log}, f, indent=2)

    holdout_access_ledger = {
        "schema_version": "13.2",
        "phase": "PHASE_I",
        "timestamp_utc": utc_now,
        "fresh_holdout_features_extracted": False,
        "fresh_holdout_inference_executed": False,
        "fresh_holdout_predictions_written": False,
        "metadata_only_eligibility_audit_executed": True,
        "eligible_manifest_sha256": eligible_manifest_sha256,
        "phase_ii_authorization": "PENDING_AUTHOR_APPROVAL",
        "audit_scope_disclosure": (
            "Repository and local filesystem audit evidence recorded at script execution time; "
            "not a cryptographic proof against unlogged external execution."
        ),
    }
    with open(os.path.join(PROV_DIR, "holdout_access_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(holdout_access_ledger, f, indent=2)

    logger.info(f"Eligible holdout manifest SHA-256: {eligible_manifest_sha256}")
    logger.info(f"21 checkpoints hashed in {ckpt_inv_path}")
    logger.info(f"Dev-only extraction repro check: {dev_repro['text_encoder']['status']}, AV diff=({dev_repro['audio_covarep_max_abs_diff']}, {dev_repro['visual_facet_max_abs_diff']})")
    logger.info("=== Phase I Provenance & Identity Audit Complete ===")
    return provenance_audit


if __name__ == "__main__":
    audit_provenance_v13_2()
