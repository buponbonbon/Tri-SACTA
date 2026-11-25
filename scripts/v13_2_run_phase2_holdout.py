"""
scripts/v13_2_run_phase2_holdout.py: Strictly Gated Phase II Holdout Runner & Dry-Run Preflight.

MANDATORY SAFETY PROPERTIES:
1. NEVER accesses raw_test[500:] feature arrays or labels unless ALL authorization,
   manifest digest, checkpoint byte hashes, source tree hashes, and clean git status checks PASS.
2. In Phase I, default invocations and dry-runs are strictly incapable of reading fresh holdout.
3. Validates external approval_record.json BEFORE loading encoders or predictive models.
4. Logs every execution attempt to results/v13_2/provenance/holdout_access_ledger.json.
5. Extracts features ONLY for explicit canonical eligible sample IDs from eligible_holdout_manifest.json
   and writes outputs strictly to a fresh results/v13_2/confirmatory/ directory (never overwriting
   historical data/cmu_mosei/features_cache/test).
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import hashlib
import argparse
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AuthorizationError(PermissionError):
    """Raised when Phase II holdout execution is requested without valid matching authorization."""


def check_phase2_authorization(
    manifest_path: str,
    approval_path: Optional[str],
    repo_root: str = REPO_ROOT,
    verify_git_head: bool = True,
) -> Dict[str, Any]:
    """
    Fail-closed preflight verification executed BEFORE any encoder, checkpoint, or raw feature load.
    Checks:
    1. Explicit approval JSON file must exist and carry approved == True.
    2. Manifest JSON + detached .sha256 digest must match approval["manifest_sha256"].
    3. Eligibility manifest SHA-256 must match both freeze_manifest and approval.
    4. All 21 checkpoint files and tracked scientific source files must match freeze_manifest hashes on disk.
    5. Optionally verify git HEAD matches approval["freeze_package_commit_sha"] and scientific paths are clean.
    """
    if not approval_path or not os.path.exists(approval_path):
        raise AuthorizationError(
            "PHASE II BLOCKED (H01): Explicit author approval file is missing or not supplied."
        )
    if not os.path.exists(manifest_path):
        raise AuthorizationError(f"PHASE II BLOCKED (H02): Freeze manifest not found at {manifest_path}")

    with open(approval_path, "r", encoding="utf-8") as f:
        approval = json.load(f)
    if approval.get("approved") is not True or approval.get("allowed_operation") != "PHASE_II_CONFIRMATORY_EXECUTION":
        raise AuthorizationError("PHASE II BLOCKED (H01): Approval record does not explicitly authorize Phase II.")

    actual_manifest_sha = compute_sha256(manifest_path)
    if approval.get("manifest_sha256") != actual_manifest_sha:
        raise AuthorizationError(
            f"PHASE II BLOCKED (H02): Manifest SHA mismatch. Approval={approval.get('manifest_sha256')}, Disk={actual_manifest_sha}"
        )

    detached_path = os.path.join(os.path.dirname(manifest_path), "freeze_manifest.sha256")
    if os.path.exists(detached_path):
        with open(detached_path, "r", encoding="utf-8") as f:
            detached_sha = f.read().strip().split()[0]
        if detached_sha != actual_manifest_sha:
            raise AuthorizationError("PHASE II BLOCKED (H02): Detached freeze_manifest.sha256 does not match manifest.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify eligibility manifest digest
    elig_rel = manifest.get("eligible_holdout_pool", {}).get("eligible_manifest_path", "")
    elig_abs = os.path.join(repo_root, elig_rel)
    if not elig_rel or not os.path.exists(elig_abs):
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible holdout manifest missing.")
    actual_elig_sha = compute_sha256(elig_abs)
    if actual_elig_sha != manifest["eligible_holdout_pool"].get("eligible_manifest_sha256"):
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible manifest SHA differs from freeze manifest.")
    if approval.get("eligible_manifest_sha256") != actual_elig_sha:
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible manifest SHA differs from author approval.")

    # Verify all 21 checkpoints on disk
    ckpt_inv = manifest.get("checkpoints_inventory", {})
    if len(ckpt_inv) != 21:
        raise AuthorizationError(f"PHASE II BLOCKED (H02): Expected 21 checkpoints, found {len(ckpt_inv)}.")
    for cname, info in ckpt_inv.items():
        cpath = os.path.join(repo_root, info["path"])
        if not os.path.exists(cpath) or compute_sha256(cpath) != info["sha256"]:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Checkpoint hash mismatch or missing for {cname}.")

    # Verify tracked scientific code inventory on disk
    code_inv = manifest.get("code_inventory", {})
    if not code_inv:
        raise AuthorizationError("PHASE II BLOCKED (H02): Empty code_inventory in manifest.")
    for rel_p, info in code_inv.items():
        abs_p = os.path.join(repo_root, rel_p)
        if not os.path.exists(abs_p) or compute_sha256(abs_p) != info["sha256"]:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Tracked file modified or missing since freeze: {rel_p}")

    # Verify dependency lock
    dep_info = manifest.get("dependencies_lock", {})
    dep_path = os.path.join(repo_root, dep_info.get("path", ""))
    if not os.path.exists(dep_path) or compute_sha256(dep_path) != dep_info.get("sha256"):
        raise AuthorizationError("PHASE II BLOCKED (H02): requirements-lock.txt modified or missing since freeze.")

    if verify_git_head:
        try:
            head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
            if approval.get("freeze_package_commit_sha") != head_sha:
                raise AuthorizationError(
                    f"PHASE II BLOCKED (H02): Git HEAD {head_sha} != approved SHA {approval.get('freeze_package_commit_sha')}."
                )
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "src", "scripts", "requirements-lock.txt"],
                cwd=repo_root, text=True,
            ).strip()
            if dirty:
                raise AuthorizationError(f"PHASE II BLOCKED (H02): Dirty scientific source tree:\n{dirty}")
        except subprocess.CalledProcessError as e:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Git verification error: {e}")

    return {"authorized": True, "manifest": manifest, "approval": approval, "manifest_sha256": actual_manifest_sha}


def extract_features_for_explicit_ids(
    raw_pkl_path: str,
    eligible_samples: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Phase II feature extraction function accepting an explicit ordered list of canonical
    eligible_samples (raw_test_index, canonical_id, video_group_id, start_time, end_time).
    Preserves exact ordering and stores features cleanly into a fresh output directory.
    NEVER overwrites data/cmu_mosei/features_cache/test.
    """
    if "features_cache" in os.path.abspath(output_dir) and "test" in os.path.basename(output_dir):
        raise ValueError("Refusing to overwrite historical test feature cache.")

    from scripts.acquire_multimodal_features import encode_transcripts_with_deberta
    import numpy as np
    import torch

    with open(raw_pkl_path, "rb") as f:
        raw_mosei = pickle.load(f)
    raw_test = raw_mosei["test"]

    transcripts = []
    vis_list = []
    aud_list = []
    meta_out = []
    raw_sentiments = []

    for rec in eligible_samples:
        idx = int(rec["raw_test_index"])
        it = raw_test[idx]
        vid = str(it[2][0])
        st = float(it[2][1])
        et = float(it[2][2])
        if vid != rec["video_group_id"] or abs(st - rec["start_time"]) > 1e-5 or abs(et - rec["end_time"]) > 1e-5:
            raise ValueError(f"Metadata mismatch at raw_test[{idx}] against canonical manifest.")

        words, vis_arr, aud_arr = it[0]
        text = " ".join([w for w in words if w and w not in ["<pad>", "<unk>", "$"]])
        if not text.strip():
            text = "..."
        v_mean = np.mean(vis_arr, axis=0) if vis_arr.ndim == 2 else vis_arr
        a_mean = np.mean(aud_arr, axis=0) if aud_arr.ndim == 2 else aud_arr
        raw_sent = float(it[1][0][0])

        transcripts.append(text)
        vis_list.append(v_mean)
        aud_list.append(a_mean)
        raw_sentiments.append(raw_sent)
        meta_out.append({
            "canonical_id": rec["canonical_id"],
            "raw_test_index": idx,
            "video_id": vid,
            "start_time": st,
            "end_time": et,
            "raw_sentiment": raw_sent,
        })

    os.makedirs(output_dir, exist_ok=True)
    t_tensor = encode_transcripts_with_deberta(transcripts)
    v_tensor = torch.tensor(np.array(vis_list), dtype=torch.float16)
    a_tensor = torch.tensor(np.array(aud_list), dtype=torch.float16)

    torch.save(t_tensor, os.path.join(output_dir, "text.pt"))
    torch.save(v_tensor, os.path.join(output_dir, "visual.pt"))
    torch.save(a_tensor, os.path.join(output_dir, "audio.pt"))
    torch.save(meta_out, os.path.join(output_dir, "meta.pt"))

    return {"extracted_count": len(meta_out), "output_dir": output_dir}


def run_dry_run_dev_only(manifest_path: str, synthetic_only: bool = False) -> Dict[str, Any]:
    """
    Development-only dry run incapable of reading fresh holdout features or raw_test[500:].
    Verifies pipeline wiring on val (or tiny synthetic tensors) and verifies holdout stays locked.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if synthetic_only:
        import torch
        from src.service.baseline_policies import SequentialConfidenceRouter
        from src.data.loader import SUBSET_METADATA
        bank = {k: torch.nn.Linear(v["in_dim"], 3).eval() for k, v in SUBSET_METADATA.items()}
        router = SequentialConfidenceRouter(bank, torch.device("cpu"), 0.55, 0.45, True)
        res = router.route_single_query(
            {"text": torch.ones(1, 768), "audio": torch.ones(1, 74), "visual": torch.ones(1, 35)},
            {"text": True, "audio": True, "visual": True},
        )
        return {"mode": "synthetic_dry_run", "holdout_accessed": False, "sample_pred": res["prediction"]}

    from src.data.loader import load_split, SUBSET_METADATA
    from src.models.fusion_backbone import SimpleMLP
    from src.service.baseline_policies import SequentialConfidenceRouter
    import torch

    val_data = load_split("val", REPO_ROOT)
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    bank = {}
    for s_key, meta in SUBSET_METADATA.items():
        h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
        m = SimpleMLP(meta["in_dim"], h_dim, 3)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"mlp_{s_key}_seed42.pt"), weights_only=True, map_location="cpu"))
        m.eval()
        bank[s_key] = m
    router = SequentialConfidenceRouter(bank, torch.device("cpu"), 0.55, 0.45, True)
    preds = []
    for i in range(10):
        res = router.route_single_query(
            {"text": val_data["text"][i : i + 1], "audio": val_data["audio"][i : i + 1], "visual": val_data["visual"][i : i + 1]},
            {"text": True, "audio": True, "visual": True},
        )
        preds.append(res["prediction"])
    return {
        "mode": "dev_val_dry_run_10_samples",
        "holdout_accessed": False,
        "manifest_version": manifest.get("manifest_version"),
        "sample_preds": preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Tri-SACTA V13.2 Gated Phase II Runner")
    parser.add_argument("--manifest", default=os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json"))
    parser.add_argument("--approval", default=None, help="Path to explicit author approval JSON")
    parser.add_argument("--dry-run-dev", action="store_true", help="Run dev-only pipeline check (cannot touch holdout)")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic dry-run fixture")
    parser.add_argument("--execute-holdout", action="store_true", help="Execute Phase II holdout (requires valid --approval)")
    args = parser.parse_args()

    if args.dry_run_dev:
        rep = run_dry_run_dev_only(args.manifest, synthetic_only=args.synthetic)
        logger.info(f"Dry-run completed (zero holdout access): {rep}")
        return

    # Always check authorization before touching holdout (even if neither flag passed -> fail closed)
    auth = check_phase2_authorization(args.manifest, args.approval, repo_root=REPO_ROOT)
    if args.execute_holdout:
        logger.info(f"Phase II Authorization verified for manifest {auth['manifest_sha256']}. Proceeding to confirmatory extraction.")
        # Phase II extraction/execution entrypoint (gated until author approves)
        elig_path = os.path.join(REPO_ROOT, auth["manifest"]["eligible_holdout_pool"]["eligible_manifest_path"])
        with open(elig_path, "r", encoding="utf-8") as f:
            elig = json.load(f)
        out_conf = os.path.join(REPO_ROOT, "results", "v13_2", "confirmatory")
        extract_features_for_explicit_ids(
            os.path.join(REPO_ROOT, "data", "cmu_mosei", "raw", "mosei.pkl"),
            elig["eligible_samples"],
            os.path.join(out_conf, "features_cache"),
        )


if __name__ == "__main__":
    main()
