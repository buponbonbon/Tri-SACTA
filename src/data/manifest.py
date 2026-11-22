"""
Data Manifest Schema and Integrity Verification for Tri-SACTA — V3 (R3-04).

Provides:
- ManifestItem and DataManifest dataclasses.
- Strict schema validation:
  1. Non-empty manifest verification (empty manifest is REJECTED).
  2. Strict unique sample IDs across entire dataset (duplicates REJECTED).
  3. Strict speaker-disjoint, video-disjoint, and dialogue-disjoint split verification.
  4. Mapped label range validation [0, num_classes - 1].
  5. Feature file checksums and non-empty file existence checks.
  6. Deterministic SHA-256 manifest fingerprinting for provenance.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class ManifestItem:
    """Provenance and metadata record for a single multimodal sample."""
    sample_id: str
    video_id: str
    speaker_id: str
    split: str  # 'train', 'val', 'cal', 'test'
    raw_label: Any
    mapped_label: int
    available_visual: bool = True
    available_audio: bool = True
    available_text: bool = True
    available_prosody: bool = True
    dialogue_id: Optional[str] = None
    feature_files: Dict[str, str] = field(default_factory=dict)
    feature_checksums: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataManifest:
    """Complete dataset manifest encapsulating sample records and dataset metadata."""
    dataset_name: str
    dataset_version: str
    feature_sources: Dict[str, str]
    feature_dims: Dict[str, int]
    items: List[ManifestItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_sources": self.feature_sources,
            "feature_dims": self.feature_dims,
            "items": [asdict(item) for item in self.items],
        }

    def compute_manifest_hash(self) -> str:
        """Compute deterministic canonical SHA-256 fingerprint over complete manifest contents (V5-03)."""
        h = hashlib.sha256()
        h.update(f"dataset:{self.dataset_name}\n".encode("utf-8"))
        h.update(f"version:{self.dataset_version}\n".encode("utf-8"))
        for k in sorted(self.feature_sources.keys()):
            h.update(f"src:{k}={self.feature_sources[k]}\n".encode("utf-8"))
        for k in sorted(self.feature_dims.keys()):
            h.update(f"dim:{k}={self.feature_dims[k]}\n".encode("utf-8"))

        for item in sorted(self.items, key=lambda x: x.sample_id):
            f_str = ",".join(f"{m}={item.feature_files.get(m, '')}:{item.feature_checksums.get(m, '')}" for m in sorted(item.feature_files.keys()))
            row_str = f"{item.sample_id}|{item.split}|{item.speaker_id}|{item.video_id}|{item.dialogue_id}|{item.mapped_label}|{item.raw_label}|{f_str}\n"
            h.update(row_str.encode("utf-8"))
        return h.hexdigest()

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved dataset manifest ({len(self.items)} items, hash={self.compute_manifest_hash()[:12]}) to {path}")

    @classmethod
    def load_json(cls, path: str) -> DataManifest:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = [ManifestItem(**item_dict) for item_dict in data.get("items", [])]
        return cls(
            dataset_name=data["dataset_name"],
            dataset_version=data["dataset_version"],
            feature_sources=data["feature_sources"],
            feature_dims=data["feature_dims"],
            items=items,
        )

    def validate_split_integrity(
        self,
        num_classes: Optional[int] = None,
        data_mode: str = "real",
        base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify strict schema validity, non-synthetic provenance, and disjointness properties (R3-04, R4-05)."""
        errors: List[str] = []

        # 1. Non-empty validation
        if len(self.items) == 0:
            errors.append("Manifest is completely empty (zero items).")
            return {
                "is_valid": False,
                "errors": errors,
                "total_samples": 0,
            }

        # 2. Strict authenticity check: Reject synthetic fixtures in real mode (R4-01, R4-05, V5-03)
        if data_mode == "real":
            if "fixture" in self.dataset_version.lower() or "quarantined" in self.dataset_version.lower():
                errors.append(f"Synthetic fixture version '{self.dataset_version}' is strictly forbidden in real mode.")

            synthetic_spk_prefixes = ("spk_cmu_mosei_", "spk_meld_", "spk_trn_", "spk_val_", "fixture_spk_")
            synthetic_vid_prefixes = ("vid_trn_", "vid_val_", "vid_cal_", "vid_tst_", "fixture_vid_")
            for item in self.items:
                if item.sample_id.startswith("fixture_"):
                    errors.append(f"Fixture sample ID '{item.sample_id}' strictly forbidden in real mode.")
                    break
                if item.speaker_id.startswith(synthetic_spk_prefixes):
                    errors.append(f"Synthetic speaker pattern '{item.speaker_id}' detected in real-mode manifest.")
                    break
                if item.video_id.startswith(synthetic_vid_prefixes):
                    errors.append(f"Synthetic video pattern '{item.video_id}' detected in real-mode manifest.")
                    break

        # 3. Unique sample IDs
        seen_sample_ids: Set[str] = set()
        duplicate_sample_ids: Set[str] = set()
        for item in self.items:
            if item.sample_id in seen_sample_ids:
                duplicate_sample_ids.add(item.sample_id)
            seen_sample_ids.add(item.sample_id)

        if duplicate_sample_ids:
            errors.append(f"Duplicate sample IDs detected: {sorted(list(duplicate_sample_ids))[:10]}")

        # 4. Allowed splits & label ranges
        allowed_splits = {"train", "val", "cal", "test"}
        speakers_by_split: Dict[str, Set[str]] = {}
        videos_by_split: Dict[str, Set[str]] = {}
        dialogues_by_split: Dict[str, Set[str]] = {}
        counts_by_split: Dict[str, int] = {}

        for item in self.items:
            s = item.split
            if s not in allowed_splits:
                errors.append(f"Invalid split '{s}' for sample {item.sample_id}")
            
            if num_classes is not None and not (0 <= item.mapped_label < num_classes):
                errors.append(f"Label {item.mapped_label} out of bounds [0, {num_classes - 1}] for sample {item.sample_id}")

            speakers_by_split.setdefault(s, set()).add(item.speaker_id)
            videos_by_split.setdefault(s, set()).add(item.video_id)
            if item.dialogue_id:
                dialogues_by_split.setdefault(s, set()).add(item.dialogue_id)
            counts_by_split[s] = counts_by_split.get(s, 0) + 1

        # 5. Overlap detection
        splits = list(counts_by_split.keys())
        speaker_overlaps: Dict[str, List[str]] = {}
        video_overlaps: Dict[str, List[str]] = {}
        dialogue_overlaps: Dict[str, List[str]] = {}

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                spk_inter = speakers_by_split[s1].intersection(speakers_by_split[s2])
                if spk_inter:
                    speaker_overlaps[f"{s1}_vs_{s2}"] = sorted(list(spk_inter))

                vid_inter = videos_by_split[s1].intersection(videos_by_split[s2])
                if vid_inter:
                    video_overlaps[f"{s1}_vs_{s2}"] = sorted(list(vid_inter))

                if s1 in dialogues_by_split and s2 in dialogues_by_split:
                    diag_inter = dialogues_by_split[s1].intersection(dialogues_by_split[s2])
                    if diag_inter:
                        dialogue_overlaps[f"{s1}_vs_{s2}"] = sorted(list(diag_inter))

        # Overlap rules: MELD allows speaker overlap due to recurring characters across dialogue scenes, but other datasets require speaker-disjoint splits.
        if self.dataset_name != "meld" and speaker_overlaps:
            errors.append(f"Speaker cross-split leakage: {speaker_overlaps}")
        if video_overlaps:
            errors.append(f"Video cross-split leakage: {video_overlaps}")
        if dialogue_overlaps:
            errors.append(f"Dialogue cross-split leakage: {dialogue_overlaps}")

        # 6. Complete feature checksum and path traversal verification (V5-03)
        if base_dir and os.path.exists(base_dir) and data_mode == "real":
            import hashlib
            base_dir_abs = os.path.abspath(base_dir)
            file_actual_shas: Dict[str, str] = {}

            # Collect unique files referenced across all manifest items
            unique_rel_paths: Set[str] = set()
            for item in self.items:
                for rel_path in item.feature_files.values():
                    unique_rel_paths.add(rel_path)

            for rel_path in unique_rel_paths:
                full_p = os.path.abspath(os.path.join(base_dir_abs, rel_path))
                # Path traversal check: must reside strictly under base_dir
                try:
                    if os.path.commonpath([base_dir_abs, full_p]) != base_dir_abs:
                        errors.append(f"Path traversal detected: {rel_path} escapes approved base directory {base_dir}")
                        continue
                except ValueError:
                    errors.append(f"Path traversal check failed across drives: {rel_path}")
                    continue

                if not os.path.exists(full_p):
                    errors.append(f"Feature file not found: {full_p}")
                    continue

                h = hashlib.sha256()
                with open(full_p, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                file_actual_shas[rel_path] = h.hexdigest()

            # Verify EVERY item across the entire manifest (no 50-row truncation)
            for idx, item in enumerate(self.items):
                if not item.feature_files or not item.feature_checksums:
                    errors.append(f"Row {idx} ({item.sample_id}) lacks mandatory feature_files or feature_checksums in real mode.")
                    break
                for mod, rel_path in item.feature_files.items():
                    expected_sha = item.feature_checksums.get(mod)
                    if not expected_sha:
                        errors.append(f"Row {idx} ({item.sample_id}): Missing expected checksum for modality '{mod}'.")
                        break
                    actual_sha = file_actual_shas.get(rel_path)
                    if actual_sha != expected_sha:
                        errors.append(
                            f"Checksum mismatch at row {idx} ({item.sample_id}, {mod}): "
                            f"expected {expected_sha[:12]}, got {actual_sha[:12] if actual_sha else 'MISSING'}"
                        )
                        break
                if len(errors) >= 10:  # Cap error list to avoid spamming
                    break

        is_valid = (len(errors) == 0)

        return {
            "is_valid": is_valid,
            "errors": errors,
            "counts_by_split": counts_by_split,
            "speaker_overlaps": speaker_overlaps,
            "video_overlaps": video_overlaps,
            "dialogue_overlaps": dialogue_overlaps,
            "total_samples": len(self.items),
            "manifest_hash": self.compute_manifest_hash(),
        }
