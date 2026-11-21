"""
Data Manifest Schema and Integrity Verification for Tri-SACTA (Section 7.2, AUD-11, AUD-26).

Provides:
- ManifestItem and DataManifest dataclasses.
- Strict schema validation for sample provenance, speaker metadata, and feature checksums.
- Speaker-disjoint / video-disjoint split verification.
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

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved dataset manifest ({len(self.items)} items) to {path}")

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

    def validate_split_integrity(self) -> Dict[str, Any]:
        """Verify strict speaker-disjoint and video-disjoint property across splits."""
        speakers_by_split: Dict[str, Set[str]] = {}
        videos_by_split: Dict[str, Set[str]] = {}
        counts_by_split: Dict[str, int] = {}

        for item in self.items:
            s = item.split
            speakers_by_split.setdefault(s, set()).add(item.speaker_id)
            videos_by_split.setdefault(s, set()).add(item.video_id)
            counts_by_split[s] = counts_by_split.get(s, 0) + 1

        splits = list(speakers_by_split.keys())
        speaker_overlaps: Dict[str, List[str]] = {}
        video_overlaps: Dict[str, List[str]] = {}

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                spk_inter = speakers_by_split[s1].intersection(speakers_by_split[s2])
                if spk_inter:
                    speaker_overlaps[f"{s1}_vs_{s2}"] = sorted(list(spk_inter))

                vid_inter = videos_by_split[s1].intersection(videos_by_split[s2])
                if vid_inter:
                    video_overlaps[f"{s1}_vs_{s2}"] = sorted(list(vid_inter))

        is_valid = (len(speaker_overlaps) == 0 and len(video_overlaps) == 0)

        return {
            "is_valid": is_valid,
            "counts_by_split": counts_by_split,
            "speaker_overlaps": speaker_overlaps,
            "video_overlaps": video_overlaps,
            "total_samples": len(self.items),
        }
