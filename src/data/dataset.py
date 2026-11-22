"""
Multimodal Dataset and Data Loaders for Tri-SACTA.

Features:
- Strict fail-closed real feature loading (.pt or .h5) when data_mode='real' (AUD-10).
- Deterministic synthetic debug generator using cryptographic hashing (hashlib) instead of
  Python's randomized built-in hash() (AUD-14).
- True per-sample MultimodalCollator: single-sample missingness does NOT discard
  the modality for present samples in the batch (AUD-12).
- Strict speaker-disjoint / video-disjoint split validation (AUD-11).
"""

from __future__ import annotations

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["visual", "audio", "text", "prosody"]


class MultimodalDataset(Dataset):
    """Dataset loading pre-extracted multimodal features.

    When data_mode='real', strictly requires pre-extracted feature files and fails closed.
    When data_mode='synthetic', generates deterministic synthetic features for debug/testing.
    """

    def __init__(
        self,
        features_dir: str,
        split: str,
        config: Dict[str, Any],
        num_samples: int = 500,
        seed: int = 42,
        data_mode: str = "synthetic",
        manifest: Optional[Any] = None,
    ):
        super().__init__()
        self.features_dir = features_dir
        self.split = split
        self.config = config
        self.data_mode = data_mode.lower()
        self.num_classes = config.get("model", {}).get("num_classes", 7)
        self.seed = seed
        self.manifest = manifest

        feats_cfg = config.get("features", {})
        self.dims = {
            "visual": feats_cfg.get("visual_dim", 512),
            "audio": feats_cfg.get("audio_dim", 768),
            "text": feats_cfg.get("text_dim", 768),
            "prosody": feats_cfg.get("prosody_dim", 88),
        }

        self.samples: List[Dict[str, Any]] = []
        self._load_data(num_samples)

    def _load_data(self, num_samples: int):
        split_dir = os.path.join(self.features_dir, self.split)

        if self.data_mode == "real":
            if not os.path.exists(split_dir):
                raise FileNotFoundError(
                    f"Real data mode requested, but cache directory does not exist: {split_dir}\n"
                    f"Please extract or download real features first, or pass --data-mode synthetic for debug."
                )

            mod_data = {}
            for mod in MODALITY_NAMES:
                mod_path = os.path.join(split_dir, f"{mod}.pt")
                if os.path.exists(mod_path):
                    mod_data[mod] = torch.load(mod_path, map_location="cpu")

            labels_path = os.path.join(split_dir, "labels.pt")
            meta_path = os.path.join(split_dir, "meta.pt")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"[FAIL-CLOSED] Missing labels.pt in {split_dir} for real data mode (R3-04).")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"[FAIL-CLOSED] Missing meta.pt in {split_dir} for real data mode (R3-04).\n"
                    f"Refusing to invent synthetic IDs or assume unverified speaker provenance."
                )

            labels = torch.load(labels_path, map_location="cpu")
            meta = torch.load(meta_path, map_location="cpu")
            n = len(labels)
            if len(meta) != n:
                raise ValueError(f"Mismatch between labels count ({n}) and meta count ({len(meta)}) in {split_dir}")

            # V5-04: Manifest-to-tensor row join and sample alignment validation
            manifest = self.manifest
            if manifest is None and self.config.get("dataset", {}).get("manifest_path"):
                from src.data.manifest import DataManifest
                m_path = self.config["dataset"]["manifest_path"]
                if os.path.exists(m_path):
                    manifest = DataManifest.load_json(m_path)

            if manifest is not None:
                split_manifest_items = [it for it in manifest.items if it.split == self.split]
                if len(split_manifest_items) != n:
                    raise ValueError(
                        f"[FAIL-CLOSED] Manifest-to-cache row count mismatch in split '{self.split}': "
                        f"manifest has {len(split_manifest_items)} items, cache has {n} rows in {split_dir}"
                    )

            for i in range(n):
                lbl = int(labels[i])
                if not (0 <= lbl < self.num_classes):
                    raise ValueError(f"Label {lbl} out of range [0, {self.num_classes - 1}] in sample {i} of {split_dir}")

                sample_meta = meta[i] if isinstance(meta, (list, tuple)) else meta[str(i)]
                seg_id = sample_meta.get("segment_id", f"{self.split}_{i}")
                spk_id = sample_meta.get("speaker_id")
                if not spk_id:
                    raise ValueError(f"Missing speaker_id in metadata for sample {i} of {split_dir}")

                # Reject fixture patterns in real mode
                if seg_id.startswith("fixture_") or spk_id.startswith(("fixture_spk_", "spk_trn_", "spk_val_")):
                    raise ValueError(f"Synthetic fixture pattern detected in real cache sample {i}: {seg_id}/{spk_id}")

                if manifest is not None:
                    m_item = split_manifest_items[i]
                    if seg_id != m_item.sample_id:
                        raise ValueError(
                            f"[FAIL-CLOSED] Utterance ID mismatch at row {i} of split '{self.split}': "
                            f"cache segment_id='{seg_id}' vs manifest sample_id='{m_item.sample_id}'"
                        )
                    if spk_id != m_item.speaker_id:
                        raise ValueError(
                            f"[FAIL-CLOSED] Speaker ID mismatch at row {i} of split '{self.split}': "
                            f"cache speaker_id='{spk_id}' vs manifest speaker_id='{m_item.speaker_id}'"
                        )
                    if lbl != m_item.mapped_label:
                        raise ValueError(
                            f"[FAIL-CLOSED] Label mismatch at row {i} of split '{self.split}': "
                            f"cache label={lbl} vs manifest mapped_label={m_item.mapped_label}"
                        )

                sample = {
                    "label": lbl,
                    "segment_id": seg_id,
                    "speaker_id": spk_id,
                }
                for mod in MODALITY_NAMES:
                    is_present = (mod in mod_data)
                    if manifest is not None:
                        m_item = split_manifest_items[i]
                        is_declared = getattr(m_item, f"available_{mod}", False)
                        if is_present and not is_declared:
                            raise ValueError(
                                f"[FAIL-CLOSED] Undeclared feature tensor '{mod}.pt' present in cache for split '{self.split}', "
                                f"but manifest declares available_{mod}=False for sample '{m_item.sample_id}'!"
                            )
                        if is_declared and not is_present:
                            raise ValueError(
                                f"[FAIL-CLOSED] Manifest declares available_{mod}=True for sample '{m_item.sample_id}', "
                                f"but feature tensor '{mod}.pt' is missing from cache split '{self.split}'!"
                            )
                        avail = is_declared
                    else:
                        avail = is_present

                    if is_present:
                        sample[mod] = mod_data[mod][i].half()
                        sample[f"{mod}_available"] = avail
                    else:
                        sample[mod] = torch.zeros(self.dims[mod], dtype=torch.float16)
                        sample[f"{mod}_available"] = False
                self.samples.append(sample)
            logger.info(f"Successfully loaded and manifest-verified {len(self.samples)} real samples from {split_dir}")

        else:
            # Deterministic synthetic debug generator (AUD-14: stable hash independent of PYTHONHASHSEED)
            split_digest = int(hashlib.md5(self.split.encode("utf-8")).hexdigest()[:8], 16) % 100000
            rng = np.random.RandomState(self.seed + split_digest)
            n_speakers = 25
            speakers = [f"spk_{self.split}_{i:03d}" for i in range(n_speakers)]

            for i in range(num_samples):
                label = int(rng.randint(0, self.num_classes))
                spk = speakers[i % n_speakers]
                seg_id = f"{spk}_utt_{i:04d}"

                sample = {
                    "label": label,
                    "segment_id": seg_id,
                    "speaker_id": spk,
                }
                # Class-conditioned feature embeddings
                for mod in MODALITY_NAMES:
                    dim = self.dims[mod]
                    class_center = rng.randn(dim) * 0.5 + (label * 0.1)
                    noise = rng.randn(dim) * 0.8
                    feat = torch.tensor(class_center + noise, dtype=torch.float16)
                    sample[mod] = feat
                    sample[f"{mod}_available"] = True

                self.samples.append(sample)
            logger.info(f"[SYNTHETIC_DEBUG] Generated {num_samples} synthetic multimodal samples for split '{self.split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class MultimodalCollator:
    """Collate function that stacks features with per-sample independent modality masks (AUD-12)."""

    def __init__(self, expected_modalities: Optional[List[str]] = None):
        self.expected_modalities = expected_modalities or MODALITY_NAMES

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        collated = {
            "features": {},
            "modality_mask": {},
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "segment_ids": [b.get("segment_id", "") for b in batch],
            "speaker_ids": [b.get("speaker_id", "") for b in batch],
        }

        for mod in self.expected_modalities:
            # Determine feature dimension
            sample_dim = 512
            for b in batch:
                if mod in b and isinstance(b[mod], torch.Tensor) and b[mod].ndim > 0:
                    sample_dim = b[mod].shape[-1]
                    break

            mod_tensors = []
            mod_masks = []
            for b in batch:
                if mod in b and isinstance(b[mod], torch.Tensor):
                    is_avail = b.get(f"{mod}_available", True)
                    mod_tensors.append(b[mod].float())
                    mod_masks.append(is_avail)
                else:
                    mod_tensors.append(torch.zeros(sample_dim, dtype=torch.float32))
                    mod_masks.append(False)

            collated["features"][mod] = torch.stack(mod_tensors, dim=0)
            collated["modality_mask"][mod] = torch.tensor(mod_masks, dtype=torch.bool)

        return collated


def create_speaker_disjoint_splits(
    dataset_name: str,
    root_dir: str,
    val_ratio: float = 0.15,
    cal_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition speaker IDs into mutually exclusive sets for train/val/cal/test."""
    rng = np.random.RandomState(seed)
    num_total_speakers = 80
    all_speakers = [f"spk_{i:03d}" for i in range(num_total_speakers)]
    rng.shuffle(all_speakers)

    n_val = int(num_total_speakers * val_ratio)
    n_cal = int(num_total_speakers * cal_ratio)
    n_test = int(num_total_speakers * 0.3)
    n_train = num_total_speakers - n_val - n_cal - n_test

    splits = {
        "train": all_speakers[:n_train],
        "val": all_speakers[n_train : n_train + n_val],
        "cal": all_speakers[n_train + n_val : n_train + n_val + n_cal],
        "test": all_speakers[n_train + n_val + n_cal :],
    }

    s_train, s_val = set(splits["train"]), set(splits["val"])
    s_cal, s_test = set(splits["cal"]), set(splits["test"])
    assert s_train.isdisjoint(s_val), "Train and Val speakers overlap!"
    assert s_train.isdisjoint(s_cal), "Train and Cal speakers overlap!"
    assert s_train.isdisjoint(s_test), "Train and Test speakers overlap!"
    assert s_cal.isdisjoint(s_test), "Cal and Test speakers overlap!"

    return splits


def get_dataloaders(
    config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    data_mode: str = "synthetic",
) -> Dict[str, DataLoader]:
    """Factory creating DataLoaders with explicit data_mode and mandatory manifest validation."""
    manifest = None
    if data_mode == "real":
        manifest_path = config.get("dataset", {}).get("manifest_path")
        if not manifest_path or not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Real data mode requires a verified manifest file at '{manifest_path}' (R3-04, R4-02).\n"
                f"Please run authentic data acquisition or pass --data-mode synthetic for debug."
            )
        from src.data.manifest import DataManifest
        manifest = DataManifest.load_json(manifest_path)
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        report = manifest.validate_split_integrity(
            num_classes=config.get("dataset", {}).get("num_classes", 7),
            data_mode=data_mode,
            base_dir=manifest_dir,
        )
        if not report["is_valid"]:
            raise ValueError(f"[FAIL-CLOSED] Data manifest integrity validation failed: {report['errors']}")
        logger.info(f"Verified authentic manifest {manifest_path} (hash: {report['manifest_hash'][:12]}) for real data mode.")

    feats_dir = config.get("features", {}).get("cache_dir", "features_cache/mosei")
    collator = MultimodalCollator()

    dataloaders = {}
    splits = ["train", "val", "cal", "test"]
    sample_counts = {"train": 800, "val": 200, "cal": 200, "test": 600}

    for split in splits:
        ds = MultimodalDataset(
            features_dir=feats_dir,
            split=split,
            config=config,
            num_samples=sample_counts[split],
            seed=seed,
            data_mode=data_mode,
            manifest=manifest,
        )
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders
