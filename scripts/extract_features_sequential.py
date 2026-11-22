"""
extract_features_sequential.py: Stage 1 Sequential Offline Feature Extraction Pipeline — V5 (V5-02).

Mandatory Scientific Integrity Rules (V5-02, V5-08):
1. Zero Gaussian or random tensor generation in real mode. Real mode strictly requires genuine assets.
2. If raw media assets (audio/video) are unavailable, fail closed with explicit FileNotFoundError / BLOCKED status.
3. Authentic text feature extraction executes real batched forward passes using pretrained DeBERTa/MiniLM.
4. All generated tensor shards are verified with SHA-256 checksums and bound to a DataManifest.
5. Fixtures for synthetic testing are completely quarantined in src/data/fixtures.py and NEVER written here.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import hashlib
import argparse
import logging
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.manifest import DataManifest, ManifestItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Rigorous GPU memory release barrier."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SequentialFeatureExtractor:
    """Manages sequential FP16 feature extraction respecting 4GB GPU budget."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        manifest_path: str,
        dataset_root: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 7,
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.dataset_root = dataset_root
        self.device = torch.device(device)
        self.num_classes = num_classes
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

        self.dims = {
            "visual": 512,
            "audio": 768,
            "text": 768,
            "prosody": 88,
        }

    def extract_text_stage_authentic(
        self,
        transcripts: List[str],
        model_name_or_path: Optional[str] = None,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Stage 1.4: Authentic Text Feature Extraction with sequential cleanup.
        
        Executes genuine forward pass through pretrained Transformer encoder
        with attention-masked mean pooling.
        """
        logger.info(f"--- [Stage 1.4] Extracting Authentic Text Features ({len(transcripts)} utterances) ---")
        N = len(transcripts)
        if N == 0:
            return torch.zeros((0, self.dims["text"]), dtype=torch.float16)

        # Locate model: prefer cached local snapshot or fallback to standard HF model
        candidate_paths = [
            os.path.expanduser("~/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"),
            "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "microsoft/deberta-v3-base",
        ]
        chosen_path = model_name_or_path
        if not chosen_path:
            for p in candidate_paths:
                if os.path.exists(p) or "/" in p:
                    chosen_path = p
                    break

        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading text encoder from: {chosen_path}")
        tokenizer = AutoTokenizer.from_pretrained(chosen_path)
        model = AutoModel.from_pretrained(chosen_path).half().to(self.device).eval()

        pooled_tensors: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, N, batch_size):
                batch_text = [t if str(t).strip() else "..." for t in transcripts[i : i + batch_size]]
                encoded = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                out = model(**encoded)
                # Attention-masked mean pooling
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                hidden = out.last_hidden_state * mask
                pooled = hidden.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
                pooled_tensors.append(pooled.cpu().half())

        del model
        del tokenizer
        clear_gpu_memory()

        features = torch.cat(pooled_tensors, dim=0)
        logger.info(f"[Stage 1.4] Authentic text extraction complete: shape {features.shape}.")
        return features

    def extract_visual_stage(
        self,
        sample_ids: List[str],
        video_paths: Optional[Dict[str, str]] = None,
        model_name: str = "openai/clip-vit-base-patch16",
    ) -> torch.Tensor:
        """Stage 1.1: Visual Feature Extraction (Fail-Closed if raw media missing)."""
        logger.info(f"--- [Stage 1.1] Visual Feature Extraction ({model_name}) ---")
        if not video_paths or not any(os.path.exists(p) for p in video_paths.values()):
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Genuine visual media assets missing for dataset '{self.dataset_name}' (V5-02).\n"
                f"Refusing to generate synthetic random noise or Gaussian fallback in real mode.\n"
                f"Raw video archive required. Status: BLOCKED."
            )
        from transformers import CLIPVisionModelWithProjection, AutoProcessor
        # Forward pass on actual frames
        raise NotImplementedError("Raw video decode and frame extraction requires unpacked video archive.")

    def extract_audio_stage(
        self,
        sample_ids: List[str],
        audio_paths: Optional[Dict[str, str]] = None,
        model_name: str = "microsoft/wavlm-base",
    ) -> torch.Tensor:
        """Stage 1.2: Audio Feature Extraction (Fail-Closed if raw media missing)."""
        logger.info(f"--- [Stage 1.2] Audio Feature Extraction ({model_name}) ---")
        if not audio_paths or not any(os.path.exists(p) for p in audio_paths.values()):
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Genuine audio media assets missing for dataset '{self.dataset_name}' (V5-02).\n"
                f"Refusing to generate synthetic random noise in real mode.\n"
                f"Raw audio archive required. Status: BLOCKED."
            )
        raise NotImplementedError("Raw audio waveform extraction requires unpacked audio archive.")

    def extract_prosody_stage(
        self,
        sample_ids: List[str],
        audio_paths: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Stage 1.3: Prosody Extraction via openSMILE (Fail-Closed if raw media missing)."""
        logger.info("--- [Stage 1.3] Prosody Feature Extraction via openSMILE ---")
        if not audio_paths or not any(os.path.exists(p) for p in audio_paths.values()):
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Raw audio missing for openSMILE extraction in '{self.dataset_name}' (V5-02).\n"
                f"Status: BLOCKED."
            )
        raise NotImplementedError("openSMILE requires audio .wav files.")

    def extract_authentic_text_split(
        self,
        raw_csv_path: str,
        split_name: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], List[ManifestItem]]:
        """Extract genuine text features and metadata from authentic dataset CSV annotations."""
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Authentic raw annotation file not found: {raw_csv_path}")

        df = pd.read_csv(raw_csv_path)
        if max_samples and len(df) > max_samples:
            df = df.iloc[:max_samples].copy()

        N = len(df)
        logger.info(f"Processing authentic {self.dataset_name} ({split_name}): {N} rows from {raw_csv_path}")

        meta: List[Dict[str, Any]] = []
        manifest_items: List[ManifestItem] = []
        transcripts: List[str] = []
        labels_list: List[int] = []

        if self.dataset_name == "meld":
            # Emotion to 7-class label mapping:
            # 0: neutral, 1: joy/happy, 2: sadness, 3: anger, 4: surprise, 5: disgust, 6: fear
            meld_emo_map = {
                "neutral": 0, "joy": 1, "happy": 1, "sadness": 2, "sad": 2,
                "anger": 3, "surprise": 4, "disgust": 5, "fear": 6
            }
            # Cleanly partition val vs cal by dialogue to avoid dialogue leakage (Section 6 item 2)
            if split_name == "val":
                df = df[df["Dialogue_ID"] % 2 == 0].copy()
            elif split_name == "cal":
                df = df[df["Dialogue_ID"] % 2 == 1].copy()

            if max_samples and len(df) > max_samples:
                df = df.iloc[:max_samples].copy()

            for idx, row in df.iterrows():
                utt_id = f"meld_{split_name}_{int(row['Dialogue_ID']):04d}_{int(row['Utterance_ID']):03d}"
                spk = str(row["Speaker"]).strip()
                diag = f"meld_{split_name}_diag_{int(row['Dialogue_ID']):04d}"
                vid = diag
                raw_emo = str(row["Emotion"]).strip().lower()
                mapped_lbl = meld_emo_map.get(raw_emo, 0)
                text = str(row["Utterance"]).strip()

                transcripts.append(text)
                labels_list.append(mapped_lbl)

                meta.append({
                    "segment_id": utt_id,
                    "speaker_id": spk,
                    "video_id": vid,
                    "dialogue_id": diag,
                    "label": mapped_lbl,
                    "raw_text": text,
                })
                manifest_items.append(
                    ManifestItem(
                        sample_id=utt_id,
                        video_id=vid,
                        speaker_id=spk,
                        split=split_name,
                        raw_label=mapped_lbl,
                        mapped_label=mapped_lbl,
                        dialogue_id=diag,
                    )
                )

        elif self.dataset_name == "cmu_mosei":
            # Partition val vs cal by video to ensure 100% video-disjointness
            unique_vids = df["video"].drop_duplicates().tolist()
            mid = len(unique_vids) // 2
            if split_name == "val":
                allowed_vids = set(unique_vids[:mid])
                df = df[df["video"].isin(allowed_vids)].copy()
            elif split_name == "cal":
                allowed_vids = set(unique_vids[mid:])
                df = df[df["video"].isin(allowed_vids)].copy()

            if max_samples and len(df) > max_samples:
                df = df.iloc[:max_samples].copy()
            # 7 emotion classes from MOSEI emotions/sentiment:
            # Columns: 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear', 'sentiment'
            for idx, row in df.iterrows():
                vid = str(row["video"]).strip()
                start_t = float(row.get("start_time", 0.0))
                end_t = float(row.get("end_time", 0.0))
                utt_id = f"mosei_{vid}_{idx:05d}"
                spk = f"mosei_spk_{vid[:8]}"  # YouTube video ID deterministically identifies speaker/channel
                text = str(row.get("text", row.get("ASR", ""))).strip()

                # Determine dominant emotion class or sentiment
                emo_scores = [
                    float(row.get("happy", 0.0)),
                    float(row.get("sad", 0.0)),
                    float(row.get("anger", 0.0)),
                    float(row.get("surprise", 0.0)),
                    float(row.get("disgust", 0.0)),
                    float(row.get("fear", 0.0)),
                ]
                max_score = max(emo_scores)
                if max_score > 0.0:
                    mapped_lbl = emo_scores.index(max_score) + 1  # 1..6
                else:
                    mapped_lbl = 0  # neutral

                transcripts.append(text)
                labels_list.append(mapped_lbl)

                meta.append({
                    "segment_id": utt_id,
                    "speaker_id": spk,
                    "video_id": vid,
                    "dialogue_id": None,
                    "label": mapped_lbl,
                    "raw_text": text,
                    "start_time": start_t,
                    "end_time": end_t,
                })
                manifest_items.append(
                    ManifestItem(
                        sample_id=utt_id,
                        video_id=vid,
                        speaker_id=spk,
                        split=split_name,
                        raw_label=mapped_lbl,
                        mapped_label=mapped_lbl,
                    )
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        text_features = self.extract_text_stage_authentic(transcripts)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return text_features, labels_tensor, meta, manifest_items


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Sequential Feature Extractor (V5)")
    parser.add_argument("--dataset", type=str, default="meld", choices=["cmu_mosei", "meld"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extract-text-only", action="store_true", help="Extract authentic text features from official CSV annotations")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or f"data/{args.dataset}/features_cache"
    manifest_p = args.manifest or f"data/{args.dataset}/data_manifest.json"

    extractor = SequentialFeatureExtractor(
        dataset_name=args.dataset,
        output_dir=out_dir,
        manifest_path=manifest_p,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    if args.extract_text_only:
        raw_dir = f"data/{args.dataset}/raw"
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Raw annotation directory not found at {raw_dir}. Run acquire_authentic_data.py first.")

        if args.dataset == "meld":
            csv_map = {
                "train": os.path.join(raw_dir, "train_sent_emo.csv"),
                "val": os.path.join(raw_dir, "dev_sent_emo.csv"),
                "cal": os.path.join(raw_dir, "dev_sent_emo.csv"),  # Calibration partitioned from dev
                "test": os.path.join(raw_dir, "test_sent_emo.csv"),
            }
        else:
            csv_map = {
                "train": os.path.join(raw_dir, "train.csv"),
                "val": os.path.join(raw_dir, "val.csv"),
                "cal": os.path.join(raw_dir, "val.csv"),
                "test": os.path.join(raw_dir, "test.csv"),
            }

        all_manifest_items: List[ManifestItem] = []
        for split, path in csv_map.items():
            split_dir = os.path.join(out_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            t_feat, labels, meta, items = extractor.extract_authentic_text_split(
                raw_csv_path=path,
                split_name=split,
                max_samples=args.max_samples,
            )

            # Save split tensors atomically
            torch.save(t_feat, os.path.join(split_dir, "text.pt"))
            torch.save(labels, os.path.join(split_dir, "labels.pt"))
            torch.save(meta, os.path.join(split_dir, "meta.pt"))

            # Compute checksums with canonical path relative to manifest base directory (V5-03)
            manifest_base = os.path.dirname(os.path.abspath(manifest_p))
            rel_prefix = os.path.relpath(out_dir, manifest_base).replace("\\", "/")
            feature_files = {"text": f"{rel_prefix}/{split}/text.pt"}
            feature_checksums = {"text": compute_sha256(os.path.join(split_dir, "text.pt"))}

            for item in items:
                item.feature_files = dict(feature_files)
                item.feature_checksums = dict(feature_checksums)
                all_manifest_items.append(item)

        manifest = DataManifest(
            dataset_name=args.dataset,
            dataset_version="1.0-authentic-text-only",
            feature_sources={"text": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"},
            feature_dims={"text": 768},
            items=all_manifest_items,
        )
        report = manifest.validate_split_integrity(num_classes=7, data_mode="real", base_dir=out_dir)
        manifest.save_json(manifest_p)
        logger.info(f"[SUCCESS] Authentic text extraction complete ({len(all_manifest_items)} items). Manifest hash: {report['manifest_hash'][:12]}")
    else:
        logger.info(f"Extractor ready for dataset {args.dataset}. Specify --extract-text-only or provide multimodal raw media.")


if __name__ == "__main__":
    main()
