import os
import torch

data_dir = "data/cmu_mosei/features_cache"
splits = ["train", "val", "cal", "test"]

print("=== CMU-MOSEI Features Cache Audit ===")
for sp in splits:
    sp_dir = os.path.join(data_dir, sp)
    lbl = torch.load(os.path.join(sp_dir, "labels.pt"), map_location="cpu")
    meta = torch.load(os.path.join(sp_dir, "meta.pt"), map_location="cpu")
    txt = torch.load(os.path.join(sp_dir, "text.pt"), map_location="cpu")
    aud = torch.load(os.path.join(sp_dir, "audio.pt"), map_location="cpu")
    vis = torch.load(os.path.join(sp_dir, "visual.pt"), map_location="cpu")

    # Inspect video / speaker IDs in meta
    # meta is typically a list of dicts
    video_ids = set()
    segment_ids = set()
    for m in meta:
        if isinstance(m, dict):
            vid = m.get("video_id") or m.get("id", "").split("[")[0]
            video_ids.add(vid)
            segment_ids.add(m.get("segment_id", m.get("id")))
        else:
            segment_ids.add(str(m))

    print(f"Split [{sp}]:")
    print(f"  Samples: {len(lbl)}")
    print(f"  Text shape: {txt.shape}, Audio shape: {aud.shape}, Visual shape: {vis.shape}")
    print(f"  Unique segments/utterances: {len(segment_ids)}")
    print(f"  Unique videos/dialogues: {len(video_ids)}")
    print(f"  Class distribution: {torch.bincount(lbl.long(), minlength=7).tolist()}")
