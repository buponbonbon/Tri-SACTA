import os
import torch

data_dir = "data/cmu_mosei/features_cache"
splits = ["train", "val", "cal", "test"]
split_videos = {}

for sp in splits:
    sp_dir = os.path.join(data_dir, sp)
    meta = torch.load(os.path.join(sp_dir, "meta.pt"), map_location="cpu", weights_only=False)
    vids = set()
    for m in meta:
        vid = m.get("video_id") or m.get("id", "").split("[")[0]
        vids.add(vid)
    split_videos[sp] = vids

print("=== Cross-Split Video Overlap Audit ===")
for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        s1, s2 = splits[i], splits[j]
        overlap = split_videos[s1].intersection(split_videos[s2])
        print(f"Overlap between {s1} ({len(split_videos[s1])} videos) and {s2} ({len(split_videos[s2])} videos): {len(overlap)} videos")
        if overlap:
            print(f"  Overlapping video IDs: {list(overlap)[:5]}...")
