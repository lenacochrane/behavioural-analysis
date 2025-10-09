import os
import re
import cv2
import math
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from statistics import mode


# ---- load data ---- #
videos_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/videos-for-model'
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/grid-videos/keypoint-100'
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/grid-videos/keypoint-100/moseq_df.csv')

# ---- parse video_base / track_id from name ---- #
extracted = df['name'].str.extract(r'^(?P<file>.+?)(?:_track(?P<track>\d+))?(?:\.\w+)?$')
df['file'] = extracted['file']
df['track']   = extracted['track'].fillna(0).astype('int16')

df = df.sort_values(['file','track','frame_index']).reset_index(drop=True)

print(df.columns)


# ================================
# ---- make bouts dfs ----
# ================================

# make sure onset is boolean
df['onset'] = df['onset'].astype(bool)

# ---- compute bouts from onsets (one row per bout) ----
# start = onset frame; end = next onset - 1 (or last frame in that track)
onsets = (
    df.loc[df['onset'], ['file','track','syllable','frame_index']]
      .sort_values(['file','track','frame_index'])
      .rename(columns={'frame_index':'start_frame'})
)

# next onset (within same file/track)
onsets['next_onset_frame'] = (
    onsets.groupby(['file','track'])['start_frame'].shift(-1)
)

# last frame in each (file, track)
track_last = (
    df.groupby(['file','track'])['frame_index']
      .max()
      .rename('last_frame')
      .reset_index()
)

# assemble bouts
bouts_df = onsets.merge(track_last, on=['file','track'], how='left')

# safe end_frame: fill NaNs with last_frame+1, then -1, then cast to int
bouts_df['end_frame'] = (
    bouts_df['next_onset_frame']
          .fillna(bouts_df['last_frame'] + 1)
          .sub(1)
          .astype('int64')
)

bouts_df['length'] = bouts_df['end_frame'] - bouts_df['start_frame'] + 1

# final tidy columns
bouts_df = (
    bouts_df[['file','track','syllable','start_frame','end_frame','length']]
      .sort_values(['file','track','syllable','start_frame'])
      .reset_index(drop=True)
)

print("bouts_df:", bouts_df.shape, "rows")
print(bouts_df.head())

# ==========================
# ---- save bouts only ----
# ==========================
tables_dir = os.path.join(output_dir, "tables")
os.makedirs(tables_dir, exist_ok=True)

bouts_path = os.path.join(tables_dir, "bouts.parquet")
bouts_df.to_parquet(bouts_path, index=False)
print(f"✅ wrote {bouts_path}")


# --- sanity check averages ---

print("\n=== From bouts_df (parquet version) ===")
bouts_avg = (
    bouts_df.groupby('syllable')['length']
            .mean()
            .round(2)
)
print(bouts_avg)


# --- how many bouts last at least 1 second (per syllable) ---

bouts_df['duration_s'] = bouts_df['length'] 

# counts PER SYLLABLE of bouts >= 1s
per_syll_counts = (
    bouts_df[bouts_df['duration_s'] >= 1]
    .groupby('syllable')
    .size()
    .sort_values(ascending=False)
)

print("\n=== Bouts ≥ 1s per syllable ===")
print(per_syll_counts)

# total number of such bouts
print("\nTotal bouts ≥ 1s:", int((bouts_df['duration_s'] >= 1).sum()))

# number of syllable IDs that have at least one ≥ 1s bout
syllables_with_1s = per_syll_counts.index.tolist()
print("Syllables with ≥1s bouts (count):", len(syllables_with_1s))
print("Syllables list:", syllables_with_1s)



# ==========================
# ---- chose 20 random bouts per syllable (1/file) ----
# ==========================

# set seed if you want reproducibility; remove random_state for true randomness
examples_per_syllable = {}  # syllable -> 20-row DataFrame

for syll in sorted(bouts_df['syllable'].unique()):
    g = bouts_df[bouts_df['syllable'] == syll]

    # pick one random bout per file
    per_file_random = (
        g.groupby('file', group_keys=False)
         .apply(lambda x: x.sample(n=1))   # <-- no random_state
         .reset_index(drop=True)
    )

    if len(per_file_random) < 10:
        print(f"⚠️ syllable {syll}: only {len(per_file_random)} files; skipping")
        continue

    # randomly pick 20 files from those
    chosen = per_file_random.sample(n=10).reset_index(drop=True)

    examples_per_syllable[syll] = chosen
    # print(f"🎯 syllable {syll}: randomly selected 20 files")



# --------------------------------------------
# ---- helpers: safe_crop + extract clip ----
# --------------------------------------------
fps        = 3         
# crop_size  = 400
dot_radius = 3
dot_thick  = -1  # filled


# ---- fixed-per-clip window anchored on centroid stats ----
BUFFER = 300          # pixels around the anchor in x and y (tweak)
MIN_W, MIN_H = 400, 400  # minimum tile size (ensures visibility)

def compute_fixed_bbox(coords_df, frame_w, frame_h, mode="median"):
    """
    coords_df: columns frame_index, centroid_x, centroid_y (may have NaNs)
    mode: 'median' (stable) or 'first' (first valid point)
    returns (x0, y0, x1, y1) inclusive-exclusive bbox for cropping, clamped to frame.
    """
    # pick an anchor
    if mode == "first":
        valid = coords_df.dropna(subset=['centroid_x','centroid_y'])
        if valid.empty:
            # center of frame if no coords
            ax, ay = frame_w // 2, frame_h // 2
        else:
            ax, ay = int(valid.iloc[0]['centroid_x']), int(valid.iloc[0]['centroid_y'])
    else:  # median is default and robust
        ax = int(np.nanmedian(coords_df['centroid_x'].to_numpy()))
        ay = int(np.nanmedian(coords_df['centroid_y'].to_numpy()))
        if np.isnan(ax) or np.isnan(ay):
            ax, ay = frame_w // 2, frame_h // 2

    # desired box size
    half_w = max(BUFFER, MIN_W // 2)
    half_h = max(BUFFER, MIN_H // 2)

    x0 = ax - half_w
    y0 = ay - half_h
    x1 = ax + half_w
    y1 = ay + half_h

    # clamp to frame bounds
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(frame_w, x1); y1 = min(frame_h, y1)

    # ensure minimum size after clamping
    if (x1 - x0) < MIN_W:
        need = MIN_W - (x1 - x0)
        shift_left  = min(need // 2, x0)
        shift_right = min(need - shift_left, frame_w - x1)
        x0 -= shift_left; x1 += shift_right
    if (y1 - y0) < MIN_H:
        need = MIN_H - (y1 - y0)
        shift_up   = min(need // 2, y0)
        shift_down = min(need - shift_up, frame_h - y1)
        y0 -= shift_up; y1 += shift_down

    return int(x0), int(y0), int(x1), int(y1)


def extract_bout_clip(video_path, coords_df, start_f, end_f, anchor_mode="median"):
    import cv2, numpy as np, pandas as pd
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # probe frame size
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # build contiguous frame index and align coords
    idx = pd.RangeIndex(start_f, end_f + 1)
    coords = (
        coords_df.set_index('frame_index')[['centroid_x','centroid_y']]
                 .reindex(idx)
    )

    # compute ONE fixed bbox for the entire bout from coords
    x0, y0, x1, y1 = compute_fixed_bbox(coords, frame_w, frame_h, mode=anchor_mode)

    tiles, last = [], None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    for _, row in coords.iterrows():
        ok, frame = cap.read()
        if not ok:
            break

        # draw centroid if present
        cx, cy = row['centroid_x'], row['centroid_y']
        if not (pd.isna(cx) or pd.isna(cy)):
            cv2.circle(frame, (int(cx), int(cy)), dot_radius, (0, 0, 255), dot_thick)

        # fixed crop (same bbox every frame)
        tile = frame[y0:y1, x0:x1].copy()

        # pad if the bbox hits borders (should be rare given clamping, but safe)
        th, tw = tile.shape[:2]
        if th != (y1 - y0) or tw != (x1 - x0):
            out = np.zeros((y1 - y0, x1 - x0, 3), dtype=frame.dtype)
            out[:th, :tw] = tile
            tile = out

        tiles.append(tile)
        last = tile

    cap.release()
    return tiles


def pad_to_size(img, H, W):
    """Pad img (h,w,3) with black to exactly (H,W,3) without scaling."""
    h, w = img.shape[:2]
    if h == H and w == W:
        return img
    out = np.zeros((H, W, 3), dtype=img.dtype)
    out[:h, :w] = img  # top-left align; change to center if you prefer
    return out



def write_grid_5x2_pad_to_longest(output_path, clips, fps):
    if len(clips) != 10:
        print(f"⚠️ Need exactly 10 clips; got {len(clips)}")
        return False

    Lmax = max(len(c) for c in clips)
    if Lmax == 0:
        print("⚠️ All clips are empty")
        return False

    # pad each clip by freezing its last frame to match longest length
    padded = []
    for c in clips:
        if len(c) == 0:
            print("⚠️ Empty clip encountered")
            return False
        if len(c) < Lmax:
            c = c + [c[-1]] * (Lmax - len(c))
        padded.append(c)

    # Determine a common target size (no resizing; we will pad smaller tiles)
    # Use the MAX height/width across all clips' first frames
    target_h = max(c[0].shape[0] for c in padded)
    target_w = max(c[0].shape[1] for c in padded)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (target_w * 5, target_h * 2)
    )

    for i in range(Lmax):
        row1 = np.hstack([pad_to_size(padded[j][i], target_h, target_w) for j in range(0, 5)])
        row2 = np.hstack([pad_to_size(padded[j][i], target_h, target_w) for j in range(5, 10)])
        grid = np.vstack([row1, row2])
        writer.write(grid)

    writer.release()
    return True



# --------------------------------------------
# ---- build & save grids for ALL syllables ---
# --------------------------------------------
os.makedirs(os.path.join(output_dir, "grid_videos"), exist_ok=True)

for syll, chosen in examples_per_syllable.items():
    print(f"\n=== Syllable {syll} ===")
    clips = []

    for row in chosen.itertuples(index=False):
        video_path = os.path.join(videos_dir, f"{row.file}.mp4")
        if not os.path.exists(video_path):
            print(f"⚠️ Missing video: {video_path} — skipping syllable {syll}")
            clips = []
            break

        # coords for this (file, track) and bout window
        coords_df = df[
            (df['file'] == row.file) &
            (df['track'] == row.track) &
            (df['frame_index'] >= row.start_frame) &
            (df['frame_index'] <= row.end_frame)
        ][['frame_index','centroid_x','centroid_y']]

        clip = extract_bout_clip(video_path, coords_df, int(row.start_frame), int(row.end_frame))
        if not clip:
            print(f"⚠️ Empty clip for {row.file} — skipping syllable {syll}")
            clips = []
            break

        clips.append(clip)

    # need exactly 20 usable clips for a 5x4 grid
    if len(clips) != 10:
        print(f"⚠️ Not enough usable clips ({len(clips)}) for syllable {syll} — skipping.")
        continue

    # write grid named by cluster (syllable) id
    grid_dir  = os.path.join(output_dir, "grid_videos")
    out_path  = os.path.join(grid_dir, f"cluster_{syll}.mp4")  # ← filename per your cluster naming
    ok = write_grid_5x2_pad_to_longest(out_path, clips, fps)
    print(("✅ Wrote " if ok else "⚠️ Failed: ") + out_path)



