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
output_dir = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/grid-videos/keypoint-20'
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/grid-videos/keypoint-20/moseq_df.csv')

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

    if len(per_file_random) < 20:
        print(f"⚠️ syllable {syll}: only {len(per_file_random)} files; skipping")
        continue

    # randomly pick 20 files from those
    chosen = per_file_random.sample(n=20).reset_index(drop=True)

    examples_per_syllable[syll] = chosen
    print(f"🎯 syllable {syll}: randomly selected 20 files")



# --------------------------------------------
# ---- helpers: safe_crop + extract clip ----
# --------------------------------------------
fps        = 3         
crop_size  = 400
dot_radius = 3
dot_thick  = -1  # filled

def safe_crop(frame, cx, cy, size):
    import numpy as np
    h, w = frame.shape[:2]
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    out = np.zeros((size, size, 3), dtype=frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = frame[sy0:sy1, sx0:sx1]
    return out

def extract_bout_clip(video_path, coords_df, start_f, end_f):
    """Sequential read (seek once), overlay centroid, crop 400x400.
       Pads *missing coords within the bout* by reusing the last tile.
       Returns: list of frames (tiles)."""
    import cv2, numpy as np, pandas as pd
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    idx = pd.RangeIndex(start_f, end_f + 1)
    coords = (
        coords_df.set_index('frame_index')[['centroid_x','centroid_y']]
                 .reindex(idx)           # NaN where coords missing
                 .to_numpy()
    )
    tiles, last = [], None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)  # one seek
    for xy in coords:
        ok, frame = cap.read()
        if not ok:
            break
        if not np.isnan(xy).any():
            cx, cy = int(xy[0]), int(xy[1])
            cv2.circle(frame, (cx, cy), dot_radius, (0, 255, 0), dot_thick)
            tile = safe_crop(frame, cx, cy, crop_size)
            tiles.append(tile); last = tile
        else:
            if last is not None:
                tiles.append(last)
    cap.release()
    return tiles

# --- minimal grid writer: pad each clip to the longest by freezing its last frame ---
import cv2, numpy as np

def write_grid_5x4_pad_to_longest(output_path, clips, fps):
    if len(clips) != 20:
        print(f"⚠️ Need exactly 20 clips; got {len(clips)}")
        return False

    Lmax = max(len(c) for c in clips)
    if Lmax == 0:
        print("⚠️ All clips are empty")
        return False

    # pad shorter clips by repeating their last frame
    padded = []
    for c in clips:
        if len(c) == 0:
            print("⚠️ Empty clip encountered")
            return False
        if len(c) < Lmax:
            c = c + [c[-1]] * (Lmax - len(c))
        padded.append(c)

    H, W = padded[0][0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W * 5, H * 4)
    )

    for i in range(Lmax):
        row1 = np.hstack([padded[j][i] for j in range(0, 5)])
        row2 = np.hstack([padded[j][i] for j in range(5, 10)])
        row3 = np.hstack([padded[j][i] for j in range(10, 15)])
        row4 = np.hstack([padded[j][i] for j in range(15, 20)])
        grid = np.vstack([row1, row2, row3, row4])
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
    if len(clips) != 20:
        print(f"⚠️ Not enough usable clips ({len(clips)}) for syllable {syll} — skipping.")
        continue

    # write grid named by cluster (syllable) id
    grid_dir  = os.path.join(output_dir, "grid_videos")
    out_path  = os.path.join(grid_dir, f"cluster_{syll}.mp4")  # ← filename per your cluster naming
    ok = write_grid_5x4_pad_to_longest(out_path, clips, fps)
    print(("✅ Wrote " if ok else "⚠️ Failed: ") + out_path)



