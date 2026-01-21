import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# -------------------- USER CONFIG --------------------
MOSEQ_CSV = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA10/2025_07_30-10_00_13/moseq_df.csv'
VIDEO_ROOT = '/Users/cochral/Desktop/MOSEQ/videos'  # folder containing original videos
OUTPUT_MP4 = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/behavioural_syllable_videos'

SYLLABLE_TO_PLOT = 5            # <-- choose the syllable ID you want
N_EXAMPLES = 12                  # grid cells
PRE_FRAMES = 15                  # frames before the center
POST_FRAMES = 15                 # frames after the center
MIN_GAP = 10                     # enforce spacing between sampled centers (avoid near-duplicates)

GRID_ROWS, GRID_COLS = 3, 4      # 3x4 = 12
CELL_SIZE = 300                  # each cell (square) after crop/rescale
FPS = 25                         # output fps (match your data if possible)

# If you want to crop around the larvae, set crop size (in px) of the box half-width
CROP_HALF = 150                  # results in a (2*CROP_HALF) box per cell, then resized to CELL_SIZE

# Body parts to draw: map name -> (x_column, y_column)
# Adjust these to your actual columns if you have more keypoints.
KEYPOINTS = {
    'head': ('head_x', 'head_y'),
    'tail': ('tail_x', 'tail_y'),
}
CENTROID_COLS = ('centroid_x', 'centroid_y')  # used for crop center & dot

# Colors (BGR). Keep it simple.
COLOR_CENTROID = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_KP = (0, 0, 255)
RADIUS_CENTROID = 3
RADIUS_KP = 3
# -----------------------------------------------------

df = pd.read_csv(MOSEQ_CSV)

# Ensure sorted
df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)

# Filter rows with the desired syllable and valid centroids
syldf = df[(df['syllable'] == SYLLABLE_TO_PLOT) & df[list(CENTROID_COLS)].notna().all(axis=1)]
if syldf.empty:
    raise ValueError(f"No rows found for syllable {SYLLABLE_TO_PLOT}")

# Keep centers with spacing (MIN_GAP) so the 12 examples aren't adjacent frames
centers = []
last_idx = defaultdict(lambda: -1e9)  # per-video last chosen frame
for _, row in syldf.sample(frac=1, random_state=0).iterrows():
    nm, fr = row['name'], int(row['frame_index'])
    if fr - last_idx[nm] >= MIN_GAP:
        centers.append((nm, fr))
        last_idx[nm] = fr
    if len(centers) >= 3*N_EXAMPLES:  # oversample a bit for window validity checks
        break

# Validate clip windows and finalize 12 examples
examples = []
for nm, fr in centers:
    # We’ll validate once we can open video
    examples.append((nm, fr))
    if len(examples) >= 2*N_EXAMPLES:
        break
examples = examples[:2*N_EXAMPLES]  # at most 24 candidates

# Build a quick map from name -> video path (assumes name + ".mp4")
def video_path_for_name(name: str) -> str:
    # If your names already end with .mp4, adjust accordingly
    # Your example looked like "N1-GH_2025-02-24_15-16-50_td7"
    return str(Path(VIDEO_ROOT) / f"{name}.mp4")

# Helper to read a specific frame (returns None if fails)
def read_frame_at(path, idx):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

# Validate window existence (pre/post frames) and keep up to N_EXAMPLES
valid = []
window_len = PRE_FRAMES + POST_FRAMES + 1
for nm, fr in examples:
    path = video_path_for_name(nm)
    if not os.path.exists(path):
        continue
    # Try to probe first and last frame
    first = max(0, fr - PRE_FRAMES)
    last = fr + POST_FRAMES
    # A light check: attempt reading first and last frame
    f0 = read_frame_at(path, first)
    f1 = read_frame_at(path, last)
    if f0 is None or f1 is None:
        continue
    valid.append((nm, fr, path))
    if len(valid) >= N_EXAMPLES:
        break

if len(valid) < N_EXAMPLES:
    raise RuntimeError(f"Only found {len(valid)} valid examples for syllable {SYLLABLE_TO_PLOT}")

# Prepare per-example per-frame info (video-specific)
# We’ll lazily read frames in the main loop to avoid storing all clips in memory.
clips = []
for nm, center_fr, path in valid:
    # Keep the rows for this video name only, so we can get keypoints per frame fast
    sub = df[df['name'] == nm].set_index('frame_index', drop=False)
    clips.append({
        'name': nm,
        'path': path,
        'center': center_fr,
        'start': center_fr - PRE_FRAMES,
        'end': center_fr + POST_FRAMES,
        'rows': sub,  # dataframe indexed by frame_index
    })

# Pre-create VideoCaptures to speed up random access (one per unique video)
caps = {}
for path in {c['path'] for c in clips}:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    caps[path] = cap

def read_with_cap(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None

# Canvas writer
out_w = GRID_COLS * CELL_SIZE
out_h = GRID_ROWS * CELL_SIZE + 40  # + banner for title
title_text = f"Syllable {SYLLABLE_TO_PLOT}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = str(Path(OUTPUT_MP4) / f"grid_syllable_{SYLLABLE_TO_PLOT}.mp4")
writer = cv2.VideoWriter(out_path, fourcc, FPS, (out_w, out_h))

# Utility to crop around (cx, cy) safely
def crop_around(frame, cx, cy, half=CROP_HALF):
    h, w = frame.shape[:2]
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, int(cx) + half)
    y2 = min(h, int(cy) + half)
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1)

# Main assembly loop over frames in the window
for t in range(window_len):
    # Compose grid
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Draw title banner
    cv2.rectangle(canvas, (0, 0), (out_w, 40), (30, 30, 30), thickness=-1)
    cv2.putText(canvas, title_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)

    for i, clip in enumerate(clips):
        r = i // GRID_COLS
        c = i % GRID_COLS
        y0 = 40 + r * CELL_SIZE
        x0 = c * CELL_SIZE

        frame_idx = clip['start'] + t
        cap = caps[clip['path']]
        frame = read_with_cap(cap, frame_idx)
        if frame is None:
            # If out of range, draw a black tile with a message
            tile = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
            cv2.putText(tile, "missing", (10, CELL_SIZE//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            canvas[y0:y0+CELL_SIZE, x0:x0+CELL_SIZE] = tile
            continue

        # Get row for this frame (to fetch centroid + keypoints)
        if frame_idx in clip['rows'].index:
            row = clip['rows'].loc[frame_idx]
        else:
            # No tracking for this frame
            row = None

        # Default: no crop center if we don't have a row
        if row is not None and pd.notna(row[CENTROID_COLS[0]]) and pd.notna(row[CENTROID_COLS[1]]):
            cx, cy = float(row[CENTROID_COLS[0]]), float(row[CENTROID_COLS[1]])
        else:
            # fallback to center of frame
            h, w = frame.shape[:2]
            cx, cy = w/2, h/2

        # Crop
        crop, (ox, oy) = crop_around(frame, cx, cy, CROP_HALF)
        if crop.size == 0:
            crop = np.zeros((2*CROP_HALF, 2*CROP_HALF, 3), dtype=np.uint8)

        # Draw centroid and keypoints (convert to crop coords)
        if row is not None:
            if pd.notna(row[CENTROID_COLS[0]]) and pd.notna(row[CENTROID_COLS[1]]):
                cx_i, cy_i = int(float(row[CENTROID_COLS[0]]) - ox), int(float(row[CENTROID_COLS[1]]) - oy)
                if 0 <= cx_i < crop.shape[1] and 0 <= cy_i < crop.shape[0]:
                    cv2.circle(crop, (cx_i, cy_i), RADIUS_CENTROID, COLOR_CENTROID, -1)

            # draw keypoints if present
            for part, (xcol, ycol) in KEYPOINTS.items():
                if xcol in row and ycol in row and pd.notna(row[xcol]) and pd.notna(row[ycol]):
                    kx, ky = int(float(row[xcol]) - ox), int(float(row[ycol]) - oy)
                    if 0 <= kx < crop.shape[1] and 0 <= ky < crop.shape[0]:
                        cv2.circle(crop, (kx, ky), RADIUS_KP, COLOR_KP, -1)

        # Resize to cell
        tile = cv2.resize(crop, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)

        # Add per-cell text: video name and the syllable id + relative t
        label1 = f"{Path(clip['name']).name}"
        label2 = f"syll {SYLLABLE_TO_PLOT} | t={frame_idx - clip['center']:+d}"
        cv2.putText(tile, label1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(tile, label2, (6, CELL_SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        canvas[y0:y0+CELL_SIZE, x0:x0+CELL_SIZE] = tile

    writer.write(canvas)

# Cleanup
writer.release()
for cap in caps.values():
    cap.release()

print(f"Saved: {out_path}")
