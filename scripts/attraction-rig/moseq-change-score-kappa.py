
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


folder = '/Users/cochral/Desktop/MOSEQ/w'
files = glob.glob(os.path.join(folder, '*.csv'))

dfs = []
for f in files:
    df = pd.read_csv(f)
    df['file'] = os.path.basename(f)
    dfs.append(df)

# Combine all into one DataFrame
data = pd.concat(dfs, ignore_index=True)

# Ensure sorted by file, track, frame
data = data.sort_values(['file', 'track', 'frame_idx'])

# Compute per-frame absolute delta for body.x and body.y
def compute_change_score(group):
    dx = group['body.x'].diff().abs()
    dy = group['body.y'].diff().abs()
    return (dx + dy)

# Apply per file & track
data['change_score'] = data.groupby(['file', 'track'], group_keys=False).apply(compute_change_score)

data['change_score'] = data['change_score'].fillna(0)



# Now you can sum across tracks if you want a single per-frame score per file:
file_frame_scores = (
    data.groupby(['file', 'frame_idx'])['change_score']
    .sum()
    .reset_index()
)



# pick a few example files
example_files = file_frame_scores['file'].unique()[:3]

for f in example_files:
    sub = file_frame_scores[file_frame_scores['file'] == f]
    
    plt.figure(figsize=(12, 4))
    plt.plot(sub['frame_idx'], sub['change_score'], lw=0.8)
    plt.title(f"Change score per frame — {f}")
    plt.xlabel("Frame")
    plt.ylabel("Change score")
    plt.tight_layout()
    plt.show()



min_distance = 3  # in frames = seconds
prominence_factor = 0.01  # tweak if too many/few peaks

rows = []
example_files = np.random.choice(file_frame_scores['file'].unique(), 4, replace=False)

for file_name, sub in file_frame_scores[file_frame_scores['file'].isin(example_files)].groupby('file'):

    frames = sub['frame_idx'].to_numpy()
    raw = sub['change_score'].to_numpy()

    prom = np.nanmedian(raw) + prominence_factor * np.nanstd(raw)  # no smoothing
    peaks, _ = find_peaks(raw, prominence=prom, distance=min_distance)

    if len(peaks) > 1:
        gaps_frames = np.diff(frames[peaks])
        median_gap_frames = float(np.median(gaps_frames))
    else:
        median_gap_frames = np.nan

    rows.append({
        'file': file_name,
        'n_peaks': int(len(peaks)),
        'median_gap_frames': median_gap_frames
    })

    plt.figure(figsize=(12, 4))
    plt.plot(frames, raw, lw=1, alpha=0.8, label='raw')
    plt.plot(frames[peaks], raw[peaks], 'rx', ms=6, label='peaks')
    plt.title(f"{file_name} — median gap ≈ {median_gap_frames:.1f} frames")
    plt.xlabel('Frame'); plt.ylabel('Change score')
    plt.legend(); plt.tight_layout(); plt.show()

peak_summary = pd.DataFrame(rows)
print(peak_summary)


print(peak_summary['median_gap_frames'].mean())



