import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import itertools

"""

1. IMAGE OVERLAY WITH TRACK NODES 

2. EXAMPLE OF CUMULATIVE TRACKS 

"""

""" 1. IMAGE OVERLAY WITH TRACK NODES """

video_path = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-07_14-59-00_td11.mp4'

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-07_14-59-00_td11.tracks.feather')
df = df.sort_values('frame')
df = df[df['frame'] < 300]


output = '/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/tracks'

# =========================
# 1) Frame 20 overlay -> PDF
# =========================
FRAME_TO_EXPORT = 22
frame_df = df[df['frame'] == FRAME_TO_EXPORT].copy()

# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_EXPORT)
# ok, frame_img = cap.read()
# cap.release()

cap = cv2.VideoCapture(video_path)
for _ in range(FRAME_TO_EXPORT + 1):
    ok, frame_img = cap.read()
cap.release()


# BGR -> RGB for matplotlib
frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

# black_thresh = 60  # 0–255

# mask = np.all(frame_rgb < black_thresh, axis=2)
# frame_rgb[mask] = [255, 255, 255]
# plt.figure(figsize=(8, 8)) 
# plt.imshow(frame_rgb)
plt.figure(figsize=(6, 6), dpi=600)          # 14*100 = 1400 px
plt.imshow(frame_rgb, interpolation="nearest") # stops smoothing blur


plt.axis("off")

# overlay head/body/tail points (and labels) per track
for tid, sub in frame_df.groupby("track_id"):
    # head
    xh, yh = sub["x_head"].iloc[0], sub["y_head"].iloc[0]
    xb, yb = sub["x_body"].iloc[0], sub["y_body"].iloc[0]
    xt, yt = sub["x_tail"].iloc[0], sub["y_tail"].iloc[0]

    # skip if missing
    if np.isnan([xh, yh, xb, yb, xt, yt]).any():
        continue

    plt.plot(
        [xh, xb, xt],
        [yh, yb, yt],
        color="navy",
        linewidth=0.6,
        alpha=0.4,
        zorder=1
    )


    # points
    plt.scatter([xh], [yh], s=1, color='lightskyblue')
    plt.scatter([xb], [yb], s=1, color='steelblue')
    plt.scatter([xt], [yt], s=1, color='cornflowerblue')


    # # tiny labels
    # plt.text(xh + 4, yh + 4, "H", fontsize=6,  color='darkred', fontweight='bold')
    # plt.text(xb + 4, yb + 4, "B", fontsize=6,  color='indianred', fontweight='bold')
    # plt.text(xt + 4, yt + 4, "T", fontsize=6,  color='lightcoral', fontweight='bold')

out_pdf_frame = f"{output}/overlay{FRAME_TO_EXPORT}.pdf"
plt.savefig(out_pdf_frame, format="pdf", bbox_inches="tight", pad_inches=0)
plt.close()




# ==========================================
# 2) Cumulative body paths (0–99) -> PDF
# ==========================================
# Use video size so coordinates align naturally

df = pd.read_feather('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/2025-03-03_11-40-39_td1.tracks.feather')
df = df.sort_values('frame')
df = df[df['frame'] < 600]

cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

frames = sorted(df["frame"].unique())
n_frames = len(frames)

track_ids = sorted(df["track_id"].dropna().unique())
cmap = cm.get_cmap("Blues_r")  # red palette

CONTACT_MM  = 1.00
CONVERSION  = 90 / 1050          # mm per pixel
CONTACT_PX  = CONTACT_MM / CONVERSION
FIG_INCHES  = 2.0                # figure size; vector output, so this only sets scale
LINE_PX     = 3           # track width, in video pixels
BOX_HALF    = 11                 # half side-length of the contact marker, px
BOX_THICK   = 4              # outline thickness (hollow box), px
BOX_COLOR   = (1.0, 0.647, 0.0)  # orange, RGB
PARTS       = ['head', 'body', 'tail']
PT_PER_PX   = 72 * FIG_INCHES / W   # video px -> points, so widths match the old raster

contact_points = []
in_contact = set()           # pairs touching last frame -> one dot per encounter

last_xy = {}
segments   = []   # one line segment per track per frame step
seg_colors = []

vals = np.linspace(0, 1, len(track_ids))

for i, frame in enumerate(frames):
    fdf = df[df["frame"] == frame]

    t = i / (n_frames - 1)
    max_val = 0.8
    color = cmap(max_val * t)

    for tid, sub in fdf.groupby("track_id"):
        xb, yb = sub["x_body"].iloc[0], sub["y_body"].iloc[0]
        if np.isnan([xb, yb]).any():
            continue

        if tid in last_xy:
            segments.append([last_xy[tid], (xb, yb)])
            seg_colors.append(color)
        last_xy[tid] = (xb, yb)

    # orange box at the onset of each close-range contact (<1mm, as in interaction_type_bout)
    pos = {}
    for tid, sub in fdf.groupby("track_id"):
        pts = {pt: np.array([sub[f"x_{pt}"].iloc[0], sub[f"y_{pt}"].iloc[0]]) for pt in PARTS}
        if not np.isnan(np.concatenate(list(pts.values()))).any():
            pos[tid] = pts

    still = set()
    for id1, id2 in itertools.combinations(sorted(pos), 2):
        d, best = min(
            (np.linalg.norm(pos[id1][p1] - pos[id2][p2]), (p1, p2))
            for p1 in PARTS for p2 in PARTS
        )
        if d < CONTACT_PX:
            still.add((id1, id2))
            if (id1, id2) not in in_contact:
                p1, p2 = best
                contact_points.append((frame, id1, id2, (pos[id1][p1] + pos[id2][p2]) / 2))
    in_contact = still


# genuine head-to-head contacts whose marker falls in the gap between the two
# body paths, so it reads as a stray box on the figure
SKIP_CONTACTS = {(20, 7, 8)}   # (frame, track_id_1, track_id_2)

out_pdf_path = f"{output}/tracks.pdf"
fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.add_collection(LineCollection(
    segments,
    colors=seg_colors,
    linewidths=LINE_PX * PT_PER_PX,
    capstyle="round",
    joinstyle="round",
    zorder=1,
))

for frame, id1, id2, (cx, cy) in contact_points:
    if (frame, id1, id2) in SKIP_CONTACTS:
        continue
    ax.add_patch(Rectangle(
        (cx - BOX_HALF, cy - BOX_HALF), 2 * BOX_HALF, 2 * BOX_HALF,
        fill=False, edgecolor=BOX_COLOR, linewidth=BOX_THICK * PT_PER_PX,
        zorder=3,
    ))

ax.set_xlim(0, W)
ax.set_ylim(H, 0)            # image convention: y increases downwards
ax.set_aspect("equal")
ax.axis("off")
fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
plt.close(fig)








