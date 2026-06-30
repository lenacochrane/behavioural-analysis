import os
import cv2
import pandas as pd
import numpy as np

# =============================================================================
# paths
# =============================================================================

csv_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/another-video/2026-02-18_10-40-54_td16_behaviour.csv"
video_path = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/another-video/2026-02-18_10-40-54_td16.mp4"
save_dir = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/another-video/track_videos"

os.makedirs(save_dir, exist_ok=True)

# =============================================================================
# load data
# =============================================================================

df = pd.read_csv(csv_path)


pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head', 'y_head_corrected', 'x_head_corrected', 'y_tail_corrected', 'x_tail_corrected']
df[pixel_columns] = df[pixel_columns] / 0.086


df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

track_ids = df["track_id"].unique()

# =============================================================================
# process each track
# =============================================================================

for track_id in track_ids:

    print(f"Processing track {track_id}")

    track_df = df[df["track_id"] == track_id].copy()
    track_df = track_df.set_index("frame")

    # open video fresh for each track
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(save_dir, f"track_{track_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 5, (width, height))  # 5 fps

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if frame_idx in track_df.index:
            row = track_df.loc[frame_idx]

            # coordinates
            x_head = int(round(row["x_head_corrected"]))
            y_head = int(round(row["y_head_corrected"]))
            x_body = int(round(row["x_body"]))
            y_body = int(round(row["y_body"]))
            x_tail = int(round(row["x_tail_corrected"]))
            y_tail = int(round(row["y_tail_corrected"]))

            # draw points
            cv2.circle(display, (x_head, y_head), 4, (255, 0, 0), -1)
            cv2.circle(display, (x_body, y_body), 4, (255, 0, 0), -1)
            cv2.circle(display, (x_tail, y_tail), 4, (255, 0, 0), -1)

            # behaviour label (next to body)
            behaviour = str(row["behaviour"])

            cv2.putText(
                display,
                behaviour,
                (x_body + 10, y_body + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ((0, 255, 0)),
                1,
                cv2.LINE_AA
            )

        writer.write(display)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved: {output_path}")