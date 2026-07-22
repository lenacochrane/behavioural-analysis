import os
import cv2
import pandas as pd
import numpy as np


def make_check_digging_videos(directory, fps=25, coord_scale=1032/90):
    """
    For each video mentioned in behaviour_detection.csv:
    - find the matching .mp4
    - overlay digging points only on frames where digging_status == True
    - draw a dot at x_body, y_body
    - write track_id above the dot
    - save annotated videos into directory/check_digging
    """

    behaviour_path = os.path.join(directory, "behaviour_detection.csv")

    if not os.path.exists(behaviour_path):
        raise FileNotFoundError(f"Could not find behaviour_detection.csv in:\n{directory}")

    output_dir = os.path.join(directory, "check_digging")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(behaviour_path)

    required_cols = ["file", "frame", "track_id", "x_body", "y_body", "digging_status"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in behaviour_detection.csv: {missing}")

    # Make digging_status robust to True/"True"/1/"1"
    df["digging_status"] = (
        df["digging_status"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    digging_df = df[df["digging_status"] == True].copy()

    if digging_df.empty:
        print("No digging_status == True rows found.")
        return

    # Clean file names in case file column still contains .tracks.feather
    digging_df["video_base"] = (
        digging_df["file"]
        .astype(str)
        .str.replace(".tracks.feather", "", regex=False)
        .str.replace(".track.feather", "", regex=False)
        .str.replace(".feather", "", regex=False)
    )

    mp4_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

    def find_matching_mp4(video_base):
        """
        Match:
        file column: 2026-xx-xx_td1
        mp4:         2026-xx-xx_td1.mp4
        or any mp4 that contains the base name.
        """
        exact = video_base + ".mp4"

        if exact in mp4_files:
            return os.path.join(directory, exact)

        matches = [f for f in mp4_files if video_base in f]

        if len(matches) == 1:
            return os.path.join(directory, matches[0])

        if len(matches) > 1:
            print(f"Multiple mp4 matches for {video_base}, using first: {matches[0]}")
            return os.path.join(directory, matches[0])

        print(f"No matching mp4 found for {video_base}")
        return None

    # One output video per video_base
    for video_base, video_digging in digging_df.groupby("video_base"):

        mp4_path = find_matching_mp4(video_base)

        if mp4_path is None:
            continue

        print(f"Processing: {os.path.basename(mp4_path)}")

        cap = cv2.VideoCapture(mp4_path)

        if not cap.isOpened():
            print(f"Could not open video: {mp4_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use detected FPS if available, otherwise default to 25
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps is None or video_fps <= 0 or np.isnan(video_fps):
            video_fps = fps

        output_path = os.path.join(output_dir, f"{video_base}_check_digging.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

        # Group digging rows by frame for quick lookup
        digging_by_frame = {
            int(frame): rows
            for frame, rows in video_digging.groupby("frame")
        }

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx in digging_by_frame:
                rows = digging_by_frame[frame_idx]

                for _, row in rows.iterrows():

                    x = row["x_body"] * coord_scale
                    y = row["y_body"] * coord_scale

                    if pd.isna(x) or pd.isna(y):
                        continue

                    x = int(round(x))
                    y = int(round(y))

                    track_id = row["track_id"]

                    # Skip points outside video frame
                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue

                    # Dot
                    cv2.circle(
                        frame,
                        (x, y),
                        radius=5,
                        color=(0, 0, 255),
                        thickness=-1
                    )

                    # Track ID text above dot
                    cv2.putText(
                        frame,
                        str(track_id),
                        (x + 6, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        print(f"Saved: {output_path}")

    print("Done.")


# Example usage:
# make_check_digging_videos(
#     "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated"
# )

# make_check_digging_videos("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated")
# make_check_digging_videos("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed")
# make_check_digging_videos("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated")

make_check_digging_videos("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/fed-starved")
# make_check_digging_videos("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/agarose-plates/socially-isolated/starved-starved")