import os
import glob
import cv2
import pandas as pd


VIDEO_DIR = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/RIG-DEVELOPMENT/STARTLE/DEVELOPMENT/videos/new-square-white-thing/0.5mm-100ul"
FPS = 25
FLASH_DURATION = 20  # number of frames the white box stays on #4s # 5fps

RECT_X = 20
RECT_Y = 20
RECT_W = 100
RECT_H = 100


def find_csv_for_video(video_path):
    stem = os.path.splitext(video_path)[0]
    exact_csv = stem + ".csv"

    if os.path.exists(exact_csv):
        return exact_csv

    matches = glob.glob(stem + "*.csv")
    return sorted(matches)[0] if matches else None


def load_flash_frames(csv_path):
    df = pd.read_csv(csv_path)

    if "flash_start_frame" not in df.columns:
        raise ValueError(
            f"'flash_start_frame' column not found. Columns found: {list(df.columns)}"
        )

    starts = (
        pd.to_numeric(df["flash_start_frame"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    flash_frames = set()
    for start in starts:
        for f in range(start, start + FLASH_DURATION):
            flash_frames.add(f)

    return flash_frames


def draw_white_rectangle(frame):
    cv2.rectangle(
        frame,
        (RECT_X, RECT_Y),
        (RECT_X + RECT_W, RECT_Y + RECT_H),
        (255, 255, 255),
        thickness=-1
    )
    return frame


def process_video(video_path):
    csv_path = find_csv_for_video(video_path)

    if csv_path is None:
        print(f"[SKIP] No CSV found for {os.path.basename(video_path)}")
        return

    try:
        flash_frames = load_flash_frames(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed reading CSV for {os.path.basename(video_path)}: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.splitext(video_path)[0] + "_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in flash_frames:
            frame = draw_white_rectangle(frame)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] {output_path}")


def main():
    for video_path in sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))):
        if not video_path.endswith("_overlay.mp4"):
            process_video(video_path)


if __name__ == "__main__":
    main()