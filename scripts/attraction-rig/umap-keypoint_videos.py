import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2



def make_cluster_videos(csv_path, output_root='cluster_videos', fps=15, dot_radius=4, canvas_size=(800, 800)):
    df = pd.read_csv(csv_path)

    grouped = df.groupby(['Interaction Number', 'cluster'])

    for (interaction_num, cluster), group in grouped:
        group = group.sort_values("Frame")  # ensure chronological order

        folder = os.path.join(output_root, f"cluster_{cluster}")
        os.makedirs(folder, exist_ok=True)
        filename = f"interaction{interaction_num}_cluster{cluster}.mp4"
        filepath = os.path.join(folder, filename)

        x_cols = [col for col in group.columns if 'x_' in col]
        y_cols = [col for col in group.columns if 'y_' in col]

        all_x = group[x_cols].values.flatten()
        all_y = group[y_cols].values.flatten()

        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]

        if len(all_x) == 0 or len(all_y) == 0:
            print(f"Skipping interaction {interaction_num}, cluster {cluster} due to missing keypoints.")
            continue

        # Calculate fixed center and scale for the entire interaction
        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2

        canvas_w, canvas_h = canvas_size
        scale = 1.0
        if bbox_w > 0 and bbox_h > 0:
            scale = min((canvas_w - 40) / bbox_w, (canvas_h - 40) / bbox_h)

        writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, canvas_size)

        for _, row in group.iterrows():
            frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for track, color in [('Track_1', (255, 0, 0)), ('Track_2', (0, 0, 255))]:
                for part in ['tail', 'body', 'head']:
                    x = row.get(f'{track} x_{part}', np.nan)
                    y = row.get(f'{track} y_{part}', np.nan)
                    if not np.isnan(x) and not np.isnan(y):
                        sx = int((x - cx) * scale + canvas_w / 2)
                        sy = int((y - cy) * scale + canvas_h / 2)
                        cv2.circle(frame, (sx, sy), dot_radius, color, -1, lineType=cv2.LINE_AA)

            writer.write(frame)

        writer.release()
        print(f"Saved: {filepath}")



make_cluster_videos(
    '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-umap/csv.csv',
    output_root='/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-umap/videos'
)

