import pandas as pd
import numpy as np
import cv2

# Load CSV
df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA10/2025_07_08-11_32_57/moseq_df.csv')

print(df.head(10))
f = df[df['name'] == 'N1-GH_2025-02-24_15-16-50_td7']

print(f.head(10))


f = f.sort_values('frame_index')

# Update column names here if needed
coord_columns = ['centroid_x', 'centroid_y']  # replace with your actual centroid column names


# Set video parameters
image_size = 1400
original_video = cv2.VideoCapture('/Users/cochral/Desktop/MOSEQ/videos/N1-GH_2025-02-24_15-16-50_td7.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/moseq/behavioural_syllable_videos/N1-GH_2025-02-24_15-16-50_td7_10.mp4', fourcc, 25.0, (image_size, image_size))

# Sort by frame for consistency
f = f.sort_values('frame_index')

frame_number = 0
while original_video.isOpened():
    ret, frame = original_video.read()
    if not ret:
        break

    frame_df = f[f['frame_index'] == frame_number]

    for _, row in frame_df.iterrows():
        x = row['centroid_x']
        y = row['centroid_y']
        if np.isnan([x, y]).any():
            continue

        x, y = int(x), int(y)

        # Optional: Change color based on condition
        color = (0, 255, 0)

        # Draw circle at centroid
        cv2.circle(frame, (x, y), radius=4, color=color, thickness=-1)

        # Get syllable (or any column you want to annotate)
        syllable = str(row['syllable'])  # make sure it's a string

        # Put text above the centroid
        cv2.putText(frame, syllable, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    video_output.write(frame)
    frame_number += 1

original_video.release()
video_output.release()