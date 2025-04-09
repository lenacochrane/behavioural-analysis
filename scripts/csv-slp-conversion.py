

### can convert h5 to slp easily but csv to slp is hard - something like different format flattened etc 

from sleap import Labels, Skeleton
from sleap.instance import Instance, Point
from sleap.label import LabeledFrame

import pandas as pd
import numpy as np


# Load your CSV
csv_path = "/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-csv-slp/2025-02-28_12-32-46_td12.csv"
df = pd.read_csv(csv_path)



# Define skeleton nodes and edges
node_names = ["head", "body", "tail"]
edges = [("head", "body"), ("body", "tail")]
skeleton = Skeleton(node_names=node_names, edges=edges)

labeled_frames = []

# Group by frame index
for frame_idx, group in df.groupby("frame_idx"):
    instances = []
    
    for _, row in group.iterrows():
        points = []
        for node in node_names:
            x = row.get(f"{node}.x", np.nan)
            y = row.get(f"{node}.y", np.nan)
            score = row.get(f"{node}.score", np.nan)
            points.append(Point(x=x, y=y, score=score))
        
        instance = Instance(points=points)
        instance.track = row["track"]  # e.g., "track_0"
        instances.append(instance)
    
    lf = LabeledFrame(video=None, frame_idx=int(frame_idx), instances=instances)
    labeled_frames.append(lf)

# Create Labels object
labels = Labels(labeled_frames=labeled_frames, skeleton=skeleton)

# Save as .slp
output_path = csv_path.replace(".csv", "_converted.slp")
labels.save_file(output_path)
print(f"✅ Saved SLP to: {output_path}")
