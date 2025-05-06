import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/distance_over_time.csv")  # or concat'd df

# Create unique track ID across files (important if track numbers repeat per file)
df['larva_id'] = df['file'].astype(str) + "_track" + df['track'].astype(str)

# Pivot table: rows = larva, columns = frame, values = distance
pivot = df.pivot_table(index='larva_id', columns='frame', values='distance_from_centre')

# Sort by larva_id for consistent plotting
pivot = pivot.sort_index()

# Plot as heatmap
plt.figure(figsize=(14, 10))
im = plt.imshow(pivot, aspect='auto', cmap='viridis', vmin=0, vmax=50)

# Colorbar for scale
cbar = plt.colorbar(im, label='Distance from Centre (mm)')

# Labeling
plt.xlabel("Frame")
plt.ylabel("Larva")
plt.title("Larval Distance from Centre Over Time")
plt.tight_layout()
plt.show()