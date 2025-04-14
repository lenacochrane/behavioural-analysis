
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load your dataframe
df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/distance_over_time.csv")  # Replace with your path

# Step 1: Get mean distance from center for each file/track
grouped = df.groupby(['file', 'track'])['distance_from_centre'].mean().reset_index()

# Step 2: Generate angles for plotting (one per larva)
n = len(grouped)
grouped['angle'] = np.linspace(0, 2*np.pi, n, endpoint=False)

# Step 3: Start the polar plot
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})

# Plot each larva’s mean distance
ax.scatter(grouped['angle'], grouped['distance_from_centre'], color='black', alpha=0.7)

# Step 4: Add circular reference lines
# Circle for mean distance
mean_dist = grouped['distance_from_centre'].mean()
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(theta, [mean_dist]*100, linestyle='dotted', color='blue', linewidth=1.5, label='Mean radius')

# Circle for petri dish boundary (45 mm)
ax.plot(theta, [45]*100, linestyle='solid', color='grey', alpha=0.3, linewidth=2, label='Dish edge (45mm)')

# Step 5: Clean up plot
ax.set_ylim(0, 50)
ax.set_title("Pseudo Group Larvae Mean Distance from Centre", fontsize=14, weight='bold')
ax.set_yticks([0, 10, 20, 30, 40, 45])
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()