import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/nearest_neighbour.csv')
df['role'] = df['track_id'].map({0: 'fed', 1: 'starved'})


bins = np.arange(0, 91, 1)   # edges: 0,1,2,...,90 (1mm bins)
df['bin'] = pd.cut(df['body-body'], bins=bins, include_lowest=True)
df['bin_center'] = df['bin'].apply(lambda x: x.mid)

grouped_speed = (
    df
    .groupby(['file', 'role', 'bin_center'])['speed']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped_speed, x='bin_center', y='speed', hue='role', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Speed', fontsize=12, fontweight='bold')

plt.title('Speed vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/nearest_neighbour_distance_speed.png', dpi=300, bbox_inches='tight')
plt.close()


grouped_angle = (
    df
    .groupby(['file', 'role', 'bin_center'])['angle']
    .mean()
    .reset_index()
)

sns.lineplot(data=grouped_angle, x='bin_center', y='angle', hue='role', errorbar='sd')

plt.xlabel('Nearest Neighbour Distance (mm)', fontsize=12, fontweight='bold')
plt.ylabel('Angle', fontsize=12, fontweight='bold')

plt.title('Angle vs Nearest Neighbour Distance', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])


plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/nearest_neighbour_distance_angle.png', dpi=300, bbox_inches='tight')
plt.close()



plt.figure(figsize=(8,8))
sns.histplot(
    data=df,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,8))
sns.histplot(
    data=df,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle.png', dpi=300, bbox_inches='tight')
plt.close()



#### WITHIN 10MM ONLY
df_within_10mm = df[df['body-body'] <= 10]

plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_10mm,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed_within_10mm.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_10mm,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle_within_10mm.png', dpi=300, bbox_inches='tight')
plt.close()


#### WITHIN 5MM ONLY
df_within_5mm = df[df['body-body'] <= 5]

plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_5mm,
    x='speed',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Speed Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/speed_within_5mm.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,8))
sns.histplot(
    data=df_within_5mm,
    x='angle',              # <-- your speed column
    hue='role',
    bins=50,
    stat='density',         # compare shapes not raw counts
    common_norm=False,      # don't force same total area
    element='step',
    fill=False
)


plt.xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')

plt.title('Angle Distribution', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.xticks(rotation=45)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/angle_within_5mm.png', dpi=300, bbox_inches='tight')
plt.close()
