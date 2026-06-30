import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/behaviour_detection.csv")
df["condition"] = np.where(df["track_id"] <= 4, "isolated", "grouped")


grouped = (
    df.groupby(["condition", "file", "behaviour"])
    .size()
    .reset_index(name="count")
)

grouped["prop"] = grouped.groupby(["condition", "file"])["count"].transform(lambda x: x / x.sum())

plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x="behaviour", y="prop", hue="condition")

plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/track_videos/plots/behaviour_proportions.pdf")
plt.close()



df = df[df['head_distance'] < 10]
df = df[df["behaviour"] != "digging"]

grouped = (
    df.groupby(["condition", "file", "behaviour"])
    .size()
    .reset_index(name="count")
)

grouped["prop"] = grouped.groupby(["condition", "file"])["count"].transform(lambda x: x / x.sum())

plt.figure(figsize=(8, 5))
sns.barplot(data=grouped, x="behaviour", y="prop", hue="condition")

plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/track_videos/plots/behaviour_proportions_10mm.pdf")
plt.close()




df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/behaviour_detection.csv")
df["condition"] = np.where(df["track_id"] <= 4, "isolated", "grouped")

sharp_turns = df[df["behaviour"] == "sharp_turn"]
sharp_turns = sharp_turns[sharp_turns["head_distance"] >= 1]

plt.figure(figsize=(6, 4))

sns.histplot(
    data=sharp_turns,
    x="head_distance",
    hue="condition",
    bins=60,
    stat="density",
    common_norm=False
)

plt.xlabel("Head distance to nearest larva (mm)")
plt.ylabel("Density")
plt.title("Sharp turns vs neighbour distance")

plt.tight_layout()
plt.savefig("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/track_videos/plots/sharp_turn_head_distance_by_condition.pdf")
plt.show()






df = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated/behaviour_detection.csv")
df["condition"] = np.where(df["track_id"] <= 4, "isolated", "grouped")

turn_behaviours = ["turn", "sharp_turn", "small_turn"]

df["turn"] = df["behaviour"].isin(turn_behaviours)

df = df[df["behaviour"] != "digging"]

# keep only turns
turn_df = df[df["turn"]]

# remove very small distances
turn_df = turn_df[turn_df["head_distance"] >= 1]

plt.figure(figsize=(6, 4))

sns.histplot(
    data=turn_df,
    x="head_distance",
    hue="condition",
    bins=60,
    stat="density",
    common_norm=False
)

plt.xlabel("Head distance to nearest larva (mm)")
plt.ylabel("Density")
plt.title("Turns vs neighbour distance (≥1 mm)")

plt.tight_layout()
plt.savefig("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/track_videos/plots/all_turn_head_distance_by_condition.pdf")
plt.show()