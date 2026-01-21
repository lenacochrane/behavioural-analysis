
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/counts.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/counts.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)



# assume df already loaded + concatenated as you showed
# columns: file, time, hole, inside_count, condition

df = df.sort_values(["condition", "file", "hole", "time"]).reset_index(drop=True)

# define runs: break when count / hole / file changes or time skips
new_run = (
    (df["condition"] != df["condition"].shift()) |
    (df["file"] != df["file"].shift()) |
    (df["hole"] != df["hole"].shift()) |
    (df["inside_count"] != df["inside_count"].shift()) |
    ((df["time"] - df["time"].shift()) != 1)
)

df["run_id"] = new_run.cumsum()

# get duration of each occupancy run
runs = (
    df.groupby(["condition", "file", "hole", "run_id"], as_index=False)
      .agg(
          inside_count=("inside_count", "first"),
          duration_frames=("time", "size")
      )
)

# plot: how long occupancy N persists before changing
# plt.figure(figsize=(8,6))
# sns.lineplot(
#     data=runs,
#     x="inside_count",
#     y="duration_frames",
#     hue="condition",
#     errorbar="sd",
#     marker="o"
# )

# plt.ylim(0, None)
# plt.xlabel("Number of larvae in hole")
# plt.ylabel("Duration before change (frames)")
# plt.title("Occupancy stability vs hole occupancy")
# plt.tight_layout()
# plt.show()

# --- classify WHY the run ended: entry (+1) or exit (-1) ---
runs = runs.sort_values(["condition", "file", "hole"]).reset_index(drop=True)

runs["next_inside_count"] = runs.groupby(["condition","file","hole"])["inside_count"].shift(-1)
runs["delta"] = runs["next_inside_count"] - runs["inside_count"]

runs["change_type"] = np.select(
    [runs["delta"] == 1, runs["delta"] == -1],
    ["entry (+1)", "exit (-1)"],
    default="other"
)

# drop last run per hole (no "next"), and drop weird jumps if you want
runs = runs.dropna(subset=["next_inside_count"])
runs = runs[runs["change_type"].isin(["entry (+1)", "exit (-1)"])]

# --- plot two separate plots (one for entry-ending runs, one for exit-ending runs) ---
for change in ["entry (+1)", "exit (-1)"]:
    d = runs[runs["change_type"] == change]

    plt.figure(figsize=(7,5))
    sns.lineplot(
        data=d,
        x="inside_count",
        y="duration_frames",
        hue="condition",
        errorbar="sd",
        marker="o"
    )
    plt.ylim(0, None)
    plt.xlabel("Number in hole during stable period (N)")
    plt.ylabel("How long it stayed at N (frames)")
    plt.title(f"Stability at N before {change}")
    plt.tight_layout()
    plt.show()