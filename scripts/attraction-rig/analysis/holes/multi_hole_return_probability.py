
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/GROUP-HOUSED/returns.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-4-HOLE/SOCIAL-ISOLATION/returns.csv')
df2['condition'] = 'SI'

plt.figure(figsize=(8,8))

df = pd.concat([df1, df2], ignore_index=True)

df["returned_same_hole"] = df["returned_same_hole"].astype(bool)

prob_returned = (
    df.groupby(['condition', 'file'])['returned_same_hole']
      .mean()
      .reset_index())

sns.barplot(data=prob_returned, x='condition', y='returned_same_hole', edgecolor='black', linewidth=2, ci='sd', color='#2E8B57', alpha=0.6)

plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('Return Probability', fontsize=12, fontweight='bold')

# plt.ylim(0, 10)

plt.title('Return Probability', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.xticks(fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/returns/return_probability.png', dpi=300, bbox_inches='tight')
plt.close()


##### PROB OF RETURN FIRST VERSUS REST OF TIME EXITING 

# df already loaded and concatenated, with:
# columns: condition, file, track, exit frame, returned_same_hole

df = df.sort_values(["condition", "file", "track", "exit frame"]).reset_index(drop=True)
df["exit_number"] = df.groupby(["condition", "file", "track"]).cumcount() + 1
df["exit_type"] = np.where(df["exit_number"] == 1, "first_exit", "later_exits")




prob_first_vs_later = (
    df.groupby(["condition", "file", "exit_type"])["returned_same_hole"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(6,5))
sns.barplot(
    data=prob_first_vs_later,
    x="exit_type",
    y="returned_same_hole",
    hue="condition",
    errorbar="sd",
    edgecolor="black",
    linewidth=1.5
)

plt.ylabel("P(returned to same hole)")
plt.xlabel("")
plt.title("Return probability after first vs later exits")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()











bin_size = 600
total_frames = 3600
bins = np.arange(0, total_frames + bin_size, bin_size)   # 0,600,...,3600
bin_labels = [f"{b}-{b+bin_size}" for b in bins[:-1]]

# -----------------------
# A) Same-hole prob vs EXIT TIME bin
# -----------------------
df["exit_bin"] = pd.cut(df["exit frame"], bins=bins, labels=bin_labels, right=False)

prob_by_exit = (
    df.groupby(["condition", "exit_bin", 'file'], as_index=False)["returned_same_hole"]
      .mean()
      .rename(columns={"returned_same_hole": "prob_same_return"})
)

plt.figure(figsize=(7,4))
sns.lineplot(
    data=prob_by_exit,
    x="exit_bin", y="prob_same_return",
    hue="condition", marker="o", ci="sd"
)
plt.ylim(0, 1)
plt.xlabel("Exit time bin (frames)")
plt.ylabel("P(returned to same hole)")
plt.title("Same-hole return probability vs exit time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/returns/return_probability_by_exit_frame.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/returns/return_probability_by_exit_frame.pdf', format='pdf', bbox_inches='tight')
plt.close()




df_lag = df.copy()
df_lag["return_time"] = pd.to_numeric(df_lag["return_time"], errors="coerce")
df_lag = df_lag.dropna(subset=["return_time"])

max_lag = df_lag["return_time"].max()
lag_bins = np.arange(0, max_lag + bin_size, bin_size)
lag_labels = [f"{b}-{b+bin_size}" for b in lag_bins[:-1]]

df_lag["lag_bin"] = pd.cut(df_lag["return_time"], bins=lag_bins, labels=lag_labels, right=False)

prob_by_lag = (
    df_lag.groupby(["condition", 'file', "lag_bin"], as_index=False)["returned_same_hole"]
      .mean()
      .rename(columns={"returned_same_hole": "prob_same_return"})
)

plt.figure(figsize=(7,4))
sns.lineplot(
    data=prob_by_lag,
    x="lag_bin", y="prob_same_return",
    hue="condition", marker="o", ci="sd"
)
plt.ylim(0, 1)
plt.xlabel("Return lag bin (frames)")
plt.ylabel("P(returned to same hole)")
plt.title("Same-hole return probability vs return lag")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/holes_multi/returns/return_probability_by_return_time.png', dpi=300, bbox_inches='tight')
plt.close()