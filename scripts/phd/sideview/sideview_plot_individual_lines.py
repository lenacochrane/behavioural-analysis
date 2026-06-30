import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

output = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-sideview'

df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/SIDEVIEW/ANALYSIS/percent_in_clusters.csv')

bin_size = 300 # 5fps *  60s
df["frame_bin"] = (df["frame"] // bin_size) * bin_size

per_file = (
    df.groupby(["condition", "file_id", "frame_bin"], as_index=False)
      ["percent_in_clusters"]
      .mean()
)

for cond, plot_df in per_file.groupby("condition"):

    plt.figure(figsize=(7, 4))

    sns.lineplot(
        data=plot_df,
        x="frame_bin",
        y="percent_in_clusters",
        hue="file_id",
        estimator=None,      # important: draw each file individually
        legend=False,
        linewidth=0.5,
        palette='Blues'
    )

    plt.xlabel("Frame")
    plt.ylabel("Percent in clusters")
    plt.title(f"{cond}: percentage in clusters per file")
    sns.despine()

    outpath = os.path.join(output, f"{cond}__percent_in_clusters_per_file.pdf")
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()