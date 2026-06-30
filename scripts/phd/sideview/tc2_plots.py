
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

clusters = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/SIDEVIEW/ANALYSIS/percent_in_clusters.csv')

number_of_clusters = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/SIDEVIEW/ANALYSIS/n_clusters_over_time.csv')

conditions_to_keep = ['T21-A08', 'T21-B09']

clusters = clusters[
    clusters['condition'].isin(conditions_to_keep)
]

number_of_clusters = number_of_clusters[
    number_of_clusters['condition'].isin(conditions_to_keep)
]

palette = {
    'T21-A08': "#0679BC",
    'T21-B09': "#00C6F7"
}


bin_size = 300 # 5fps *  60s
clusters["frame_bin"] = (clusters["frame"] // bin_size) * bin_size

average_df = (
    clusters.groupby(["condition", "file_id", "frame_bin"], as_index=False)["percent_in_clusters"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))


plt.figure(figsize=(7, 4))
sns.lineplot(
    data=average_df,
    x="frame", y="percent_in_clusters",
    hue="condition",
    palette=palette,
    errorbar=("ci", 95)   # if seaborn old, replace with ci=95
)

plt.xlabel("Frame")
plt.ylabel("Percent in clusters")
plt.title(f"Percentage in Clusters")
plt.legend(title="Condition")
sns.despine()

outpath = os.path.join('/Users/cochral/repos/behavioural-analysis/plots/phd/tc2', "percent_in_clusters_over_time.pdf")
plt.savefig(outpath, format="pdf", bbox_inches="tight")
plt.close()



bin_size = 300 # 5fps *  60s
number_of_clusters["frame_bin"] = (number_of_clusters["frame"] // bin_size) * bin_size

number_of_clusters = (
    number_of_clusters.groupby(["condition", "file_id", "frame_bin"], as_index=False)["n_clusters"]
      .mean()
      .rename(columns={"frame_bin": "frame"}))




plt.figure(figsize=(7, 4))
sns.lineplot(
    data=number_of_clusters,
    x="frame", y="n_clusters",
    hue="condition",
    palette=palette,
    errorbar=("ci", 95)   # if seaborn old, replace with ci=95
)

plt.xlabel("Frame")
plt.ylabel("Number of Clusters")
plt.title(f"Number of Clusters")
plt.legend(title="Condition")
sns.despine()

outpath = os.path.join('/Users/cochral/repos/behavioural-analysis/plots/phd/tc2', "number_of_clusters_over_time.pdf")
plt.savefig(outpath, format="pdf", bbox_inches="tight")
plt.close()