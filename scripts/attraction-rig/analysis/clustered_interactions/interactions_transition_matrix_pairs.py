
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl
import networkx as nx
from matplotlib.patches import FancyArrowPatch


df_interaction = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/cropped_interactions.csv')

df_cluster = pd.read_csv("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/pca-data2-F18.csv")

cluster_name = "Yhat.idt.pca"

df = pd.merge(
            df_interaction, 
            df_cluster[['interaction_id', cluster_name]], 
            on='interaction_id', 
            how='inner')


df = df[df['Normalized Frame'] == 0]



keep_cols = [
    "file", "condition", "Interaction Pair",
    "Interaction Number", "Frame", "Normalized Frame",
    cluster_name  # = "Yhat.idt.pca"
]

df_pairs = df[keep_cols].copy()


df_pairs = df_pairs.sort_values(["condition", "Interaction Number", "file", "Interaction Pair", "Frame"])

df_pairs["interaction_count"] = (df_pairs.groupby(["file", "condition", "Interaction Pair"]).cumcount() + 1)

df_pairs = df_pairs.rename(columns={cluster_name: "cluster"})

print(df_pairs.head())

df_pairs.to_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_transition_matrix_pairs.csv', index=False)


transitions = (
    df_pairs
    .sort_values(["file", "condition", "Interaction Pair", "interaction_count"])
    .groupby(['file', "condition", "Interaction Pair"])
    .apply(
        lambda g: pd.DataFrame({
            "from_cluster": g["cluster"].iloc[:-1].values,
            "to_cluster":   g["cluster"].iloc[1:].values,
            "interaction_k": g["interaction_count"].iloc[:-1].values
        })
    )
    .reset_index()
)

transition_counts = (
    transitions
    .groupby(["from_cluster", "to_cluster"])
    .size()
    .unstack(fill_value=0)
)

transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

print(transition_probs)

for cond in ["iso", "group"]:
    sub = transitions[transitions["condition"] == cond]

    counts = (sub.groupby(["from_cluster","to_cluster"])
                .size().unstack(fill_value=0))
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(probs, cmap="viridis", vmin=0, vmax=0.25, square=True)
    plt.xlabel("To cluster (k+1)")
    plt.ylabel("From cluster (k)")
    plt.title(f"{cond}: Transition likelihood P(next | current)")
    plt.tight_layout()
    plt.savefig(
        f"/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/transition_matrix/transition_likelihood_{cond}.png",
        dpi=300
    )
    plt.close()











# ----- settings -----
bin_size = 4
clusters = range(1, 13)   # force 12x12 for comparability

# make k bins: [1,3), [3,5), [5,7) ... (i.e. k=1-2, 3-4, 5-6 ...)
max_k = int(transitions["interaction_k"].max())
edges = np.arange(1, max_k + bin_size + 1, bin_size)
k_bins = pd.cut(transitions["interaction_k"], bins=edges, right=False)

# add bin label column
transitions = transitions.copy()
transitions["k_bin"] = k_bins

bin_labels = transitions["k_bin"].cat.categories  # all bins in order

# ----- precompute matrices + global vmax (so colors comparable across ALL panels) -----
mats = {}   # (cond, bin) -> probs matrix
vmax_global = 0

for cond in ["iso", "group"]:
    for b in bin_labels:
        sub = transitions[(transitions["condition"] == cond) & (transitions["k_bin"] == b)]
        if sub.empty:
            continue

        counts = sub.groupby(["from_cluster", "to_cluster"]).size().unstack(fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)

        # enforce full 12x12 grid
        probs = probs.reindex(index=clusters, columns=clusters, fill_value=0)

        mats[(cond, b)] = probs
        vmax_global = max(vmax_global, probs.to_numpy().max())

# ----- plot grid: 2 rows (iso/group) x nbins columns -----
ncols = len(bin_labels)
fig, axes = plt.subplots(2, ncols, figsize=(3.2*ncols, 7), constrained_layout=True, sharex=True, sharey=True)

for r, cond in enumerate(["iso", "group"]):
    for c, b in enumerate(bin_labels):
        ax = axes[r, c] if ncols > 1 else axes[r]

        probs = mats.get((cond, b), None)
        if probs is None:
            ax.axis("off")
            continue

        sns.heatmap(
            probs,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=vmax_global,     # SAME scale everywhere
            cbar=False,
            square=True
        )

        if r == 0:
            ax.set_title(str(b))  # shows [1, 3), [3, 5) etc
        if c == 0:
            ax.set_ylabel(f"{cond}\nFrom (k)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")

# one shared colorbar
mappable = axes[0, 0].collections[0] if ncols > 1 else axes[0].collections[0]
cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.01)
cbar.set_label("P(next | current)")

# bottom x-labels
for ax in (axes[1, :] if ncols > 1 else [axes[1]]):
    ax.set_xlabel("To cluster (k+1)")

plt.suptitle("Transition likelihood over interaction history (bins of 2)", y=1.02)
plt.savefig(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/transition_matrix/transition_evolution_bins4.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
