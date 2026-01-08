
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


# ============================================================
# DURATION-CLASS TRANSITIONS (short/medium/long) from df_pairs
# Makes:
#   1) iso & group heatmaps (P(next_dur | curr_dur))
#   2) difference heatmap (Group - Iso)
#   3) fair-weighted difference circlegraph
# Put this right after df_pairs["interaction_count"] is created.
# ============================================================

dur_map = {
    1: "medium",
    2: "short",
    3: "long",
    4: "long",
    5: "long",
    6: "short",
    7: "medium",
    8: "short",
    9: "medium",
    10: "short",
    11: "medium",
    12: "long",
}
dur_order = ["short", "medium", "long"]

df_durT = df_pairs.copy()
df_durT["cluster"] = df_durT["cluster"].astype(int)
df_durT["dur"] = df_durT["cluster"].map(dur_map)

# next cluster within each pair, in time order (your sort already did this)
df_durT["next_cluster"] = (
    df_durT
    .groupby(["file", "condition", "Interaction Pair"])["cluster"]
    .shift(-1)
)

df_durT = df_durT.dropna(subset=["next_cluster"]).copy()
df_durT["next_cluster"] = df_durT["next_cluster"].astype(int)
df_durT["next_dur"] = df_durT["next_cluster"].map(dur_map)

df_durT = df_durT.dropna(subset=["dur", "next_dur"]).copy()

def dur_counts_probs(dsub):
    C = (
        dsub.groupby(["dur", "next_dur"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=dur_order, columns=dur_order, fill_value=0)
    )
    P = C.div(C.sum(axis=1), axis=0).fillna(0)
    return C, P

C_iso, P_iso = dur_counts_probs(df_durT[df_durT["condition"] == "iso"])
C_grp, P_grp = dur_counts_probs(df_durT[df_durT["condition"] == "group"])

# ---------- heatmaps (iso + group, shared vmax) ----------
vmax = float(max(P_iso.to_numpy().max(), P_grp.to_numpy().max()))
if vmax == 0:
    vmax = 1e-6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
hm1 = sns.heatmap(P_iso, ax=ax1, vmin=0, vmax=vmax, cmap="viridis", square=True, cbar=False)
hm2 = sns.heatmap(P_grp, ax=ax2, vmin=0, vmax=vmax, cmap="viridis", square=True, cbar=False)
ax1.set_title("iso (dur class transitions)")
ax2.set_title("group (dur class transitions)")
ax1.set_xlabel("Next"); ax2.set_xlabel("Next")
ax1.set_ylabel("Current"); ax2.set_ylabel("")

fig.subplots_adjust(right=0.88, wspace=0.35)
cax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
cbar = fig.colorbar(hm1.collections[0], cax=cax)
cbar.set_label("P(next | current)")

plt.savefig(os.path.join( "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_iso_vs_group_heatmaps.png"), dpi=300, bbox_inches="tight")
plt.close()

# ---------- difference heatmap (Group - Iso) ----------
P_diff = P_grp - P_iso
lim = float(np.nanmax(np.abs(P_diff.to_numpy())))
if lim == 0:
    lim = 1e-6

plt.figure(figsize=(5, 4))
sns.heatmap(P_diff, cmap="RdBu", center=0, vmin=-lim, vmax=lim, square=True)
plt.title("Dur-class transition diff (Group − Iso)")
plt.xlabel("To"); plt.ylabel("From")
plt.tight_layout()
plt.savefig(os.path.join('//Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_transition_diff_heatmap.png'), dpi=300)
plt.close()



# ---------- fair weighting (row support per condition) ----------
support_iso = C_iso.sum(axis=1).astype(float)
support_grp = C_grp.sum(axis=1).astype(float)

W_iso = support_iso / support_iso.max() if support_iso.max() > 0 else support_iso * 0.0
W_grp = support_grp / support_grp.max() if support_grp.max() > 0 else support_grp * 0.0
W_fair = 0.5 * (W_iso + W_grp)

D = P_diff.mul(W_fair, axis=0)

# ---------- circlegraph for weighted difference ----------
nodes = dur_order
pos = nx.circular_layout(nodes)

lim = float(np.nanmax(np.abs(D.to_numpy())))
if lim == 0:
    lim = 1e-6

cmap = plt.cm.RdBu
norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

def diff_to_graph(D, thresh=0.02):
    G = nx.DiGraph()
    for n in D.index:
        G.add_node(n)
    for i in D.index:
        for j in D.columns:
            w = float(D.loc[i, j])
            if abs(w) >= thresh:
                G.add_edge(i, j, weight=w)
    return G

Gd = diff_to_graph(D, thresh=0.02)

def w_to_lw(w, min_w=0.6, max_w=8.0):
    return min_w + (abs(w) / lim) * (max_w - min_w)

def w_to_alpha(w, min_a=0.2, max_a=0.95):
    return min_a + (abs(w) / lim) * (max_a - min_a)

fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
ax.set_title("Dur-class transition diff circle (Group − Iso)")
ax.axis("off")

nx.draw_networkx_nodes(Gd, pos, ax=ax, node_size=1300)
nx.draw_networkx_labels(Gd, pos, ax=ax, font_size=10)

for u, v, dat in Gd.edges(data=True):
    w = float(dat["weight"])
    patch = FancyArrowPatch(
        posA=pos[u], posB=pos[v],
        arrowstyle="-|>",
        mutation_scale=16,
        connectionstyle=f"arc3,rad={0.18 if u != v else 0.45}",
        linewidth=w_to_lw(w),
        color=cmap(norm(w)),
        alpha=w_to_alpha(w),
        shrinkA=22,
        shrinkB=22
    )
    ax.add_patch(patch)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.90, 0.28, 0.02, 0.45])  # small bar
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("ΔP = P(group) − P(iso)")

plt.savefig(os.path.join("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_transition_diff_circlegraph.png"), dpi=300, bbox_inches="tight")
plt.close()









df_totals = (
    df_pairs
    .groupby(["file", "condition", "Interaction Pair"], as_index=False)
    .agg(total_interactions=("interaction_count", "max"))
)

plt.figure(figsize=(6, 5))

sns.boxplot(
    data=df_totals,
    x="condition",
    y="total_interactions",
    showfliers=True
)

sns.stripplot(
    data=df_totals,
    x="condition",
    y="total_interactions",
    color="black",
    alpha=0.4,
    jitter=0.25,
    size=3
)

plt.ylabel("Total interactions per pair (per file)")
plt.xlabel("")
plt.title("Interaction count per larval pair")

plt.tight_layout()
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_counts_per_pair_boxplot.png', dpi=300)
plt.close()




bin_size = 2

max_count = df_pairs["interaction_count"].max()
bins = np.arange(0, max_count + bin_size, bin_size)

df_pairs["interaction_bin"] = pd.cut(
    df_pairs["interaction_count"],
    bins=bins,
    right=False,   # [0–3), [3–6), ...
    include_lowest=True
)


df_freq = (
    df_pairs
    .groupby(
        ["interaction_bin", "file", "condition", "cluster"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "count"})
)


g = sns.catplot(
    data=df_freq,
    x="cluster",
    y="count",
    hue="condition",
    col="interaction_bin",
    kind="bar",         
    col_wrap=3,
    sharey=False,
    height=3.5,
    aspect=1
)

g.set_titles("{col_name}")
g.set_axis_labels("Cluster", "Count per file")
g.add_legend(title="Condition")

plt.tight_layout()
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_cluster_counts_per_bin_boxplot.png', dpi=300)
plt.close()



# ============================================================
# DURATION-CLASS (short/medium/long) COUNTS PER INTERACTION BIN
# Drop this directly UNDER the cluster-counts-per-bin block above.
# It does NOT overwrite df_pairs (uses a copy) and saves a new figure.
# ============================================================

cluster_to_class = {
    1: "medium",
    2: "short",
    3: "long",
    4: "long",
    5: "long",
    6: "short",
    7: "medium",
    8: "short",
    9: "medium",
    10: "short",
    11: "medium",
    12: "long",
}

# work on a copy so we don't interfere with anything else downstream
df_dur = df_pairs.copy()

# map cluster -> duration class
df_dur["interaction_class"] = df_dur["cluster"].astype(int).map(cluster_to_class)

# (optional) enforce stable ordering on the x-axis
df_dur["interaction_class"] = pd.Categorical(
    df_dur["interaction_class"],
    categories=["short", "medium", "long"],
    ordered=True
)

# reuse the same binning (do it on df_dur only, doesn’t touch df_pairs)
bin_size = 3
max_count = df_dur["interaction_count"].max()
bins = np.arange(0, max_count + bin_size, bin_size)

df_dur["interaction_bin"] = pd.cut(
    df_dur["interaction_count"],
    bins=bins,
    right=False,
    include_lowest=True
)

# count per file, per bin, per condition, per duration class
df_dur_freq = (
    df_dur
    .groupby(["interaction_bin", "file", "condition", "interaction_class"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
)

df_dur_freq["fraction"] = (
    df_dur_freq["count"] /
    df_dur_freq.groupby(["interaction_bin", "file", "condition"])["count"].transform("sum")
)


g = sns.catplot(
    data=df_dur_freq,
    x="interaction_class",
    y="fraction",
    hue="condition",
    col="interaction_bin",
    kind="bar",
    col_wrap=3,
    sharey=False,
    height=3.5,
    aspect=1
)

g.set_titles("{col_name}")
g.set_axis_labels("Interaction duration class", "Fraction")
g.add_legend(title="Condition")
for ax in g.axes.flat:
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_duration_class_counts_per_bin-durations.png",
    dpi=300
)
plt.close()





dfp = df_pairs.copy()
dfp["pair_id"] = dfp["file"].astype(str) + " | " + dfp["Interaction Pair"].astype(str)
dfp["cluster"] = dfp["cluster"].astype(int)  # if already int, fine

def make_mat(d):
    # order rows by longest -> shortest (max interaction_count)
    order = d.groupby("pair_id")["interaction_count"].max().sort_values(ascending=False)
    pair_ids = order.index.to_list()
    max_len = int(order.max())

    mat = np.full((len(pair_ids), max_len), np.nan)  # NaN = no data
    row = {pid: i for i, pid in enumerate(pair_ids)}

    # fill: col = interaction_count-1, value = cluster
    for pid, ic, cl in d[["pair_id","interaction_count","cluster"]].itertuples(index=False):
        mat[row[pid], int(ic) - 1] = cl

    return mat

mat_iso   = make_mat(dfp[dfp["condition"] == "iso"])
mat_group = make_mat(dfp[dfp["condition"] == "group"])

fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

cmap = plt.get_cmap("tab20").copy()
cmap.set_bad("white")  # NaNs show as white

ax[0].imshow(mat_iso,   aspect="auto", interpolation="nearest", cmap=cmap)
ax[0].set_title("iso")
ax[0].set_xlabel("interaction_count")
ax[0].set_ylabel("(file, pair) rows")

ax[1].imshow(mat_group, aspect="auto", interpolation="nearest", cmap=cmap)
ax[1].set_title("group")
ax[1].set_xlabel("interaction_count")
ax[1].set_ylabel("")


# get unique clusters actually present
clusters = np.unique(dfp["cluster"])
clusters = clusters[~np.isnan(clusters)].astype(int)

# colorbar
cbar = fig.colorbar(
    ax[0].images[0],   # use one of the imshow images
    ax=ax,
    fraction=0.03,
    pad=0.02
)

cbar.set_label("Cluster")
cbar.set_ticks(clusters)
cbar.set_ticklabels([str(c) for c in clusters])
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_cluster_transition_matrix_pairs.png', dpi=300)
plt.close()




###### DURATION OF CLUSTERS ######

# SHORT = < 2
# MEDIUM = 2 - 5
# LONG = > 5


cluster_map = {
    1: "medium",
    2: "short",
    3: "long",
    4: "long",
    5: "long",
    6: "short",
    7: "medium",
    8: "short",
    9: "medium",
    10: "short",
    11: "medium",
    12: "long",}

dfp["interaction_class"] = dfp["cluster"].map(cluster_map)

dfp["class_code"] = pd.Categorical(
    dfp["interaction_class"],
    categories=["short", "medium", "long"],
    ordered=True
).codes




def make_mat(d):
    d = d.copy()
    d["pair_id"] = d["file"].astype(str) + " | " + d["Interaction Pair"].astype(str)

    # total length
    lengths = d.groupby("pair_id")["interaction_count"].max()

    # first interaction class
    first_class = (
        d.sort_values("interaction_count")
         .groupby("pair_id")["class_code"]
         .first()
    )

    order = (
        pd.DataFrame({"len": lengths, "first": first_class})
        .sort_values(["len", "first"], ascending=[False, True])
        .index
        .to_list()
    )

    max_len = int(lengths.max())
    mat = np.full((len(order), max_len), np.nan)
    row = {pid: i for i, pid in enumerate(order)}

    for pid, ic, code in d[["pair_id", "interaction_count", "class_code"]].itertuples(index=False):
        mat[row[pid], ic - 1] = code

    return mat

mat_iso   = make_mat(dfp[dfp["condition"] == "iso"])
mat_group = make_mat(dfp[dfp["condition"] == "group"])

fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

cmap = plt.get_cmap("viridis", 3).copy()   # 3 discrete colors
cmap.set_bad("white")

im0 = ax[0].imshow(mat_iso, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
ax[0].set_title("iso")
ax[0].set_xlabel("interaction_count")
ax[0].set_ylabel("(file, pair)")

im1 = ax[1].imshow(mat_group, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
ax[1].set_title("group")
ax[1].set_xlabel("interaction_count")

# legend as colorbar with labels
cbar = fig.colorbar(im0, ax=ax, fraction=0.03, pad=0.02)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["short", "medium", "long"])
cbar.set_label("Interaction class")

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_duration_transition_matrix_pairs_durations.png', dpi=300)
plt.close()



# --- TIME RASTER of interaction duration class (short/medium/long) ---

# 1) ensure dfp has the class_code already (you do above)
# dfp columns we need: file, condition, Interaction Pair, Frame (or time), class_code

dfp_time = dfp.copy()
dfp_time["pair_id"] = dfp_time["file"].astype(str) + " | " + dfp_time["Interaction Pair"].astype(str)

# choose x-axis: use "Frame" (recommended) or swap to "time" if you have it
time_col = "Frame"   # <- change to "time" if that's what you want

def make_time_mat(d, time_col="Frame"):
    d = d.copy()

    # row order: most interactions at top (same as before)
    order = d.groupby("pair_id")["interaction_count"].max().sort_values(ascending=False)
    pair_ids = order.index.to_list()
    row = {pid: i for i, pid in enumerate(pair_ids)}

    # make a continuous time axis (min..max) so “rest is white”
    tmin = int(d[time_col].min())
    tmax = int(d[time_col].max())
    T = tmax - tmin + 1

    mat = np.full((len(pair_ids), T), np.nan)

    # paint only event times; collisions (same pair + same frame) keep the last one
    for pid, t, code in d[["pair_id", time_col, "class_code"]].itertuples(index=False):
        mat[row[pid], int(t) - tmin] = code

    return mat, tmin, tmax

mat_iso_t, tmin_iso, tmax_iso = make_time_mat(dfp_time[dfp_time["condition"]=="iso"], time_col=time_col)
mat_grp_t, tmin_grp, tmax_grp = make_time_mat(dfp_time[dfp_time["condition"]=="group"], time_col=time_col)

# 2) plot (same colours + white background)
fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

from matplotlib.colors import ListedColormap

# custom bright colours for short / medium / long
cmap = ListedColormap([
    "#A158B3",  # short (purple)
    "#21918c",  # medium (teal)
    "#fde725"   # long (yellow)
])
cmap.set_bad("black")   # NaNs (no interaction) = black

im0 = ax[0].imshow(mat_iso_t, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
ax[0].set_title("iso")
ax[0].set_xlabel(time_col)
ax[0].set_ylabel("(file, pair)")

im1 = ax[1].imshow(mat_grp_t, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
ax[1].set_title("group")
ax[1].set_xlabel(time_col)

# optional: label x ticks in real frame numbers (sparse so it stays readable)
for a, tmin, tmax in [(ax[0], tmin_iso, tmax_iso), (ax[1], tmin_grp, tmax_grp)]:
    n_ticks = 6
    ticks = np.linspace(0, (tmax - tmin), n_ticks).astype(int)
    a.set_xticks(ticks)
    a.set_xticklabels((ticks + tmin).astype(int))

cbar = fig.colorbar(im0, ax=ax, fraction=0.03, pad=0.02)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["short", "medium", "long"])
cbar.set_label("Interaction class")

plt.savefig(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_duration_over_time.png",
    dpi=300
)
plt.close()












cluster_map = {
    1: "head-head",
    2: "head-head",
    3: "head-head-sustained",
    4: "head-head-sustained",
    5: "tailing",
    6: "pass-n-pause",
    7: "pass-n-follow",
    8: "passing",
    9: "passing",
    10: "pass-n-pause",
    11: "pass-n-follow",
    12: "tailing"
    }
dfp["interaction_type"] = dfp["cluster"].map(cluster_map)
dfp["type_code"] = pd.Categorical(
    dfp["interaction_type"],
    categories=["head-head", "tailing", "passing", "head-head-sustained", "pass-n-pause", "pass-n-follow"],
    ordered=True
).codes


def make_mat(d):
    d = d.copy()
    d["pair_id"] = d["file"].astype(str) + " | " + d["Interaction Pair"].astype(str)

    order = d.groupby("pair_id")["interaction_count"].max().sort_values(ascending=False)
    pair_ids = order.index.to_list()
    max_len = int(order.max())

    mat = np.full((len(pair_ids), max_len), np.nan)
    row = {pid: i for i, pid in enumerate(pair_ids)}

    for pid, ic, code in d[["pair_id", "interaction_count", "type_code"]].itertuples(index=False):
        mat[row[pid], int(ic) - 1] = code

    return mat


mat_iso   = make_mat(dfp[dfp["condition"] == "iso"])
mat_group = make_mat(dfp[dfp["condition"] == "group"])

n_types = 6  # head-head, head-head-sustained, tailing, passing, pass-n-pause, pass-n-follow

cmap = plt.get_cmap("tab10", n_types).copy()
cmap.set_bad("white")

fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

cmap = plt.get_cmap("viridis", 6).copy()   # 6 discrete colors
cmap.set_bad("white")

im0 = ax[0].imshow(mat_iso, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=n_types - 1)
ax[0].set_title("iso")
ax[0].set_xlabel("interaction_count")
ax[0].set_ylabel("(file, pair)")

im1 = ax[1].imshow(mat_group, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=n_types - 1)
ax[1].set_title("group")
ax[1].set_xlabel("interaction_count")

cbar = fig.colorbar(im0, ax=ax, fraction=0.03, pad=0.02)

cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels([
    "head-head",
    "head-head-sustained",
    "tailing",
    "passing",
    "pass-n-pause",
    "pass-n-follow"
])

cbar.set_label("Interaction type")

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/types_interactions.png', dpi=300)
plt.close()
