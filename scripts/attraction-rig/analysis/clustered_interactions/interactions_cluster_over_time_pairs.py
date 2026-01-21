
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
from matplotlib.colors import ListedColormap

# ------------------------------------------------------
# DEFINING CLUSTER TO CLUSTER TRANSITIONS FOR TRACK PAIRS
# ------------------------------------------------------
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
df_pairs = df_pairs.sort_values(["condition", "file", "Interaction Pair", "Frame", "Interaction Number"])
df_pairs["interaction_count"] = (df_pairs.groupby(["file", "condition", "Interaction Pair"]).cumcount() + 1)
df_pairs = df_pairs.rename(columns={cluster_name: "cluster"})
df_pairs.to_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/interaction_transition_matrix_pairs.csv', index=False)



# --------------------------------------------
# HEATMAP: TRANSITION BETWEEN DURATION CLASSES
# --------------------------------------------
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

df_durT = df_durT.sort_values(
    ["condition", "file", "Interaction Pair", "Frame", "Interaction Number"]
)

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
plt.savefig(os.path.join( "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_iso_vs_group_heatmaps.pdf"), format='pdf', bbox_inches="tight")
plt.close()


# -------------------------------------------------------
# HEATMAP: DURATION TRANSITON DIFFERENCE BETWEEN GH AND SI
# -------------------------------------------------------
# P_diff = P_grp - P_iso
P_iso = P_iso.reindex(index=dur_order, columns=dur_order, fill_value=0)
P_grp = P_grp.reindex(index=dur_order, columns=dur_order, fill_value=0)
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
plt.savefig(os.path.join('//Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_transition_diff_heatmap.pdf'), format='pdf', bbox_inches="tight")
plt.close()


# -------------------------------------------------
# CIRCLE GRAPH: TRANSITION BETWEEN DURATION CLASSES
# -------------------------------------------------
# ---------- fair weighting (row support per condition) ---------- downweight per row 
# support_iso = C_iso.sum(axis=1).astype(float)
# support_grp = C_grp.sum(axis=1).astype(float)

# W_iso = support_iso / support_iso.max() if support_iso.max() > 0 else support_iso * 0.0
# W_grp = support_grp / support_grp.max() if support_grp.max() > 0 else support_grp * 0.0
# W_fair = 0.5 * (W_iso + W_grp)

# D = P_diff.mul(W_fair, axis=0)

D = P_diff.copy()

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
ax.set_title("Transition Likelihood Difference")
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
cax = fig.add_axes([0.95, 0.28, 0.02, 0.45])  # small bar
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("P(group) − P(iso)")

plt.savefig(os.path.join("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_transition_diff_circlegraph.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/durclass_transition_diff_circlegraph.pdf"), 
            format='pdf', bbox_inches="tight")
plt.close()


# ------------------------------------
# BOXPLOT: INTERACTION COUNTS PER PAIR
# ------------------------------------
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


# ----------------------------------------------------------------------------
# BOXPLOT: INTERACTION COUNTS PER PAIR OVER BINNED NUMBER OF PREV INTERACTIONS
# ----------------------------------------------------------------------------
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



# --------------------------------------------------------------------------------------
# BOXPLOT: INTERACTION DURATION! COUNTS PER PAIR OVER BINNED NUMBER OF PREV INTERACTIONS
# --------------------------------------------------------------------------------------
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



# -----------------------------------------------------------
# RASTA: CLUSTER TRANSITIONS OVER INTERACTION COUNT, PER PAIR
# -----------------------------------------------------------
dfp = df_pairs.copy()
dfp["pair_id"] = dfp["file"].astype(str) + " | " + dfp["Interaction Pair"].astype(str)
dfp["cluster"] = dfp["cluster"].astype(int)  # if already int, fine

# def make_mat(d):
#     # order rows by longest -> shortest (max interaction_count)
#     order = d.groupby("pair_id")["interaction_count"].max().sort_values(ascending=False)
#     pair_ids = order.index.to_list()
#     max_len = int(order.max())

#     mat = np.full((len(pair_ids), max_len), np.nan)  # NaN = no data
#     row = {pid: i for i, pid in enumerate(pair_ids)}

#     # fill: col = interaction_count-1, value = cluster
#     for pid, ic, cl in d[["pair_id","interaction_count","cluster"]].itertuples(index=False):
#         mat[row[pid], int(ic) - 1] = cl
#     return mat

def make_mat_sorted_summary(d, n_early=3):
    # build per-pair cluster sequence in correct time order
    seq = d.groupby("pair_id")["cluster"].apply(list)

    # summary features for sorting
    def mode_or_neg1(x):
        vc = pd.Series(x).value_counts()
        return int(vc.index[0]) if len(vc) else -1

    summary = pd.DataFrame({
        "len": seq.apply(len),
        "first": seq.apply(lambda x: x[0] if len(x) else -1),
        "early": seq.apply(lambda x: tuple(x[:n_early] + [-1]*(n_early-len(x))) if len(x) else tuple([-1]*n_early)),
        "mode": seq.apply(mode_or_neg1),
    })

    # sort: early pattern -> dominant cluster -> longer sequences first
    summary = summary.sort_values(["early", "mode", "len"], ascending=[True, True, False])

    # build matrix
    pair_ids = summary.index.to_list()
    max_len = int(summary["len"].max())
    mat = np.full((len(pair_ids), max_len), np.nan)

    row = {pid: i for i, pid in enumerate(pair_ids)}
    for pid, clist in seq.items():
        r = row.get(pid, None)
        if r is None:
            continue
        mat[r, :len(clist)] = clist

    return mat, summary

mat_iso , summary_iso  = make_mat_sorted_summary(dfp[dfp["condition"] == "iso"])
mat_group , summary_group = make_mat_sorted_summary(dfp[dfp["condition"] == "group"])

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
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/RASTA_clusters.png', dpi=300)
plt.close()


# -------------------------------------------------------------------
# RASTA: CLUSTER DURATION TRANSITIONS OVER INTERACTION COUNT, PER PAIR
# -------------------------------------------------------------------

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




def make_mat_sorted(d):
    d = d.copy()
    d["pair_id"] = d["file"].astype(str) + " | " + d["Interaction Pair"].astype(str)

    # CRITICAL: ensure interaction_count order within each pair
    d = d.sort_values(["pair_id", "interaction_count"])

    summary = (
        d.groupby("pair_id")
         .agg(
             n=("interaction_count", "max"),          # length of sequence
             mean_code=("class_code", "mean"),        # mostly short -> mostly long
             frac_long=("class_code", lambda x: (x == 2).mean()),  # optional tie-break
             frac_short=("class_code", lambda x: (x == 0).mean()), # optional tie-break
         )
         .sort_values(["mean_code", "frac_long", "n"], ascending=[True, True, False])
    )

    order = summary.index.to_list()
    max_len = int(summary["n"].max())

    mat = np.full((len(order), max_len), np.nan)
    row = {pid: i for i, pid in enumerate(order)}

    for pid, ic, code in d[["pair_id", "interaction_count", "class_code"]].itertuples(index=False):
        mat[row[pid], int(ic) - 1] = int(code)

    return mat


mat_iso   = make_mat_sorted(dfp[dfp["condition"] == "iso"])
mat_group = make_mat_sorted(dfp[dfp["condition"] == "group"])

fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

# cmap = plt.get_cmap("viridis", 3).copy()   # 3 discrete colors

cmap = ListedColormap([
    "#eac7e2",  # short (purple)
    "#cd68b5",  # medium (green)
    "#ff00c3",  # long (orange)
])
cmap.set_bad("black")

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

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/RASTA_durations.png', dpi=300)
plt.close()





# ---------------------------------------------
# RASTA: CLUSTER DURATION TRANSITIONS OVER TIME
# ---------------------------------------------

# 1) ensure dfp has the class_code already (you do above)
# dfp columns we need: file, condition, Interaction Pair, Frame (or time), class_code

dfp_time = dfp.copy()
dfp_time["pair_id"] = dfp_time["file"].astype(str) + " | " + dfp_time["Interaction Pair"].astype(str)

# choose x-axis: use "Frame" (recommended) or swap to "time" if you have it
time_col = "Frame"   # <- change to "time" if that's what you want


def make_time_mat(d, time_col="Frame", expand=3):  # expand in frames (3 => ~7px wide)
    d = d.copy()

    order = d.groupby("pair_id")["interaction_count"].max().sort_values(ascending=False)
    pair_ids = order.index.to_list()
    row = {pid: i for i, pid in enumerate(pair_ids)}

    tmin = int(d[time_col].min())
    tmax = int(d[time_col].max())
    T = tmax - tmin + 1

    mat = np.full((len(pair_ids), T), np.nan)

    for pid, t, code in d[["pair_id", time_col, "class_code"]].itertuples(index=False):
        tt = int(t) - tmin
        lo = max(0, tt - expand)
        hi = min(T, tt + expand + 1)
        mat[row[pid], lo:hi] = code   # paint a band

    return mat, tmin, tmax


mat_iso_t, tmin_iso, tmax_iso = make_time_mat(dfp_time[dfp_time["condition"]=="iso"], time_col=time_col)
mat_grp_t, tmin_grp, tmax_grp = make_time_mat(dfp_time[dfp_time["condition"]=="group"], time_col=time_col)

# 2) plot (same colours + white background)
fig, ax = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)



# custom bright colours for short / medium / long
cmap = ListedColormap([
    "#eac7e2",  # short (purple)
    "#cd68b5",  # medium (green)
    "#ff00c3",  # long (orange)
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



# ------------------------------
# RASTA: "TYPES OF INTERACTIONS"
# ------------------------------
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





# -------------------------------------
# TIME BETWEEN INTERACTIONS VS DURATION
# -------------------------------------

dur_map = {
    1: "medium", 2: "short", 3: "long", 4: "long", 5: "long", 6: "short",
    7: "medium", 8: "short", 9: "medium", 10: "short", 11: "medium", 12: "long",
}
dur_order = ["short", "medium", "long"]

df_gap = df_pairs.copy()
df_gap["cluster"] = df_gap["cluster"].astype(int)

# within-pair temporal order
df_gap = df_gap.sort_values(["file", "condition", "Interaction Pair", "Frame", "Interaction Number"])

df_gap["next_frame"] = df_gap.groupby(["file", "condition", "Interaction Pair"])["Frame"].shift(-1)
df_gap["next_cluster"] = df_gap.groupby(["file", "condition", "Interaction Pair"])["cluster"].shift(-1)

df_gap["dt_frames"] = df_gap["next_frame"] - df_gap["Frame"]
df_gap["dt_frames"] = df_gap["dt_frames"].astype(float)
df_gap = df_gap.dropna(subset=["dt_frames", "next_cluster"]).copy()
df_gap = df_gap[df_gap["dt_frames"] > 0].copy()

df_gap["next_cluster"] = df_gap["next_cluster"].astype(int)
df_gap["next_dur"] = df_gap["next_cluster"].map(dur_map)
df_gap = df_gap.dropna(subset=["next_dur"]).copy()


bin_width = 600  # seconds
max_t = int(df_gap["dt_frames"].max())
sec_bins = np.arange(0, max_t + bin_width + 1, bin_width)
df_gap["gap_bin"] = pd.cut(
    df_gap["dt_frames"],
    bins=sec_bins,
    right=False,
    include_lowest=True
)


# per-file probs
tmp = (
    df_gap
    .groupby(["file", "condition", "gap_bin", "next_dur"])
    .size()
    .reset_index(name="count")
)

tmp["prob"] = tmp["count"] / tmp.groupby(["file", "condition", "gap_bin"])["count"].transform("sum")

# fill missing dur classes with 0 via merge grid (avoids your IntervalIndex error)
files = tmp["file"].unique()
conds = tmp["condition"].unique()
gap_bins = df_gap["gap_bin"].cat.categories  # <-- THIS is the missing piece

grid = pd.MultiIndex.from_product(
    [files, conds, gap_bins, dur_order],
    names=["file", "condition", "gap_bin", "next_dur"]
).to_frame(index=False)

tmp = grid.merge(tmp, on=["file", "condition", "gap_bin", "next_dur"], how="left")
tmp["count"] = tmp["count"].fillna(0)
tmp["prob"] = tmp["prob"].fillna(0)

# plot: facet by condition, hue by next_dur
g = sns.catplot(
    data=tmp,
    x="gap_bin",
    y="prob",
    hue="next_dur",
    hue_order=dur_order,
    col="condition",
    kind="point",
    errorbar='sd',
    dodge=0.35,
    height=4,
    aspect=1.25,
)

g.set_axis_labels("Time gap between meetings")
g.set_titles("{col_name}")
for ax in g.axes.flat:
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

plt.tight_layout()
plt.savefig(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/"
    "n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs/"
    "time_between_interactions_duration.png",
    dpi=300
)
plt.close()




# ---------------------------------------------------------------------
# INTERACTION DEPENDANT ON WHETHER THEY HAVE INTERACTED IN THE MEANTIME
# ---------------------------------------------------------------------

outdir = "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/youngser_2/idt1/transitions_between_clusters_pairs"
os.makedirs(outdir, exist_ok=True)

dur_map = {
    1: "medium", 2: "short", 3: "long", 4: "long", 5: "long", 6: "short",
    7: "medium", 8: "short", 9: "medium", 10: "short", 11: "medium", 12: "long",
}
dur_order = ["short", "medium", "long"]
between_order = ["neither", "one", "both"]

# --------------------------
# 0) Start from df_pairs
# --------------------------
base = df_pairs.copy()
base = base.dropna(subset=["file", "condition", "Interaction Pair", "Frame", "cluster"]).copy()
base["cluster"] = base["cluster"].astype(int)

# parse "(3, 10)" robustly
pair_nums = base["Interaction Pair"].astype(str).str.findall(r"-?\d+")
good = pair_nums.apply(lambda xs: len(xs) >= 2)
base = base.loc[good].copy()
base["id1"] = pair_nums.loc[good].apply(lambda xs: int(xs[0]))
base["id2"] = pair_nums.loc[good].apply(lambda xs: int(xs[1]))

# unordered pair key
base["a"] = base[["id1", "id2"]].min(axis=1)
base["b"] = base[["id1", "id2"]].max(axis=1)
base["pair_key"] = base["a"].astype(str) + "-" + base["b"].astype(str)

# dedupe events so each row is one meeting at one time for one unordered pair
dedupe_cols = ["file", "condition", "Frame", "pair_key"]
if "Interaction Number" in base.columns:
    dedupe_cols.append("Interaction Number")

base = base.sort_values(dedupe_cols).drop_duplicates(subset=dedupe_cols).copy()

# global event ordering per file+condition
sort_cols = ["file", "condition", "Frame"]
if "Interaction Number" in base.columns:
    sort_cols.append("Interaction Number")

base = base.sort_values(sort_cols).reset_index(drop=True)
base["event_idx"] = base.groupby(["file", "condition"]).cumcount().astype(int)

# --------------------------
# 1) Compute NEXT meeting of same pair (on FULL base timeline)
# --------------------------
meet = base.sort_values(["file", "condition", "pair_key", "event_idx"]).copy()
meet["next_event_idx"] = meet.groupby(["file", "condition", "pair_key"])["event_idx"].shift(-1)
meet["next_cluster"]   = meet.groupby(["file", "condition", "pair_key"])["cluster"].shift(-1)

meet = meet.dropna(subset=["next_event_idx", "next_cluster"]).copy()
meet["next_event_idx"] = meet["next_event_idx"].astype(int)
meet["next_cluster"]   = meet["next_cluster"].astype(int)
meet["next_dur"]       = meet["next_cluster"].map(dur_map)
meet = meet.dropna(subset=["next_dur"]).copy()
meet["next_dur"] = pd.Categorical(meet["next_dur"], categories=dur_order, ordered=True)

# --------------------------
# 2) Per file+condition: build per-larva cumulative counts over FULL timeline
#    then classify between_cat for each meet row.
# --------------------------
rows = []

for (f, cond), events in base.groupby(["file", "condition"], sort=False):
    events = events.sort_values("event_idx").copy()
    n_events = int(events["event_idx"].max()) + 1  # FULL timeline length

    id1_arr = events["id1"].to_numpy()
    id2_arr = events["id2"].to_numpy()
    ev_arr  = events["event_idx"].to_numpy()

    larvae = pd.unique(pd.concat([events["id1"], events["id2"]], ignore_index=True)).astype(int)

    # build larva_cum[L] arrays length = n_events
    larva_cum = {}
    for L in larvae:
        present = ((id1_arr == L) | (id2_arr == L)).astype(int)
        full = np.zeros(n_events, dtype=int)
        full[ev_arr] = present
        larva_cum[L] = np.cumsum(full)

    # meetings for this file+cond
    msub = meet[(meet["file"] == f) & (meet["condition"] == cond)].copy()
    if msub.empty:
        continue

    for r in msub.itertuples(index=False):
        curr = int(r.event_idx)
        nxt  = int(r.next_event_idx)
        query = nxt - 1  # last event strictly before next meeting

        # safety
        if query < curr:
            continue
        if query >= n_events:
            # should never happen now, but keep safe
            query = n_events - 1

        A = int(r.id1)
        B = int(r.id2)

        # interactions strictly between (curr, nxt): cum[query] - cum[curr]
        A_between = larva_cum[A][query] - larva_cum[A][curr]
        B_between = larva_cum[B][query] - larva_cum[B][curr]

        a_other = A_between > 0
        b_other = B_between > 0

        if (not a_other) and (not b_other):
            between_cat = "neither"
        elif a_other ^ b_other:
            between_cat = "one"
        else:
            between_cat = "both"

        rows.append({
            "file": f,
            "condition": cond,
            "pair_key": r.pair_key,
            "event_idx": curr,
            "next_event_idx": nxt,
            "A_between": int(A_between),
            "B_between": int(B_between),
            "between_cat": between_cat,
            "next_dur": r.next_dur
        })

wide = pd.DataFrame(rows)
wide["between_cat"] = pd.Categorical(wide["between_cat"], categories=between_order, ordered=True)
wide["next_dur"] = pd.Categorical(wide["next_dur"], categories=dur_order, ordered=True)

# --------------------------
# 3) Per-file probs (equal movie weighting), then seaborn pointplot
# --------------------------
counts = (
    wide.groupby(["file", "condition", "between_cat", "next_dur"])
        .size()
        .reset_index(name="count")
)

# Fill missing combos with 0
files = wide["file"].unique()
conds = wide["condition"].unique()
grid = pd.MultiIndex.from_product(
    [files, conds, between_order, dur_order],
    names=["file", "condition", "between_cat", "next_dur"]
).to_frame(index=False)

counts = grid.merge(counts, on=["file", "condition", "between_cat", "next_dur"], how="left")
counts["count"] = counts["count"].fillna(0)

counts["prob"] = (
    counts["count"] /
    counts.groupby(["file", "condition", "between_cat"])["count"].transform("sum").replace(0, np.nan)
).fillna(0)

g = sns.catplot(
    data=counts,
    x="between_cat",
    y="prob",
    hue="next_dur",
    hue_order=dur_order,
    col="condition",
    kind="point",
    order=between_order,
    dodge=0.35,
    errorbar="sd",
    height=4,
    aspect=1.25,
)

g.set_axis_labels("Other interactions between pair meetings", "P(next duration)")
g.set_titles("{col_name}")
for ax in g.axes.flat:
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "between_activity_nextdur_pointplot.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(outdir, "between_activity_nextdur_pointplot.pdf"), format="pdf", bbox_inches="tight")
plt.close()

print(wide["between_cat"].value_counts(dropna=False))
print(wide.groupby(["condition","between_cat"])["next_dur"].value_counts(normalize=True))

wide.to_csv(os.path.join(outdir, "between_activity_table.csv"), index=False)



