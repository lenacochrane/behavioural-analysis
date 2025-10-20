import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# --- load + prep ---
df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt/merged.csv')

df["both_best_score"] = (
    df["best_score"].astype(str).str.lower().eq("yes") &
    df["best_score_reverse"].astype(str).str.lower().eq("yes")
)

y_cols = ["total", "disease model", "mouse", "human", "cell line", "drosophila"]
for col in y_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

outdir = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-genes'
os.makedirs(outdir, exist_ok=True)

sns.set_style("whitegrid")

for y in y_cols:
    if y not in df.columns:
        continue

    # -------- TOP BARS: one row per Gene (NO aggregation) --------
    # keep the row with the largest value of y for that gene
    top = (df.sort_values(y, ascending=False)
             .drop_duplicates(subset="Gene", keep="first")
             .loc[:, ["Gene", y, "both_best_score"]]
             .dropna(subset=[y]))

    if top.empty:
        continue

    # order genes by this y descending for plotting
    gene_order = top.sort_values(y, ascending=False)["Gene"].tolist()
    top = top.set_index("Gene").loc[gene_order].reset_index()

    # -------- BOTTOM STRIPES: disease × gene membership (uses FULL df) --------
    mat = (df.assign(flag=1)
             .pivot_table(index="Disease", columns="Gene", values="flag",
                          aggfunc="max", fill_value=0)
             .reindex(columns=gene_order))

    diseases = mat.index.tolist()
    n_genes  = len(gene_order)
    n_dis    = len(diseases)

    # --- figure layout ---
    fig_w = max(14, 0.4 * n_genes)
    fig_h = 6 + min(2.5, 0.22 * n_dis)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.8], hspace=0.08)

    # --- top: bars with hue on both_best_score ---
    ax_top = fig.add_subplot(gs[0, 0])
    sns.barplot(
        data=top, x="Gene", y=y,
        hue="both_best_score",
        palette={True: "crimson", False: "steelblue"},
        edgecolor="black", dodge=False, ax=ax_top
    )
    ax_top.set_ylabel(f"{y.title()} PubMed count")
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(title="Both best score", loc="upper right")

    # annotate a few tallest bars to avoid clutter (top 15)
    for p in ax_top.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax_top.text(p.get_x() + p.get_width()/2, h,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)


    # --- bottom: color-coded disease stripes aligned to genes ---
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    bg = np.array([0.95, 0.95, 0.95])
    palette = sns.color_palette("Set2", len(diseases))
    img = np.tile(bg, (len(diseases), n_genes, 1))
    M = mat.values.astype(bool)
    for i, color in enumerate(palette):
        img[i, M[i], :] = color

    ax_bot.imshow(img, aspect='auto', interpolation='nearest', origin='upper')
    ax_bot.set_yticks(np.arange(len(diseases)))
    ax_bot.set_yticklabels(diseases, fontsize=10)
    ax_bot.set_xticks(np.arange(n_genes))
    ax_bot.set_xticklabels(gene_order, rotation=90, fontsize=8)
    ax_bot.set_xlabel("Gene")
    for spine in ax_bot.spines.values():
        spine.set_visible(False)

    # legend for disease colors
    handles = [Patch(facecolor=palette[i], edgecolor='none', label=diseases[i]) for i in range(len(diseases))]
    ax_bot.legend(handles=handles, title="Disease stripes",
                  bbox_to_anchor=(1.005, 1.0), loc="upper left", borderaxespad=0.)

    plt.tight_layout()

    # save (clean filename)
    safe_y = y.replace(" ", "_")
    outfile = os.path.join(outdir, f'genes__{safe_y}__with_disease_stripes.png')
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved all plots to: {outdir}")
