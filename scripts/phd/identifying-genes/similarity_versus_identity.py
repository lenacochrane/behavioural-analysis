import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors  # pip install mplcursors



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt/merged.csv')

df["both_best_score"] = (
    (df["best_score"].str.lower() == "yes") &
    (df["best_score_reverse"].str.lower() == "yes")
)

high = df[df["rank"].astype(str).str.lower().eq("high")]

plt.figure(figsize=(10,8))
sns.scatterplot(
    data=df,
    x="identity_percent",
    y="similarity_percent",
    s=60,
    hue="both_best_score",
    # color="royalblue",
    edgecolor="white"
)


plt.xlabel("Identity (%)")
plt.ylabel("Similarity (%)")
plt.title("Fly Orthologues: Identity vs Similarity")

# --- small static text labels next to each point ---
for _, row in df.iterrows():
    plt.text(
        row["identity_percent"] + 0.4,  # small x-offset so text doesn't overlap the dot
        row["similarity_percent"],
        row["Gene"],
        fontsize=7,
        alpha=0.7
    )

# --- hover tooltips (larger, more readable) ---
cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    i = sel.index
    gene = df.iloc[i]["Gene"]
    sel.annotation.set_text(gene)
    sel.annotation.get_bbox_patch().set(alpha=0.9, color="white")

plt.tight_layout()
plt.show()



outdir = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-genes'
os.makedirs(outdir, exist_ok=True)

for disease in df["Disease"].dropna().unique():
    sub = df[df["Disease"] == disease]

    plt.figure(figsize=(10,8))
    sns.scatterplot(
        data=sub,
        x="identity_percent",
        y="similarity_percent",
        s=60,
        hue="both_best_score",
        edgecolor="white"
    )
    plt.xlabel("Identity (%)")
    plt.ylabel("Similarity (%)")
    plt.title(f"Fly Orthologues: Identity vs Similarity — {disease}")

    # small gene labels
    for _, row in sub.iterrows():
        plt.text(
            row["identity_percent"] + 0.4,
            row["similarity_percent"],
            row["Gene"],
            fontsize=7, alpha=0.7
        )

    plt.tight_layout()

    # save as PNG named by disease
    safe_name = str(disease).replace("/", "_").replace(" ", "_")
    outfile = os.path.join(outdir, f"scatter_{safe_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



