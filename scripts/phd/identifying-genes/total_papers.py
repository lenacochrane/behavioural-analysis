import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/refined_attempt/merged.csv')

df["both_best_score"] = ((df["best_score"].str.lower() == "yes") &(df["best_score_reverse"].str.lower() == "yes")) ## rank == high
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
for col in ["total", "disease model", "mouse", "human", "cell line", "drosophila"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# df = df[df['Disease'] == 'Schizophrenia (SCZ)']

# # order genes by Rank (smallest → largest)
# gene_order = df.sort_values('Rank')['Gene'].tolist()

# # barplot
# plt.figure(figsize=(12, 6))
# ax = sns.barplot(
#     data=df,
#     x='Gene',
#     y='total',
#     order=gene_order,
#     color='steelblue',
#     edgecolor='black',
#     hue='both_best_score',
#     ci=None,           # no confidence intervals
#     estimator=sum      # if a gene appears more than once, sum totals
# )

# ax.set_xlabel('Gene (ordered by Rank)')
# ax.set_ylabel('Total PubMed papers')
# ax.set_title('Schizophrenia (SCZ): total papers by gene (Rank 1→15)')
# ax.tick_params(axis='x', rotation=60)

# # (optional) annotate values on bars
# for p in ax.patches:
#     ax.text(p.get_x() + p.get_width()/2, p.get_height(),
#             f"{int(p.get_height()) if pd.notnull(p.get_height()) else ''}",
#             ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.show()

directory = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-genes'


y_cols = ["total", "disease model", "mouse", "human", "cell line", "drosophila"]

# --- loop per disease ---
for disease, dsub in df.groupby("Disease", dropna=True):
    dsub = dsub.dropna(subset=["Rank"]).copy()
    if dsub.empty:
        continue

    # order genes by Rank within this disease
    gene_order = dsub.sort_values("Rank")["Gene"].tolist()

    # --- loop per y column ---
    for y in y_cols:
        if y not in dsub.columns:
            continue
        dplot = dsub.dropna(subset=[y]).copy()
        if dplot.empty:
            continue

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            data=dplot,
            x='Gene',
            y=y,
            order=gene_order,
            edgecolor='black',
            hue='both_best_score',   # your hue, as requested
            ci=None,                 # for seaborn <0.12; harmless in >=0.12
            estimator=sum            # if a gene appears multiple times, sum
            # (leave dodge=True default: you'll see two bars per gene by hue)
        )

        ax.set_xlabel('Gene (ordered by Rank)')
        ax.set_ylabel(f'PubMed count for {y}')
        ax.tick_params(axis='x', rotation=60)
        ax.set_title(f'{disease} — {y}')
        ax.legend(title="Both best score")

        # annotate bars
        for p in ax.patches:
            h = p.get_height()
            if pd.notnull(h):
                ax.text(
                    p.get_x() + p.get_width()/2, h,
                    f"{int(h)}",
                    ha='center', va='bottom', fontsize=8
                )

        plt.tight_layout()



        safe_disease = str(disease).replace("/", "_").replace("\\", "_")
        safe_y = y.replace(" ", "_")
        fname = f"{safe_disease}__{safe_y}.png"
        output = os.path.join(directory, fname)
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()