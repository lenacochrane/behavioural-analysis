
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc  # standard library for single-cell analysis
import gzip
from scipy.io import mmread
from scipy.sparse import csr_matrix
from anndata import AnnData # AnnData = a single-cell dataset structure
import seaborn as sns


"""

Single cell transcriptome atlas of the Drosophila larval brain
Clarisse Brunet Avalos, G Larisa Maier, Rémy Bruggmann, Simon G Sprecher

    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134722

        GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_matrix.mtx.gz
        GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_genes.tsv.gz
        GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_barcodes.tsv.gz

        saved to /Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE-EXPRESSION/L1-DATASET/dataset

        👉 4708 cells
        👉 17493 genes


"""

save_dir = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE-EXPRESSION/L1-DATASET"

data_dir = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE-EXPRESSION/L1-DATASET/dataset"

matrix_file = f"{data_dir}/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_matrix.mtx.gz"
genes_file = f"{data_dir}/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_genes.tsv.gz"
barcodes_file = f"{data_dir}/GSE134722_FirstInstarLarvalBrainNormalCondition_finalaggr_10X_barcodes.tsv.gz"

# Read matrix (genes x cells), then transpose to cells x genes
with gzip.open(matrix_file, "rb") as f:
    X = mmread(f).tocsr().T

# Read genes
with gzip.open(genes_file, "rt") as f:
    genes = [line.strip().split("\t") for line in f]

gene_ids = [g[0] for g in genes]
gene_symbols = [g[1] for g in genes]

# Read cell barcodes
with gzip.open(barcodes_file, "rt") as f:
    barcodes = [line.strip() for line in f]

# Build AnnData object
adata = AnnData(
    X=csr_matrix(X),
    obs=pd.DataFrame(index=barcodes),
    var=pd.DataFrame(
        {
            "gene_ids": gene_ids,
            "gene_symbols": gene_symbols
        },
        index=gene_symbols
    )
)

# Make duplicate gene names unique if needed
adata.var_names_make_unique()



genes_of_interest = [
    "Nrx-1", "para", "Atg17", "Cul1",
    "GluRIA", "Prosap", "Syx1A", "Nmdar2",
    "dlg1", "kis", "p120ctn", "Pp2C1",
    "trio", "ctrip", "Ca-alpha1T", "Iml1",
    "lid", "raskol"
]

# raskol is missing
missing = [g for g in genes_of_interest if g not in adata.var_names]
print("Missing:", missing)





genes_of_interest = [
    "Nrx-1", "para", "Atg17", "Cul1",
    "GluRIA", "Prosap", "Syx1A", "Nmdar2",
    "dlg1", "kis", "p120ctn", "Pp2C1",
    "trio", "ctrip", "Ca-alpha1T", "Iml1", "lid"
]



summary = []

for gene in genes_of_interest:
    x = adata[:, gene].X

    # convert sparse column to a flat numpy array
    x = x.toarray().flatten()

    n_cells = (x > 0).sum()
    frac_cells = n_cells / len(x)
    mean_all_cells = x.mean()
    mean_expressing_cells = x[x > 0].mean() if n_cells > 0 else 0

    summary.append({
        "gene": gene,
        "n_cells_expressing": int(n_cells),
        "fraction_cells_expressing": frac_cells,
        "mean_expression_all_cells": mean_all_cells,
        "mean_expression_expressing_cells": mean_expressing_cells
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{save_dir}/genes_of_interest_summary.csv", index=False)
print(summary_df)






# build binary matrix (cells × genes)
binary = {}

for gene in genes_of_interest:
    x = adata[:, gene].X.toarray().flatten()
    binary[gene] = (x > 0).astype(int)

binary_df = pd.DataFrame(binary)
binary_df.to_csv(f"{save_dir}/genes_of_interest_binary_matrix.csv", index=False)

corr = binary_df.corr()
print(corr)

corr_no_diag = corr.copy()
np.fill_diagonal(corr_no_diag.values, np.nan)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_no_diag, cmap="viridis")
plt.title("Gene Co-expression")
plt.savefig(f"{save_dir}/genes_of_interest_correlation_heatmap.png")
plt.close()







# broad cell-type markers
elav = adata[:, "elav"].X.toarray().flatten()
repo = adata[:, "repo"].X.toarray().flatten()
dpn  = adata[:, "dpn"].X.toarray().flatten()

cell_type = []

for i in range(adata.n_obs):
    if elav[i] > 0:
        cell_type.append("neuron")
    elif repo[i] > 0:
        cell_type.append("glia")
    elif dpn[i] > 0:
        cell_type.append("progenitor")
    else:
        cell_type.append("unknown")

adata.obs["cell_type"] = cell_type

print(adata.obs["cell_type"].value_counts())


results = []

cell_types = ["neuron", "glia", "progenitor", "unknown"]

for gene in genes_of_interest:
    gene_all = adata[:, gene].X.toarray().flatten()
    total_gene_positive = (gene_all > 0).sum()

    for ct in cell_types:
        mask = adata.obs["cell_type"] == ct
        x = adata[mask, gene].X.toarray().flatten()

        n_cells_in_type = len(x)
        n_gene_positive_in_type = (x > 0).sum()

        results.append({
            "gene": gene,
            "cell_type": ct,
            "fraction_of_celltype_expressing": n_gene_positive_in_type / n_cells_in_type if n_cells_in_type > 0 else 0,
            "fraction_of_gene_positive_cells_in_this_type": n_gene_positive_in_type / total_gene_positive if total_gene_positive > 0 else 0,
            "mean_expression_all_cells": x.mean(),
            "mean_expression_expressing_cells": x[x > 0].mean() if n_gene_positive_in_type > 0 else 0
        })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(f"{save_dir}/genes_of_interest_by_cell_type.csv", index=False)




import seaborn as sns
import matplotlib.pyplot as plt



# # 1. Fraction of each cell type expressing the gene
# frac_celltype_plot = results_df.pivot(
#     index="gene",
#     columns="cell_type",
#     values="within_cell_type_fraction"
# )

# plt.figure(figsize=(7, 8))
# sns.heatmap(frac_celltype_plot, cmap="viridis", annot=True, fmt=".2f")
# plt.title("Fraction of each cell type expressing each gene")
# plt.xlabel("Cell type")
# plt.ylabel("Gene")
# plt.tight_layout()
# plt.savefig(f"{save_dir}/genes_of_interest_fraction_of_celltype_expressing_heatmap.png", dpi=300)
# plt.close()




# 2. Fraction of gene-positive cells belonging to each cell type
frac_gene_plot = results_df.pivot(
    index="gene",
    columns="cell_type",
    values="fraction_of_gene_positive_cells_in_this_type"
)

plt.figure(figsize=(7, 8))
sns.heatmap(frac_gene_plot, cmap="cividis", annot=True, fmt=".2f")
plt.title("Fraction of gene-positive cells in each cell type")
plt.xlabel("Cell type")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(f"{save_dir}/genes_of_interest_fraction_of_gene_positive_cells_in_each_type_heatmap.png", dpi=300)
plt.close()



# 3. Mean expression among expressing cells only
mean_expr_plot = results_df.pivot(
    index="gene",
    columns="cell_type",
    values="mean_expression_expressing_cells"
)

plt.figure(figsize=(7, 8))
sns.heatmap(mean_expr_plot, cmap="magma", annot=True, fmt=".2f")
plt.title("Mean expression in expressing cells only")
plt.xlabel("Cell type")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(f"{save_dir}/genes_of_interest_mean_expression_expressing_cells_heatmap.png", dpi=300)
plt.close()
















neuron_mask = adata.obs["cell_type"] == "neuron"
adata_neurons = adata[neuron_mask].copy()

adata_neurons.write(f"{save_dir}/adata_neurons.h5ad")

import scanpy as sc

sc.pp.normalize_total(adata_neurons)
sc.pp.log1p(adata_neurons)

adata_neurons.raw = adata_neurons

sc.pp.highly_variable_genes(adata_neurons, n_top_genes=2000)
adata_neurons = adata_neurons[:, adata_neurons.var.highly_variable]

sc.pp.pca(adata_neurons)
sc.pp.neighbors(adata_neurons)

sc.tl.leiden(adata_neurons)

adata_neurons.obs["leiden"]

sc.tl.umap(adata_neurons)

sc.pl.umap(adata_neurons, color="leiden")
plt.savefig(f"{save_dir}/umap_leiden.png", dpi=300)
plt.show()




sc.pl.umap(
    adata_neurons,
    color=genes_of_interest,
    use_raw=True,   # 🔥 THIS is the key
    show=False
)

plt.savefig(f"{save_dir}/umap_genes_of_interest.png", dpi=300, bbox_inches="tight")
plt.close()










ChAT = adata[:, "ChAT"].X.toarray().flatten()
VGlut = adata[:, "VGlut"].X.toarray().flatten()
Gad1 = adata[:, "Gad1"].X.toarray().flatten()

subtype = []

for i in range(adata.n_obs):
    labels = []

    if ChAT[i] > 0:
        labels.append("cholinergic")
    if Gad1[i] > 0:
        labels.append("GABAergic")
    if VGlut[i] > 0:
        labels.append("glutamatergic")

    if len(labels) == 0:
        labels.append("other")

    # join labels into a single string
    subtype.append("_".join(labels))

adata.obs["neuron_subtype_marker_based"] = subtype




subtypes = adata.obs["neuron_subtype_marker_based"].unique()

results_subtype = []

for gene in genes_of_interest:
    gene_all = adata[:, gene].X.toarray().flatten()
    total_gene_positive = (gene_all > 0).sum()

    for st in subtypes:
        mask = adata.obs["neuron_subtype_marker_based"] == st
        x = adata[mask, gene].X.toarray().flatten()

        n_cells_in_subtype = len(x)
        n_gene_positive_in_subtype = (x > 0).sum()

        results_subtype.append({
            "gene": gene,
            "subtype": st,
            "fraction_of_subtype_expressing": n_gene_positive_in_subtype / n_cells_in_subtype if n_cells_in_subtype > 0 else 0,
            "fraction_of_gene_positive_cells_in_subtype": n_gene_positive_in_subtype / total_gene_positive if total_gene_positive > 0 else 0,
            "mean_expression_all_cells": x.mean(),
            "mean_expression_expressing_cells": x[x > 0].mean() if n_gene_positive_in_subtype > 0 else 0
        })

results_subtype_df = pd.DataFrame(results_subtype)
results_subtype_df.to_csv(f"{save_dir}/genes_of_interest_by_marker_based_subtype.csv", index=False)
print(results_subtype_df)












# 1. Fraction of each subtype expressing each gene
frac_subtype_plot = results_subtype_df.pivot(
    index="gene",
    columns="subtype",
    values="fraction_of_subtype_expressing"
)

plt.figure(figsize=(7, 8))
sns.heatmap(frac_subtype_plot, cmap="viridis", annot=True, fmt=".2f")
plt.title("Fraction of each marker-based subtype expressing each gene")
plt.xlabel("Subtype")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(f"{save_dir}/genes_of_interest_fraction_of_subtype_expressing_heatmap.png", dpi=300)
plt.show()


# 2. Fraction of gene-positive cells in each subtype
frac_gene_subtype_plot = results_subtype_df.pivot(
    index="gene",
    columns="subtype",
    values="fraction_of_gene_positive_cells_in_subtype"
)

plt.figure(figsize=(7, 8))
sns.heatmap(frac_gene_subtype_plot, cmap="cividis", annot=True, fmt=".2f")
plt.title("Fraction of gene-positive cells in each marker-based subtype")
plt.xlabel("Subtype")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(f"{save_dir}/genes_of_interest_fraction_of_gene_positive_cells_in_subtype_heatmap.png", dpi=300)
plt.show()


# 3. Mean expression in expressing cells
mean_subtype_plot = results_subtype_df.pivot(
    index="gene",
    columns="subtype",
    values="mean_expression_expressing_cells"
)

plt.figure(figsize=(7, 8))
sns.heatmap(mean_subtype_plot, cmap="magma", annot=True, fmt=".2f")
plt.title("Mean expression in expressing cells by marker-based subtype")
plt.xlabel("Subtype")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(f"{save_dir}/genes_of_interest_mean_expression_by_subtype_heatmap.png", dpi=300)
plt.show()