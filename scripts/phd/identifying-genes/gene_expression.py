import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import mplcursors  
import re
import ast
from matplotlib import cm
import gzip


""" GENE EXPRESSION ANALYSIS

This analysis uses the modENCODE developmental bulk RNA-seq dataset
distributed by FlyBase (gene_rpkm_report_fb_*.tsv.gz). 

DOWNLOADED:
cd ~/Downloads
curl -L -O https://s3ftp.flybase.org/releases/current/precomputed_files/genes/gene_rpkm_report_fb_2025_05.tsv.gz

PAPER:
https://flybase.org/reports/FBrf0225793 

- RPKM_value: quantitative expression level (Reads Per Kilobase per Million)
- Bin_value: FlyBase-defined expression bins (1–8), derived from RPKM
  and used for visualisation of expression patterns



"""



path = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE_EXPRESSION/modENCODE_mRNA-Seq_development.tsv.gz"  # adjust if needed

cols = [
    "Release_ID",
    "FBgn",
    "GeneSymbol",
    "Parent_library_FBlc",
    "Parent_library_name",
    "RNASource_FBlc",
    "RNASource_name",
    "RPKM_value",
    "Bin_value",
    "Unique_exon_base_count",
    "Total_exon_base_count",
    "Count_used",
]

df = pd.read_csv(
    path,
    sep="\t",
    comment="#",
    names=cols
)

df = df[
    ["FBgn", "GeneSymbol", "Parent_library_name",
     "RNASource_FBlc", "RNASource_name",
     "RPKM_value", "Bin_value"]
].rename(columns={
    "FBgn": "fbgn",
    "GeneSymbol": "gene",
    "Parent_library_name": "dataset",
    "RNASource_FBlc": "experiment_id",
    "RNASource_name": "stage",
    "RPKM_value": "rpkm",
    "Bin_value": "bin"
})

df = df[df["dataset"] == "modENCODE_mRNA-Seq_development"]


fbgn_to_name = {
    "FBgn0001624": 'DLG4',
    'FBgn0260794': 'TRIP12',
    'FBgn0022768': 'PPM1D',
    "FBgn0038975": "NRXN1",
    "FBgn0285944": "SCN2A",
    "FBgn0266557": "CHD8",
    "FBgn0263231": "DDX3X",
    "FBgn0040752": "SHANK3",
    "FBgn0031759": "KDM5B",
    "FBgn0260799": "CTNND2",
    "FBgn0035049": "MMP16",
    "FBgn0040022": "SETD1A",
    "FBgn0015509": "CUL1",
    "FBgn0024277": "TRIO",
    "FBgn0264386": "CACNA1G",
    "FBgn0004619": "GRIA3",
    "FBgn0053513": "GRIN2A",
    "FBgn0037363": "RB1CC1",
    "FBgn0035227": "DEPDC5",
    "FBgn0285944": "SCN1A",
    "FBgn0261570": "SYNGAP1",
    "FBgn0013343": "STX1B"
}


df = df[df["fbgn"].isin(fbgn_to_name)].copy()

# 3) add a friendly name column
df["gene"] = df["fbgn"].map(fbgn_to_name)

# (optional) check none are missing
missing = df["gene"].isna().sum()
print("Missing labels:", missing)

df["stage"] = df["stage"].str.replace("mE_mRNA_", "", regex=False)

# df["stage"] has values like: em0-2hr, L1, P5, AdF_Ecl_1days, AdM_Ecl_1days, etc.

stage_map = {
    # embryo
    "em0-2hr":  "embryo 0–2 h",
    "em2-4hr":  "embryo 2–4 h",
    "em4-6hr":  "embryo 4–6 h",
    "em6-8hr":  "embryo 6–8 h",
    "em8-10hr": "embryo 8–10 h",
    "em10-12hr":"embryo 10–12 h",
    "em12-14hr":"embryo 12–14 h",
    "em14-16hr":"embryo 14–16 h",
    "em16-18hr":"embryo 16–18 h",
    "em18-20hr":"embryo 18–20 h",
    "em20-22hr":"embryo 20–22 h",
    "em22-24hr":"embryo 22–24 h",

    # larva
    "L1": "larva L1",
    "L2": "larva L2",
    "L3_12hr": "larva L3 12 h",
    "L3_PS1-2": "larva L3 puff stage 1–2",
    "L3_PS3-6": "larva L3 puff stage 3–6",
    "L3_PS7-9": "larva L3 puff stage 7–9",

    # prepupa/pupa
    "WPP": "white prepupa",
    "P5": "pupa stage 5",
    "P6": "pupa stage 6",
    "P8": "pupa stage 8",
    "P9-10": "pupa stage 9–10",
    "P15": "pupa stage 15",

    # adults (female+male mapped to SAME label)
    "AdF_Ecl_1days":  "adult day 1",
    "AdM_Ecl_1days":  "adult day 1",
    "AdF_Ecl_5days":  "adult day 5",
    "AdM_Ecl_5days":  "adult day 5",
    "AdF_Ecl_30days": "adult day 30",
    "AdM_Ecl_30days": "adult day 30",
}

df["stage_label"] = df["stage"].map(stage_map).fillna(df["stage"])

print(sorted(df["stage_label"].unique()))

stage_order = [
    "embryo 0–2 h", "embryo 2–4 h", "embryo 4–6 h", "embryo 6–8 h",
    "embryo 8–10 h", "embryo 10–12 h", "embryo 12–14 h", "embryo 14–16 h",
    "embryo 16–18 h", "embryo 18–20 h", "embryo 20–22 h", "embryo 22–24 h",
    "larva L1", "larva L2",
    "larva L3 12 h", "larva L3 puff stage 1–2",
    "larva L3 puff stage 3–6", "larva L3 puff stage 7–9",
    "white prepupa",
    "pupa stage 5", "pupa stage 6", "pupa stage 8",
    "pupa stage 9–10", "pupa stage 15",
    "adult day 1", "adult day 5", "adult day 30"
]

df["stage_label"] = pd.Categorical(
    df["stage_label"],
    categories=stage_order,
    ordered=True
)


print(df.head(10))


heatmap_df = (
    df
    .pivot_table(
        index="gene",
        columns="stage_label",
        values="bin",
        aggfunc="mean"   # handles replicates if any
    )
)
print(heatmap_df.head(10))

plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_df,
    cmap="YlOrRd",
    vmin=1,
    vmax=8,
    cbar_kws={"label": "Expression bin"}
)


plt.yticks(fontweight="bold")
plt.xlabel("Developmental stage")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig("/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE_EXPRESSION/gene_expression_heatmap_YlOrRd.png", dpi=300, bbox_inches="tight")
plt.savefig("/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENE_EXPRESSION/gene_expression_heatmap_YlOrRd.pdf", format="pdf", bbox_inches="tight")
plt.show()