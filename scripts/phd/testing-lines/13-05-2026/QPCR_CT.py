
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/13-05-2026/_lena_2026-05-13_115054.xls"
output = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/13-05-2026"


well_map = {
    "C4": "N1_nxr",
    "C5": "N1_nxr",
    "C6": "N1_nxr",
    "C7": "N1_rpl32",
    "C8": "N1_rpl32",
    "C9": "N1_rpl32",

    "D4": "N2_nxr",
    "D5": "N2_nxr",
    "D6": "N2_nxr",
    "D7": "N2_rpl32",
    "D8": "N2_rpl32",

    "E4": "C1_nxr",
    "E5": "C1_nxr",
    "E6": "C1_nxr",
    "E7": "C1_rpl32",
    "E8": "C1_rpl32",
    "E9": "C1_rpl32",

    "F4": "C2_nxr",
    "F5": "C2_nxr",
    "F6": "C2_nxr",
    "F7": "C2_rpl32",
    "F8": "C2_rpl32",

    "G4": "temp_nxr",
    "G5": "temp_nxr",
    "G6": "temp_nxr",
}


# read ugly Results tab raw first
raw = pd.read_excel(path, sheet_name="Results", header=None, engine="xlrd")

# find the true header row
header_row = raw.index[
    raw.apply(
        lambda row: row.astype(str).str.contains("Well Position", case=False, na=False).any(),
        axis=1
    )
][0]

# now read properly
df = pd.read_excel(path, sheet_name="Results", header=header_row, engine="xlrd")

# map only your wells
df["well_label"] = df["Well Position"].map(well_map)
qpcr = df[df["well_label"].notna()].copy()

# split into sample and gene
qpcr[["sample", "gene"]] = qpcr["well_label"].str.rsplit("_", n=1, expand=True)

# proof we accessed it properly
print("Header row found:", header_row)
print("Mapped wells found:", len(qpcr), "out of", len(well_map))
print(qpcr[["Well Position", "well_label", "sample", "gene", "CT", "Amp Status", "Cq Conf", "Tm1"]])


######## CT DATA

qpcr_ct = qpcr[
    [
        "Well Position",
        "sample",
        "gene",
        "CT"
    ]
].copy()

qpcr_ct["CT"] = pd.to_numeric(qpcr_ct["CT"], errors="coerce")

replicate_summary = (
    qpcr_ct
    .groupby(["sample", "gene"])
    .agg(
        mean_CT=("CT", "mean"),
        sd_CT=("CT", "std"),
        min_CT=("CT", "min"),
        max_CT=("CT", "max"),
        n=("CT", "count")
    )
    .reset_index()
)

print(replicate_summary)

replicate_summary.to_csv("/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/13-05-2026/qPCR_summary.csv", index=False)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=qpcr_ct,
    x="sample",
    y="CT",
    hue="gene",
    palette='Blues'
)

plt.ylabel("Ct")
plt.xlabel("Sample")
plt.title("qPCR Ct values")

plt.tight_layout()

plt.savefig(
    os.path.join(output, "qpcr_ct_scatter.png"),
    dpi=300
)

plt.close()


delta_ct = (
    replicate_summary
    .pivot(
        index="sample",
        columns="gene",
        values="mean_CT"
    )
    .reset_index()
)

delta_ct = delta_ct[delta_ct["sample"] != "temp"].copy()

delta_ct["delta_CT"] = delta_ct["nxr"] - delta_ct["rpl32"]

print(delta_ct)

delta_ct.to_csv(
    os.path.join(output, "qPCR_delta_CT.csv"),
    index=False
)


# assign condition labels
delta_ct["condition"] = delta_ct["sample"].map({
    "C1": "control",
    "C2": "control",
    "N1": "neurexin_knockdown",
    "N2": "neurexin_knockdown"
})

# mean delta Ct per condition
condition_summary = (
    delta_ct
    .groupby("condition", as_index=False)
    .agg(
        mean_delta_CT=("delta_CT", "mean")
    )
)

print(condition_summary)


# calculate delta delta Ct
control_mean = condition_summary.loc[
    condition_summary["condition"] == "control",
    "mean_delta_CT"
].iloc[0]

knockdown_mean = condition_summary.loc[
    condition_summary["condition"] == "neurexin_knockdown",
    "mean_delta_CT"
].iloc[0]

delta_delta_CT = knockdown_mean - control_mean

# fold change
fold_change = 2 ** (-delta_delta_CT)

print("delta_delta_CT:", delta_delta_CT)
print("fold_change:", fold_change)

# save
condition_summary.to_csv(
    os.path.join(output, "qPCR_condition_summary.csv"),
    index=False
)