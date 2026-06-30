
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/lucy/recomputed_ct_results_selected_wells.csv')

well_map = {
    "C4": "N1_nxr",
    "C5": "N1_nxr",
    "C6": "N1_nxr",
    "C7": "N1_rpl32",
    "C8": "N1_rpl32",
    "C9": "N1_rpl32",

    "D4": "N2_rpl32",
    "D5": "N2_rpl32",
    "D6": "N2_rpl32",
    "D7": "N2_nxr",
    "D8": "N2_nxr",
    "D9": "N2_nxr",

    "E4": "C1_nxr",
    "E5": "C1_nxr",
    "E6": "C1_nxr",
    "E7": "C1_rpl32",
    "E8": "C1_rpl32",
    "E9": "C1_rpl32",

    "F4": "C2_rpl32",
    "F5": "C2_rpl32",
    "F6": "C2_rpl32",
    "F7": "C2_nxr",
    "F8": "C2_nxr",
    "F9": "C2_nxr",
}


df["well_label"] = df["Well Position"].map(well_map)

df = df[df["well_label"].notna()].copy()

df[["sample", "gene"]] = df["well_label"].str.rsplit("_", n=1, expand=True)

df["Ct"] = pd.to_numeric(df["Ct"], errors="coerce")

output = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/lucy"

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="sample",
    y="Ct",
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



mean_ct_table = (
    df
    .groupby(["sample", "gene"], as_index=False)
    .agg(
        mean_Ct=("Ct", "mean")
    )
)

print(mean_ct_table)

mean_ct_table.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/lucy/mean_ct_table.csv",
    index=False
)

delta_ct_table = (
    mean_ct_table
    .pivot(
        index="sample",
        columns="gene",
        values="mean_Ct"
    )
    .reset_index()
)

delta_ct_table["delta_Ct"] = (
    delta_ct_table["nxr"] - delta_ct_table["rpl32"]
)

print(delta_ct_table)
delta_ct_table.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/lucy/delta_ct_table.csv",
    index=False
)


condition_delta_ct = (
    delta_ct_table
    .assign(
        condition=lambda x: x["sample"].str[0].map({
            "C": "control",
            "N": "neurexin_knockdown"
        })
    )
    .groupby("condition", as_index=False)
    .agg(
        mean_delta_Ct=("delta_Ct", "mean")
    )
)

print(condition_delta_ct)

# calculate delta-delta Ct
control_mean = condition_delta_ct.loc[
    condition_delta_ct["condition"] == "control",
    "mean_delta_Ct"
].iloc[0]

knockdown_mean = condition_delta_ct.loc[
    condition_delta_ct["condition"] == "neurexin_knockdown",
    "mean_delta_Ct"
].iloc[0]

delta_delta_Ct = knockdown_mean - control_mean

# fold change
fold_change = 2 ** (-delta_delta_Ct)

# final table
final_qpcr = pd.DataFrame({
    "control_mean_delta_Ct": [control_mean],
    "knockdown_mean_delta_Ct": [knockdown_mean],
    "delta_delta_Ct": [delta_delta_Ct],
    "fold_change": [fold_change]
})

print(final_qpcr)

# save
final_qpcr.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/lucy/final_qpcr_results.csv",
    index=False
)