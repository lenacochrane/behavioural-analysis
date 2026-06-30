import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026/2026-05-06_144739.xls"
output = "/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/qPCR/06-05-2026"

well_map = {
    "C4": "N1_nxr", "C5": "N1_nxr", "C6": "N1_nxr",
    "C7": "N1_rpl32", "C8": "N1_rpl32", "C9": "N1_rpl32",
    "D4": "N2_rpl32", "D5": "N2_rpl32", "D6": "N2_rpl32",
    "D7": "N2_nxr", "D8": "N2_nxr", "D9": "N2_nxr",
    "E4": "C1_nxr", "E5": "C1_nxr", "E6": "C1_nxr",
    "E7": "C1_rpl32", "E8": "C1_rpl32", "E9": "C1_rpl32",
    "F4": "C2_rpl32", "F5": "C2_rpl32", "F6": "C2_rpl32",
    "F7": "C2_nxr", "F8": "C2_nxr", "F9": "C2_nxr",
    "G4": "temp_nxr", "G5": "temp_nxr", "G6": "temp_nxr",
    "G7": "temp_rpl32", "G8": "temp_rpl32", "G9": "temp_rpl32",
}

# check sheet names
xls = pd.ExcelFile(path, engine="xlrd")

raw = pd.read_excel(path, sheet_name="Amplification Data", header=None, engine="xlrd")

header_row = raw.index[
    raw.apply(
        lambda row: row.astype(str).str.contains("Well Position", case=False, na=False).any(),
        axis=1
    )
][0]

amp = pd.read_excel(path, sheet_name="Amplification Data", header=header_row, engine="xlrd")

print("Header row:", header_row)
print(amp.columns)
print(amp.head())

amp["well_label"] = amp["Well Position"].map(well_map)
amp = amp[amp["well_label"].notna()].copy()

amp[["sample", "gene"]] = amp["well_label"].str.rsplit("_", n=1, expand=True)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=amp,
    x="Cycle",
    y="Delta Rn",
    hue="well_label",
    estimator=None,
    # palette="coolwarm",
)

plt.xlabel("Cycle")
plt.ylabel("Delta Rn")
plt.title("qPCR amplification curves")

plt.tight_layout()

plt.savefig(
    os.path.join(output, "qPCR_amplification_curves_deltaRn.png"),
    dpi=300
)

plt.close()
