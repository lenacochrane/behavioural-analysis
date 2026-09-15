
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

""" 
ROSEPLOT APPROACH ANGLE X SPEED DIFFERENCE FOR GH VS PSEUDO POPULATION
"""

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/nearest_neighbour.csv')
df1['condition'] = 'GH'
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/nearest_neighbour.csv')
df2['condition'] = 'PSEUDO'

df = pd.concat([df1, df2], ignore_index=True)
df["condition"] = df["condition"].astype(str)


## ANGLE BINS
df["approach_angle"] = pd.to_numeric(df["approach_angle"], errors="coerce")
angle_edges = np.arange(0, 181, 30)  # 0,30,60,90,120,150,180
df['angle_bin'] = pd.cut(df['approach_angle'], angle_edges, include_lowest=True, right=False)
df = df.dropna(subset=["angle_bin"])
df['angle_bin'] = df['angle_bin'].astype(str)


## NORMALISE SPEED TO EACH FILE'S OWN 10-20 MM BASELINE
baseline = (
    df.loc[df["closest_node_distance"].between(10, 20, inclusive="both")]
      .groupby(["condition", "filename"])["speed_tail"]
      .mean()
      .rename("mean_speed_10_20")
      .reset_index()
)

df = df.merge(baseline, on=["condition", "filename"], how="left")
df = df.dropna(subset=["mean_speed_10_20"]).copy()
df["speed_norm"] = df["speed_tail"] / df["mean_speed_10_20"]


## DISTANCE BINS 1-5 MM
df = df[(df["closest_node_distance"] > 1) & (df["closest_node_distance"] <= 5)].copy()

bins = np.arange(1, 5.1, 1.0)
df["bin"] = pd.cut(df["closest_node_distance"], bins, include_lowest=True, right=True)
df["bin_center"] = df["bin"].apply(lambda x: x.mid).astype(float)


## ONE SPEED CURVE PER FILE
replicates = (
    df.groupby(["condition", "filename", "angle_bin", "bin_center"], observed=True)["speed_norm"]
      .mean()
      .reset_index(name="mean_speed_norm")
)


## ONE AUC PER FILE X ANGLE BIN
def auc_per_file(group):
    group = group.sort_values("bin_center")
    x = group["bin_center"].to_numpy(dtype=float)
    y = group["mean_speed_norm"].to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan
    return np.trapz(y, x)

auc = (
    replicates.groupby(["condition", "filename", "angle_bin"], observed=True)
              .apply(auc_per_file)
              .reset_index(name="auc")
              .dropna(subset=["auc"])
)

# sanity check: should be 10 GH + 10 PSEUDO per angle bin
print(auc.groupby(["angle_bin", "condition"])["auc"].count().unstack(fill_value=0))


## MANN-WHITNEY PER ANGLE BIN, FDR CORRECTED ACROSS BINS
rows = []
for angle_bin, sub in auc.groupby("angle_bin"):
    gh = sub.loc[sub["condition"] == "GH", "auc"]
    pseudo = sub.loc[sub["condition"] == "PSEUDO", "auc"]
    if gh.size == 0 or pseudo.size == 0:
        continue
    u, p = mannwhitneyu(gh, pseudo, alternative="two-sided")
    rows.append({
        "angle_bin": angle_bin,
        "n_GH": gh.size,
        "n_PSEUDO": pseudo.size,
        "mean_GH": gh.mean(),
        "mean_PSEUDO": pseudo.mean(),
        "relative_area": pseudo.mean() - gh.mean(),
        "sem_GH": gh.sem(),
        "sem_PSEUDO": pseudo.sem(),
        "sem_diff": np.sqrt(gh.sem()**2 + pseudo.sem()**2),
        "sd_GH": gh.std(),
        "sd_PSEUDO": pseudo.std(),
        "sd_diff": np.sqrt(gh.std()**2 + pseudo.std()**2),
        "u_stat": u,
        "p_raw": p,
    })

results = pd.DataFrame(rows)
passed, p_corr, _, _ = multipletests(results["p_raw"], alpha=0.05, method="fdr_bh")
results["p_corrected"] = p_corr
results["significant"] = passed
print(results)

results.to_csv('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-1/ghXpseudo/roseplot_stats.csv', index=False)


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

results = pd.read_csv('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-1/ghXpseudo/roseplot_stats.csv')

# --- angle centres ---
def angle_center(label):
    left = float(label.split(",")[0].strip("[("))
    right = float(label.split(",")[1].strip(" )]"))
    return (left + right) / 2

results["angle_center_deg"] = results["angle_bin"].apply(angle_center)
results = results.sort_values("angle_center_deg")

theta = np.deg2rad(results["angle_center_deg"].to_numpy())
width = np.deg2rad(30)

# magnitude of the effect (PSEUDO - GH)
r = np.abs(results["relative_area"].to_numpy())

# error bars: +/- 1 SD of the difference between the two group means
sd = results["sd_diff"].to_numpy()
ci_low_plot = np.clip(r - sd, 0, None)
ci_high_plot = r + sd


# --- stars from the corrected p-values ---
def star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# --- plot ---
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection="polar")

ax.bar(
    theta,
    r,
    width=width,
    align="center",
    color="steelblue",
    edgecolor="0.2",
    linewidth=1.5,
)

# error bars
for th, low, high in zip(theta, ci_low_plot, ci_high_plot):
    ax.plot([th, th], [low, high], color="black", linewidth=2, solid_capstyle="round")

# formatting
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(0)
ax.set_thetamax(180)

ax.set_yticklabels([])
ax.yaxis.grid(False)
ax.xaxis.grid(True, linewidth=1, alpha=0.8, color="0.3")
ax.spines["polar"].set_linewidth(1)

rmax = ci_high_plot.max() * 1.2
ax.set_rlim(0, rmax)

for th, high, p in zip(theta, ci_high_plot, results["p_corrected"].to_numpy()):
    label = star(p)
    if label:
        ax.text(th, high + rmax * 0.05, label, ha="center", va="center", fontsize=18, fontweight="bold")

ax.set_frame_on(False)

ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax.set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°", "180°"])

ax.set_title(
    "Speed Difference",
    pad=20
)

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/lrs_paper/FIGURE-1/ghXpseudo/roseplot_4.pdf',
            format='pdf', bbox_inches='tight')
plt.show()


# %%
