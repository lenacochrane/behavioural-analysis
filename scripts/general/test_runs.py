import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyarrow.feather as feather
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import wkt
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/epi/Epi25_gene_results_top.csv')
df['gda_x_weighted'] = df['gda_score'] * df['weighted_score']

output_directory = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/GENES/disease-mining/refined/epi/plots'

sns.scatterplot(data=df, x="weighted_score",
        y="gda_score")

for _, row in df.iterrows():
    plt.text(
        row["weighted_score"] + 0.3,  # small x-offset so text doesn't overlap the dot
        row["gda_score"],
        row["gene_symbol"],
        fontsize=7,
        alpha=0.7)


plt.tight_layout()
plt.legend()

# --- save static version before interactive part ---
save_path = os.path.join(f"{output_directory}/weigh.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.close()


sns.scatterplot(data=df, x="weighted_score",
        y="gda_score")

for _, row in df.iterrows():
    plt.text(
        row["weighted_score"] + 0.3,  # small x-offset so text doesn't overlap the dot
        row["gda_score"],
        row["gene_symbol"],
        fontsize=7,
        alpha=0.7)


plt.tight_layout()
plt.legend()

# --- save static version before interactive part ---
save_path = os.path.join(f"{output_directory}/weight.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.close()




df_sorted = df.sort_values('gda_x_weighted', ascending=False)

sns.barplot(data=df_sorted, x='gene_symbol', y='gda_x_weighted')
plt.xticks(rotation=90)

save_path = os.path.join(f"{output_directory}/bar.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.close()


