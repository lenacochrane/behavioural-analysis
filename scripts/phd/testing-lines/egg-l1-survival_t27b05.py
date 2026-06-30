import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/TESTING-LINES/EGG-L1/ELAV_T27-B05/egg-l1-survival.xlsx')
df['lines_crossed'] = df['neuronal_expression']  + "_" + df['cross']
df['condition'] = df['lines_crossed'] + "-temp_" + df['staged_temp'].astype(str)

df['percentage_survived'] = df['number_survival_24h'] / df['number_transferred'] * 100 

save_path = "/Users/cochral/repos/behavioural-analysis/plots/phd/testing-lines/egg-l1-survival"


""" PLOTTING THE SURVIVAL OF EGG-L1 AT DIFFERENT TEMPERATURES 
       THIS PLOT IS THE SURVIVAL NEXT MORNING SO LIKE 40H AFTER STAGING AS 24H WAS NOT ENOUGH FOR 22 AND 18D DEGREES
"""

plt.figure(figsize=(8, 6))

sns.stripplot(
    data=df,
    x='condition',
    y='percentage_survived',
    jitter=True,
    color='cornflowerblue',
    alpha=0.7,
    zorder=1
)

sns.pointplot(
    data=df,
    x='condition',
    y='percentage_survived',
    join=False,
    color='navy',
    zorder=5
)

plt.xlabel("")
plt.ylabel("Percentage Survived", fontweight='bold', fontsize=12)
plt.title("Survival of EGG-L1 at Different Temperatures", fontweight='bold', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.ylim(0, 100)
sns.despine()
plt.savefig(f"{save_path}/egg_l1_survived_next_morning_t27b05.pdf", format='pdf')
plt.show()

