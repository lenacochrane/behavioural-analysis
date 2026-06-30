
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T21-D02/experiments.xlsx', sheet_name='exp')

unique_df = df.drop_duplicates(
    subset=['date_of_exp','stock', 'collection_week', 'no_elav_virgin', 'no_males']
)



plt.figure(figsize=(6, 4))
sns.pointplot(data=unique_df, x='stock', y='number_survived', color='blue', linestyle='none')
sns.stripplot(
    data=unique_df,
    x='stock',
    y='number_survived', color='lightblue', alpha=0.5, jitter=True
)
plt.ylim(0,100)
sns.despine()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/phd/testing-lines/l1-l3/l1-l3.pdf', format='pdf', bbox_inches='tight')
plt.show()









