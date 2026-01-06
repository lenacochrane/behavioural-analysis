
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Adobe / Illustrator friendly PDFs ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---- Seaborn default "deep" colours ----
pal = sns.color_palette("deep", 2)

PALETTE = {
    "group-housed": pal[0],     # blue
    "socially-isolated": pal[1] # orange
}

HUE_ORDER = ["group-housed", "socially-isolated"]

### N10
df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated/time_average_msd.csv')
df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed/time_average_msd.csv')
### N1
# df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/socially-isolated/time_average_msd.csv')
# df4 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n1/group-housed/time_average_msd.csv')
# ### N2
# df5 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated/time_average_msd.csv')
# df6 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed/time_average_msd.csv')


# ### PSEUDO N10
# df7 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated/time_average_msd.csv')
# df8 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed/time_average_msd.csv')
# ### PSEUDO N2
# df9 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated/time_average_msd.csv')
# df10 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed/time_average_msd.csv')


plt.figure(figsize=(8,8))

### N10
sns.lineplot(data=df1, x='tau', y='msd', ci=None, label='SI', color=PALETTE["socially-isolated"])
sns.lineplot(data=df2, x='tau', y='msd', ci=None, label='GH', color=PALETTE["group-housed"])

# ### N1
# sns.lineplot(data=df3, x='tau', y='msd', ci=None, label='si_n1')
# sns.lineplot(data=df4, x='tau', y='msd', ci=None, label='gh_n1')
### N2
# sns.lineplot(data=df5, x='tau', y='msd', ci=None, label='si_n2')
# sns.lineplot(data=df6, x='tau', y='msd', ci=None, label='gh_n2')


### PSEUDO N10
# sns.lineplot(data=df7, x='tau', y='msd', ci=None, label='pseudo-si_n10')
# sns.lineplot(data=df8, x='tau', y='msd', ci=None, label='pseudo-gh_n10')

### PSEUDO N2
# sns.lineplot(data=df9, x='tau', y='msd', ci=None, label='pseudo-si_n2')
# sns.lineplot(data=df10, x='tau', y='msd', ci=None, label='pseudo-gh_n2')



plt.xlabel('Tau', fontsize=12,fontweight='bold')
plt.ylabel('MSD', fontsize=12,fontweight='bold')


# plt.yscale('log')
# plt.xscale('log')


plt.title('Time Average Mean Squared Distance', fontsize=16, fontweight='bold')

plt.legend(title='Number of Larvae')

plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/time-average-msd/n10.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/time-average-msd/n10.pdf', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()
