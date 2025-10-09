
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/group-housed/inter_bout_dynamics.csv')
df1['condition'] = 'GH'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n10/socially-isolated/inter_bout_dynamics.csv')
df2['condition'] = 'SI'



palette = {
    'SI': '#fca35d',  # yellow-orange
    'GH': '#0d75b9'        # blue-green
}


df = pd.concat([df1, df2], ignore_index=True)

### TIME BETWEEN BOUTS 

plt.figure(figsize=(8,8))

sns.barplot(data=df, x='condition', y='time_since_last_bout',  edgecolor='black', linewidth=2, errorbar='sd', palette=palette, alpha=0.8)

plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('Time (s)', fontsize=12, fontweight='bold')

plt.title('Time Between Proximal <1mm Bouts', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[1, 1, 1, 1])

plt.ylim(0, None)

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/time-between-bouts.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


#### LIKELHOOD OF INTERACTING WITH SAME PARTNER POST <1MM BOUT 


partner_repeat_summary = (
    df.groupby(['condition', 'file'])['same_partner']
    .mean()
    .reset_index()
)

plt.figure(figsize=(8,8))

sns.barplot(
    data=partner_repeat_summary,
    x='condition', y='same_partner',
    palette=palette, edgecolor='black', linewidth=2, errorbar='sd', alpha=0.8 
)

plt.ylabel('Proportion Same Partner', fontsize=12, fontweight='bold')
plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.title('Likelihood of Repeated Partner Across Bouts', fontsize=14, fontweight='bold')

plt.ylim(0, 1)
plt.tight_layout()

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/repeat-partner-probability.png', dpi=300, bbox_inches='tight')
plt.show()


### INDIVIDUALITY: ARE SOME LARVAE MORE LIKELY TO INTERACT THAN OTHERS?


# Make sure 'condition' is in df and carried over
larva_bout_counts = (
    df.groupby(['condition', 'file', 'larva_id'])
    .size()
    .reset_index(name='num_bouts')
)

files = larva_bout_counts['file'].unique()
n_files = len(files)

n_cols = 4
n_rows = int(np.ceil(n_files / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

for idx, file in enumerate(files):
    ax = axes[idx // n_cols][idx % n_cols]
    sub_df = larva_bout_counts[larva_bout_counts['file'] == file]

    sns.barplot(
        data=sub_df,
        x='larva_id', y='num_bouts',
        hue='condition', dodge=False,
        ax=ax,
        palette=palette,
        edgecolor='black'
    )

    ax.set_title(f'{file}', fontsize=10)
    ax.set_xlabel('Larva ID')
    ax.set_ylabel('# Bouts')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.get_legend().remove()

# Add one legend outside the loop
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='Condition', loc='upper right', bbox_to_anchor=(1.12, 0.95))

# Hide unused axes
for i in range(n_files, n_rows * n_cols):
    fig.delaxes(axes[i // n_cols][i % n_cols])

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/individual_larval_bout_number.png', dpi=300)
plt.show()


### VARIABILITY BETWEEN LARVAE ! coefficient of variation (CV):

variability = (
    larva_bout_counts
    .groupby(['condition', 'file'])
    .agg(mean_bouts=('num_bouts', 'mean'),
         std_bouts=('num_bouts', 'std'))
    .reset_index()
)

variability['cv'] = variability['std_bouts'] / variability['mean_bouts']

plt.figure(figsize=(8, 6))

sns.barplot(
    data=variability,
    x='condition', y='cv',
    palette=palette, edgecolor='black', linewidth=1.5,
    errorbar='sd'  # shows SD across files per condition
)

plt.ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.title('Variability in Larval Interaction Bouts per Video', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/individual-variability.png', dpi=300)
plt.show()


### LARVAL PREFERENCES BASED ON NUMBER OF INTERACTIONS

pref_matrices = {
    (group['condition'].iloc[0], file): group.pivot_table(
        index='larva_id',
        columns='partner_id',
        values='bout_number',
        aggfunc='count',
        fill_value=0
    )
    for file, group in df.groupby('file')
}

n_files = len(pref_matrices)
n_cols = 4
n_rows = int(np.ceil(n_files / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)

# Define color maps per condition
condition_cmaps = {
    'GH': 'Blues',
    'SI': 'Oranges'
}

for idx, ((condition, file), matrix) in enumerate(pref_matrices.items()):
    ax = axes[idx // n_cols][idx % n_cols]
    
    cmap = condition_cmaps.get(condition, 'Greys')  # fallback if unknown condition
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax, square=True)
    ax.set_title(f"{condition} | {file}", fontsize=10)
    ax.set_xlabel("Partner ID")
    ax.set_ylabel("Larva ID")

# Hide any unused axes
for i in range(n_files, n_rows * n_cols):
    fig.delaxes(axes[i // n_cols][i % n_cols])

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/pref-matrix.png', dpi=300)
plt.show()




### IS THERE A LINK BETWEEN LARVAL PREFERENCE AND INCREASED LIKLEHOOD OF OVERALL INTERACTION?
### LARVAL PREFERENCES BASED ON LARVAL ACTIVITY (E.G. ARE THEY JUST INTERACTING MORE AND THEREFORE HAVE HIGHER PREFERENCE)
# observed / expected


observed_matrices = {
    (cond, file): group.pivot_table(
        index='larva_id',
        columns='partner_id',
        values='bout_number',
        aggfunc='count',
        fill_value=0
    )
    for (cond, file), group in df.groupby(['condition', 'file'])
}

larva_totals = df.groupby(['condition', 'file', 'larva_id'])['bout_number'].count().reset_index()

from collections import defaultdict

larva_total_dict = defaultdict(dict)
for _, row in larva_totals.iterrows():
    key = (row['condition'], row['file'])
    larva_total_dict[key][row['larva_id']] = row['bout_number']


expected_matrices = {}

for key, observed in observed_matrices.items():
    total = sum(larva_total_dict[key].values())
    expected = pd.DataFrame(index=observed.index, columns=observed.columns)

    for i in observed.index:
        for j in observed.columns:
            b_i = larva_total_dict[key].get(i, 0)
            b_j = larva_total_dict[key].get(j, 0)
            expected.loc[i, j] = (b_i * b_j) / total if total > 0 else 0

    expected_matrices[key] = expected.astype(float)


preference_matrices = {
    key: observed_matrices[key] / expected_matrices[key]
    for key in observed_matrices.keys()
}

n_files = len(preference_matrices)
n_cols = 4
n_rows = int(np.ceil(n_files / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)

for idx, ((condition, file), matrix) in enumerate(preference_matrices.items()):
    ax = axes[idx // n_cols][idx % n_cols]

    cmap = '#fca35d' if condition == 'SI' else '#0d75b9'

    sns.heatmap(
        matrix.astype(float),
        annot=False,
        cmap=sns.light_palette(cmap, as_cmap=True),
        vmin=0.0, vmax=2.0,
        square=True,
        cbar=False,
        ax=ax
    )

    ax.set_title(f"{condition} | {file}", fontsize=10)
    ax.set_xlabel("Partner ID")
    ax.set_ylabel("Larva ID")

# Remove empty axes
for i in range(n_files, n_rows * n_cols):
    fig.delaxes(axes[i // n_cols][i % n_cols])

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/interactions/interaction-bout-dynamics/pref-matrix-expected-interaction.png', dpi=300)
plt.show()
