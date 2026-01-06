import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import matplotlib as mpl

# ---- Adobe-friendly fonts (must be set BEFORE plotting) ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/head_head_contacts_kinematics_over_time.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/head_head_contacts_kinematics_over_time.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/head_head_contacts_kinematics_over_time.csv')
df3['condition'] = 'starved-starved'



df = pd.concat([df1, df2, df3], ignore_index=True)

df['heading_angle_change'] = (
    df
    .sort_values(['file', 'interaction_number', 'track_id', 'frame'])
    .groupby(['file', 'condition', 'interaction_number', 'track_id'])['heading_angle']
    .diff()
    .abs() )


df['interaction_group'] = np.where(df['interaction_number'] == 1, 'first', 'other')



interactions_per_file = (
    df.groupby(['condition', 'file'])['interaction_number']
      .max()
      .reset_index(name='number_of_interactions')
)


plt.figure(figsize = (8, 8))

sns.barplot(data=interactions_per_file, x='condition', y='number_of_interactions', ci='sd')

plt.xlabel('Condition', fontsize=12, fontweight='bold')
plt.ylabel('Number of Interactions', fontsize=12, fontweight='bold')
plt.title('Number of Head-Head Interaction "Bouts" per File', fontsize=16, fontweight='bold')

plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/number_of_interactions.png', dpi=300, bbox_inches='tight')
plt.close()




fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)

for ax, cond in zip(axes, df['condition'].unique()):
    sub = df[df['condition'] == cond]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='speed',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(cond)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Speed' if ax == axes[0] else '')

    # keep legend only on last panel (optional)
    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/speed.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/speed.pdf',
             format='pdf', bbox_inches='tight')
plt.close()



fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)

for ax, cond in zip(axes, df['condition'].unique()):
    sub = df[df['condition'] == cond]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='heading_angle_change',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(cond)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Heading angle change' if ax == axes[0] else '')

    # keep legend only on last panel (optional)
    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/heading_angle_change.png', dpi=300, bbox_inches='tight')
plt.close()



fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)

for ax, cond in zip(axes, df['condition'].unique()):
    sub = df[df['condition'] == cond]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='heading_angle',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(cond)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Heading angle' if ax == axes[0] else '')

    # keep legend only on last panel (optional)
    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/heading_angle.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/heading_angle.pdf',
             format='pdf', bbox_inches='tight')
plt.close()



fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)

for ax, cond in zip(axes, df['condition'].unique()):
    sub = df[df['condition'] == cond]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='min_distance',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(cond)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Minimum distance' if ax == axes[0] else '')

    # keep legend only on last panel (optional)
    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/min_distance.png', dpi=300, bbox_inches='tight')
plt.close()




default_pal = sns.color_palette("deep", 4)

COND_PALETTE = {
    "first": default_pal[2],         # blue
    "other": default_pal[3],     # orange
}

# (optional) enforce condition order everywhere
COND_ORDER = ["first", "other"]


###### JUST FED AND STARVED

df_subset = df[df['condition'] == 'fed-starved']

df_subset['role'] = df_subset['track_id'].map({
    0: 'fed',
    1: 'starved'
})



fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

for ax, role in zip(axes, ['fed', 'starved']):
    sub = df_subset[df_subset['role'] == role]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='speed',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax, palette=COND_PALETTE, hue_order=COND_ORDER
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(role)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Speed' if role == 'fed' else '')
    ax.set_xlim(0, 15)

    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_speed.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_speed.pdf',
             format='pdf', bbox_inches='tight')
plt.close()





fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

for ax, role in zip(axes, ['fed', 'starved']):
    sub = df_subset[df_subset['role'] == role]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='min_distance',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(role)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Minimum distance' if role == 'fed' else '')

    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_min_distance.png', dpi=300, bbox_inches='tight')
plt.close()




fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

for ax, role in zip(axes, ['fed', 'starved']):
    sub = df_subset[df_subset['role'] == role]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='heading_angle',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax, palette=COND_PALETTE, hue_order=COND_ORDER
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(role)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Heading angle' if role == 'fed' else '')
    ax.set_xlim(0, 15)

    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_heading_angle.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_heading_angle.pdf',
             format='pdf', bbox_inches='tight')
plt.close()



fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

for ax, role in zip(axes, ['fed', 'starved']):
    sub = df_subset[df_subset['role'] == role]

    sns.lineplot(
        data=sub,
        x='rel_frame',
        y='heading_angle_change',
        hue='interaction_group',
        errorbar=('ci', 95),
        ax=ax, palette=COND_PALETTE, hue_order=COND_ORDER
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(role)
    ax.set_xlabel('Relative frame')
    ax.set_ylabel('Heading angle change' if role == 'fed' else '')
    ax.set_xlim(0, 15)

    if ax != axes[-1]:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_heading_angle_change.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/h-h-first_other/fed_starved_heading_angle_change.pdf',
             format='pdf', bbox_inches='tight')
plt.close()