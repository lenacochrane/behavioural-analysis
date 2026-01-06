
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os

"""
“Seen” interactions are cases where both larvae are oriented toward each other before contact, 
whereas “unseen” interactions occur when one larva approaches from behind while the other is not facing it; 
in these unseen cases, the approacher is the larva moving toward the interaction, 
and the receiver is the larva being approached

"""


df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/cropped_interactions.csv')

print(df.columns)

df['track1_approach_angle_edited'] = 180 - df['track1_approach_angle']
df['track2_approach_angle_edited'] = 180 - df['track2_approach_angle']

## difference (large differences surely reflect one approaching and the other not relaising idk come back to it)
df['approach_angle_diff'] = (df['track1_approach_angle_edited'] - df['track2_approach_angle_edited']).abs()


bins = [0, 30, 60, 90, 120, 150, 180]

df['track1_approach_bin'] = pd.cut(
    df['track1_approach_angle_edited'],
    bins=bins,
    include_lowest=True,
    right=False)

df['track2_approach_bin'] = pd.cut(
    df['track2_approach_angle_edited'],
    bins=bins,
    include_lowest=True,
    right=False)


pre = df[(df['Normalized Frame'] >= -10) & (df['Normalized Frame'] <= -1)]


def classify_interaction(x):
    # fractions in pre-window
    t1_approacher = (x['track1_approach_angle_edited'].between(0, 30)).mean()
    t2_approacher = (x['track2_approach_angle_edited'].between(0, 30)).mean()

    t1_receiver = (x['track1_approach_angle_edited'].between(150, 180)).mean()
    t2_receiver = (x['track2_approach_angle_edited'].between(150, 180)).mean()

    # track1 approaches unseen
    if t1_approacher >= 0.6 and t2_receiver >= 0.6:
        return pd.Series({'approach_type': 'unseen', 'approacher_track': 1, 'receiver_track': 2})
    # track2 approaches unseen
    if t2_approacher >= 0.6 and t1_receiver >= 0.6:
        return pd.Series({'approach_type': 'unseen', 'approacher_track': 2, 'receiver_track': 1})

    # seen cases
    if t1_approacher >= 0.6 and (x['track2_approach_angle_edited'].between(0, 30)).mean() >= 0.6:
        return pd.Series({'approach_type': 'seen', 'approacher_track': None, 'receiver_track': None})
    if t2_approacher >= 0.6 and (x['track1_approach_angle_edited'].between(0, 30)).mean() >= 0.6:
        return pd.Series({'approach_type': 'seen', 'approacher_track': None, 'receiver_track': None})
    return pd.Series({'approach_type': pd.NA, 'approacher_track': pd.NA, 'receiver_track': pd.NA})


labels = (
    pre.groupby('interaction_id')
      .apply(classify_interaction)
      .reset_index()
)

labels = labels[labels['approach_type'].notna()]   # <-- filter here

df = df.merge(labels, on='interaction_id', how='inner')  # inner = only labeled interactions

# output = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision'
# df.to_csv(os.path.join(output, 'unseen_versus_seen_interactions.csv'), index=False)



##### UNSEEN APPROACH VERSUS RECEIVED #####

df_unseen = df[df['approach_type'] == 'unseen'].copy()

# receiver speed column
df_unseen['receiver_speed'] = np.where(
    df_unseen['receiver_track'] == 1,
    df_unseen['track1_speed'],
    df_unseen['track2_speed']
)

df_unseen['approacher_speed'] = np.where(
    df_unseen['approacher_track'] == 1,
    df_unseen['track1_speed'],
    df_unseen['track2_speed']
)

# receiver heading_angle column
df_unseen['receiver_heading_angle'] = np.where(
    df_unseen['receiver_track'] == 1,
    df_unseen['track1_angle'],
    df_unseen['track2_angle']
)

df_unseen['approacher_heading_angle'] = np.where(
    df_unseen['approacher_track'] == 1,
    df_unseen['track1_angle'],
    df_unseen['track2_angle']
)


def angular_change(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 180 - diff)

df_unseen = df_unseen.sort_values(['interaction_id', 'Normalized Frame'])


df_unseen['receiver_turn'] = (
    df_unseen
    .groupby('interaction_id')['receiver_heading_angle']
    .transform(lambda x: angular_change(x, x.shift()))
)


df_unseen['approacher_turn'] = (
    df_unseen
    .groupby('interaction_id')['approacher_heading_angle']
    .transform(lambda x: angular_change(x, x.shift()))
)





sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_speed',
    errorbar=('ci', 95), color='green', label='Receiver'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='approacher_speed',
    errorbar=('ci', 95), color='red', label='Approacher'
)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_overtime.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_overtime.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()




sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_speed',
    errorbar=('ci', 95), color='green', label='Receiver'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='approacher_speed',
    errorbar=('ci', 95), color='red', label='Approacher'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='min_distance',
    errorbar=('ci', 95), color='gray', label='Min Distance'
)

plt.ylim(0,2)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_overtime_with_min_distance.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_overtime_with_min_distance.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()




sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_heading_angle',
    errorbar=('ci', 95), color='green', label='Receiver'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='approacher_heading_angle',
    errorbar=('ci', 95), color='red', label='Approacher'
)

plt.ylim(0,180)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_angle.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_angle.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()



sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_turn',
    errorbar=('ci', 95), color='green', label='Receiver'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='approacher_turn',
    errorbar=('ci', 95), color='red', label='Approacher'
)

plt.ylim(0,30)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_angle_change.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/receiver_vs_approacher_angle_change.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()




##### UNSEEN VERSUS SEEN #####


seen_interactions_df = (
    df[df['approach_type'] == 'seen'][['interaction_id','Normalized Frame','track1_speed','track2_speed']]
    .melt(id_vars=['interaction_id','Normalized Frame'],
          value_vars=['track1_speed','track2_speed'],
          value_name='response_speed')
    .assign(approach_type='seen')
)


seen_angles = (
    df[df['approach_type'] == 'seen']
    [['interaction_id','Normalized Frame','track1_angle','track2_angle']]
    .melt(id_vars=['interaction_id','Normalized Frame'],
        value_vars=['track1_angle','track2_angle'],
        value_name='heading_angle')
    .assign(approach_type='seen')
)

seen_angles = seen_angles.sort_values(
    ['interaction_id', 'variable', 'Normalized Frame']
)

seen_angles['turn'] = (
    seen_angles
    .groupby(['interaction_id', 'variable'])['heading_angle']
    .transform(lambda x: angular_change(x, x.shift()))
)





sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_speed',
    errorbar=('ci', 95), color='blue', label='Unseen Interactions (Receiver)'
)

sns.lineplot(
    data=seen_interactions_df,
    x='Normalized Frame',
    y='response_speed',
    errorbar=('ci', 95), color='orange', label='Seen Interactions'
)

sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='min_distance',
    errorbar=('ci', 95), color='black', label='Min Distance'
)

plt.ylim(0,2)

plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/seen_v_unseen.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/seen_v_unseen.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()





sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_heading_angle',
    errorbar=('ci', 95), color='blue', label='Unseen (Receiver)'
)

sns.lineplot(
    data=seen_angles,
    x='Normalized Frame',
    y='heading_angle',
    errorbar=('ci', 95), color='orange', label='Seen'
)

plt.ylim(0,180)

plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/unseen_versus_seen_angle.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/unseen_versus_seen_angle.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()



sns.lineplot(
    data=df_unseen,
    x='Normalized Frame',
    y='receiver_turn',
    errorbar=('ci', 95), color='blue', label='Unseen (Receiver)'
)

sns.lineplot(
    data=seen_angles,
    x='Normalized Frame',
    y='turn',
    errorbar=('ci', 95), color='orange', label='Seen'
)

# sns.lineplot(
#     data=df_unseen,
#     x='Normalized Frame',
#     y='min_distance',
#     errorbar=('ci', 95), color='black', label='Min Distance'
# )


plt.ylim(0,30)
# plt.legend(
#     loc='upper left',
#     bbox_to_anchor=(1.02, 0.5),
#     frameon=False
# )
plt.savefig('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/unseen_versus_seen_angle_change.png', dpi=300, bbox_inches='tight')
plt.savefig(
    '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/umap-pipeline/is_it_vision/unseen_versus_seen_angle_change.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()