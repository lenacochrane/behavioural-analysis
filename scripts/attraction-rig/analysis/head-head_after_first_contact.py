import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/head_head_first_contact_kinematics.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/head_head_first_contact_kinematics.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/head_head_first_contact_kinematics.csv')
df3['condition'] = 'starved-starved'


##### ALL CONDITIONS TOGETHER

df = pd.concat([df1, df2, df3], ignore_index=True)

df = df.sort_values(['condition', 'file', 'track_id', 'rel_frame'])

df['heading_angle_change'] = (
    df.groupby(['condition', 'file', 'track_id'])['heading_angle']
      .diff().abs() 
)

plt.figure(figsize=(6,4))

sns.lineplot(
    data=df,
    x='rel_frame',
    y='speed',
    hue='condition',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Speed')
plt.title('Speed around first head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/speed.png', dpi=300, bbox_inches='tight')
plt.close()



plt.figure(figsize=(6,4))

sns.lineplot(
    data=df,
    x='rel_frame',
    y='heading_angle',
    hue='condition',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Heading Angle')
plt.title('Heading Angle around first head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/heading_angle.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(6,4))

sns.lineplot(
    data=df,
    x='rel_frame',
    y='heading_angle_change',
    hue='condition',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Heading Angle')
plt.title('Heading Angle Change around first head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/heading_angle_change.png', dpi=300, bbox_inches='tight')
plt.close()



plt.figure(figsize=(6,4))

sns.lineplot(
    data=df,
    x='rel_frame',
    y='min_distance',
    hue='condition',
    errorbar=('ci', 95)
)


plt.ylim(0,3)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Min Distance')
plt.title('Min Distance around first head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/min_distance.png', dpi=300, bbox_inches='tight')
plt.close()






##### FED-STARVED ONLY

df2['role'] = df2['track_id'].map({0: 'fed', 1: 'starved'})
df2 = df2.sort_values(['file', 'track_id', 'rel_frame'])
df2['heading_angle_change'] = (
    df2.groupby(['role', 'file', 'track_id'])['heading_angle']
      .diff().abs()
)


plt.figure(figsize=(6,4))

sns.lineplot(
    data=df2,
    x='rel_frame',
    y='speed',
    hue='role',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Speed')
plt.title('Speed around after head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/fed_starved_speed.png', dpi=300, bbox_inches='tight')
plt.close()





plt.figure(figsize=(6,4))

sns.lineplot(
    data=df2,
    x='rel_frame',
    y='heading_angle',
    hue='role',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Heading Angle')
plt.title('Heading Angle around after head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/fed_starved_heading_angle.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(6,4))

sns.lineplot(
    data=df2,
    x='rel_frame',
    y='heading_angle_change',
    hue='role',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Heading Angle Change')
plt.title('Heading Angle Change around after head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/fed_starved_heading_angle_change.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(6,4))

sns.lineplot(
    data=df2,
    x='rel_frame',
    y='min_distance',
    hue='role',
    errorbar=('ci', 95)
)


plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Frames relative after first head–head contact')
plt.ylabel('Min Distance')
plt.title('Min Distance around after head–head contact')
plt.tight_layout()
plt.savefig('/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/fed_starved/first-h-h/fed_starved_min_distance.png', dpi=300, bbox_inches='tight')
plt.close()