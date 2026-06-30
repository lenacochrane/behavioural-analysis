import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm


###### INTERACTION TYPE - ALL NODE-NODE PER FRAME CONTACTS ######

df1 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/fed-fed/closest_contacts_1mm_overtime.csv')
df1['condition'] = 'fed-fed'

df2 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-fed/closest_contacts_1mm_overtime.csv')
df2['condition'] = 'fed-starved'

df3 = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/starved-starved/closest_contacts_1mm_overtime.csv')
df3['condition'] = 'starved-starved'



def plot_binary_raster(df, condition):
    d = df.copy()

    # make sure touching is 0/1
    d['touch_val'] = d['touching'].astype(int)

    # rows = file, columns = frame
    mat = d.pivot_table(
        index='file',
        columns='frame',
        values='touch_val',
        aggfunc='max',
        fill_value=0
    ).sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(8, 4))

    cmap = ListedColormap(['black', 'orange'])  # 0 = no contact, 1 = contact

    plt.imshow(mat.to_numpy(), aspect='auto', cmap=cmap, interpolation='nearest')
    plt.title(f'{condition} – contact (< 1 mm)')
    plt.ylabel('file')
    plt.xlabel('frame')

    plt.yticks(range(len(mat.index)), mat.index, fontsize=6)
    plt.xticks([])  # frames usually too many to label nicely

    plt.tight_layout()
    plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/rasta/{condition}.png', dpi=300, bbox_inches='tight')
    plt.show()



plot_binary_raster(df1, 'fed-fed')
plot_binary_raster(df2, 'fed-starved')
plot_binary_raster(df3, 'starved-starved')




def plot_type_raster(df, condition):
    d = df.copy()

    # define a stable ordering of types
    type_order = [
        'head_head',
        'head_tail',
        'head_body',
        'body_body',
        'body_tail',
        'tail_tail',
    ]

    # map type → int code; 0 = no contact / not touching
    type_to_code = {t: i+1 for i, t in enumerate(type_order)}

    # start with 0 everywhere
    d['type_code'] = 0

    # only assign codes where touching == True and we have a relevant type
    mask = d['touching'] & d['Closest Interaction Type'].notna()
    d.loc[mask, 'type_code'] = (
        d.loc[mask, 'Closest Interaction Type']
        .map(type_to_code)
        .fillna(0)
        .astype(int)
    )

    # pivot: rows = file, cols = frame
    mat = d.pivot_table(
        index='file',
        columns='frame',
        values='type_code',
        aggfunc='max',
        fill_value=0
    ).sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(8, 4))

    # build a colormap: 0 = black (no contact), then one color per type
    colors = [
        'black',    # 0 = no contact
        "#ef0004",  # head_head
        "#ec9033",  # head_tail
        "#8aa786",  # head_body
        "#5ca6d3",  # body_body
        "#aeb2e5",  # body_tail
        "#967e99",  # tail_tail
    ]
    cmap = ListedColormap(colors[:len(type_order) + 1])

    # boundaries so each integer gets its own color
    bounds = np.arange(len(type_order) + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(mat.to_numpy(), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    plt.title(f'{condition} – interaction type', fontsize=12, fontweight='bold')
    plt.ylabel('file')
    plt.xlabel('frame')

    plt.yticks(range(len(mat.index)), mat.index, fontsize=6)
    plt.xticks([])

    # legend
    patches = [
        mpatches.Patch(color=cmap(code), label=t)
        for t, code in type_to_code.items()
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=6,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f'/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/rasta/{condition}_type.png', dpi=300, bbox_inches='tight')
    plt.show()



plot_type_raster(df1, 'fed-fed')
plot_type_raster(df2, 'fed-starved')
plot_type_raster(df3, 'starved-starved')




def plot_relevant_type_raster(df, condition):
    d = df.copy()

    # only care about these two types
    type_order = [
        'head_head',
        'head_tail',
    ]

    # map type → int code; 0 = no contact / not relevant
    type_to_code = {t: i+1 for i, t in enumerate(type_order)}

    # start with 0 everywhere
    d['type_code'] = 0

    # only assign codes where touching == True AND Relevant Interaction Type is one of the two
    mask = (
        d['touching'] &
        d['Relevant Interaction Type'].isin(type_order)
    )

    d.loc[mask, 'type_code'] = (
        d.loc[mask, 'Relevant Interaction Type']
        .map(type_to_code)
        .fillna(0)
        .astype(int)
    )

    # pivot: rows = file, cols = frame
    mat = d.pivot_table(
        index='file',
        columns='frame',
        values='type_code',
        aggfunc='max',
        fill_value=0
    ).sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(8, 4))

    # 0 = black (no relevant contact), then one color per type
    colors = [
        'black',    # 0 = no contact / no HH/HT
        "#ef0004",  # 1 = head_head
        "#25b33f",  # 2 = head_tail
    ]
    cmap = ListedColormap(colors[:len(type_order) + 1])

    bounds = np.arange(len(type_order) + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(mat.to_numpy(), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    plt.title(f'{condition} – head-head / head-tail over time', fontsize=12, fontweight='bold')
    plt.ylabel('file')
    plt.xlabel('frame')

    plt.yticks(range(len(mat.index)), mat.index, fontsize=6)
    plt.xticks([])

    # legend (only the two types)
    patches = [
        mpatches.Patch(color=cmap(code), label=t)
        for t, code in type_to_code.items()
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=6,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(
        f'/Users/cochral/repos/behavioural-analysis/plots/socially-isolated/head-head/rasta/{condition}_HH_HT_type.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

plot_relevant_type_raster(df1, 'fed-fed')
plot_relevant_type_raster(df2, 'fed-starved')
plot_relevant_type_raster(df3, 'starved-starved')