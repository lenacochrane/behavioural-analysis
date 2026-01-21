import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from PIL import Image
import math
from matplotlib.colors import LinearSegmentedColormap


def translate_rotate_syllables(df):

    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True) # kappa shd be included if multiple models

    def translate_bout(bout):
        # first frame of this bout
        first = bout.iloc[0]
        x = first['centroid_x']
        y = first['centroid_y']

        # subtract from all frames in this bout
        bout['rel_centroid_x'] = bout['centroid_x'] - x
        bout['rel_centroid_y'] = bout['centroid_y'] - y
        return bout

    df = df.groupby('bout_id', group_keys=False).apply(translate_bout)

    ## ROTATION: USE HEADING TO ROTATE ALL COORDINATES SO THAT THE ANIMAL IS FACING 'UP' (Y AXIS POSITIVE) 
    # +1.57 radians is up so want the first frame heading to be this

    def rotate_bout(bout):
        # first frame of this bout
        first = bout.iloc[0]
        h0 = first['heading']          # starting heading
        target = np.pi / 2             # you want all bouts to start "up" (≈ 1.57) # +1.57 radians

        delta = target - h0            # how much to rotate this bout by
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        x = bout['rel_centroid_x'].to_numpy()
        y = bout['rel_centroid_y'].to_numpy()

        x_rot = x * cos_d - y * sin_d
        y_rot = x * sin_d + y * cos_d

        bout['rotated_centroid_x'] = x_rot
        bout['rotated_centroid_y'] = y_rot

        # optional: rotated heading, so they all start near +pi/2
        bout['heading_rotated'] = bout['heading'] + delta

        return bout

    df = df.groupby('bout_id', group_keys=False).apply(rotate_bout)

    return df



def syllables_with_traces(df, output_directory, syllable_groups):

    def plot_time_colored_mean(ax, x, y, lw=2):
        x = np.asarray(x)
        y = np.asarray(y)

        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        t = np.linspace(0, 1, len(x) - 1)

        lc = LineCollection(segs, cmap='plasma')
        lc.set_array(t)
        lc.set_clim(0, 1)
        lc.set_linewidth(lw)
        ax.add_collection(lc)
        return lc

    os.makedirs(output_directory, exist_ok=True)

    # map syllable -> group name
    syll_to_group = {}
    for gname, syls in syllable_groups.items():
        for s in syls:
            syll_to_group[int(s)] = gname

    df = df.copy()
    df['bout_id'] = df['onset'].astype(int).cumsum()

    translated_df = translate_rotate_syllables(df)

    translated_df = translated_df.sort_values(['bout_id', 'frame_index'])
    translated_df['rel_frame'] = translated_df.groupby('bout_id').cumcount()

    # ---------- NEW: compute x/y limits per group ----------
    group_limits = {g: [None, None, None, None] for g in syllable_groups}  # xmin,xmax,ymin,ymax

    # keep per-syllable plot inputs so we don't recompute later
    per_syll = {}

    for syll, g_syl in translated_df.groupby('syllable', sort=True):

        syll = int(syll)
        if syll not in syll_to_group:
            continue  # ignore syllables not in your grouping dict

        summary = (g_syl.groupby('rel_frame').agg(
            mean_x=('rotated_centroid_x', 'mean'),
            mean_y=('rotated_centroid_y', 'mean'),
            sd_x=('rotated_centroid_x', 'std'),
            sd_y=('rotated_centroid_y', 'std'),
            n_bouts=('bout_id', 'nunique'),
            n_frames=('bout_id', 'size'),
        ).reset_index())

        total_bouts = g_syl['bout_id'].nunique()
        threshold_bouts = max(1, int(np.ceil(0.05 * total_bouts)))
        mean_trace = summary[summary['n_bouts'] >= threshold_bouts].copy()

        # save canonical CSV (unchanged)
        out_csv = os.path.join(output_directory, f"syllable_{syll}.csv")
        summary.to_csv(out_csv, index=False)

        # raw traces selection (unchanged default)
        filter_raw_within_1sd = False

        if filter_raw_within_1sd:
            stats_for_merge = summary[['rel_frame', 'mean_x', 'mean_y', 'sd_x', 'sd_y']].copy()
            g_tmp = g_syl.merge(stats_for_merge, on='rel_frame', how='left')

            g_tmp['within_1sd'] = (
                g_tmp['rotated_centroid_x'].between(g_tmp['mean_x'] - g_tmp['sd_x'],
                                                    g_tmp['mean_x'] + g_tmp['sd_x'])
                &
                g_tmp['rotated_centroid_y'].between(g_tmp['mean_y'] - g_tmp['sd_y'],
                                                    g_tmp['mean_y'] + g_tmp['sd_y'])
            )

            good_bouts_mask = g_tmp.groupby('bout_id')['within_1sd'].transform('all')
            g_plot = g_tmp[good_bouts_mask].copy()
        else:
            g_plot = g_syl

        # store what we need to plot later
        per_syll[syll] = (g_plot, mean_trace)

        # update group limits using RAW traces + mean trace
        gname = syll_to_group[syll]
        xmin, xmax, ymin, ymax = group_limits[gname]

        xvals = g_plot['rotated_centroid_x'].to_numpy()
        yvals = g_plot['rotated_centroid_y'].to_numpy()

        this_xmin = float(np.nanmin(xvals))
        this_xmax = float(np.nanmax(xvals))
        this_ymin = float(np.nanmin(yvals))
        this_ymax = float(np.nanmax(yvals))

        # also include mean_trace so it never clips
        if len(mean_trace):
            this_xmin = min(this_xmin, float(np.nanmin(mean_trace['mean_x'].to_numpy())))
            this_xmax = max(this_xmax, float(np.nanmax(mean_trace['mean_x'].to_numpy())))
            this_ymin = min(this_ymin, float(np.nanmin(mean_trace['mean_y'].to_numpy())))
            this_ymax = max(this_ymax, float(np.nanmax(mean_trace['mean_y'].to_numpy())))

        xmin = this_xmin if xmin is None else min(xmin, this_xmin)
        xmax = this_xmax if xmax is None else max(xmax, this_xmax)
        ymin = this_ymin if ymin is None else min(ymin, this_ymin)
        ymax = this_ymax if ymax is None else max(ymax, this_ymax)

        group_limits[gname] = [xmin, xmax, ymin, ymax]
    # ------------------------------------------------------

    # ---------- NEW: second loop only to SAVE (same plots, now with fixed limits) ----------
    for syll, (g_plot, mean_trace) in per_syll.items():

        gname = syll_to_group[syll]
        xmin, xmax, ymin, ymax = group_limits[gname]

        fig, ax = plt.subplots(figsize=(4, 4))

        for bout_id, g_bout in g_plot.groupby('bout_id'):
            ax.plot(
                g_bout['rotated_centroid_x'],
                g_bout['rotated_centroid_y'],
                color='gray',
                alpha=0.15,
                linewidth=0.7
            )

        plot_time_colored_mean(ax, mean_trace['mean_x'], mean_trace['mean_y'], lw=2.0)
        ax.scatter([0], [0], s=10, zorder=10, color='black')

        ax.set_aspect('equal', 'box')

        # HERE is the whole point:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(f"{syll}")
        ax.set_xlabel("")
        ax.set_ylabel("")

        out_png = os.path.join(output_directory, f"syllable_{syll}.png")
        out_pdf = os.path.join(output_directory, f"syllable_{syll}.pdf")

        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(out_pdf, dpi=600, bbox_inches='tight')
        plt.close(fig)
    # --------------------------------------------------------------------------------------






def plot_syllable_groups(syllable_folder, syllable_groups, output_folder):

    for group_name, syllables in syllable_groups.items():

        n = len(syllables)
        # ncols = min(n, 4)

        if n == 1:
            ncols = 1
        elif n <= 3:
            ncols = 1
        elif n <= 6:
            ncols = 2
        else:
            ncols = 4

        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.2, nrows * 2.2)
        )

        # flatten axes safely
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax in axes:
            ax.axis("off")

        for ax, syll in zip(axes, syllables):
            path = os.path.join(syllable_folder, f"syllable_{syll}.png")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")

            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title('')

        fig.suptitle('')

        plt.tight_layout(pad=0.05)

        out_png = os.path.join(
            output_folder, f"group_{group_name}.png"
        )
        out_pdf = os.path.join(
            output_folder, f"group_{group_name}.pdf"
        )

        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight", dpi=600)
        plt.close(fig)



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def syllable_group_trace_grids(df, output_root, syllable_groups):

    os.makedirs(output_root, exist_ok=True)
    df = df.copy()

    # 1) bout_id per animal (so bouts don't mix across tracks)
    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
    # df['bout_id'] = df.groupby('name')['onset'].cumsum()

    df['bout_id'] = (
    df.groupby('name')['onset'].cumsum()
    .astype(str)
)
    df['bout_id'] = df['name'] + "_bout" + df['bout_id']

    # 2) filter out bouts < 2 frames (still per animal, because bout_id was per name)
    bout_len = df.groupby(['name', 'bout_id']).size()
    good = bout_len[bout_len >= 2].reset_index()[['name', 'bout_id']]
    df = df.merge(good, on=['name', 'bout_id'], how='inner')



    # --- duration per bout (frames) + syllable label for that bout ---
    # (assumes each bout has one syllable; if not, use mode)
    bout_dur = (
        df.groupby('bout_id')
        .agg(
            syllable=('syllable', lambda s: int(s.mode().iloc[0]) if s.notna().any() else np.nan),
            n_frames=('frame_index', 'size')
        )
        .dropna(subset=['syllable'])
        .reset_index()
    )

    # boxplot: durations per syllable (matplotlib)
    syllables_sorted = sorted(bout_dur['syllable'].unique())
    syllables_sorted = [s for s in syllables_sorted if s <= 45]
    data = [bout_dur.loc[bout_dur['syllable'] == s, 'n_frames'].values for s in syllables_sorted]

    plt.figure(figsize=(max(12, len(syllables_sorted) * 0.35), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(syllables_sorted) + 1), syllables_sorted, rotation=90)
    plt.xlabel("Syllable")
    plt.ylabel("Bout duration (frames)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "bout_duration_boxplot_per_syllable.png"), dpi=300, bbox_inches="tight")
    plt.close()



    # --- compute central 95% bounds per syllable ---
    bounds = (
        bout_dur.groupby('syllable')['n_frames']
                .quantile([0.05, 0.95])
                .unstack()
                .rename(columns={0.05: 'lo', 0.95: 'hi'})
                .reset_index()
    )

    bout_dur = bout_dur.merge(bounds, on='syllable', how='left')

    # keep only bouts within bounds
    bout_keep = bout_dur[(bout_dur['n_frames'] >= bout_dur['lo']) & (bout_dur['n_frames'] <= bout_dur['hi'])]

    keep_uids = set(bout_keep['bout_id'])
    df = df[df['bout_id'].isin(keep_uids)].copy()



    # 3) use YOUR existing translate+rotate (expects bout_id column)
    df = translate_rotate_syllables(df)

    
    # 4) make grids per group/syllable
    for group_name, syllables in syllable_groups.items():
        group_folder = os.path.join(output_root, str(group_name))
        os.makedirs(group_folder, exist_ok=True)

        for syll in map(int, syllables):
            sub = df[df['syllable'] == syll].copy()
            if sub.empty:
                print(f"[skip] {group_name} syllable {syll}: no data")
                continue

            # collect up to 144 bouts (12x12)
            bouts = []
            for (nm, bid), bout in sub.groupby(['name', 'bout_id'], sort=False):
                bout = bout.sort_values('frame_index')
                if len(bout) >= 2:
                    bouts.append(bout)
                if len(bouts) == 144:
                    break

            if not bouts:
                print(f"[skip] {group_name} syllable {syll}: no bouts")
                continue

            # consistent limits per syllable
            allx = np.concatenate([b['rotated_centroid_x'].to_numpy() for b in bouts])
            ally = np.concatenate([b['rotated_centroid_y'].to_numpy() for b in bouts])
            xmin, xmax = float(np.nanmin(allx)), float(np.nanmax(allx))
            ymin, ymax = float(np.nanmin(ally)), float(np.nanmax(ally))
            padx = 0.05 * (xmax - xmin + 1e-9)
            pady = 0.05 * (ymax - ymin + 1e-9)
            xmin, xmax = xmin - padx, xmax + padx
            ymin, ymax = ymin - pady, ymax + pady

            fig, axes = plt.subplots(12, 12, figsize=(12 * 1.2, 12 * 1.2))
            axes = axes.ravel()

            for i, ax in enumerate(axes):
                row = i // 12
                col = i % 12

                ax.set_aspect('equal', 'box')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                # default: hide everything
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

                # hide all spines by default
                for sp in ax.spines.values():
                    sp.set_visible(False)

                # LEFT column: show Y axis + labels
                if col == 0:
                    ax.set_yticks(np.linspace(ymin, ymax, 3))
                    ax.tick_params(left=True, labelleft=True, length=3, labelsize=3)

                # BOTTOM row: show X axis + labels
                if row == 11:
                    ax.set_xticks(np.linspace(xmin, xmax, 3))
                    ax.tick_params(bottom=True, labelbottom=True, length=3, labelsize=3)
           

            for ax, bout in zip(axes, bouts):
                x = bout['rotated_centroid_x'].to_numpy()
                y = bout['rotated_centroid_y'].to_numpy()

                pts = np.column_stack([x, y]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                t = np.linspace(0, 1, len(x) - 1)

                start_color = "cornflowerblue"
                end_color   = "darkred"
                cmap = LinearSegmentedColormap.from_list("start_end", [start_color, end_color])

                pts = np.column_stack([x, y]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                t = np.linspace(0, 1, len(x) - 1)

                lc = LineCollection(segs, cmap=cmap)
                lc.set_array(t)
                lc.set_linewidth(1.2)
                ax.add_collection(lc)


                ax.scatter([0], [0], s=0.5, color='black', zorder=5)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect('equal', 'box')
                # ax.axis("off")
         

                n_frames = len(bout)
                ax.text(
                    0.02, 0.98,
                    f"{n_frames}f",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=4
                )

            fig.suptitle(f"{group_name} — syllable {syll}", y=0.995)
            plt.tight_layout(pad=0.05)

            out_png = os.path.join(group_folder, f"syllable_{syll}_grid.png")
            out_pdf = os.path.join(group_folder, f"syllable_{syll}_grid.pdf")
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
            plt.close(fig)

            print(f"[saved] {out_png}")







import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def syllable_group_trace_grids_arclen_normalized(
    df,
    output_root,
    syllable_groups,
    n_resample=25,
    grid_n=12,
    max_bouts=None,
    min_total_len=1e-6,
    duration_quantiles=(0.05, 0.95),  # central 90% by default
    max_syllable=45,                  # set None if you don't want this filter
):
    """
    Grid plot, but each bout is:
      1) filtered to bouts with duration within central quantiles per syllable (default 5–95%)
      2) translate+rotated by your translate_rotate_syllables(df)
      3) arc-length reparameterized (uniform sampling along the path)
      4) scaled to unit path length (small bouts enlarge, big bouts compress)

    Saves per group/syllable into:
      output_root/<group_name>/syllable_<syll>_grid_arclen_normalized.png/.pdf
    """

    os.makedirs(output_root, exist_ok=True)
    df = df.copy()

    # ------------------------------------------------------------
    # 1) bout_id per animal
    # ------------------------------------------------------------
    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
    bout_num = df.groupby('name')['onset'].cumsum().astype(int)
    df['bout_id'] = df['name'].astype(str) + "_bout" + bout_num.astype(str)

    # filter out bouts < 2 frames
    bout_len = df.groupby('bout_id').size()
    df = df[df['bout_id'].isin(bout_len[bout_len >= 2].index)].copy()

    # optional: only keep syllables up to max_syllable
    if max_syllable is not None:
        df = df[df['syllable'].astype(float) <= max_syllable].copy()

    # ------------------------------------------------------------
    # 2) CENTRAL 90% DURATION FILTER (per syllable)
    # ------------------------------------------------------------
    # bout duration + syllable label (mode per bout)
    bout_dur = (
        df.groupby('bout_id')
          .agg(
              syllable=('syllable', lambda s: int(s.mode().iloc[0]) if s.notna().any() else np.nan),
              n_frames=('frame_index', 'size')
          )
          .dropna(subset=['syllable'])
          .reset_index()
    )
    bout_dur['syllable'] = bout_dur['syllable'].astype(int)

    qlo, qhi = duration_quantiles  # e.g. (0.05, 0.95)
    bounds = (
        bout_dur.groupby('syllable')['n_frames']
                .quantile([qlo, qhi])
                .unstack()
                .rename(columns={qlo: 'lo', qhi: 'hi'})
                .reset_index()
    )

    bout_dur = bout_dur.merge(bounds, on='syllable', how='left')

    # keep bouts within central quantiles for THEIR syllable
    bout_keep = bout_dur[
        (bout_dur['n_frames'] >= bout_dur['lo']) &
        (bout_dur['n_frames'] <= bout_dur['hi'])
    ].copy()

    keep_ids = set(bout_keep['bout_id'])
    df = df[df['bout_id'].isin(keep_ids)].copy()

    # ------------------------------------------------------------
    # 3) translate + rotate (your existing function)
    # ------------------------------------------------------------
    df = translate_rotate_syllables(df)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def arclen_resample_and_scale(x, y, n=n_resample, min_len=min_total_len):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return None, None

        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx * dx + dy * dy)

        L = float(np.nansum(ds))
        if (not np.isfinite(L)) or (L < min_len):
            return None, None

        s = np.concatenate([[0.0], np.cumsum(ds)])  # length len(x)
        if s[-1] <= 0:
            return None, None

        u_old = s / s[-1]                  # 0..1 along arc-length
        u_new = np.linspace(0.0, 1.0, n)   # target 0..1

        x_new = np.interp(u_new, u_old, x)
        y_new = np.interp(u_new, u_old, y)

        # unit-length scaling (so total path length -> 1)
        x_new = x_new / L
        y_new = y_new / L

        return x_new, y_new

    def add_time_colored_line(ax, x, y, cmap, lw=1.2):
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(x) - 1)
        lc = LineCollection(segs, cmap=cmap)
        lc.set_array(t)
        lc.set_linewidth(lw)
        ax.add_collection(lc)

    start_color = "cornflowerblue"
    end_color   = "darkred"
    cmap = LinearSegmentedColormap.from_list("start_end", [start_color, end_color])

    if max_bouts is None:
        max_bouts = grid_n * grid_n

    # ------------------------------------------------------------
    # 4) grids per group/syllable (ARC-LENGTH normalized + scaled)
    # ------------------------------------------------------------
    for group_name, syllables in syllable_groups.items():
        group_folder = os.path.join(output_root, str(group_name))
        os.makedirs(group_folder, exist_ok=True)

        for syll in map(int, syllables):
            if (max_syllable is not None) and (syll > max_syllable):
                continue

            sub = df[df['syllable'] == syll].copy()
            if sub.empty:
                print(f"[skip] {group_name} syllable {syll}: no data after duration filter")
                continue

            bouts_xy = []
            for bid, bout in sub.groupby('bout_id', sort=False):
                bout = bout.sort_values('frame_index')
                x = bout['rotated_centroid_x'].to_numpy()
                y = bout['rotated_centroid_y'].to_numpy()

                xr, yr = arclen_resample_and_scale(x, y, n=n_resample)
                if xr is None:
                    continue

                bouts_xy.append((xr, yr))
                if len(bouts_xy) >= max_bouts:
                    break

            if not bouts_xy:
                print(f"[skip] {group_name} syllable {syll}: no valid bouts after arc-length filter")
                continue

            # limits from normalized+scaled data
            allx = np.concatenate([b[0] for b in bouts_xy])
            ally = np.concatenate([b[1] for b in bouts_xy])
            xmin, xmax = float(np.nanmin(allx)), float(np.nanmax(allx))
            ymin, ymax = float(np.nanmin(ally)), float(np.nanmax(ally))
            padx = 0.05 * (xmax - xmin + 1e-9)
            pady = 0.05 * (ymax - ymin + 1e-9)
            xmin, xmax = xmin - padx, xmax + padx
            ymin, ymax = ymin - pady, ymax + pady

            fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 1.2, grid_n * 1.2))
            axes = axes.ravel()

            for i, ax in enumerate(axes):
                row = i // grid_n
                col = i % grid_n

                ax.set_aspect('equal', 'box')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

                

                for sp in ax.spines.values():
                    sp.set_visible(False)

                if col == 0:
                    ax.set_yticks(np.linspace(ymin, ymax, 3))
                    ax.tick_params(left=True, labelleft=True, length=3, labelsize=3)

                if row == grid_n - 1:
                    ax.set_xticks(np.linspace(xmin, xmax, 3))
                    ax.tick_params(bottom=True, labelbottom=True, length=3, labelsize=3)

            for ax, (x, y) in zip(axes, bouts_xy):
                add_time_colored_line(ax, x, y, cmap=cmap, lw=1.2)
                ax.scatter([0], [0], s=0.15, color='black', zorder=5)

            qlo, qhi = duration_quantiles
            fig.suptitle(
                f"{group_name} — syllable {syll} (arc-length normalized, unit-length)\n"
                f"durations within central {int((qhi-qlo)*100)}% ({qlo:.2f}–{qhi:.2f}); n_resample={n_resample}",
                y=0.995
            )
            plt.tight_layout(pad=0.05)

            out_png = os.path.join(group_folder, f"syllable_{syll}_grid_arclen_normalized.png")
            out_pdf = os.path.join(group_folder, f"syllable_{syll}_grid_arclen_normalized.pdf")
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
            plt.close(fig)

            print(f"[saved] {out_png}")





# def syllable_group_quantifications(df, output_root, syllable_groups):
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     os.makedirs(output_root, exist_ok=True)
#     df = df.copy()

#     # -------------------------------
#     # 0) map syllable -> group label
#     # -------------------------------
#     syll_to_group = {}
#     for gname, syls in syllable_groups.items():
#         for s in syls:
#             syll_to_group[int(s)] = str(gname)

#     df['syllable'] = pd.to_numeric(df['syllable'], errors='coerce')
#     df = df.dropna(subset=['syllable']).copy()
#     df['syllable'] = df['syllable'].astype(int)

#     # If you only care up to 45:
#     df = df[df['syllable'] <= 45].copy()

#     df['syllable_group'] = df['syllable'].map(syll_to_group)
#     df = df.dropna(subset=['syllable_group']).copy()

#     # ------------------------------------------------------------
#     # 1) Make bout_id per animal (name) so bouts never mix animals
#     # ------------------------------------------------------------
#     df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
#     bout_num = df.groupby('name')['onset'].cumsum().astype(int)
#     df['bout_id'] = df['name'].astype(str) + "_bout" + bout_num.astype(str)

#     # filter out bouts < 2 frames
#     bout_len = df.groupby('bout_id').size()
#     df = df[df['bout_id'].isin(bout_len[bout_len >= 2].index)].copy()

#     # ------------------------------------------------------------
#     # 2) Bout duration per syllable + central 90% bounds (5–95%)
#     # ------------------------------------------------------------
#     bout_info = (
#         df.groupby('bout_id')
#           .agg(
#               syllable=('syllable', lambda s: int(s.mode().iloc[0]) if s.notna().any() else np.nan),
#               syllable_group=('syllable_group', lambda s: str(s.mode().iloc[0]) if s.notna().any() else np.nan),
#               n_frames=('frame_index', 'size')
#           )
#           .dropna(subset=['syllable', 'syllable_group'])
#           .reset_index()
#     )
#     bout_info['syllable'] = bout_info['syllable'].astype(int)

#     bounds = (
#         bout_info.groupby('syllable')['n_frames']
#                  .quantile([0.05, 0.95])
#                  .unstack()
#                  .rename(columns={0.05: 'lo', 0.95: 'hi'})
#                  .reset_index()
#     )
#     bout_info = bout_info.merge(bounds, on='syllable', how='left')

#     # keep only bouts within central 90% for THEIR syllable
#     bout_keep = bout_info[
#         (bout_info['n_frames'] >= bout_info['lo']) &
#         (bout_info['n_frames'] <= bout_info['hi'])
#     ].copy()

#     keep_ids = set(bout_keep['bout_id'])
#     df = df[df['bout_id'].isin(keep_ids)].copy()

#     # ------------------------------------------------------------
#     # 2b) SAVE duration boxplot (syllables <= 45)
#     # ------------------------------------------------------------
#     syllables_sorted = sorted(bout_keep['syllable'].unique())
#     data = [bout_keep.loc[bout_keep['syllable'] == s, 'n_frames'].values for s in syllables_sorted]

#     plt.figure(figsize=(max(12, len(syllables_sorted) * 0.35), 5))
#     plt.boxplot(data, showfliers=False)
#     plt.xticks(range(1, len(syllables_sorted) + 1), syllables_sorted, rotation=90)
#     plt.xlabel("Syllable")
#     plt.ylabel("Bout duration (frames)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_root, "bout_duration_boxplot_per_syllable_le45.png"), dpi=300, bbox_inches="tight")
#     plt.close()

#     # ------------------------------------------------------------
#     # 3) Compute speed per frame (px/frame) inside each bout
#     # ------------------------------------------------------------
#     df = df.sort_values(['bout_id', 'frame_index']).copy()
#     dx = df.groupby('bout_id')['centroid_x'].diff()
#     dy = df.groupby('bout_id')['centroid_y'].diff()
#     df['speed'] = np.sqrt(dx**2 + dy**2)

#     # IMPORTANT: first frame per bout stays NaN, so t=0 is effectively dropped
#     # (you already discovered this is what you want)
#     # df['speed'] = df['speed'].fillna(np.nan)  # not needed, but harmless

#     # ------------------------------------------------------------
#     # 3b) Bout-level median speed summaries (for boxplots)
#     # ------------------------------------------------------------
#     bout_speed = (
#         df.groupby('bout_id')
#           .agg(
#               syllable=('syllable', lambda s: int(s.mode().iloc[0])),
#               syllable_group=('syllable_group', lambda s: str(s.mode().iloc[0])),
#               median_speed=('speed', 'median')
#           )
#           .dropna(subset=['median_speed'])
#           .reset_index()
#     )

#     # boxplot: median speed by GROUP
#     groups_sorted = list(syllable_groups.keys())
#     data_g = [bout_speed.loc[bout_speed['syllable_group'] == g, 'median_speed'].values for g in groups_sorted if g in set(bout_speed['syllable_group'])]

#     plt.figure(figsize=(max(10, len(data_g) * 0.8), 5))
#     plt.boxplot(data_g, showfliers=False)
#     plt.xticks(range(1, len(data_g) + 1), [g for g in groups_sorted if g in set(bout_speed['syllable_group'])], rotation=45, ha='right')
#     plt.xlabel("Syllable group")
#     plt.ylabel("Bout median speed (px/frame)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_root, "median_speed_boxplot_by_group.png"), dpi=300, bbox_inches="tight")
#     plt.close()

#     # boxplot: median speed by SYLLABLE (<=45)
#     syllables_sorted = sorted(bout_speed['syllable'].unique())
#     data_s = [bout_speed.loc[bout_speed['syllable'] == s, 'median_speed'].values for s in syllables_sorted]

#     plt.figure(figsize=(max(12, len(syllables_sorted) * 0.35), 5))
#     plt.boxplot(data_s, showfliers=False)
#     plt.xticks(range(1, len(syllables_sorted) + 1), syllables_sorted, rotation=90)
#     plt.xlabel("Syllable")
#     plt.ylabel("Bout median speed (px/frame)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_root, "median_speed_boxplot_by_syllable_le45.png"), dpi=300, bbox_inches="tight")
#     plt.close()


#     # ------------------------------------------------------------
#     # 3c) Per-GROUP boxplots: median speed per syllable
#     # ------------------------------------------------------------
#     for group_name, syllables in syllable_groups.items():
#         group_folder = os.path.join(output_root, str(group_name))
#         os.makedirs(group_folder, exist_ok=True)

#         syls = [int(s) for s in syllables if int(s) <= 45]
#         sub = bout_speed[bout_speed['syllable'].isin(syls)].copy()

#         if sub.empty:
#             print(f"[skip] {group_name}: no data for boxplot")
#             continue

#         # order syllables numerically
#         syls_sorted = sorted(sub['syllable'].unique())
#         data = [
#             sub.loc[sub['syllable'] == s, 'median_speed'].values
#             for s in syls_sorted
#         ]

#         plt.figure(figsize=(max(6, len(syls_sorted) * 0.6), 4))
#         plt.boxplot(data, showfliers=True)
#         plt.xticks(range(1, len(syls_sorted) + 1), syls_sorted, rotation=45)
#         plt.xlabel("Syllable")
#         plt.ylabel("Bout median speed (px / frame)")
#         plt.title(f"{group_name} — median speed per syllable")
#         plt.tight_layout()

#         out_png = os.path.join(group_folder, "median_speed_boxplot_per_syllable.png")
#         out_pdf = os.path.join(group_folder, "median_speed_boxplot_per_syllable.pdf")
#         plt.savefig(out_png, dpi=300, bbox_inches="tight")
#         plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
#         plt.close()

#         print(f"[saved] {out_png}")


#     # ------------------------------------------------------------
#     # 4) Save per-syllable raw speed traces (t=0 dropped automatically)
#     #    inside output_root/<group_name>/
#     # ------------------------------------------------------------
#     for group_name, syllables in syllable_groups.items():
#         group_folder = os.path.join(output_root, str(group_name))
#         os.makedirs(group_folder, exist_ok=True)

#         for syll in map(int, syllables):
#             if syll > 45:
#                 continue

#             sub = df[df['syllable'] == syll].copy()
#             if sub.empty:
#                 print(f"[skip] {group_name} syllable {syll}: no bouts after 90% duration filter")
#                 continue

#             plt.figure(figsize=(7, 4))

#             n_plotted = 0
#             for bid, bout in sub.groupby('bout_id', sort=False):
#                 bout = bout.sort_values('frame_index')
#                 y = bout['speed'].to_numpy()
#                 x = np.arange(len(y))

#                 # Drop t=0 (NaN speed) and any other NaNs
#                 mask = ~np.isnan(y)
#                 if mask.sum() < 2:
#                     continue

#                 plt.plot(x[mask], y[mask], linewidth=0.8, alpha=0.20)
#                 n_plotted += 1

#             plt.xlabel("Time (frames)")
#             plt.ylabel("Speed (px / frame)")
#             plt.title(
#                 f"{group_name} — syllable {syll}\n"
#                 f"raw speed traces (n bouts = {n_plotted})\n"
#                 f"durations within central 90% (5–95%)"
#             )
#             plt.tight_layout()

#             out_png = os.path.join(group_folder, f"syllable_{syll}_speed_raw.png")
#             out_pdf = os.path.join(group_folder, f"syllable_{syll}_speed_raw.pdf")
#             plt.savefig(out_png, dpi=300, bbox_inches="tight")
#             plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
#             plt.close()

#             print(f"[saved] {out_png}")


def syllable_group_quantifications(df, output_root, syllable_groups):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(output_root, exist_ok=True)
    df = df.copy()

    # -------------------------------
    # 0) map syllable -> group label
    # -------------------------------
    syll_to_group = {}
    for gname, syls in syllable_groups.items():
        for s in syls:
            syll_to_group[int(s)] = str(gname)

    df['syllable'] = pd.to_numeric(df['syllable'], errors='coerce')
    df = df.dropna(subset=['syllable']).copy()
    df['syllable'] = df['syllable'].astype(int)

    # Only care up to and including 45
    df = df[df['syllable'] <= 45].copy()

    df['syllable_group'] = df['syllable'].map(syll_to_group)
    df = df.dropna(subset=['syllable_group']).copy()

    # ------------------------------------------------------------
    # 1) Make bout_id per animal (name) so bouts never mix animals
    # ------------------------------------------------------------
    df = df.sort_values(['name', 'frame_index']).reset_index(drop=True)
    bout_num = df.groupby('name')['onset'].cumsum().astype(int)
    df['bout_id'] = df['name'].astype(str) + "_bout" + bout_num.astype(str)

    # filter out bouts < 2 frames
    bout_len = df.groupby('bout_id').size()
    df = df[df['bout_id'].isin(bout_len[bout_len >= 2].index)].copy()

    # ------------------------------------------------------------
    # 2) Bout duration per syllable + central 90% bounds (5–95%)
    # ------------------------------------------------------------
    bout_info = (
        df.groupby('bout_id')
          .agg(
              syllable=('syllable', lambda s: int(s.mode().iloc[0]) if s.notna().any() else np.nan),
              syllable_group=('syllable_group', lambda s: str(s.mode().iloc[0]) if s.notna().any() else np.nan),
              n_frames=('frame_index', 'size')
          )
          .dropna(subset=['syllable', 'syllable_group'])
          .reset_index()
    )
    bout_info['syllable'] = bout_info['syllable'].astype(int)

    bounds = (
        bout_info.groupby('syllable')['n_frames']
                 .quantile([0.05, 0.95])
                 .unstack()
                 .rename(columns={0.05: 'lo', 0.95: 'hi'})
                 .reset_index()
    )
    bout_info = bout_info.merge(bounds, on='syllable', how='left')

    bout_keep = bout_info[
        (bout_info['n_frames'] >= bout_info['lo']) &
        (bout_info['n_frames'] <= bout_info['hi'])
    ].copy()

    keep_ids = set(bout_keep['bout_id'])
    df = df[df['bout_id'].isin(keep_ids)].copy()

    # ------------------------------------------------------------
    # 2b) SAVE duration boxplot (syllables <= 45)
    # ------------------------------------------------------------
    syllables_sorted = sorted(bout_keep['syllable'].unique())
    data = [bout_keep.loc[bout_keep['syllable'] == s, 'n_frames'].values for s in syllables_sorted]

    plt.figure(figsize=(max(12, len(syllables_sorted) * 0.35), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(syllables_sorted) + 1), syllables_sorted, rotation=90)
    plt.xlabel("Syllable")
    plt.ylabel("Bout duration (frames)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "bout_duration_boxplot_per_syllable_le45.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # 3) Compute per-frame speed + acceleration inside each bout
    #    + heading from centroid motion + |Δθ|
    # ------------------------------------------------------------
    df = df.sort_values(['bout_id', 'frame_index']).copy()

    # dx/dy per frame (per bout)
    df['dx'] = df.groupby('bout_id')['centroid_x'].diff()
    df['dy'] = df.groupby('bout_id')['centroid_y'].diff()

    # speed (px/frame) — first frame per bout = NaN
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # accel (px/frame^2) — first TWO frames per bout will be NaN
    df['accel'] = df.groupby('bout_id')['speed'].diff()

    # heading from centroid motion: atan2(dy, dx)
    # first frame per bout NaN; also any stationary step (dx=dy=0) => heading=0 by atan2(0,0) but we set to NaN
    df['heading_rad'] = np.arctan2(df['dy'], df['dx'])
    df.loc[(df['dx'] == 0) & (df['dy'] == 0), 'heading_rad'] = np.nan

    # Δθ per frame (per bout), wrapped to [-pi, pi]
    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    df['dtheta'] = df.groupby('bout_id')['heading_rad'].diff()
    df['dtheta'] = wrap_to_pi(df['dtheta'].to_numpy())
    df['abs_dtheta'] = np.abs(df['dtheta'])

        # ------------------------------------------------------------
    # 3c) Net displacement + path length per frame (centroid-based)
    # ------------------------------------------------------------
    # path length per bout = sum of per-frame step lengths (speed), ignoring NaNs
    path_len = df.groupby('bout_id')['speed'].transform(lambda s: np.nansum(s.to_numpy()))
    df['path_len'] = path_len

    # net displacement per bout = distance between first and last centroid positions
    first_x = df.groupby('bout_id')['centroid_x'].transform('first')
    first_y = df.groupby('bout_id')['centroid_y'].transform('first')
    last_x  = df.groupby('bout_id')['centroid_x'].transform('last')
    last_y  = df.groupby('bout_id')['centroid_y'].transform('last')
    df['net_disp'] = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)

    # (optional but very informative) straightness ratio: 1 = perfectly straight
    df['straightness'] = df['net_disp'] / (df['path_len'] + 1e-9)


    # ------------------------------------------------------------
    # 3b) Bout-level summaries: median + max for each metric
    # ------------------------------------------------------------
    def mode_int(x):
        x = x.dropna()
        return int(x.mode().iloc[0]) if len(x) else np.nan

    def mode_str(x):
        x = x.dropna()
        return str(x.mode().iloc[0]) if len(x) else np.nan

    bout_metrics = (
        df.groupby('bout_id')
          .agg(
              syllable=('syllable', mode_int),
              syllable_group=('syllable_group', mode_str),

              median_speed=('speed', 'median'),
              max_speed=('speed', 'max'),

              median_abs_accel=('accel', lambda s: np.nanmedian(np.abs(s.to_numpy()))),
              max_abs_accel=('accel', lambda s: np.nanmax(np.abs(s.to_numpy()))),

              median_abs_dtheta=('abs_dtheta', 'median'),
              max_abs_dtheta=('abs_dtheta', 'max'),

              net_disp=('net_disp', 'first'),
              path_len=('path_len', 'first'),
              straightness=('straightness', 'first'),
          )
          .reset_index()
    )
    bout_metrics = bout_metrics.dropna(subset=['syllable', 'syllable_group']).copy()

    # ------------------------------------------------------------
    # 5) OVERALL BOXPLOTS: by GROUP and by SYLLABLE (<=45)
    # ------------------------------------------------------------
    def save_boxplot(values_by_cat, labels, ylabel, outpath, title=None, rotate=45, showfliers=False):
        plt.figure(figsize=(max(10, len(labels) * 0.8), 5))
        plt.boxplot(values_by_cat, showfliers=showfliers)
        plt.xticks(range(1, len(labels) + 1), labels, rotation=rotate,
                   ha='right' if rotate else 'center')
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()

    groups_sorted = [str(g) for g in syllable_groups.keys()
                     if str(g) in set(bout_metrics['syllable_group'])]
    sylls_sorted_all = sorted([int(s) for s in bout_metrics['syllable'].dropna().unique() if int(s) <= 45])

    metric_specs = [
        ("median_speed",      "Bout median speed (px/frame)",      "median_speed"),
        ("max_speed",         "Bout max speed (px/frame)",         "max_speed"),
        ("median_abs_accel",  "Bout median |accel| (px/frame²)",    "median_abs_accel"),
        ("max_abs_accel",     "Bout max |accel| (px/frame²)",       "max_abs_accel"),
        ("median_abs_dtheta", "Bout median |Δθ| (rad/frame)",       "median_abs_dtheta"),
        ("max_abs_dtheta",    "Bout max |Δθ| (rad/frame)",          "max_abs_dtheta"),
        ("net_disp",       "Bout net displacement (px)",          "net_disp"),
        ("path_len",       "Bout path length (px)",               "path_len"),
        ("straightness",   "Bout straightness (net/path)",        "straightness") ]


    # by GROUP
    for col, ylabel, stem in metric_specs:
        data_g = [bout_metrics.loc[bout_metrics['syllable_group'] == g, col].dropna().values for g in groups_sorted]
        save_boxplot(
            data_g, groups_sorted, ylabel,
            os.path.join(output_root, f"{stem}_boxplot_by_group.png"),
            title=f"{ylabel} by syllable group",
            rotate=45,
            showfliers=False
        )

    # by SYLLABLE
    for col, ylabel, stem in metric_specs:
        data_s = [bout_metrics.loc[bout_metrics['syllable'] == s, col].dropna().values for s in sylls_sorted_all]
        save_boxplot(
            data_s, [str(s) for s in sylls_sorted_all], ylabel,
            os.path.join(output_root, f"{stem}_boxplot_by_syllable_le45.png"),
            title=f"{ylabel} by syllable (≤45)",
            rotate=90,
            showfliers=False
        )

    # ------------------------------------------------------------
    # 6) PER-GROUP BOXPLOTS: per syllable inside each group folder
    # ------------------------------------------------------------
    for group_name, syllables in syllable_groups.items():
        group_folder = os.path.join(output_root, str(group_name))
        os.makedirs(group_folder, exist_ok=True)

        syls = sorted([int(s) for s in syllables if int(s) <= 45])
        subm = bout_metrics[bout_metrics['syllable'].isin(syls)].copy()
        if subm.empty:
            print(f"[skip] {group_name}: no data for per-syllable boxplots")
            continue

        syls_present = sorted(subm['syllable'].unique())

        for col, ylabel, stem in metric_specs:
            data = [subm.loc[subm['syllable'] == s, col].dropna().values for s in syls_present]
            out_png = os.path.join(group_folder, f"{stem}_boxplot_per_syllable.png")

            save_boxplot(
                data, [str(s) for s in syls_present], ylabel, out_png,
                title=f"{group_name} — {ylabel} per syllable",
                rotate=45,
                showfliers=True  # keep fliers here so you can see skew
            )

            # PDF
            out_pdf = os.path.join(group_folder, f"{stem}_boxplot_per_syllable.pdf")
            plt.figure(figsize=(max(6, len(syls_present) * 0.6), 4))
            plt.boxplot(data, showfliers=True)
            plt.xticks(range(1, len(syls_present) + 1), [str(s) for s in syls_present], rotation=45)
            plt.ylabel(ylabel)
            plt.title(f"{group_name} — {ylabel} per syllable")
            plt.tight_layout()
            plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
            plt.close()

            print(f"[saved] {out_png}")

    # ------------------------------------------------------------
    # 7) RAW TRACES PER SYLLABLE (within each group folder)
    # ------------------------------------------------------------
    trace_specs = [
        ("speed",      "Speed (px / frame)",         "speed_raw"),
        ("accel",      "Acceleration (px / frame²)", "accel_raw"),
        ("abs_dtheta", "|Δθ| (rad / frame)",         "abs_dtheta_raw"),
    ]

    for group_name, syllables in syllable_groups.items():
        group_folder = os.path.join(output_root, str(group_name))
        os.makedirs(group_folder, exist_ok=True)

        for syll in map(int, syllables):
            if syll > 45:
                continue

            sub = df[df['syllable'] == syll].copy()
            if sub.empty:
                print(f"[skip] {group_name} syllable {syll}: no data after duration filter")
                continue

            for ycol, ylabel, stem in trace_specs:
                plt.figure(figsize=(7, 4))

                n_plotted = 0
                for bid, bout in sub.groupby('bout_id', sort=False):
                    bout = bout.sort_values('frame_index')
                    y = bout[ycol].to_numpy()
                    x = np.arange(len(y))

                    mask = ~np.isnan(y)
                    if mask.sum() < 2:
                        continue

                    plt.plot(x[mask], y[mask], linewidth=0.8, alpha=0.20)
                    n_plotted += 1

                plt.xlabel("Time (frames)")
                plt.ylabel(ylabel)
                plt.title(
                    f"{group_name} — syllable {syll}\n"
                    f"raw traces: {ycol} (n bouts = {n_plotted})\n"
                    f"durations within central 90% (5–95%)"
                )
                plt.tight_layout()

                out_png = os.path.join(group_folder, f"syllable_{syll}_{stem}.png")
                out_pdf = os.path.join(group_folder, f"syllable_{syll}_{stem}.pdf")
                plt.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
                plt.close()

                print(f"[saved] {out_png}")

    print("[done] speed + accel + centroid-heading |Δθ| quantifications complete.")



df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
output_directory = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram/individual'

syllable_groups = {
    "orange1": [33, 43], ##orange
    "orange": [44, 45], ##orange
    "orange2": [2, 19], ##orange
    "orange3": [28, 37], ##orange
    'orange4': [6, 11], ##orange
    'blue': [1], ##blue
    'green': [16], ##green
    'green2': [39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42], ##green
    'green3': [5, 36, 13, 15, 4 ,21, ], ##green
    'green4': [3, 32], ##green
    'green5': [0, 14, 38, 18, 20], ##green
    'green6': [34, 8, 41, 7, 31], ##green
}
syllables_with_traces(df, output_directory, syllable_groups)



syllable_folder = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram/individual'
output_folder = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram'
syllable_groups = {
    "orange1": [33, 43], ##orange
    "orange": [44, 45], ##orange
    "orange2": [2, 19], ##orange
    "orange3": [28, 37], ##orange
    'orange4': [6, 11], ##orange
    'blue': [1], ##blue
    'green': [16], ##green
    'green2': [39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42], ##green
    'green3': [5, 36, 13, 15, 4 ,21, ], ##green
    'green4': [3, 32], ##green
    'green5': [0, 14, 38, 18, 20], ##green
    'green6': [34, 8, 41, 7, 31], ##green
}

plot_syllable_groups(syllable_folder, syllable_groups, output_folder)





df = pd.read_csv('/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/moseq_df.csv')
output_root = '/Users/cochral/Desktop/MOSEQ/KEYPOINT-KAPPA1500/plots/grouped-dendrogram/testing-groups'
syllable_groups = {
    "orange1": [33, 43], ##orange
    "orange": [44, 45], ##orange
    "orange2": [2, 19], ##orange
    "orange3": [28, 37], ##orange
    'orange4': [6, 11], ##orange
    'blue': [1], ##blue
    'green': [16], ##green
    'green2': [39, 27, 40, 26, 25, 22, 30, 9, 23, 10, 24, 12, 17, 35, 29, 42], ##green
    'green3': [5, 36, 13, 15, 4 ,21, ], ##green
    'green4': [3, 32], ##green
    'green5': [0, 14, 38, 18, 20], ##green
    'green6': [34, 8, 41, 7, 31], ##green
}




syllable_group_trace_grids(df, output_root, syllable_groups)
syllable_group_trace_grids_arclen_normalized(df, output_root, syllable_groups)
syllable_group_quantifications(df, output_root, syllable_groups)