import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import os

def compute_digging_grid_search_plot(
    df,
    ground_truth_df,
    save_dir,
    is_moving_thresholds=[0.1, 0.2, 0.3, 0.4],
    cumulative_displacement_windows=[5, 10, 15],
    displacement_rate_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
    std_windows=[5, 10, 15],
    std_thresholds=[0.1, 0.2, 0.3, 0.4],
    digging_windows=[20, 30, 40, 50]
):
    os.makedirs(save_dir, exist_ok=True)
    param_grid = list(product(
        is_moving_thresholds,
        cumulative_displacement_windows,
        displacement_rate_thresholds,
        std_windows,
        std_thresholds,
        digging_windows
    ))

    for (is_move_thres, cum_disp_win, disp_rate_thres, std_win, std_thres, dig_win) in param_grid:
        temp_df = df.copy()

        # Smooth x and y
        temp_df['x'] = temp_df['x_body'].rolling(window=5, min_periods=1).mean()
        temp_df['y'] = temp_df['y_body'].rolling(window=5, min_periods=1).mean()

        # dx, dy, distance, is_moving
        temp_df['dx'] = temp_df.groupby('track_id')['x'].diff().fillna(0)
        temp_df['dy'] = temp_df.groupby('track_id')['y'].diff().fillna(0)
        temp_df['distance'] = np.sqrt(temp_df['dx']**2 + temp_df['dy']**2)
        temp_df['is_moving'] = temp_df['distance'] > is_move_thres

        # Cumulative displacement and rate
        temp_df['cumulative_displacement'] = temp_df.groupby('track_id')['distance'].cumsum()
        temp_df['cumulative_displacement_rate'] = (
            temp_df.groupby('track_id')['cumulative_displacement']
            .apply(lambda x: x.diff(cum_disp_win) / cum_disp_win)
            .fillna(0)
        )

        # Standard deviation
        temp_df['x_std'] = temp_df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=std_win, min_periods=1).std())
        temp_df['y_std'] = temp_df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=std_win, min_periods=1).std())
        temp_df['overall_std'] = np.sqrt(temp_df['x_std']**2 + temp_df['y_std']**2)

        # Final movement
        temp_df['final_movement'] = (
            (temp_df['cumulative_displacement_rate'] > disp_rate_thres) |
            ((temp_df['overall_std'] > std_thres) & (temp_df['is_moving']))
        )

        # Digging status
        temp_df['digging_status'] = (
            temp_df.groupby('track_id')['final_movement']
            .transform(lambda x: (~x).rolling(window=dig_win, center=False)
            .apply(lambda r: r.sum() >= (dig_win / 2)).fillna(0).astype(bool))
        )

        # Compute predicted number digging per frame
        digging_counts = temp_df.groupby('frame')['digging_status'].sum().reset_index()

        # -------------------
        # Plotting: Two sns.lineplot calls
        # -------------------
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.lineplot(data=digging_counts, x='frame', y='digging_status', label='Predicted', ax=ax)
        sns.lineplot(data=ground_truth_df, x='frame', y='number_digging', label='Ground Truth', ax=ax)

        ax.set_title(f'Digging Over Time\n'
                     f'is_moving>{is_move_thres}, disp_win={cum_disp_win}, rate>{disp_rate_thres}, '
                     f'std_win={std_win}, std>{std_thres}, dig_win={dig_win}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number Digging')
        ax.grid(True)
        ax.legend()

        # Save the plot
        filename = (f"digging_plot_m{is_move_thres}_cdw{cum_disp_win}_cdrt{disp_rate_thres}_"
                    f"sw{std_win}_st{std_thres}_dw{dig_win}.png").replace('.', 'p')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()


df = pd.read_feather('/camp/lab/windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-paramaters-gridsearch/n10/2025-03-03_11-40-39_td1.tracks.feather')
ground_truth_d = pd.read_excel('/camp/lab/windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-paramaters-gridsearch/n10/ground-truth.xlsx')
save_dir = '/camp/lab/windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-paramaters-gridsearch/n10'