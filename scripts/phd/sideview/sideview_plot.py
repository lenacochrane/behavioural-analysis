import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
import matplotlib as mpl

################################################################
####### --- PLOTTING SIDEVIEW ANALYSIS FOR NDD-GENES --- #######
################################################################

output = '/Users/cochral/repos/behavioural-analysis/plots/phd/ndd-sideview'

condition_labels = {
    "T21-A08": "Control",
    'T27-A01': 'Control',
    
    "T21-B09": "NRXN1 KD",
    "T25-C03": "CTNND2 KD",
    "T25-C02": "CHD8 KD",
    'T25-B09': 'DLG4 KD',
    'T25-B02': 'SHANK3 KD',
    'T24-C09': 'SETD1A KD', 
    'T24-C01': 'CUL1 KD', 
    'T25-C07': 'TRIP12 KD',
}


################################################################
######## --------- PLOT PERCENTAGE IN CLUSTERS -------- ########
################################################################

def plot_percent_in_clusters():

    df = pd.read_csv('/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/percent_in_clusters.csv')
    print('Percentage in clusters data loaded:')


    palette = {
        "Control": "#0d82d5",
        "NRXN1 KD": "#52cdfa",
        "CTNND2 KD": "#52cdfa",
        "CHD8 KD": "#52cdfa",
        "DLG4 KD": "#52cdfa",
        "SHANK3 KD": "#52cdfa",
        "SETD1A KD": "#52cdfa",
        "TRIP12 KD": "#52cdfa",
        'CUL1 KD': "#52cdfa",

    }

    bin_size = 300 #60s
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    average_df = (
        df.groupby(["control", "condition", "file_id", "frame_bin"], as_index=False)["percent_in_clusters"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(average_df["control"].unique()):

        control_df = average_df[average_df["control"] == control].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="percent_in_clusters",
                hue="plot_label",
                palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Percentage in Clusters")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(output, f"{cond_label}__percent_in_clusters.pdf")
            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()


################################################################
########## --------- PLOT NUMBER IN CLUSTERS -------- ##########
################################################################
def plot_number_in_clusters():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/percent_in_clusters.csv'
    )

    print('Number in clusters data loaded:')

    palette = {
        "Control": "#0d82d5",
        "NRXN1 KD": "#52cdfa",
        "CTNND2 KD": "#52cdfa",
        "CHD8 KD": "#52cdfa",
        "DLG4 KD": "#52cdfa",
        "SHANK3 KD": "#52cdfa",
        "SETD1A KD": "#52cdfa",
        "TRIP12 KD": "#52cdfa",
        "CUL1 KD": "#52cdfa",
    }

    bin_size = 300  # 60 s
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    average_df = (
        df.groupby(
            ["control", "condition", "file_id", "frame_bin"],
            as_index=False
        )["n_larvae_in_clusters"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(average_df["control"].unique()):

        control_df = average_df[
            average_df["control"] == control
        ].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = (
                plot_df["condition"]
                .replace(condition_labels)
            )

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="n_larvae_in_clusters",
                hue="plot_label",
                palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Number in Clusters")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__number_in_clusters.pdf"
            )

            plt.savefig(
                outpath,
                format="pdf",
                bbox_inches="tight"
            )

            plt.close()






################################################################
###### --------- PLOT NUMBER OF CLUSTERS / TIME -------- #######
################################################################

def plot_number_of_clusters_over_time():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/n_clusters_over_time.csv'
    )

    print('Number of clusters data loaded:')

    bin_size = 300
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    number_of_clusters = (
        df.groupby(
            ["control", "condition", "file_id", "frame_bin"],
            as_index=False
        )["n_clusters"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(number_of_clusters["control"].unique()):

        control_df = number_of_clusters[
            number_of_clusters["control"] == control
        ].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = (
                plot_df["condition"]
                .replace(condition_labels)
            )

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="n_clusters",
                hue="plot_label",
                # palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Number of Clusters")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__number_of_clusters_over_time.pdf"
            )

            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()





################################################################
####### --------- PLOT SIZE OF CLUSTERS / TIME -------- ########
################################################################
def plot_average_cluster_size_over_time():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/avg_cluster_size_over_time.csv'
    )

    print('Average cluster size data loaded:')

    bin_size = 300
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    cluster_size = (
        df.groupby(
            ["control", "condition", "file_id", "frame_bin"],
            as_index=False
        )["avg_cluster_size"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(cluster_size["control"].unique()):

        control_df = cluster_size[
            cluster_size["control"] == control
        ].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="avg_cluster_size",
                hue="plot_label",
                # palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Average Cluster Size")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__average_cluster_size_over_time.pdf"
            )

            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()





# ################################################################
# ####### --------- PLOT MAX CLUSTER SIZE / TIME -------- ########
# ################################################################
def plot_max_cluster_size_over_time():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/max_cluster_size_over_time.csv'
    )

    print('Max cluster size data loaded:')

    bin_size = 300
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    cluster_size = (
        df.groupby(
            ["control", "condition", "file_id", "frame_bin"],
            as_index=False
        )["max_cluster_size"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(cluster_size["control"].unique()):

        control_df = cluster_size[
            cluster_size["control"] == control
        ].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="max_cluster_size",
                hue="plot_label",
                # palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Max Cluster Size")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__max_cluster_size_over_time.pdf"
            )

            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()






# ################################################################
# ######### --------- DEPTH DIFFERENCE IN FOOD -------- ##########
# ################################################################
def plot_depth_difference_food():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/depth_difference_food.csv'
    )

    print('Depth difference in food data loaded:')

    df["plot_label"] = df["condition"].replace(condition_labels)

    control_mask = df["condition"] == df["control"]

    df.loc[control_mask, "plot_label"] = (
        "Control " + df.loc[control_mask, "condition"]
    )

    plt.figure(figsize=(7, 4))

    sns.stripplot(
        data=df,
        x="plot_label",
        y="depth_difference_food",
        color="grey",
        alpha=0.6,
        jitter=True
    )

    sns.pointplot(
        data=df,
        x="plot_label",
        y="depth_difference_food",
        errorbar="sd",
        color="green",
        linestyle="none"
    )

    plt.xlabel("Condition")
    plt.ylabel("Food Depth Difference")

    plt.xticks(rotation=45, ha="right")

    sns.despine()

    outpath = os.path.join(
        output,
        "depth_difference_food.pdf"
    )

    plt.savefig(
        outpath,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()


# ################################################################
# ######### --------- AVERAGE NEAREST NEIGHBOR -------- ##########
# ################################################################
def plot_mean_nearest_neighbour_over_time():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/nearest_neighbour.csv'
    )

    print('Nearest neighbour data loaded:')

    bin_size = 300
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    nearest_neighbour = (
        df.groupby(
            ["control", "condition", "file_id", "frame_bin"],
            as_index=False
        )["mean_nearest_neighbour_all"]
        .mean()
        .rename(columns={"frame_bin": "frame"})
    )

    for control in sorted(nearest_neighbour["control"].unique()):

        control_df = nearest_neighbour[
            nearest_neighbour["control"] == control
        ].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 4))

            sns.lineplot(
                data=plot_df,
                x="frame",
                y="mean_nearest_neighbour_all",
                hue="plot_label",
                # palette=palette,
                errorbar=("ci", 95)
            )

            plt.xlabel("Time")
            plt.ylabel("Mean Nearest Neighbour")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__mean_nearest_neighbour_over_time.pdf"
            )

            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()





# ################################################################
# ######### --------- AVERAGE NEAREST NEIGHBOR -------- ##########
# ################################################################

def plot_nearest_neighbour_over_time():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/nearest_neighbour.csv'
    )

    print('Nearest neighbour data loaded:')

    bin_size = 300
    df["frame_bin"] = (df["frame"] // bin_size) * bin_size

    nn_columns = [
        "mean_nearest_neighbour_all",
        "mean_nearest_neighbour_outside_clusters",
        "mean_nearest_neighbour_within_clusters"
    ]

    for nn_col in nn_columns:

        nearest_neighbour = (
            df.groupby(
                ["control", "condition", "file_id", "frame_bin"],
                as_index=False
            )[nn_col]
            .mean()
            .rename(columns={"frame_bin": "frame"})
        )

        for control in sorted(nearest_neighbour["control"].unique()):

            control_df = nearest_neighbour[
                nearest_neighbour["control"] == control
            ].copy()

            conditions = sorted([
                c for c in control_df["condition"].unique()
                if c != control
            ])

            for cond in conditions:

                print(f"Plotting {cond} against its control {control} - {nn_col}...")

                plot_df = control_df[
                    control_df["condition"].isin([control, cond])
                ].copy()

                plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

                control_label = condition_labels.get(control, control)
                cond_label = condition_labels.get(cond, cond)

                plot_df["plot_label"] = pd.Categorical(
                    plot_df["plot_label"],
                    categories=[control_label, cond_label],
                    ordered=True
                )

                plt.figure(figsize=(7, 4))

                sns.lineplot(
                    data=plot_df,
                    x="frame",
                    y=nn_col,
                    hue="plot_label",
                    # palette=palette,
                    errorbar=("ci", 95)
                )

                plt.xlabel("Time")
                plt.ylabel(nn_col.replace("_", " ").title())
                plt.legend(title="Condition")
                sns.despine()

                outpath = os.path.join(
                    output,
                    f"{cond_label}__{nn_col}_over_time.pdf"
                )

                plt.savefig(outpath, format="pdf", bbox_inches="tight")
                plt.close()






# ################################################################
# ######### ----- CLUSTER SIZE X NEAREST NEIGHBOR ----- ##########
# ################################################################
def plot_cluster_size_vs_nearest_neighbour():

    df = pd.read_csv(
        '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW/ANALYSIS/cluster_size_vs_nearest_neighbour.csv'
    )

    print('Cluster size vs nearest neighbour data loaded:')

    for control in sorted(df["control"].unique()):

        control_df = df[df["control"] == control].copy()

        conditions = sorted([
            c for c in control_df["condition"].unique()
            if c != control
        ])

        for cond in conditions:

            print(f"Plotting {cond} against its control {control}...")

            plot_df = control_df[
                control_df["condition"].isin([control, cond])
            ].copy()

            plot_df["plot_label"] = plot_df["condition"].replace(condition_labels)

            control_label = condition_labels.get(control, control)
            cond_label = condition_labels.get(cond, cond)

            plot_df["plot_label"] = pd.Categorical(
                plot_df["plot_label"],
                categories=[control_label, cond_label],
                ordered=True
            )

            plt.figure(figsize=(7, 5))

            sns.scatterplot(
                data=plot_df,
                x="cluster_size",
                y="mean_nn_within_cluster",
                hue="plot_label",
                # palette=palette,
                alpha=0.5
            )

            plt.xlabel("Cluster Size")
            plt.ylabel("Mean Nearest Neighbour Distance")
            plt.legend(title="Condition")
            sns.despine()

            outpath = os.path.join(
                output,
                f"{cond_label}__cluster_size_vs_nearest_neighbour.pdf"
            )

            plt.savefig(outpath, format="pdf", bbox_inches="tight")
            plt.close()





plot_percent_in_clusters()
plot_number_in_clusters()
plot_number_of_clusters_over_time()
plot_average_cluster_size_over_time()
plot_max_cluster_size_over_time()
plot_depth_difference_food()
plot_mean_nearest_neighbour_over_time()
plot_nearest_neighbour_over_time()
plot_cluster_size_vs_nearest_neighbour()