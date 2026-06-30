from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import os
from scipy.spatial.distance import cdist
import matplotlib as mpl

##### --- SIDEVIEW CLUSTER PERCENTAGE --- #####
def cluster_percentage(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [
                f for f in os.listdir(cluster_folder)
                if f.endswith(".feather")
            ]

            dfs = []

            for cluster_file in cluster_files:
                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))

                file_id = cluster_file.split(".predictions")[0]
                df["file_id"] = file_id

                dfs.append(df)

            if not dfs:
                print(f"No feather files found in {condition}")
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df["frame"] = combined_df["frame"].astype(int)

            result = (
                combined_df
                .groupby(["file_id", "frame"])
                .agg(
                    n_larvae_total=("cluster", "size"),
                    n_larvae_in_clusters=("cluster", lambda x: (x != -1).sum())
                )
                .reset_index()
            )

            result["percent_in_clusters"] = (
                result["n_larvae_in_clusters"] / result["n_larvae_total"] * 100
            )
            result.insert(0, "condition", condition)
            result.insert(0, "control", control)

            all_results.append(result)

            result.to_csv(
                os.path.join(cluster_folder, "percent_in_clusters.csv"),
                index=False
            )

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "percent_in_clusters.csv"),
            index=False
        )







##### --- SIDEVIEW NUMBER OF CLUSTERS OVER TIME --- #####
def number_of_clusters(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.split("CONTROL_")[-1]

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            dfs = []

            for cluster_file in cluster_files:
                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
                file_id = cluster_file.split(".predictions")[0]
                df["file_id"] = file_id
                dfs.append(df)

            if not dfs:
                print(f"No feather files found in {condition}")
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df["frame"] = combined_df["frame"].astype(int)

            # count unique clusters per frame (excluding -1)
            result = (
                combined_df[combined_df["cluster"] != -1]
                .groupby(["file_id", "frame"])["cluster"]
                .nunique()
                .reset_index(name="n_clusters")
            )

            # if a frame had only -1, it won't appear above; optionally fill those with 0:
            all_frames = combined_df.groupby(["file_id", "frame"]).size().reset_index()[["file_id", "frame"]]
            result = all_frames.merge(result, on=["file_id", "frame"], how="left").fillna({"n_clusters": 0})

            result.insert(0, "condition", condition)
            result.insert(0, "control", control)
            all_results.append(result)

            result.to_csv(os.path.join(cluster_folder, "n_clusters_over_time.csv"), index=False)
        

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(analysis_folder, "n_clusters_over_time.csv"), index=False)



##### --- SIDEVIEW AVERAGE CLUSTER SIZE OVER TIME --- #####
def average_cluster_size(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            dfs = []

            for cluster_file in cluster_files:
                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
                file_id = cluster_file.split(".predictions")[0]
                df["file_id"] = file_id
                dfs.append(df)

            if not dfs:
                print(f"No feather files found in {condition}")
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df["frame"] = combined_df["frame"].astype(int)

            clustered = combined_df[combined_df["cluster"] != -1].copy()

            cluster_sizes = (
                clustered
                .groupby(["file_id", "frame", "cluster"])
                .size()
                .reset_index(name="cluster_size")
            )

            result = (
                cluster_sizes
                .groupby(["file_id", "frame"])["cluster_size"]
                .mean()
                .reset_index(name="avg_cluster_size")
            )

            all_frames = (
                combined_df
                .groupby(["file_id", "frame"])
                .size()
                .reset_index()[["file_id", "frame"]]
            )

            result = all_frames.merge(result, on=["file_id", "frame"], how="left")
            result = result.fillna({"avg_cluster_size": 0})

            result.insert(0, "condition", condition)
            result.insert(0, "control", control)

            all_results.append(result)

            result.to_csv(
                os.path.join(cluster_folder, "avg_cluster_size_over_time.csv"),
                index=False
            )

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "avg_cluster_size_over_time.csv"),
            index=False
        )



##### --- SIDEVIEW AVERAGE CLUSTER SIZE OVER TIME --- #####
##### --- SIDEVIEW MAX CLUSTER SIZE OVER TIME --- #####
def max_cluster_size(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            dfs = []

            for cluster_file in cluster_files:
                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
                file_id = cluster_file.split(".predictions")[0]
                df["file_id"] = file_id
                dfs.append(df)

            if not dfs:
                print(f"No feather files found in {condition}")
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df["frame"] = combined_df["frame"].astype(int)

            clustered = combined_df[combined_df["cluster"] != -1].copy()

            cluster_sizes = (
                clustered
                .groupby(["file_id", "frame", "cluster"])
                .size()
                .reset_index(name="cluster_size")
            )

            result = (
                cluster_sizes
                .groupby(["file_id", "frame"])["cluster_size"]
                .max()
                .reset_index(name="max_cluster_size")
            )

            all_frames = (
                combined_df
                .groupby(["file_id", "frame"])
                .size()
                .reset_index()[["file_id", "frame"]]
            )

            result = all_frames.merge(result, on=["file_id", "frame"], how="left")
            result = result.fillna({"max_cluster_size": 0})

            result.insert(0, "condition", condition)
            result.insert(0, "control", control)

            all_results.append(result)

            result.to_csv(
                os.path.join(cluster_folder, "max_cluster_size_over_time.csv"),
                index=False
            )

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "max_cluster_size_over_time.csv"),
            index=False
        )




def depth_difference_food(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            for cluster_file in cluster_files:

                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))
                file_id = cluster_file.split(".predictions")[0]

                df["frame"] = df["frame"].astype(int)

                first = df[df["frame"] < 500]["y_tail"].mean()

                last_frames = df["frame"].max()

                last_df = df[df["frame"] > (last_frames - 500)]

                last_df = last_df[last_df["y_tail"] >= first]

                last = last_df["y_tail"].mean()

                diff = last - first

                result = pd.DataFrame({
                    "control": [control],
                    "condition": [condition],
                    "file_id": [file_id],
                    "depth_difference_food": [diff]
                })

                all_results.append(result)

    if all_results:
        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "depth_difference_food.csv"),
            index=False
        )




def nearest_neighbour(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            for cluster_file in cluster_files:

                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))

                df["frame"] = df["frame"].astype(int)
                df = df[df["frame"] % 50 == 0].copy()

                print(f"  {cluster_file}: {df['frame'].nunique()} sampled frames")

                file_id = cluster_file.split(".predictions")[0]

                frames = []

                for frame, frame_df in df.groupby("frame"):

                    coords_all = frame_df[["x_head", "y_head"]].values

                    if len(coords_all) >= 2:
                        dist_matrix = cdist(coords_all, coords_all)
                        np.fill_diagonal(dist_matrix, np.inf)
                        all_nn = dist_matrix.min(axis=1).mean()
                    else:
                        all_nn = np.nan

                    outside_df = frame_df[frame_df["cluster"] == -1]
                    coords_outside = outside_df[["x_head", "y_head"]].values

                    if len(coords_outside) >= 2:
                        dist_matrix = cdist(coords_outside, coords_outside)
                        np.fill_diagonal(dist_matrix, np.inf)
                        outside_nn = dist_matrix.min(axis=1).mean()
                    else:
                        outside_nn = np.nan

                    cluster_nn_values = []

                    clustered_df = frame_df[frame_df["cluster"] != -1]

                    for cluster_id, cluster_df in clustered_df.groupby("cluster"):

                        coords_cluster = cluster_df[["x_head", "y_head"]].values

                        if len(coords_cluster) < 2:
                            continue

                        dist_matrix = cdist(coords_cluster, coords_cluster)
                        np.fill_diagonal(dist_matrix, np.inf)

                        nearest = dist_matrix.min(axis=1)
                        cluster_nn_values.append(nearest.mean())

                    if cluster_nn_values:
                        within_cluster_nn = np.mean(cluster_nn_values)
                    else:
                        within_cluster_nn = np.nan

                    frames.append({
                        "control": control,
                        "condition": condition,
                        "file_id": file_id,
                        "frame": frame,
                        "mean_nearest_neighbour_all": all_nn,
                        "mean_nearest_neighbour_outside_clusters": outside_nn,
                        "mean_nearest_neighbour_within_clusters": within_cluster_nn
                    })

                if frames:
                    all_results.append(pd.DataFrame(frames))

    if all_results:

        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "nearest_neighbour.csv"),
            index=False
        )



def cluster_size_vs_nearest_neighbour(parent_folder):

    all_results = []

    for control_folder in os.listdir(parent_folder):

        control_path = os.path.join(parent_folder, control_folder)

        if not os.path.isdir(control_path):
            continue

        if not control_folder.startswith("CONTROL_"):
            continue

        control = control_folder.replace("CONTROL_", "")

        for condition in os.listdir(control_path):

            cluster_folder = os.path.join(control_path, condition, "clustering")

            if not os.path.isdir(cluster_folder):
                print(f"Skipping {condition} in {control_folder}, no clustering folder.")
                continue

            print(f"Processing control={control}, condition={condition}")

            cluster_files = [f for f in os.listdir(cluster_folder) if f.endswith(".feather")]

            for cluster_file in cluster_files:

                df = pd.read_feather(os.path.join(cluster_folder, cluster_file))

                df["frame"] = df["frame"].astype(int)
                df = df[df["frame"] % 50 == 0].copy()

                file_id = cluster_file.split(".predictions")[0]

                rows = []

                clustered_df = df[df["cluster"] != -1].copy()

                for (frame, cluster_id), cluster_df in clustered_df.groupby(["frame", "cluster"]):

                    cluster_size = len(cluster_df)

                    if cluster_size < 2:
                        mean_nn = np.nan
                    else:
                        coords = cluster_df[["x_head", "y_head"]].values

                        dist_matrix = cdist(coords, coords)
                        np.fill_diagonal(dist_matrix, np.inf)

                        nearest = dist_matrix.min(axis=1)
                        mean_nn = nearest.mean()

                    rows.append({
                        "control": control,
                        "condition": condition,
                        "file_id": file_id,
                        "frame": frame,
                        "cluster": cluster_id,
                        "cluster_size": cluster_size,
                        "mean_nn_within_cluster": mean_nn
                    })

                if rows:
                    all_results.append(pd.DataFrame(rows))

    if all_results:

        analysis_folder = os.path.join(parent_folder, "ANALYSIS")
        os.makedirs(analysis_folder, exist_ok=True)

        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_csv(
            os.path.join(analysis_folder, "cluster_size_vs_nearest_neighbour.csv"),
            index=False
        )



##### --- RUNNING ANALYSIS FOR SIDEVIEW VIDEOS --- #####

parent_folder = '/Volumes/lab-windingm/home/users/cochral/PhD/NDD/EXPERIMENTS/ELAV_T27-B05/SIDEVIEW'
cluster_percentage(parent_folder)
number_of_clusters(parent_folder)
average_cluster_size(parent_folder)
max_cluster_size(parent_folder)
depth_difference_food(parent_folder)
nearest_neighbour(parent_folder)    
cluster_size_vs_nearest_neighbour(parent_folder)


