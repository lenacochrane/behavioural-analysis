import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely import wkt
from scipy.spatial.distance import cdist

# =============================================================================
# process_directory: returns mm converted track files
# =============================================================================
def process_directory(directory):

    track_files = sorted(
        [f for f in os.listdir(directory) if f.endswith("tracks.feather")]
    )
    video_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".mp4")]
    )

    perimeter_files = sorted(
        [f for f in os.listdir(directory) if f.endswith("_perimeter.wkt")]
    )

    matched_pairs = []

    for track_file in track_files:
        track_prefix = track_file.replace(".tracks.feather", "")
        print(track_prefix)

        matched_video = None
        matched_perimeter = None

        for video_file in video_files:
            video_prefix = video_file.replace(".mp4", "")
            print(video_prefix)
            if track_prefix == video_prefix:
                matched_video = video_file
                break
        
        for perimeter_file in perimeter_files:
            perimeter_prefix = perimeter_file.replace("_perimeter.wkt", "")
            print(perimeter_prefix)
            if track_prefix == perimeter_prefix:
                matched_perimeter = perimeter_file
                break

        matched_pairs.append({
            "track_file": track_file,
            "video_file": matched_video,
            "perimeter_file": matched_perimeter
        })


    track_data = {}

    for match in matched_pairs:

        perimeter_file = match.get('perimeter_file')
    
        if perimeter_file:

            perimeter_path = os.path.join(directory, perimeter_file)
            with open(perimeter_path, 'r') as f:
                perimeter_wkt = f.read()

            perimeter_polygon = wkt.loads(perimeter_wkt)

            # Calculate the diameter of the perimeter 
            minx, miny, maxx, maxy = perimeter_polygon.bounds
            diameter = maxx - minx  # This assumes the perimeter is a circle and uses its width as the diameter.

            conversion_factor = 90 / diameter # 90mm 

            # IF PERIMETER DETECTED BADLY 
            threshold = 0.09 #
            if conversion_factor > threshold:
                print(f"Conversion factor {conversion_factor:.3f} is above threshold for {match['track_file']}. Using default conversion factor:")
                conversion_factor = 90 / 1032  # Use the old conversion factor
            
            # scaled_perimeter_polygon = scale(perimeter_polygon, xfact=conversion_factor, yfact=conversion_factor,  origin=(0, 0))
            perimeter_coordinates = np.array(perimeter_polygon.exterior.coords)
            perimeter_coordinates *= conversion_factor
            scaled_perimeter_polygon = Polygon(perimeter_coordinates)

            match['perimeter_polygon'] = scaled_perimeter_polygon  # Update the scaled polygon.

            track_file = match['track_file']
            track_file_data = pd.read_feather(os.path.join(directory, track_file))

            pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            track_file_data[pixel_columns] = track_file_data[pixel_columns] * conversion_factor
            print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")



            centre_x = scaled_perimeter_polygon.centroid.x
            centre_y = scaled_perimeter_polygon.centroid.y

            minx, miny, maxx, maxy = scaled_perimeter_polygon.bounds
            radius = (maxx - minx) / 2


            track_file_data["dist_body"] = np.sqrt(
                (track_file_data["x_body"] - centre_x)**2 +
                (track_file_data["y_body"] - centre_y)**2
            )

            track_file_data["dist_head"] = np.sqrt(
                (track_file_data["x_head"] - centre_x)**2 +
                (track_file_data["y_head"] - centre_y)**2
            )

            track_file_data["dist_tail"] = np.sqrt(
                (track_file_data["x_tail"] - centre_x)**2 +
                (track_file_data["y_tail"] - centre_y)**2
            )

            centre_threshold = 41  # mm

            track_file_data["within_walls"] = (
                (track_file_data["dist_body"] < centre_threshold) &
                (track_file_data["dist_head"] < centre_threshold) &
                (track_file_data["dist_tail"] < centre_threshold)
            )

            track_file_data.drop(columns=["dist_body", "dist_head", "dist_tail"], inplace=True)


            # get video path
            video_file = match.get("video_file")
            video_path = os.path.join(directory, video_file)
            video_name = os.path.splitext(video_file)[0]

            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

            if ret:
                # convert centre (mm → pixels)
                centre_px = (
                    int(centre_x / conversion_factor),
                    int(centre_y / conversion_factor)
                )

                # choose your threshold (mm)
                centre_threshold = 41
                radius_px = int(centre_threshold / conversion_factor)
                cv2.circle(frame, centre_px, radius_px, (0, 0, 255), 1)

                save_path = os.path.join(directory, f"{video_name}_hunch_threshold.png")
                cv2.imwrite(save_path, frame)

            cap.release()




            track_data[track_file] = track_file_data

        else:
            print(f"no perimeter file detected for {match['track_file']}")

    return track_data


# =============================================================================
# identify_perimeters: identifies + creates perimeter polygons for video files
# =============================================================================
def identify_perimeters(directory):

    def detect_largest_circle(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=100,
            param1=500,
            param2=50,
            minRadius=400,
            maxRadius=600
        )

        if circles is not None:
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            return largest_circle  # x, y, r

        return None

    def circle_to_polygon(x, y, radius, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
        return Polygon(points)

    video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        wkt_file_path = os.path.join(directory, f"{video_name}_perimeter.wkt")
        png_file_path = os.path.join(directory, f"{video_name}_perimeter.png")

        # skip if perimeter already exists
        if os.path.exists(wkt_file_path):
            continue

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ret, frame = cap.read()

        if ret:
            circle = detect_largest_circle(frame)

            if circle is not None:
                x, y, r = circle
                petri_dish_boundary = circle_to_polygon(x, y, r)

                with open(wkt_file_path, "w") as f:
                    f.write(petri_dish_boundary.wkt)

                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.imwrite(png_file_path, frame)

                print(f"Perimeter created for {video_file}")
            else:
                print(f"No perimeter detected for {video_file}")
        else:
            print(f"Failed to extract frame 10 from {video_file}")

        cap.release()






# =============================================================================
# correct_head_tail_swaps: head–tail orientation correction (per track)
# =============================================================================

def correct_head_tail_swaps(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # raw first-pass vector check, only used to trigger a 6-frame window
    df["vector_tail_head_x"] = df["x_head"] - df["x_tail"]
    df["vector_tail_head_y"] = df["y_head"] - df["y_tail"]

    df["vector_head_tail_x"] = -df["vector_tail_head_x"]
    df["vector_head_tail_y"] = -df["vector_tail_head_y"]

    df["prev_vector_x"] = df["vector_tail_head_x"].shift(1)
    df["prev_vector_y"] = df["vector_tail_head_y"].shift(1)

    df["align_tail_head"] = (
        df["vector_tail_head_x"] * df["prev_vector_x"] +
        df["vector_tail_head_y"] * df["prev_vector_y"]
    )

    df["align_head_tail"] = (
        df["vector_head_tail_x"] * df["prev_vector_x"] +
        df["vector_head_tail_y"] * df["prev_vector_y"]
    )
    
    

    df["flipped"] = (
        (df["align_head_tail"] > df["align_tail_head"]) &
        (df["instance_score"].notna())
    )

    # corrected columns
    df["x_head_corrected"] = np.nan
    df["y_head_corrected"] = np.nan
    df["x_tail_corrected"] = np.nan
    df["y_tail_corrected"] = np.nan
    df["flipped_corrected"] = False
    df["checked_window"] = False

    # debug columns
    df["align_tail_head_corrected"] = np.nan
    df["align_head_tail_corrected"] = np.nan
    df["tail_dist_noflip"] = np.nan
    df["tail_dist_flip"] = np.nan

    df["align_difference"] = np.nan
    

    # first frame stays as-is
    df.loc[0, "x_head_corrected"] = df.loc[0, "x_head"]
    df.loc[0, "y_head_corrected"] = df.loc[0, "y_head"]
    df.loc[0, "x_tail_corrected"] = df.loc[0, "x_tail"]
    df.loc[0, "y_tail_corrected"] = df.loc[0, "y_tail"]

    i = 1
    while i < len(df):

        # if this row does not trigger a check window, just copy raw values
        if df.loc[i, "flipped"] == False:
            df.loc[i, "x_head_corrected"] = df.loc[i, "x_head"]
            df.loc[i, "y_head_corrected"] = df.loc[i, "y_head"]
            df.loc[i, "x_tail_corrected"] = df.loc[i, "x_tail"]
            df.loc[i, "y_tail_corrected"] = df.loc[i, "y_tail"]
            i += 1
            continue

        # raw True at row i -> check this row and next 5 rows
        end_idx = min(i + 6, len(df) - 1)

        for j in range(i, end_idx + 1):

            # if this frame has no instance_score, keep raw and move on
            if pd.isna(df.loc[j, "instance_score"]):
                df.loc[j, "x_head_corrected"] = df.loc[j, "x_head"]
                df.loc[j, "y_head_corrected"] = df.loc[j, "y_head"]
                df.loc[j, "x_tail_corrected"] = df.loc[j, "x_tail"]
                df.loc[j, "y_tail_corrected"] = df.loc[j, "y_tail"]
                df.loc[j, "checked_window"] = True
                continue

            # previous corrected frame
            prev_head_x = df.loc[j - 1, "x_head_corrected"]
            prev_head_y = df.loc[j - 1, "y_head_corrected"]
            prev_tail_x = df.loc[j - 1, "x_tail_corrected"]
            prev_tail_y = df.loc[j - 1, "y_tail_corrected"]

            # previous corrected vector: tail -> head
            prev_vector_x = prev_head_x - prev_tail_x
            prev_vector_y = prev_head_y - prev_tail_y

            # current raw coordinates
            curr_head_x = df.loc[j, "x_head"]
            curr_head_y = df.loc[j, "y_head"]
            curr_tail_x = df.loc[j, "x_tail"]
            curr_tail_y = df.loc[j, "y_tail"]

            # no-flip orientation
            tail_head_x = curr_head_x - curr_tail_x
            tail_head_y = curr_head_y - curr_tail_y

            # flipped orientation
            head_tail_x = -tail_head_x
            head_tail_y = -tail_head_y

            # vector alignment scores
            align_tail_head = (
                tail_head_x * prev_vector_x +
                tail_head_y * prev_vector_y
            )

            align_head_tail = (
                head_tail_x * prev_vector_x +
                head_tail_y * prev_vector_y
            )

            # tail continuity scores
            tail_dist_noflip = np.hypot(curr_tail_x - prev_tail_x, curr_tail_y - prev_tail_y)
            tail_dist_flip = np.hypot(curr_head_x - prev_tail_x, curr_head_y - prev_tail_y)

            align_difference = align_head_tail - align_tail_head

            df.loc[j, "checked_window"] = True
            df.loc[j, "align_tail_head_corrected"] = align_tail_head
            df.loc[j, "align_head_tail_corrected"] = align_head_tail
            df.loc[j, "tail_dist_noflip"] = tail_dist_noflip
            df.loc[j, "tail_dist_flip"] = tail_dist_flip

            df.loc[j, "align_difference"] = align_difference

            # min_align_distance_change = 400  #tweak if necessary PIXELS
            min_align_distance_change = 2.89  #tweak if necessary MM

            # final decision
            # if (align_head_tail > align_tail_head) and (tail_dist_flip < tail_dist_noflip):
            if (align_difference > min_align_distance_change) and (tail_dist_flip < tail_dist_noflip):
                df.loc[j, "flipped_corrected"] = True
                df.loc[j, "x_head_corrected"] = curr_tail_x
                df.loc[j, "y_head_corrected"] = curr_tail_y
                df.loc[j, "x_tail_corrected"] = curr_head_x
                df.loc[j, "y_tail_corrected"] = curr_head_y
            else:
                df.loc[j, "x_head_corrected"] = curr_head_x
                df.loc[j, "y_head_corrected"] = curr_head_y
                df.loc[j, "x_tail_corrected"] = curr_tail_x
                df.loc[j, "y_tail_corrected"] = curr_tail_y

        i = end_idx + 1

    return df





# =============================================================================
# speed: calculated speed for head, body and tail coordinates (per track)
# =============================================================================

def speed(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    def calc_speed(x, y):
        dx = df[x].diff()
        dy = df[y].diff()
        return np.sqrt(dx**2 + dy**2)

    df["head_speed"] = calc_speed("x_head_corrected", "y_head_corrected")
    df["tail_speed"] = calc_speed("x_tail_corrected", "y_tail_corrected")
    df["body_speed"] = calc_speed("x_body", "y_body")

    df["head_tail_speed_diff"] = df["head_speed"] - df["tail_speed"]

    return df

# =============================================================================
# acceleration: calculated acceleration for head, body and tail (per track)
# =============================================================================

def acceleration(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    df["head_acceleration"] = df["head_speed"].diff()
    df["tail_acceleration"] = df["tail_speed"].diff()
    df["body_acceleration"] = df["body_speed"].diff()

    return df

# =============================================================================
# tail_head_orientation: tail-head orientation and turn rate (per track)
# =============================================================================

def tail_head_orientation(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # tail -> head vector
    dx = df["x_head_corrected"] - df["x_tail_corrected"]
    dy = df["y_head_corrected"] - df["y_tail_corrected"]

    angle = np.arctan2(dy, dx)

    # overall tail-head direction
    df["tail_head_direction_angle"] = np.degrees(angle)

    # frame-to-frame angle change
    angle_diff = angle.diff()

    # wrap to [-pi, pi] so angle changes are correct across boundary
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    df["tail_head_turn_rate"] = np.degrees(angle_diff)

    return df


# turning = change in direction over time
# bending = shape of the body at a frame

# =============================================================================
# bending: computes body bending (angle and magnitude) (per track)
# =============================================================================

def bending(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # tail -> body vector
    tb_x = df["x_body"] - df["x_tail_corrected"]
    tb_y = df["y_body"] - df["y_tail_corrected"]

    # body -> head vector
    bh_x = df["x_head_corrected"] - df["x_body"]
    bh_y = df["y_head_corrected"] - df["y_body"]

    # angles of each segment
    tail_body_angle = np.arctan2(tb_y, tb_x)
    body_head_angle = np.arctan2(bh_y, bh_x)

    # angle between the two body segments
    bend_angle = body_head_angle - tail_body_angle

    # wrap to [-pi, pi]
    bend_angle = (bend_angle + np.pi) % (2 * np.pi) - np.pi

    bend_angle_deg = np.degrees(bend_angle)

    df["bend_angle"] = bend_angle_deg
    df["bend_magnitude"] = np.abs(bend_angle_deg)

    return df

# =============================================================================
# head_lateral_displacement: sideways head movement relative to body axis 
# =============================================================================

def head_lateral_displacement(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # tail -> head axis
    axis_x = df["x_head_corrected"] - df["x_tail_corrected"]
    axis_y = df["y_head_corrected"] - df["y_tail_corrected"]

    # normalize axis
    norm = np.sqrt(axis_x**2 + axis_y**2)
    axis_x = axis_x / norm
    axis_y = axis_y / norm

    # perpendicular axis
    perp_x = -axis_y
    perp_y = axis_x

    # head movement (frame-to-frame)
    dx = df["x_head_corrected"].diff()
    dy = df["y_head_corrected"].diff()

    # projection onto perpendicular axis
    df["head_lateral_displacement"] = dx * perp_x + dy * perp_y

    # optional magnitude
    df["head_lateral_magnitude"] = np.abs(df["head_lateral_displacement"])

    return df

# =============================================================================
# aligned_movement: measures if movement is forward/backward to tail-head axis 
# =============================================================================

def aligned_movement(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # body axis (tail → head)
    axis_x = df["x_head_corrected"] - df["x_tail_corrected"]
    axis_y = df["y_head_corrected"] - df["y_tail_corrected"]

    # normalize axis
    norm = np.sqrt(axis_x**2 + axis_y**2)
    axis_x = axis_x / norm
    axis_y = axis_y / norm

    # body movement (centroid movement)
    dx = df["x_body"].diff()
    dy = df["y_body"].diff()

    # projection onto axis (parallel component)
    df["alignment"] = dx * axis_x + dy * axis_y

    return df

# =============================================================================
# body_length: measures t-h, t-b, b-h lengths (per track)
# =============================================================================

def body_length(track_df):
    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # tail → head (full length)
    dx_th = df["x_head_corrected"] - df["x_tail_corrected"]
    dy_th = df["y_head_corrected"] - df["y_tail_corrected"]
    df["tail_head_length"] = np.sqrt(dx_th**2 + dy_th**2)

    # tail → body
    dx_tb = df["x_body"] - df["x_tail_corrected"]
    dy_tb = df["y_body"] - df["y_tail_corrected"]
    df["tail_body_length"] = np.sqrt(dx_tb**2 + dy_tb**2)

    # body → head
    dx_bh = df["x_head_corrected"] - df["x_body"]
    dy_bh = df["y_head_corrected"] - df["y_body"]
    df["body_head_length"] = np.sqrt(dx_bh**2 + dy_bh**2)

    return df




def digging(track_df):

    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # Smooth body position slightly
    df["x"] = df["x_body"].rolling(window=5, min_periods=1).mean()
    df["y"] = df["y_body"].rolling(window=5, min_periods=1).mean()

    # Frame-to-frame movement
    df["dx"] = df["x"].diff().fillna(0)
    df["dy"] = df["y"].diff().fillna(0)
    df["distance"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # XY confinement
    df["x_std"] = df["x"].rolling(window=10, min_periods=1).std().fillna(0)
    df["y_std"] = df["y"].rolling(window=10, min_periods=1).std().fillna(0)
    df["overall_std"] = np.sqrt(df["x_std"]**2 + df["y_std"]**2)

    # Candidate digging frames
    df["digging_candidate"] = (
        (df["distance"] < 0.2) &
        (df["overall_std"] < 0.50)
    )

    df["digging_status"] = False

    rolling_mean = df["digging_candidate"].rolling(100).mean()

    for i in range(len(df) - 99):
        if rolling_mean.iloc[i + 99] >= 0.9:
            df.iloc[i:i+100, df.columns.get_loc("digging_status")] = True
        

    return df


def hunching(track_df):
    df = track_df.copy()

    # recent baseline body length before this frame
    df["tail_head_length_baseline"] = (
        df["tail_head_length"]
        .rolling(window=100, min_periods=10)
        .median()
    )

    # how much shorter is the body compared with recent baseline?
    df["body_shortening"] = (
        df["tail_head_length_baseline"] - df["tail_head_length"]
    )

    df["body_shortening_fraction"] = (
        df["body_shortening"] / df["tail_head_length_baseline"]
    )

    df["is_hunching"] = (
        df["within_walls"] &
        (df["bend_magnitude"] < 30) &
        (df["body_shortening_fraction"] > 0.20)
    )

    return df

    

# =============================================================================
# head_nearest_neighbour: nearest node to focal head across other larvae
# =============================================================================

# def nearest_neighbour(file_df):
#     df = file_df.copy()
#     df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

#     df["nearest_neighbour_id"] = np.nan
#     df["closest_node_to_head"] = None
#     df["head_distance"] = np.nan

#     parts = ["head", "body", "tail"]

#     for frame, frame_df in df.groupby("frame"):
#         if frame_df["track_id"].nunique() < 2:
#             continue

#         node_rows = []
#         for idx, row in frame_df.iterrows():
#             for part in parts:
#                 if part == "head":
#                     x = row["x_head_corrected"]
#                     y = row["y_head_corrected"]
#                 elif part == "tail":
#                     x = row["x_tail_corrected"]
#                     y = row["y_tail_corrected"]
#                 else:
#                     x = row["x_body"]
#                     y = row["y_body"]

#                 node_rows.append({
#                     "index": idx,
#                     "track_id": row["track_id"],
#                     "part": part,
#                     "x": x,
#                     "y": y
#                 })

#         nodes = pd.DataFrame(node_rows)

#         for focal_idx, focal_row in frame_df.iterrows():
#             focal_track = focal_row["track_id"]

#             focal_head = np.array([[
#                 focal_row["x_head_corrected"],
#                 focal_row["y_head_corrected"]
#             ]], dtype=float)

#             other_nodes = nodes[nodes["track_id"] != focal_track].copy()
#             if other_nodes.empty:
#                 continue

#             B = other_nodes[["x", "y"]].to_numpy(float)
#             D = cdist(focal_head, B)

#             if np.isnan(D).all():
#                 continue

#             b = int(np.nanargmin(D))
#             nearest = other_nodes.iloc[b]

#             df.at[focal_idx, "nearest_neighbour_id"] = nearest["track_id"]
#             df.at[focal_idx, "closest_node_to_head"] = nearest["part"]
#             df.at[focal_idx, "head_distance"] = float(D[0, b])

#     return df

def nearest_neighbour(file_df):
    df = file_df.copy()
    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

    # any node-node nearest neighbour
    df["closest_node_node_other_id"] = np.nan
    df["closest_nodes"] = None
    df["closest_node_node_distance"] = np.nan

    # focal head to nearest other node
    df["head_nearest_id"] = np.nan
    df["closest_node_to_head"] = None
    df["head_distance"] = np.nan

    parts = ["head", "body", "tail"]

    def get_xy(row, part):
        if part == "head":
            return row["x_head_corrected"], row["y_head_corrected"]
        elif part == "tail":
            return row["x_tail_corrected"], row["y_tail_corrected"]
        else:
            return row["x_body"], row["y_body"]

    def unify_nodes(p1, p2):
        return "-".join(sorted([p1, p2]))

    for frame, frame_df in df.groupby("frame"):
        if frame_df["track_id"].nunique() < 2:
            continue

        node_rows = []

        for idx, row in frame_df.iterrows():
            for part in parts:
                x, y = get_xy(row, part)

                node_rows.append({
                    "index": idx,
                    "track_id": row["track_id"],
                    "part": part,
                    "x": x,
                    "y": y
                })

        nodes = pd.DataFrame(node_rows)

        for focal_idx, focal_nodes in nodes.groupby("index"):
            focal_track = focal_nodes["track_id"].iloc[0]

            # exclude same larva
            other_nodes = nodes[nodes["track_id"] != focal_track]
            if other_nodes.empty:
                continue

            # -----------------------------
            # 1. closest any node-node
            # -----------------------------
            A = focal_nodes[["x", "y"]].to_numpy(float)
            B = other_nodes[["x", "y"]].to_numpy(float)

            D = cdist(A, B)

            if not np.isnan(D).all():
                a, b = np.unravel_index(np.nanargmin(D), D.shape)

                focal_part = focal_nodes.iloc[a]["part"]
                nearest = other_nodes.iloc[b]

                df.at[focal_idx, "closest_node_node_other_id"] = nearest["track_id"]
                df.at[focal_idx, "closest_nodes"] = unify_nodes(
                    focal_part,
                    nearest["part"]
                )
                df.at[focal_idx, "closest_node_node_distance"] = float(D[a, b])

            # -----------------------------
            # 2. closest other node to head
            # -----------------------------
            focal_head = focal_nodes[focal_nodes["part"] == "head"][["x", "y"]].to_numpy(float)

            if focal_head.shape[0] != 0:
                Dh = cdist(focal_head, B)

                if not np.isnan(Dh).all():
                    b_h = int(np.nanargmin(Dh))
                    nearest_h = other_nodes.iloc[b_h]

                    df.at[focal_idx, "head_nearest_id"] = nearest_h["track_id"]
                    df.at[focal_idx, "closest_node_to_head"] = nearest_h["part"]
                    df.at[focal_idx, "head_distance"] = float(Dh[0, b_h])

    return df



def behavioural_states(track_df):

    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    # df["is_moving"] = df["tail_speed"] > 10
    # df['is_moving_backwards'] = df['alignment'] < -10 
    df["is_moving"] = df["tail_speed"] > 0.5
    df['is_moving_backwards'] = df['alignment'] < -0.5
    df["is_changing_direction"] = df["tail_head_turn_rate"].abs() > 20
    df["is_bending"] = df["bend_magnitude"] > 40
    df["bend_type"] = ""
    df.loc[(df["bend_magnitude"] >= 40) & (df["bend_magnitude"] < 90), "bend_type"] = "bend"
    df.loc[df["bend_magnitude"] >= 90, "bend_type"] = "large_bend"


    # threshold = 15 # CHANGE TO 18 PIXELS

    threshold = 1.275 # CHANGE TO 18 PIXELS


    # sign of lateral movement
    df["lateral_sign"] = np.sign(df["head_lateral_displacement"])

    # active frames (strong sideways movement)
    df["lateral_active"] = df["head_lateral_magnitude"] > threshold

    # sign change between frames
    df["sign_change"] = df["lateral_sign"] != df["lateral_sign"].shift(1)

    # final casting condition
    df["is_casting"] = (
        df["lateral_active"] &
        df["lateral_active"].shift(1) &
        df["sign_change"]
    )

    # not solved hunching- need examples


    df["behavioural_states"] = ""

    for idx, row in df.iterrows():
        states = []

        if row["is_moving"]:
            states.append("forward_run")
        else:
            states.append("stationary")

        if row["is_moving_backwards"]:
            states.append("backward")

        if row["is_changing_direction"]:
            states.append("changing_direction")

        if row["bend_type"] == "bend":
            states.append("bend")

        if row["bend_type"] == "large_bend":
            states.append("large_bend")

        if row["is_casting"]:
            states.append("casting")

        df.at[idx, "behavioural_states"] = "|".join(states) if states else "unknown"
    

    df["behaviour"] = df["behavioural_states"]

    df.loc[df["behaviour"] == "forward_run|backward", "behaviour"] = "backward"
    df.loc[df["behaviour"] == "stationary|backward", "behaviour"] = "backward"

    # # small turn
    # df.loc[
    #     df["behavioural_states"].str.contains("stationary", na=False) &
    #     df["behavioural_states"].str.contains("changing_direction", na=False),
    #     "behaviour"
    # ] = "small_turn"


    df.loc[
    (df["behaviour"] == df["behavioural_states"]) &
    (df["behavioural_states"] == "stationary|changing_direction"),
    "behaviour"
    ] = "small_turn"

    df.loc[
        (df["behaviour"] == df["behavioural_states"]) &
        (df["behavioural_states"] == "stationary|backward|changing_direction"),
        "behaviour"
    ] = "small_turn"


    df.loc[df["behavioural_states"] == "stationary|backward|bend", "behaviour"] = "small_turn"
    df.loc[df["behavioural_states"] == "forward_run|backward|bend", "behaviour"] = "small_turn"

    df.loc[df["behavioural_states"] == "stationary|backward|large_bend", "behaviour"] = "sharp_turn"
    df.loc[df["behavioural_states"] == "forward_run|backward|large_bend", "behaviour"] = "sharp_turn"


    df.loc[
    (df["behavioural_states"] == "forward_run|changing_direction") &
    (df["bend_magnitude"] <= 20),
    "behaviour"
    ] = "forward_run"


    df.loc[
        (df["behavioural_states"] == "forward_run|changing_direction") &
        (df["bend_magnitude"] > 20),
        "behaviour"] = "steering"

    df.loc[df["behavioural_states"] == "forward_run|bend", "behaviour"] = "steering"



    # df.loc[
    #     df["behaviour"].str.contains("changing_direction") &
    #     df["behaviour"].str.contains("large_bend"),
    #     "behaviour"
    # ] = "sharp_turn"

    # df.loc[
    #     df["behaviour"].str.contains("changing_direction") &
    #     df["behaviour"].str.contains("bend"),
    #     "behaviour"
    # ] = "turn"

    # turn-related anchor frames
    df["is_turn_anchor"] = (
        df["behavioural_states"].str.contains("changing_direction", na=False) &
        df["behavioural_states"].str.contains("bend", na=False)
    )

    df["is_sharp_turn_anchor"] = (
        df["behavioural_states"].str.contains("changing_direction", na=False) &
        df["behavioural_states"].str.contains("large_bend", na=False)
    )

    # any frame that can belong to a turn sequence
    df["is_turn_related"] = (
        df["behavioural_states"].str.contains("changing_direction", na=False) |
        df["behavioural_states"].str.contains("bend", na=False) |
        df["behavioural_states"].str.contains("large_bend", na=False)
    )




    i = 0
    while i < len(df):

        if not df.loc[i, "is_turn_related"]:
            i += 1
            continue

        start = i
        while i < len(df) and df.loc[i, "is_turn_related"]:
            i += 1
        end = i - 1

        block = df.loc[start:end]

        if block["is_sharp_turn_anchor"].any():
            df.loc[start:end, "behaviour"] = "sharp_turn"
        elif block["is_turn_anchor"].any():
            df.loc[start:end, "behaviour"] = "turn"

    

    # detect pattern: bend → changing_direction (forward_run context)

    cond1 = df["behavioural_states"] == "forward_run|bend"
    cond2 = df["behavioural_states"].shift(-1) == "forward_run|changing_direction"

    turn_pair = cond1 & cond2

    # label both frames
    df.loc[turn_pair, "behaviour"] = "turn"
    df.loc[turn_pair.shift(1, fill_value=False), "behaviour"] = "turn"


        # additional stationary turn / sharp_turn sequences
    df["is_stationary_change"] = (
        df["behavioural_states"].str.contains("stationary", na=False) &
        df["behavioural_states"].str.contains("changing_direction", na=False)
    )

    df["is_stationary_bend"] = (
        df["behavioural_states"].str.contains("stationary", na=False) &
        df["behavioural_states"].str.contains("bend", na=False) &
        ~df["behavioural_states"].str.contains("large_bend", na=False)
    )

    df["is_stationary_large_bend"] = (
        df["behavioural_states"].str.contains("stationary", na=False) &
        df["behavioural_states"].str.contains("large_bend", na=False)
    )

    df["is_stationary_turn_related"] = (
        df["is_stationary_change"] |
        df["is_stationary_bend"] |
        df["is_stationary_large_bend"]
    )

    i = 0
    while i < len(df):

        if not df.loc[i, "is_stationary_turn_related"]:
            i += 1
            continue

        start = i
        while i < len(df) and df.loc[i, "is_stationary_turn_related"]:
            i += 1
        end = i - 1

        block = df.loc[start:end]

        if block["is_stationary_change"].any() and block["is_stationary_large_bend"].any():
            df.loc[start:end, "behaviour"] = "sharp_turn"
        elif block["is_stationary_change"].any() and block["is_stationary_bend"].any():
            df.loc[start:end, "behaviour"] = "turn"

    

    df.loc[df["behaviour"] == "stationary|bend", "behaviour"] = "turn"
    df.loc[df["behaviour"] == "stationary|large_bend", "behaviour"] = "sharp_turn"

    df.loc[df["behaviour"] == "stationary|backward|large_bend", "behaviour"] = "sharp_turn"
    df.loc[df["behaviour"] == "forward_run|backward|large_bend", "behaviour"] = "sharp_turn"


    df.loc[
    df["behaviour"] == "forward_run|backward|changing_direction",
    "behaviour"
    ] = "forward_run"

    df.loc[
        df["behaviour"] == "forward_run|large_bend",
        "behaviour"
    ] = "sharp_turn"


    # override: if hunching present → behaviour = hunching
    df.loc[df["is_hunching"], "behaviour"] = "hunching"
        
    # override: if casting present → behaviour = casting
    df.loc[df["behavioural_states"].str.contains("casting"), "behaviour"] = "casting" #might need to see if this is affected w the turn logic

    ## override: if digging present → behaviour = digging
    df.loc[df["digging_status"], "behaviour"] = "digging"

    return df




def run_metrics(track_df):
    df = correct_head_tail_swaps(track_df)
    df = speed(df)
    df = acceleration(df)
    df = tail_head_orientation(df)
    df = bending(df)
    df = head_lateral_displacement(df)
    df = aligned_movement(df)
    df = body_length(df)
    df = hunching(df)
    df = digging(df)
    df = behavioural_states(df)
    return df



""" 
"""






directory = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed'

identify_perimeters(directory)
track_data = process_directory(directory)

dfs = []

for track_file, df in track_data.items():
    df = (
        df
        .groupby("track_id", group_keys=False)
        .apply(run_metrics)
        .reset_index(drop=True))
    
    print("Raw trigger Trues:", df["flipped"].sum())
    print("Actually flipped after checking:", df["flipped_corrected"].sum())

    df = nearest_neighbour(df)

    df["file"] = track_file.replace(".tracks.feather", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

output_path = os.path.join(directory, 'behaviour_detection.csv')
data.to_csv(output_path, index=False)


directory = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated'

identify_perimeters(directory)
track_data = process_directory(directory)

dfs = []

for track_file, df in track_data.items():
    df = (
        df
        .groupby("track_id", group_keys=False)
        .apply(run_metrics)
        .reset_index(drop=True))
    
    print("Raw trigger Trues:", df["flipped"].sum())
    print("Actually flipped after checking:", df["flipped_corrected"].sum())

    df = nearest_neighbour(df)

    df["file"] = track_file.replace(".tracks.feather", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

output_path = os.path.join(directory, 'behaviour_detection.csv')
data.to_csv(output_path, index=False)



directory = '/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated'

identify_perimeters(directory)
track_data = process_directory(directory)

dfs = []

for track_file, df in track_data.items():
    df = (
        df
        .groupby("track_id", group_keys=False)
        .apply(run_metrics)
        .reset_index(drop=True))
    
    print("Raw trigger Trues:", df["flipped"].sum())
    print("Actually flipped after checking:", df["flipped_corrected"].sum())

    df = nearest_neighbour(df)

    df["file"] = track_file.replace(".tracks.feather", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

output_path = os.path.join(directory, 'behaviour_detection.csv')
data.to_csv(output_path, index=False)