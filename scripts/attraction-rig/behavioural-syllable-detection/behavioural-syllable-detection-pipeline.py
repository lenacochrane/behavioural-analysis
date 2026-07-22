import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

            min_align_distance_change = 400 #tweak if necessary 

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
    df = df.sort_values('frame').reset_index(drop=True)

    df["x"] = df["x_body"].rolling(window=5, min_periods=1).mean()
    df["y"] = df["y_body"].rolling(window=5, min_periods=1).mean()

    # Differences
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)

    # Distance and moving status
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['is_moving'] = df['distance'] > 0.1

    # Cumulative and std
    df["cumulative_displacement"] = df["distance"].cumsum()
    df["cumulative_displacement_rate"] = df["cumulative_displacement"].diff(10) / 10
    df["cumulative_displacement_rate"] = df["cumulative_displacement_rate"].fillna(0)

    df["x_std"] = df["x"].rolling(window=10, min_periods=1).std()
    df["y_std"] = df["y"].rolling(window=10, min_periods=1).std()

    df["overall_std"] = np.sqrt(df["x_std"]**2 + df["y_std"]**2)

    df['movement_score'] = df['cumulative_displacement_rate'] * df['overall_std']

    df['final_movement'] = (df['cumulative_displacement_rate'] > 0.1) | (df['movement_score'] > 0.25)

    ## smoothed final movement
    window_size = 50
    # df['digging_status'] = (
    #     df.groupby('track_id')['final_movement']
    #     .transform(lambda x: (~x).rolling(window=window_size, center=False).apply(lambda r: r.sum() >= (window_size * 0.8)).fillna(0).astype(bool))
    # )

    df["digging_status"] = (
    (~df["final_movement"])
    .rolling(window=window_size, min_periods=1)
    .apply(lambda r: r.sum() >= (window_size * 0.8))
    .fillna(0)
    .astype(bool)
)



    ### backfilling TRUE for larvae that actually end up digging 
    df["prev"] = df["digging_status"].shift(1).fillna(False)

    df['false_true'] = df['digging_status'] & ~df['prev'] # digging status = True ; prev frame digging status = False

    
    df["future_digging"] = (df["digging_status"]
    .rolling(window=50, min_periods=50)
    .sum()
    .shift(-49))
    

    df['long_digging'] = df['false_true'] & (df['future_digging'] >= 50)

    # 1) Initialize backfill column
    df['backfill'] = False

    # 2) Loop per track
    # for track_id, group in df.groupby('track_id'):
    #     idx   = group.index
    #     starts = idx[group.loc[idx, 'long_digging']]
    #     for s in starts:
    #         pre = max(idx.min(), s - 30)
    #         df.loc[pre:s-1, 'backfill'] = True  # back-fill up to the frame *before* 

    idx = df.index
    starts = idx[df.loc[idx, "long_digging"]]

    for s in starts:
        pre = max(idx.min(), s - 30)
        df.loc[pre:s-1, "backfill"] = True

    df['digging_status'] = df['digging_status'] | df['backfill']

    df.drop(columns=['backfill', 'long_digging', 'false_true', 'future_digging'], inplace=True)

    return df




def behavioural_states(track_df):

    df = track_df.copy()
    df = df.sort_values("frame").reset_index(drop=True)

    df["is_moving"] = df["tail_speed"] > 10
    df['is_moving_backwards'] = df['alignment'] < -10 
    df["is_changing_direction"] = df["tail_head_turn_rate"].abs() > 20
    df["is_bending"] = df["bend_magnitude"] > 40
    df["bend_type"] = ""
    df.loc[(df["bend_magnitude"] >= 40) & (df["bend_magnitude"] < 90), "bend_type"] = "bend"
    df.loc[df["bend_magnitude"] >= 90, "bend_type"] = "large_bend"


    threshold = 15 # CHANGE TO 18
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


    df.loc[
    df["behaviour"] == "forward_run|backward|changing_direction",
    "behaviour"
    ] = "forward_run"

    df.loc[
        df["behaviour"] == "forward_run|large_bend",
        "behaviour"
    ] = "sharp_turn"
        

    # override: if casting present → behaviour = casting
    df.loc[df["behavioural_states"].str.contains("casting"), "behaviour"] = "casting" #might need to see if this is affected w the turn logic

    ## override: if digging present → behaviour = digging
    df.loc[df["digging_status"], "behaviour"] = "digging"

    return df






""" 

PURPOSE:

This pipeline is designed to correct head–tail orientation swaps in fly tracking data. It identifies potential swaps based on vector alignment and instance scores, then applies a windowed correction approach to ensure consistent head–tail orientation across frames. The corrected coordinates are stored in new columns, allowing for accurate downstream behavioural analysis.


INPUT:



USAGE:

correct_head_tail_swaps = Designed to correct head-tail orientation swaps in cleaned tracking data. It takes a DataFrame for a single track. (mb i shd do for each function)



"""




df = pd.read_feather("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17.tracks.feather")

def run_metrics(track_df):
    df = correct_head_tail_swaps(track_df)
    df = speed(df)
    df = acceleration(df)
    df = tail_head_orientation(df)
    df = bending(df)
    df = head_lateral_displacement(df)
    df = aligned_movement(df)
    df = body_length(df)
    df = digging(df)
    df = behavioural_states(df)
    return df

df = (
    df
    .groupby("track_id", group_keys=False)
    .apply(run_metrics)
    .reset_index(drop=True)
)


print("Raw trigger Trues:", df["flipped"].sum())
print("Actually flipped after checking:", df["flipped_corrected"].sum())



df.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-flipped.csv",
    index=False
)   
print(df['behaviour'].unique())

df9 = df[df['track_id'] == 9]
df9.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_9-flipped.csv",
    index=False
    )

df6 = df[df['track_id'] == 6]
df6.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_6-flipped.csv",
    index=False
    )


df4 = df[df['track_id'] == 4]
df4.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_4-flipped.csv",
    index=False
    )

df8 = df[df['track_id'] == 8]
df8.to_csv(
    "/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/behavioural-detection-testing/2026-02-18_10-44-42_td17-track_8-flipped.csv",
    index=False
    )
