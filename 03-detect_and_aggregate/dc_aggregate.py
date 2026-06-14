"""
dc_aggregate.py
Tyler Edwards
Contains functions that translate YOLO Download Complete text outputs into a functional dataset
"""

import os
import time
import math
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------ UTILITY -------------------------------------------------------------------

def get_file_logger(name, log_filename, level=logging.INFO):
    """
    Returns a logger that writes exclusively to log_filename
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def timestamp(message):
    """
    Simple function to print out timestamp messages throughout run
    """
    ts = time.time()
    print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %I:%M:%S %p'), "||",message)


def frame_to_ts(frame):
    """
    Converts the raw frame of video integer to a more readable timestamp
    """
    total_seconds = frame / 60
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


# ------------------------------------------ READ & MERGE YOLO TEXT FILES ---------------------------------------------

def roster_frames(filepath, model_classes, chars, shoto=False,debug=False):  # (filepath to ROSTER detection output folder, roster_m classes)
    """
    Converts the frame by frame text file output from the YOLO ROSTER detection model into a dataframe
    """
    logger = get_file_logger("roster_frames", "logs/roster_frames.log", level=logging.DEBUG)
    logger.info(f"Extracting frame detections from {filepath}")

    _, _, files = next(os.walk(filepath)) # Get list of all file names in path
    file_count = len(files)  # Used for debug printouts

    # Initializing columns
    df = pd.DataFrame(columns = ["video", "frame", "time", "character", "x_center", "y_center", "width", "height", "area", "confidence"])

    progress = 1  # Used for progress bar
    # Parsing over YOLO texts and converting them to rows of data
    for file in tqdm(os.listdir(filepath), desc="Converting ROSTER detection .txt files into dataframe", unit="file"):
        with open(str(filepath + "/" + file), encoding = 'utf-8') as f:
            lines = f.readlines()  # Rows in text file
        # Splitting filename string to grab the frame # of video
        filesplit = file.split("_")
        # Video name is going to be all the text in the label filenames except the frame number at the end
        i = 1
        video = filesplit[0]
        while i < len(filesplit) - 1:
            video = video + "_" + filesplit[i]
            i = i + 1
        # Frame num will always be the last value in the list regardless of the video name
        frame = int(filesplit[-1].replace(".txt", "")) - 1

        # Turning the lines inside the text file into a list
        for d in lines:
            detection = d.split()
            cls = detection[0]

            # Class / character move or action
            char = str(model_classes[int(cls)])

            x = float(detection[1])
            y = float(detection[2])
            w = float(detection[3])
            h = float(detection[4])
            area = w*h
            conf = float(detection[5])

            if debug:
                logger.debug("--------------------------------------------------------------------")
                logger.debug(f"{progress}/{file_count}") # Can't use frame for progress counter since they're not actually in order in the folder
                logger.debug(filesplit)
                logger.debug(f"VIDEO: {video}")
                logger.debug(f"FRAME: {frame}")
                logger.debug(f"CHARACTER: {char}")
                logger.debug(f"CHAR_POSITION: {x},{y},{w},{h}")
                logger.debug(f"CONFIDENCE: {conf}")

            row = [str(video),int(frame),frame_to_ts(frame),char,x,y,w,h,area,conf]
            df.loc[len(df)] = row
        progress = progress + 1  # Used for debug printouts


    # Replace shoto detections with the one shoto we want if necessary
    shotos_selected = list(set(chars).intersection(['AKUMA', 'KEN', 'RYU', 'SEAN']))
    if len(shotos_selected) > 1:  # Shoto mirror
        print("Shoto mirror, detection might not be great")
    elif len(shotos_selected) == 1:  # Only one of characters in match is a shoto
        # Replace all shoto detections with the one we want
        df.loc[df['character'].isin(['AKUMA', 'KEN', 'RYU', 'SEAN']), 'character'] = shotos_selected[0]

    # Merge multiple detections to one
    # Convert center/dimensions to bounding box corners
    df['x_min'] = df['x_center'] - (df['width'] / 2)
    df['y_min'] = df['y_center'] - (df['height'] / 2)
    df['x_max'] = df['x_center'] + (df['width'] / 2)
    df['y_max'] = df['y_center'] + (df['height'] / 2)

    # Group and aggregate the "highest" values
    merged = df.groupby(['video', 'frame', 'character']).agg({
        'time': 'first',
        'x_min': 'min',  # True absolute top-left X
        'y_min': 'min',  # True absolute top-left Y
        'x_max': 'max',  # True absolute bottom-right X
        'y_max': 'max',  # True absolute bottom-right Y
        'confidence': 'max'  # Keep the highest confidence rating among the duplicates
    }).reset_index()

    # Recalculate original YOLO format and area
    merged['width'] = merged['x_max'] - merged['x_min']
    merged['height'] = merged['y_max'] - merged['y_min']
    merged['x_center'] = merged['x_min'] + (merged['width'] / 2)
    merged['y_center'] = merged['y_min'] + (merged['height'] / 2)
    merged['area'] = merged['width'] * merged['height']
    # Format back to original column order
    final_columns = ["video", "frame", "time", "character", "x_center", "y_center", "width", "height", "area","confidence"]
    df_cleaned = merged[final_columns]
    return df_cleaned


def character_frames(filepath, model_classes, vfx=False, debug=False):  # (filepath to CHARACTER detection output folder, character model)
    """
    Converts the frame by frame text file output from the YOLO CHARACTER detection model into a dataframe
    (Effectively the same as the roster function with slight variances. Is also used for VFX detections)
    """

    # Find name of character we're looking at
    if vfx == True:
        character = 'VFX'
    else:
        character = model_classes[0].split("-")[0]

    logger = get_file_logger(f"{character}_frames", f"logs/{character}_frames.log", level=logging.DEBUG)
    if debug:
        logger.info(f"Extracting frame detections from {filepath}")

    _, _, files = next(os.walk(filepath))
    file_count = len(files)

    # Initializing columns
    df = pd.DataFrame(columns = ["video", "frame", "time", "character", "action", "description", "x_center", "y_center", "width", "height", "area", "confidence"])

    progress = 1
    # Parsing over YOLO text file output and creating rows of data
    for file in tqdm(os.listdir(filepath), desc=f"Converting {character} .txt files into dataframe", unit="file"):
        with open(str(filepath + "/" + file), encoding = 'utf-8') as f:
            lines = f.readlines()
        # Splitting filename string to grab the frame # of video
        filesplit = file.split("_")
        # Video name is going to be all the text in the label filenames except the frame number at the end
        i = 1
        video = filesplit[0]
        while i < len(filesplit) - 1:
            video = video + "_" + filesplit[i]
            i = i + 1
        # Will always be the last value in the list regardless of the video name
        frame = int(filesplit[-1].replace(".txt", "")) - 1

        # Turning the lines inside the text file into a list
        for d in lines:
            detection = d.split()

            cls = detection[0]
            # Changing out the class numbers for the actual class names
            actiondesc = model_classes[int(cls)].split("-")
            # Grabbing character name
            character2 = actiondesc[0]

            # Extra values for jumps, blocking description, etc. *Dependent on consistent class naming format
            action = actiondesc[1]; desc = ""  # Defaults
            if len(actiondesc) > 2:
                desc = actiondesc[2]
                if len(actiondesc) == 4:
                    action = desc + actiondesc[3]
                    desc = ""

            # Description label cleanup
            if desc == "stand":
                desc = "standing"
            elif desc == "crouch":
                desc = "crouching"
            elif desc == "8":
                desc = "neutral"
            elif desc == "79":
                desc = "forward/backward"

            x = float(detection[1])
            y = float(detection[2])
            w = float(detection[3])
            h = float(detection[4])
            area = w*h
            conf = float(detection[5])

            if debug:
                logger.debug("--------------------------------------------------------------------")
                logger.debug(f"{progress}/{file_count}")  # Can't use frame for progress counter since they're not actually in order in the folder
                logger.debug(f"VIDEO: {video}")
                logger.debug(f"FRAME: {frame}")
                logger.debug(f"CHARACTER: {character2}")
                logger.debug(f"ACTIONDESC: {actiondesc}")
                logger.debug(f"ACTION: {action}")
                logger.debug(f"DESCRIPTION: {desc}")
                logger.debug(f"CONFIDENCE: {conf}")
                logger.debug(f"RAW DETECTION {detection}")

            row = [str(video), int(frame), frame_to_ts(frame), str(character2), str(action), str(desc), x, y, w, h, area, conf]
            df.loc[len(df)] = row
        progress = progress + 1
    return df


# ------------------------------------------ CHARACTER FRAMES VALIDATION ----------------------------------------------

def validate_characters(r, c_frames, iou_threshold=0.1):
    """
    Validates character action models against a baseline roster model using Vectorized IoU.
    """

    # Rename columns in roster df
    r = r[['video', 'frame', 'character', 'x_center', 'y_center', 'width', 'height']].rename(
                    columns={'x_center': 'truth_x', 'y_center': 'truth_y', 'width': 'truth_w', 'height': 'truth_h'})

    # Vectorized inner merge (Instantly drops false positive detections)
    merged = pd.merge(c_frames, r,on=['video', 'frame', 'character'],how='inner')

    # Convert both detections boxes to minx/max corners
    # Character action corners
    c_xmin = merged['x_center'] - (merged['width'] / 2)
    c_xmax = merged['x_center'] + (merged['width'] / 2)
    c_ymin = merged['y_center'] - (merged['height'] / 2)
    c_ymax = merged['y_center'] + (merged['height'] / 2)

    # Roster corners
    r_xmin = merged['truth_x'] - (merged['truth_w'] / 2)
    r_xmax = merged['truth_x'] + (merged['truth_w'] / 2)
    r_ymin = merged['truth_y'] - (merged['truth_h'] / 2)
    r_ymax = merged['truth_y'] + (merged['truth_h'] / 2)

    # Calculate intersecting rectangle (highest mins and lowest maxes)
    inter_xmin = np.maximum(c_xmin, r_xmin)
    inter_ymin = np.maximum(c_ymin, r_ymin)
    inter_xmax = np.minimum(c_xmax, r_xmax)
    inter_ymax = np.minimum(c_ymax, r_ymax)

    # Calculate area of overlap
    # .clip(lower=0) ensures that if the boxes DO NOT overlap, the width/height becomes 0 instead of a negative number
    inter_width = (inter_xmax - inter_xmin).clip(lower=0)
    inter_height = (inter_ymax - inter_ymin).clip(lower=0)
    inter_area = inter_width * inter_height

    # Calculate IoU ratio
    action_area = merged['width'] * merged['height']
    truth_area = merged['truth_w'] * merged['truth_h']
    union_area = action_area + truth_area - inter_area
    merged['iou'] = inter_area / union_area

    # Drop anything that doesn't physically overlap enough
    valid_detections = merged[merged['iou'] >= iou_threshold].copy()
    # Sort by confidence first then IoU for tiebreakers
    valid_detections = valid_detections.sort_values(['confidence', 'iou'], ascending=[False, False])
    # Drop duplicates by keeping first (best conf and iou)
    valid_detections = valid_detections.drop_duplicates(subset=['video', 'frame', 'character'], keep='first')
    # Sort again by time/frame so the order of the df makes sense when you look at it
    valid_detections = valid_detections.sort_values(['frame', 'time'], ascending=[True, True])
    # Drop temporary truth and math columns
    final_cols = [col for col in valid_detections.columns if not col.startswith('truth_') and col != 'iou']

    return valid_detections[final_cols].reset_index(drop=True)


# ------------------------------------------ ACTION MERGING ------------------------------------------------------------

# Takes character frame by frame data and converts it into full "actions".
# Ex.] A sprite appearing on screen for 10 frames consecutively turns into one "action" / row

def merge_frames(df, ts_func=frame_to_ts, min_frames=4, min_conf=0.50, max_gap_frames=6):
    """
    Merges individual detections of each frame into one row/action. Runs on each character separately.
    Ex.] 15 consecutive rows of "KEN-2MK" turns into 1 row with the frame is started and ended on.
    """

    # --- PASS 1 : Look at the data by action and then merge based on how many frames away the next duplicate action is

    # Sanitize Strings (Prevents NaN comparison failures)
    df = df.copy()
    df['character'] = df['character'].fillna("UNKNOWN").astype(str)
    df['action'] = df['action'].fillna("UNKNOWN").astype(str)
    df['description'] = df['description'].fillna("").astype(str)

    # Sort by actions first, then by frame
    # Instead of looking at the data in chronological order, sort by action and then merge based on how many frames away the next duplicate action is
    sort_cols = ['character', 'action', 'description', 'frame']
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Identify when a row changes
    char_changed = (df['character'] != df['character'].shift())
    action_changed = (df['action'] != df['action'].shift())
    desc_changed = (df['description'] != df['description'].shift())

    # Calculate frame gap between rows
    frame_gap = df.groupby(['character', 'action', 'description'])['frame'].diff()
    # Fill group starter NaNs with 0, then check if frame gap was exceeded
    gap_exceeded = frame_gap.fillna(0) > max_gap_frames
    # Combine markers to increment unique track block indices
    df['block_id'] = (char_changed | action_changed | desc_changed | gap_exceeded).cumsum()

    # Compress rows into single actions
    agg_df = df.groupby('block_id').agg(
        startup_frame=('frame', 'min'),
        ending_frame=('frame', 'max'),
        total_frames_detected=('frame', 'count'),  # Actual detected frame frames
        character=('character', 'first'),
        action=('action', 'first'),
        description=('description', 'first'),
        avg_confidence=('confidence', 'mean'),
        start_x=('x_center', 'first'),
        start_y=('y_center', 'first'),
        end_x=('x_center', 'last'),
        end_y=('y_center', 'last')
    ).reset_index(drop=True)

    # Filtering out actions with total_frames < min_frames and conf less than min_conf
    filtered_df = agg_df[(agg_df["total_frames_detected"] >= min_frames) &(agg_df["avg_confidence"] >= min_conf)].copy()

    # Sort back into order by frame/time
    filtered_df = filtered_df.sort_values(by=['startup_frame']).reset_index(drop=True)

    # Calculate/format, total_frames, time, starting_position, ending_position, and distance_moved columns
    filtered_df['total_frames'] = filtered_df['ending_frame'] - filtered_df['startup_frame'] + 1
    filtered_df['time'] = filtered_df['startup_frame'].apply(ts_func)
    filtered_df['starting_position'] = list(zip(filtered_df['start_x'], filtered_df['start_y']))
    filtered_df['ending_position'] = list(zip(filtered_df['end_x'], filtered_df['end_y']))
    dx = filtered_df['end_x'] - filtered_df['start_x']
    dy = filtered_df['end_y'] - filtered_df['start_y']
    filtered_df['distance_moved'] = np.sqrt(dx ** 2 + dy ** 2)

    ordered_columns = [
        "startup_frame", "ending_frame", "total_frames", "time", "character",
        "action", "description", "avg_confidence", "starting_position",
        "ending_position", "distance_moved"
    ]
    pass1 = filtered_df[ordered_columns].reset_index(drop=True)

    # --- PASS 2 : Now we sort the data chronologically and merge repeating actions again

    # Sort by frame of video
    df = pass1.sort_values(by=['startup_frame']).reset_index(drop=True)

    # Identify when a row changes
    char_changed = df['character'] != df['character'].shift()
    action_changed = df['action'] != df['action'].shift()
    desc_changed = df['description'] != df['description'].shift()

    # Calculate frame gap between consecutive rows (current startup_frame - previous row's ending_frame
    # *Can add this to the following merge but any consecutive action after the PASS 1 merge can usually safely be assumed to be the same instance of that action
    timeline_gap = df['startup_frame'] - df['ending_frame'].shift()

    # Trigger a merge split if the identity changes OR the timeline gap is exceeded
    df['collapse_id'] = (char_changed | action_changed | desc_changed).cumsum()

    # Collapse adjacent matching blocks
    pass2 = df.groupby('collapse_id').agg(
        startup_frame=('startup_frame', 'min'),
        ending_frame=('ending_frame', 'max'),
        character=('character', 'first'),
        action=('action', 'first'),
        description=('description', 'first'),
        avg_confidence=('avg_confidence', 'mean'),
        starting_position=('starting_position', 'first'),  # Keep origin of first block
        ending_position=('ending_position', 'last')  # Keep destination of last block
    ).reset_index(drop=True)

    # Calculate/format, total_frames, time, starting_position, ending_position, and distance_moved columns
    pass2['total_frames'] = pass2['ending_frame'] - pass2['startup_frame'] + 1
    pass2['time'] = pass2['startup_frame'].apply(ts_func)
    start_x = pass2['starting_position'].str[0]
    start_y = pass2['starting_position'].str[1]
    end_x = pass2['ending_position'].str[0]
    end_y = pass2['ending_position'].str[1]
    pass2['distance_moved'] = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    ordered_columns = [
        "startup_frame", "ending_frame", "total_frames", "time", "character",
        "action", "description", "avg_confidence", "starting_position",
        "ending_position", "distance_moved"
    ]

    return pass2[ordered_columns]


def load_context_rules(filepath="sf3_context.yaml"):
    def construct_yaml_tuple(loader, node):
        return tuple(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:seq', construct_yaml_tuple)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {filepath} could not be found.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return {}


def apply_advanced_sequence_context(df, sequence_rules):
    """
    Scans the timeline using a flat string representation of action+description.
    Filters by character block from the YAML, and dynamically splits the
    resolved action back into 'action' and 'description' columns.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Create a vectorized compound string array for matching
    # If description exists, make it 'action-description', else just 'action'
    has_desc = df['description'].notna() & (df['description'] != "")
    match_series = df['action'].astype(str).copy()
    match_series[has_desc] = df['action'].astype(str) + "-" + df['description'].astype(str)
    df['_match_string'] = match_series

    # 2. Iterate through the YAML character rules
    for character, rules in sequence_rules.items():
        if rules is None:
            continue

        # Character Gatekeeper: If this character's data isn't present, skip their rules
        char_mask = df['character'] == character
        if not char_mask.any():
            continue

        for pattern_list, resolved_action in rules.items():
            pattern = list(pattern_list)
            pattern_len = len(pattern)

            while True:
                match_found = False
                df_len = len(df)

                if df_len < pattern_len:
                    break

                # Slide down the chronological timeline
                for i in range(df_len - pattern_len + 1):
                    window_df = df.iloc[i: i + pattern_len]
                    current_window_states = window_df['_match_string'].tolist()

                    # Compare the timeline window directly to your flat YAML list
                    if current_window_states == pattern:
                        window_indices = window_df.index.tolist()
                        first_idx = window_indices[0]
                        last_idx = window_indices[-1]

                        # Stretch the final block back to the sequence start
                        df.loc[last_idx, 'startup_frame'] = df.loc[first_idx, 'startup_frame']
                        df.loc[last_idx, 'starting_position'] = df.loc[first_idx, 'starting_position']

                        # Parse the resolved action string back into Action and Description
                        if "-" in resolved_action:
                            act_part, desc_part = resolved_action.split("-", 1)
                        else:
                            act_part, desc_part = resolved_action, ""

                        # Assign the dynamically split components
                        df.loc[last_idx, 'action'] = act_part
                        df.loc[last_idx, 'description'] = desc_part

                        # Remove the setup/filler states
                        drop_indices = window_indices[:-1]
                        df = df.drop(drop_indices)

                        match_found = True
                        break  # Break to reset the tracking head loop due to mutated df size

                if not match_found:
                    break  # Clear to move to the next YAML rule

    # Clean up the temporary tracking column, recalculate frames, and re-index
    if '_match_string' in df.columns:
        df = df.drop(columns=['_match_string'])

    df['total_frames'] = df['ending_frame'] - df['startup_frame'] + 1
    return df.reset_index(drop=True)






#### Similar to actions but for VFX
def merge_vfx_frames(df, debug=False):
    """
    Merges individual detections of each frame into one row/action. Runs on each character separately.
    Ex.] 15 consecutive rows of "KEN-2MK" turns into 1 row with the frame is started and ended on
    """
    print()
############ CONTEXT LOGIC
# Trigger = move or condition that tells script that an action is potentially a reused animation of something else
    # VFX > Character = if you see VFX change character action row (See dudleys car = intro, see superart = super)
    # CHaracter to VFX = if you see  character do something look for VFX to confirm
    # Character ~, merge actions to one based on context actions )UREIN 236P to 236LP
# Conditions  = What conditions need to be met to confirm changes need to be made. Ex] Superflash, target combo, key animation
# Merge identified rows and use earliest startup frame, weighted averages etc.
### Where do we put context values? Next to everything else? I seperate sheet / file?
###### Remove detections from everything where Roster character aren't on screen (character select)

### Character, move to be adjusted, list of what to look for, look back int, look forward int
### merge with results?
def context(c, vfx, debug=False):
    print()






# ------------------------------------------ RESULTS MERGING ---------------------------------------------------------------

# Adding "results" to attacks in action dataframe
# *Can find the results of an attack by looking at the opposing character's animation during the attack
# Ex.] If P1 is doing 5HP and P2 is currently being hit, the 5HP's result is "hit"
def results(df):
    # Making sure the df is sorted by frame
    df = df.sort_values(by=["startup_frame"])
    df = df.reset_index(drop=True)
    df["time"] = df["startup_frame"].apply(frame_to_ts)

    # List of non-normals / attacks to filter out of hit/block checks
    states = ["standing", "walking", "crouching", "jump", "hit", "block", "dash", "jumping", 'lose', 'win',
              "forwarddash", "backdash", "knockdown", "rise", "walk", "block", "hit", "", "parry", np.nan]

    # Default result values for attacks
    df["P1_result"] = np.where(~df.action_x.isin(states), "whiff", "")
    df["P2_result"] = np.where(~df.action_y.isin(states), "whiff", "")

    # Iterating over actions/sprites and looking for the characters to be in "hit", "block", or "parry" state, then looking at the other character to see what caused that state.
    for index, row in df.iterrows():
        # Converting time doulbe into an actual datetime value

        ####  have to do something here to handle projectiles?
        if df.loc[index, "action_x"] in ["hit", "block", "parry"] and index > 0:
        # Look at P1 action, if HIT or BLOCK, look for previous P2 action that caused that state
            i = 0
            print(index, i)
            print(df.loc[index - i, "action_y"])
            while True: # Found the move that caused the hit/block
                if df.loc[index - i, "action_y"] not in states and pd.isna(df.loc[index - i, "action_y"]) == False:
                    df.loc[index - i, "P2_result"] = df.loc[index, "action_x"]
                    break
                else: # Keep searching for the move that caused the hit/block
                    i = i + 1
        elif df.loc[index, "action_y"] in ["hit", "block", "parry"] and index > 0:
        # Look at P2 action, if HIT or BLOCK, look for previous P1 action that caused that state
            i = 0
            while True: # Found the move that caused the hit/block
                if df.loc[index - i, "action_x"] not in states and pd.isna(df.loc[index - i, "action_x"]) == False:
                    df.loc[index - i, "P1_result"] = df.loc[index, "action_y"]
                    break
                else: # Keep searching for the move that caused the hit/block
                    i = i + 1

        # Reorder and rename columns
    df = df[['time', 'startup_frame', 'ending_frame_x',
             'total_frames_x', "character_x", "action_x",
             "description_x","P1_result", "avg_confidence_x",
             'starting_position_x', 'ending_position_x', 'distance_moved_x',
            'ending_frame_y', 'total_frames_y', "character_y",
             "action_y", "description_y", "P2_result", "avg_confidence_y",
             'starting_position_y', 'ending_position_y', 'distance_moved_y']]

    cols = ['timestamp', 'startup_frame', 'P1_ending_frame',
            'P1_total_frames','P1_character','P1_action',
            'P1_description', 'P1_result', 'P1_avg_confidence',
            'P1_starting_position', "P1_ending_position", "P1_distance_moved",
            'P2_ending_frame', 'P2_total_frames', 'P2_character',
            'P2_action', 'P2_description', 'P2_result', 'P2_avg_confidence',
            'P2_starting_position', "P2_ending_position", "P2_distance_moved"]
    df.columns = cols
    return df


# ------------------------------------------ ARCHIVE -------------------------------------------------------------------

def validate_characters_old(r, c1, c2, debug=False):  # df output from roster_frames and character_frames 1 and 2
    """
    Parses over each batch of character action detections in every frame and tries to validate the best detection
    to move forward with by referencing where the roster detection says that character is supposed to be
    on that given frame
    """
    # ---------------------------------------------------------------------------------------------
    # ------------------------------------------- SETUP -------------------------------------------
    # ---------------------------------------------------------------------------------------------

    logger = get_file_logger(f"validate_characters", f"logs/validate_characters.log", level=logging.DEBUG)

    # Finds the last frame of the video and uses it for the limit of the main while loop
    last_frame = max(set(list(r['frame']) + list(c1['frame']) + list(c2['frame'])))

    # Grabbing characters from each model and attempting to handle shotos
    char_c1 = c1['character'].iloc[0]  # Character from model 1
    char_c2 = c2['character'].iloc[0]  # Character from model 2
    expected_chars = [char_c1, char_c2]

    def d_stats(df, roster=False):
        """
        Descriptive statistics by character for 'area' and 'confidence'.
        Modified to handle both model types (roster/character)
        """

        # Creating descriptive statistics for area and conf for every character detected in roster model
        stats = pd.DataFrame(r.groupby('character')[['area', 'confidence']].describe())
        stats.columns = stats.columns.droplevel(0)  # Getting rid of multi-index because it's confusing
        stats = stats.reset_index()
        if debug and roster:
            stats.to_csv("logs/roster_dstats.csv", index=False)
        elif debug and not roster:
            char = df['character'].iloc[0]
            stats.to_csv(f"logs/{char}_dstats.csv", index=False)

        # Fixing column names for descriptive stats
        col = 0
        statsnames = ['area_', 'confidence_']
        for i in statsnames:
            stats.columns.values[1 + col] = i + "count"; stats.columns.values[2 + col] = i + "mean"; stats.columns.values[3 + col] = i + "std"
            stats.columns.values[4 + col] = i + "min"; stats.columns.values[5 + col] = i + "25%"; stats.columns.values[6 + col] = i + "50%"
            stats.columns.values[7 + col] = i + "75%"; stats.columns.values[8 + col] = i + "max"
            col = col + 8

        # Setting up area and confidence outliers for roster character 1
        c1_area_avg = stats[stats['character'] == char_c1]['area_mean'].values[0]
        c1_area_std = stats[stats['character'] == char_c1]['area_std'].values[0]
        c1_area_out = [c1_area_avg-(3*c1_area_std), c1_area_avg+(3*c1_area_std)]

        c1_conf_avg = stats[stats['character'] == char_c1]['confidence_mean'].values[0]
        c1_conf_std = stats[stats['character'] == char_c1]['confidence_std'].values[0]
        c1_conf_out = [c1_conf_avg-(3*c1_conf_std), c1_conf_avg+(3*c1_conf_std)]

        if roster:  # Setting up outliers for two characters
            # Setting up area and confidence outliers for roster character 2
            c2_area_avg = stats[stats['character'] == char_c2]['area_mean'].values[0]
            c2_area_std = stats[stats['character'] == char_c2]['area_std'].values[0]
            c2_area_out = [c2_area_avg - (3 * c2_area_std), c2_area_avg + (3 * c2_area_std)]

            c2_conf_avg = stats[stats['character'] == char_c2]['confidence_mean'].values[0]
            c2_conf_std = stats[stats['character'] == char_c2]['confidence_std'].values[0]
            c2_conf_out = [c2_conf_avg - (3 * c2_conf_std), c2_conf_avg + (3 * c2_conf_std)]
            cr_list = [char_c1, char_c2]

            return c1_area_out, c1_conf_out, c2_area_out, c2_conf_out, cr_list, c1_conf_std, c2_conf_std  # For roster model
        return c1_area_out, c1_conf_out, c1_conf_std  # For character models

    r_d_stats = d_stats(r, roster=True)
    # char_r_list = r_d_stats[4]  # Characters found in roster model #### redundant now
    c1_d_stats = d_stats(c1)
    c2_d_stats = d_stats(c2)
    if debug:
        logger.debug(expected_chars)
        # logger.debug(char_r_list)
        logger.debug(r_d_stats)
        logger.debug(c1_d_stats)
        logger.debug(c2_d_stats)

    # Removing all rows with low outlier confidences. *False negatives are better than false positives
    r = r[r['confidence'] > np.mean([r_d_stats[1][0], r_d_stats[3][0]])]
    c1 = c1[c1['confidence'] > c1_d_stats[1][0]]
    c2 = c2[c2['confidence'] > c2_d_stats[1][0]]
    r.to_csv('r_conf_filter_test.csv', index=False)
    if debug:
        c1.to_csv('c1_conf_filter_test.csv', index=False)
        c2.to_csv('c2_conf_filter_test.csv', index=False)

    # Setting up empty dataframes to fill with winning detections and initializing variables
    c1_val = pd.DataFrame(columns=["video", "frame", "time", "character", "action", "description", "x_center", "y_center", "width", "height", "area", "confidence"])
    c2_val = pd.DataFrame(columns=["video", "frame", "time", "character", "action", "description", "x_center", "y_center", "width", "height", "area", "confidence"])
    f = 0  # Frame of video the loop is currently looking at
    t1 = 0  # Character 1 tracking trigger counter
    t2 = 0  # Character 2 tracking trigger counter
    tracking_c1 = False  # Character 1 tracking trigger
    tracking_c2 = False  # Character 2 tracking trigger
    # Lists used during pre-tracking for getting the average confidence of the past 5 detections
    t1_prev_5_conf = []
    t2_prev_5_conf = []


    # -------------------------------------------------------------------------------------------------
    # ------------------------------------------- MAIN LOOP -------------------------------------------
    # -------------------------------------------------------------------------------------------------
    while f < last_frame:  # Iterate over frames of the video

        ####### Add progress bar
        if debug:
            # print("----------------------------",str(f) + "f || ", str(round(f/60,2)) + "s ----------------------------")
            logger.debug(f"---------------------------- {f} || {str(round(f/60,2))}s----------------------------")
            logger.debug(f"TRACKING {char_c1} -> {tracking_c1}, {t1}")
            logger.debug(f"TRACKING {char_c2} -> {tracking_c2}, {t2}")

        # Each dataframe filtered for the detections on the current frame f and sorted by highest confidence
        r_f = r[r["frame"] == f].sort_values('confidence', ascending=False).reset_index()
        c1_f = c1[c1["frame"] == f].sort_values('confidence', ascending=False).reset_index()
        c2_f = c2[c2["frame"] == f].sort_values('confidence', ascending=False).reset_index()

        # Initialize distance columns
        r_f['distance_from_r1_prev'] = 0.0
        r_f['distance_from_c1_prev'] = 0.0
        # r_f['distance_avg1'] = 0.0
        r_f['distance_from_r2_prev'] = 0.0
        r_f['distance_from_c2_prev'] = 0.0
        # r_f['distance_avg2'] = 0.0

        c1_f['distance_from_r1_prev'] = 0.0
        # c1_f['distance_from_r1_current'] = 0.0
        c1_f['distance_from_c1_prev'] = 0.0

        c2_f['distance_from_r2_prev'] = 0.0
        # c2_f['distance_from_r2_current'] = 0.0
        c2_f['distance_from_c2_prev'] = 0.0

        # Checking if tracking has been lost (Last 5 frames have been skipped)
        if t1 <= 0:
            tracking_c1 = False
            if debug:
                print("----- Tracking LOST for,", expected_chars[0], "on", f - 5, 'to', f)
        if t2 <= 0:
            tracking_c2 = False
            if debug:
                print("----- Tracking LOST for,", expected_chars[1], "on", f - 5, 'to', f)

        # Select r_current and check to that r1 and r2 current are not looking at the same thing
        j = 0
        while j < len(r_f['x']):  # Find distance of current r_f rows from previous c and r detections
            if tracking_c1:
                r_f.loc[j, 'distance_from_r1_prev'] = float(math.dist(r1_xy_prev, [r_f['x'][j], r_f['y'][j]]))
                r_f.loc[j, 'distance_from_c1_prev'] = float(math.dist(c1_xy_prev, [r_f['x'][j], r_f['y'][j]]))
            if tracking_c2:
                r_f.loc[j, 'distance_from_r2_prev'] = float(math.dist(r2_xy_prev, [r_f['x'][j], r_f['y'][j]]))
                r_f.loc[j, 'distance_from_c2_prev'] = float(math.dist(c2_xy_prev, [r_f['x'][j], r_f['y'][j]]))
            j = j + 1
        r_f['distance_avg1'] = r_f['distance_from_r1_prev'] +  r_f['distance_from_c1_prev'] / 2
        r_f['distance_avg2'] = r_f['distance_from_r2_prev'] +  r_f['distance_from_c2_prev'] / 2

        if len(r_f) > 0:  # Just to make sure this doesn't try to run at the start of the loop when frame = 0
            if tracking_c1 and len(c1_f) > 0:
                winner_r1 = r_f['distance_avg1'].idxmin()  # Coordinates of current frame's roster detection
                r1_xy_current = [r_f.iloc[winner_r1]['x'], r_f.iloc[winner_r1]['y']]  # Current detection for c1 in roster
            if tracking_c2 and len(c2_f) > 0:
                winner_r2 = r_f['distance_avg2'].idxmin()  # Coordinates of current frame's roster detection
                r2_xy_current = [r_f.iloc[winner_r2]['x'], r_f.iloc[winner_r2]['y']]  # Current detection for c2 in roster
            if (tracking_c1 and len(c1_f) > 0) and (tracking_c2 and len(c2_f) > 0):
                if winner_r1 == winner_r2:  # Confirm they're not looking at the same detection ####### can improve this
                    if debug:
                        print("----- SAME DETECTION")
                        print("----- WINNER 1 ->", winner_r1)
                        print("----- WINNER 2 ->", winner_r2)
                        print("----- DETECTED CHARACTERS ->", list(r_f['character']))
                    f = f + 1
                    t1 = t1 - 1
                    t2 = t2 - 1
                    continue

        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------- CHARACTER 1 TRACKING -------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        if tracking_c1 and len(c1_f) > 0:  ## AND character in roster_f set/list
            # Calculate distance for each row in c1 to previous c1,r1 and current r1 and choose the closest average
            # Filling distance columns
            i = 0
            while i < len(c1_f):
                c1_f.loc[i, 'distance_from_c1_prev'] = float(math.dist(c1_xy_prev, [c1_f['x'][i], c1_f['y'][i]]))
                c1_f.loc[i, 'distance_from_r1_prev'] = float(math.dist(r1_xy_prev, [c1_f['x'][i], c1_f['y'][i]]))
                c1_f.loc[i, 'distance_from_r1_current'] = float(math.dist(r1_xy_current, [c1_f['x'][i], c1_f['y'][i]]))
                c1_f.loc[i, 'distance_average'] = np.mean([c1_f.loc[i, 'distance_from_c1_prev'],
                                                          c1_f.loc[i, 'distance_from_r1_prev'],
                                                           c1_f.loc[i, 'distance_from_r1_current']])
                i = i + 1
            # Distance is less than 0.2 and confidence isn't an outlier
            if len(c1_f[(c1_f['distance_average'] < 0.1) & (c1_f['confidence'] > c1_d_stats[1][0])]) > 0:
                c1_f_d = c1_f[(c1_f['distance_average'] < 0.1) & (c1_f['confidence'] > c1_d_stats[1][0])]
                c1_f_d = c1_f_d.sort_values(['distance_average', 'confidence'], ascending=[True,False])
                if debug:
                    print(expected_chars[0], c1_f_d.iloc[0]['action'])
                c1_val.loc[len(c1_val)] = c1_f_d.iloc[0]  # Add row to final return output dataframe

                # Setup values for next loop
                c1_xy_prev = [c1_f_d.iloc[0]['x'], c1_f_d.iloc[0]['y']]
                r1_xy_prev = r1_xy_current  # Starting coordinates to begin tracking c1
                t1 = 5
            else:  ######## Good enough for now but may have to implement more logic to handle when it gets lost
                if debug:
                    print("----- NO C1 WINNER")
                    #print(c1_f_d.to_string())
                    print(list(r_f['character']))
                t1 = t1 - 1
                if t1 < 0:
                    tracking_c1 = False
                    if debug:
                        print("----- Tracking LOST for,", expected_chars[0], "on", f - 5, 'to', f)

        # Conditions to start tracking
        # Expected character is in the roster detections AND the area of that detection is not an outlier then t + 1
        elif expected_chars[0] in list(r_f['character']) and r_f['area'][r_f['character'] == expected_chars[0]].mean() > \
                r_d_stats[0][0] and len(c1_f) > 0:

            # Copies of r (entire roster dataframe) and r_f (current frame)
            t1_r_f = r_f[r_f['character'] == expected_chars[0]]  # To shorten the length of some variables below
            # t1_r = r[r['character'] == expected_chars[0]].drop_duplicates(subset=['character','frame'], keep='first').copy(deep=True)

            # If t1 reaches 5 AND average confidence for past 5 frames was > 50% -> Start tracking and redo past 5 frames
            if t1 >= 5 and np.mean(t1_prev_5_conf) >= 0.50:
                if debug:
                    print("----- Tracking for,", expected_chars[0], ",  'triggered on frames", f - 5, 'to', f)
                tracking_c1 = True
                t1_prev_5_conf = []
                f = f - 5  # Go back 5 frames and redo them
                if tracking_c2:  # Restore c2 xy values from 5 frames ago
                    c2_xy_prev = c2_xy_store
                    r2_xy_prev = r2_xy_store
                else:  # Reset t2 tracking trigger if not tracking c2 already
                    t2 = 0
                continue  # Skip everything below so don't have to worry about
            elif t1 == 1:  # Initialize values in case tracking is successful
                exp_c1_i = t1_r_f['character'][t1_r_f['character'] == expected_chars[0]].index[
                    0]  # Index of top detect?
                c1_xy_prev = [t1_r_f['x'][exp_c1_i], t1_r_f['y'][exp_c1_i]]  # Starting coordinates to begin tracking c1
                r1_xy_prev = [t1_r_f['x'][exp_c1_i], t1_r_f['y'][exp_c1_i]]  # Starting coordinates to begin tracking r1
                if tracking_c2:  # Store character 1 values to return to in case tracking for c1 is successful and we need to go back -5f
                    c2_xy_store = c2_xy_prev
                    r2_xy_store = r2_xy_prev
            t1 = t1 + 1  # Add to tracking trigger count
            t1_prev_5_conf.append(r_f['confidence'][r_f['character'] == expected_chars[0]].max())
        elif tracking_c1:  # Detections on this frame don't meet our conditions so skip it and drop a point
            t1 = t1 - 1
        else:  # If tracking not True and initialize conditions not met, reset trigger count
            t1 = 0
            t1_prev_5_conf = []

        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------- CHARACTER 2 TRACKING -------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        if tracking_c2 and len(c2_f) > 0:
            # Calculate distance for each row in c2 to previous c2 and current r2 and choose the closest based on each
            # Filling distance columns
            i = 0
            while i < len(c2_f):
                c2_f.loc[i, 'distance_from_c2_prev'] = float(math.dist(c2_xy_prev, [c2_f['x'][i], c2_f['y'][i]]))
                c2_f.loc[i, 'distance_from_r2_prev'] = float(math.dist(r2_xy_prev, [c2_f['x'][i], c2_f['y'][i]]))
                c2_f.loc[i, 'distance_from_r2_current'] = float(math.dist(r2_xy_current, [c2_f['x'][i], c2_f['y'][i]]))
                c2_f.loc[i, 'distance_average'] = np.mean([c2_f.loc[i, 'distance_from_c2_prev'],
                                                           c2_f.loc[i, 'distance_from_r2_prev'],
                                                           c2_f.loc[i, 'distance_from_r2_current']])
                i = i + 1
            # Distance is less than 0.2 and confidence isn't an outlier
            if len(c2_f[(c2_f['distance_average'] < 0.1) & (c2_f['confidence'] > c2_d_stats[1][0])]) > 0:
                c2_f_d = c2_f[(c2_f['distance_average'] < 0.1) & (c2_f['confidence'] > c2_d_stats[1][0])]
                c2_f_d = c2_f_d.sort_values(['distance_average', 'confidence'], ascending=[True, False])
                if debug:
                    print(expected_chars[1], c2_f_d.iloc[0]['action'])
                c2_val.loc[len(c2_val)] = c2_f_d.iloc[0]  # Add row to final return output dataframe

                # Setup values for next loop
                c2_xy_prev = [c2_f_d.iloc[0]['x'], c2_f_d.iloc[0]['y']]
                r2_xy_prev = r2_xy_current  # Starting coordinates to begin tracking c2
                t2 = 5
            else:  ######## Good enough for now but may have to implement more logic to handle when it gets lost
                if debug:
                    print("----- NO c2 WINNER")
                    #print(c2_f_d.to_string())
                    print(list(r_f['character']))
                t2 = t2 - 1
                if t2 < 0:
                    tracking_c2 = False
                    if debug:
                        print("----- Tracking LOST for,", expected_chars[0], "on", f - 5, 'to', f)

        # Conditions to start tracking
        # Expected character is in the roster detections AND the area of that detection is not an outlier then t + 1
        elif expected_chars[1] in list(r_f['character']) and r_f['area'][r_f['character'] == expected_chars[1]].mean() > r_d_stats[2][0] and len(c2_f) > 0:
            t2 = t2 + 1  # Add to tracking trigger count #### top or bottom of if statement?
            t2_prev_5_conf.append(r_f['confidence'][r_f['character'] == expected_chars[1]].max())

            # Copies of r (entire roster dataframe) and r_f (current frame)
            t2_r_f = r_f[r_f['character'] == expected_chars[1]]  # To shorten the length of some variables below
            # t2_r = r[r['character'] == expected_chars[1]].drop_duplicates(subset=['character','frame'], keep='first').copy(deep=True)

            # If t2 reaches 5 AND average confidence for past 5 frames was > 50% -> Start tracking and redo past 5 frames
            if t2 >= 5 and np.mean(t2_prev_5_conf) >= 0.50:
                if debug:
                    print("----- Tracking for,",expected_chars[1], ",  'triggered on frames", f - 5, 'to', f)
                tracking_c2 = True
                t2_prev_5_conf = []
                f = f - 5  # Go back 5 frames and redo them
                if tracking_c1:  # Restore c1 xy values from 5 frames ago
                    c1_xy_prev = c1_xy_store
                    r1_xy_prev = r1_xy_store
                else:  # Reset t1 tracking trigger if not tracking c1 already
                    t1 = 0
                continue  # Skip everything below so don't have to worry about
            elif t2 == 1:  # Initialize values in case tracking is successful
                exp_c2_i = t2_r_f['character'][t2_r_f['character'] == expected_chars[1]].index[0]  # Index of top detect?
                c2_xy_prev = [t2_r_f['x'][exp_c2_i], t2_r_f['y'][exp_c2_i]]  # Starting coordinates to begin tracking c2
                r2_xy_prev = [t2_r_f['x'][exp_c2_i], t2_r_f['y'][exp_c2_i]]  # Starting coordinates to begin tracking r2
                if tracking_c1:  # Store character 1 values to return to in case tracking for c2 is successful and we -5f
                    c1_xy_store = c1_xy_prev
                    r1_xy_store = r1_xy_prev
            t2 = t2 + 1  # Add to tracking trigger count
            t2_prev_5_conf.append(r_f['confidence'][r_f['character'] == expected_chars[1]].max())
        elif tracking_c2:  # Detections on this frame don't meet our conditions so skip it and drop a point
            t2 = t2 - 1
        else:  # If tracking not True and initialize conditions not met, reset trigger count
            t2 = 0
            t2_prev_5_conf = []

        f = f + 1
    return c1_val.drop_duplicates(), c2_val.drop_duplicates()  # Drop duplicates to handle re-added rows from -5fs


def merge_frames_old(df, debug=False):
    """
    Merges individual detections of each frame into one row/action. Runs on each character separately.
    Ex.] 15 consecutive rows of "KEN-2MK" turns into 1 row with the frame is started and ended on
    """

    new_df = pd.DataFrame(columns=["startup_frame", "ending_frame", "total_frames", "time", "character", "action",
                                      "description", "avg_confidence", 'starting_position', 'ending_position',
                                      'distance_moved'])
    # Initialize variables on first row
    prev_act = df["action"].iloc[0]
    prev_desc = df["description"].iloc[0]
    firstframe = df["frame"].iloc[0]
    lastframe = 0; duration = 1
    conf = df["confidence"].iloc[0]
    xy1 = [df["x_center"].iloc[0], df["y_center"].iloc[0]]
    i = 1
    # Iterating over each row/frame and merging consecutive appearances of the same sprite into one row/aciton
    while i < len(df):
        # If previous action is different from current, stop tracking current move
        if [prev_act, prev_desc] != [df["action"].iloc[i], df["description"].iloc[i]]:
            if debug:
                print(i, prev_act, prev_desc, firstframe, lastframe, duration, conf)
                print("-----------------------------------------------------------------------------------------")
            xy2 = [df["x_center"].iloc[i], df["y_center"].iloc[i]]
            dxy = float(math.dist(xy1, xy2))
            if duration <= 1:  # Only appears for one frame
                lastframe = firstframe
            else:  # Appears for multiple frames
                conf = conf / duration  # Average confidence across action
            new_row = [firstframe, lastframe, duration, frame_to_ts(firstframe), df["character"].iloc[i], prev_act,
                       prev_desc, conf, xy1, xy2, dxy]
            new_df.loc[len(new_df)] = new_row

            # Reset variables for next move
            prev_act = df["action"].iloc[i]
            prev_desc = df["description"].iloc[i]
            firstframe = df["frame"].iloc[i]
            xy1 = [df["x_center"].iloc[i], df["y_center"].iloc[i]]
            lastframe = 0
            conf = df["confidence"].iloc[i]
            duration = 1
        else:  # If previous action is the same, keep going
            lastframe = df["frame"].iloc[i]
            conf = conf + float(df["confidence"].iloc[i])
            if debug:
                print(i, prev_act, prev_desc, firstframe, lastframe, duration, conf)
            duration = duration + 1
        i = i + 1

    # Cleaning up noise
    # Clearing out rows that only appear for a few frames or have low confidence
    new_df1 = new_df[(new_df["total_frames"] >= 4) & (new_df["avg_confidence"] >= 0.5)]
    clean = False
    new_df2 = pd.DataFrame(columns=["startup_frame", "ending_frame", "total_frames", "time", "character", "action",
                                    "description", "avg_confidence", 'starting_position', 'ending_position',
                                    'distance_moved'])
    # Pass over dataframe until there are no more "duplicated actions" where its one action detected multiple times
    while not clean:
        # Initialize variables on first detection frame again
        prev_act = new_df1["action"].iloc[0]
        prev_desc = new_df1["description"].iloc[0]
        firstframe = new_df1["startup_frame"].iloc[0]
        lastframe = new_df1["ending_frame"].iloc[0]
        conf = new_df1["avg_confidence"].iloc[0]
        xy1 = new_df1['starting_position'].iloc[0]
        c = 0; j = 1; duration = 1
        while j < len(new_df1):
            if prev_act != new_df1["action"].iloc[j] and prev_desc != new_df1["description"].iloc[j]: # If previous action is different from current, stop tracking current move
                xy2 = new_df1['ending_position'].iloc[j]
                dxy = float(math.dist(xy1, xy2))
                if duration <= 1:  # Only appears for one frame
                    lastframe = new_df1["ending_frame"].iloc[j]
                else:  # Appears for multiple frames
                    conf = conf / duration
                new_row = [firstframe, lastframe, lastframe-firstframe,  frame_to_ts(firstframe), new_df1["character"].iloc[j], prev_act,
                           prev_desc, conf, xy1, xy2, dxy]
                new_df2.loc[len(new_df2)] = new_row

                # Reset for next move
                prev_act = new_df1["action"].iloc[j]
                prev_desc = new_df1["description"].iloc[j]
                firstframe = new_df1["startup_frame"].iloc[j]
                xy1 = new_df1['starting_position'].iloc[j]
                lastframe = new_df1["ending_frame"].iloc[j]
                conf = new_df1["avg_confidence"].iloc[j]
                duration = 1
            else:  # If previous action is the same, keep going
                lastframe = new_df1["ending_frame"].iloc[j]
                conf = conf + float(new_df1["avg_confidence"].iloc[j])
                duration = duration + 1
                c = c + 1
            j = j + 1
        if c == 0:
            clean = True
        new_df1 = new_df2.copy()

    return new_df2.drop_duplicates(subset=['startup_frame', 'time', 'action', 'description'], keep='last').sort_values(by=['startup_frame','time'], ascending=True)
