
import os
import pandas as pd
import numpy as np
import datetime
import math

# ------------------------------------------ ROSTER FRAMES MERGING ---------------------------------------------------------------
def roster_frames(filepath, keypath):
    _, _, files = next(os.walk(filepath))
    file_count = len(files)

    # Initializing columns
    df = pd.DataFrame(columns = ["video", "frame", "time", "character", "x", "y", "w", "h", "area", "confidence"])

    # Creating dictionary for move/sprite list since they're encoded in the text file output
    with open(keypath, encoding="utf8") as f:
        lines = f.readlines()
        i_list = list(range(0,len(lines)))
        lines2 = []
        for i in lines:
            lines2.append(i.strip())
        keys = dict(zip(i_list, lines2))

    progress = 1
    # Parsing over YOLO text file output and creating rows of data
    for file in os.listdir(filepath):
        with open(str(filepath + "/" + file)) as f:
            lines = f.readlines()  # Rows in text file
        # Splitting filename string to grab the frame # of video
        filesplit = file.split("_")
        # Video name is going to be all the text in the label filenames except the frame number at the end
        i = 1
        video = filesplit[0]
        while i < len(filesplit):
            video = video + "_" + filesplit[i]
            i = i + 1
        # Will always be the last value in the list regardless of the video name
        frame = int(filesplit[-1].replace(".txt", "")) - 1

        # Turning the lines inside the text file into a list
        for d in lines:
            detection = d.split()
            cls = detection[0]

            # Class / character move or action
            char = str(keys[int(cls)])

            x = float(detection[1])
            y = float(detection[2])
            w = float(detection[3])
            h = float(detection[4])
            area = w*h
            conf = float(detection[5])

            # Debug printouts
            # print(str(progress) + "/" + str(file_count)) # Can't use frame for progress counter since they're not actually in order in the folder
            # print("VIDEO:", video)
            # print("FRAME:", frame)
            #
            # print("CHARACTER:", char)
            # print("CHAR_POSITION:", x,y,w,h)
            # print("CONFIDENCE:", conf)
            # print("--------------------------------------------------------------------")

            row = [str(video),int(frame),round(frame/60,2),char,x,y,w,h,area,conf]
            df.loc[len(df)] = row
        progress = progress + 1
    return df

# ------------------------------------------ FRAME MERGING ---------------------------------------------------------------
# Create dataframe of frame by frame sprites on screen. (Raw output of .txt files into one dataframe)
def character_frames(filepath, keypath):
    _, _, files = next(os.walk(filepath))
    file_count = len(files)

    # Initializing columns
    df = pd.DataFrame(columns = ["video", "frame", "time", "character", "action", "description", "x", "y", "w", "h", "area", "confidence"])

    # Creating dictionary for move/sprite list since they're encoded in the text file output
    with open(keypath, encoding='utf-8') as f:
        lines = f.readlines()
        i_list = list(range(0,len(lines)))
        lines2 = []
        for i in lines:
            lines2.append(i.strip())
        keys = dict(zip(i_list, lines2))

    progress = 1
    # Parsing over YOLO text file output and creating rows of data
    for file in os.listdir(filepath):
        with open(str(filepath + "/" + file), encoding = 'utf-8') as f:
            lines = f.readlines()
        # Splitting filename string to grab the frame # of video
        filesplit = file.split("_")
        # Video name is going to be all the text in the label filenames except the frame number at the end
        i = 1
        video = filesplit[0]
        while i < len(filesplit):
            video = video + "_" + filesplit[i]
            i = i + 1
        # Will always be the last value in the list regardless of the video name
        frame = int(filesplit[-1].replace(".txt", "")) - 1

        # Turning the lines inside the text file into a list
        for d in lines:
            detection = d.split()

            cls = detection[0]
            # Changing out the class numbers for the actual class names
            actiondesc = keys[int(cls)].split("-")
            # Grabbing character name
            character = actiondesc[0]

            # Extra values for jumps, blocking description, etc. *Dependent on consistent class naming conventions
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

            # Debug printouts
            # print(str(progress) + "/" + str(file_count))  # Can't use frame for progress counter since they're not actually in order in the folder
            # print("VIDEO:", video)
            # print("FRAME:", frame)
            # print("CHARACTER:", character)
            # print(actiondesc)
            # print("ACTION:", action)
            # print("DESCRIPTION:", desc)
            # print("CONFIDENCE:", conf)
            # print("RAW DETECTION", detection)
            # print("--------------------------------------------------------------------")

            row = [str(video), int(frame), round(frame/60,2), str(character), str(action), str(desc), x, y, w, h, area, conf]
            df.loc[len(df)] = row
        progress = progress + 1

    return df

# ------------------------------------------ FRAME MERGING ---------------------------------------------------------------
######## CLEANS up character frames by using roster frames and selects a single detection for each frame/character combination using roster frames to validate
#### GOAL IS TO SELECT ONE VALIDATED DETECTION
def validate_characters(r, c1, c2, debug=False):

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- SETUP -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    # Finds the latest frame reached and uses that for iterations
    last_frame = max(set(list(r['frame']) + list(c1['frame']) + list(c1['frame'])))

    # Descriptive statistics by character for 'area' and 'confidence'. Modified to handle both model types
    def d_stats(df, roster=False):
        stats = pd.DataFrame(r.groupby('character')[['area', 'confidence']].describe())
        stats.columns = stats.columns.droplevel(0)  # Getting rid of multi-index because it's confusing
        stats = stats.reset_index()

        # Fixing column names for descriptive stats
        col = 0
        statsnames = ['area_', 'confidence_']
        for i in statsnames:
            stats.columns.values[1 + col] = i + "count"; stats.columns.values[2 + col] = i + "mean"; stats.columns.values[3 + col] = i + "std"
            stats.columns.values[4 + col] = i + "min"; stats.columns.values[5 + col] = i + "25%"; stats.columns.values[6 + col] = i + "50%"
            stats.columns.values[7 + col] = i + "75%"; stats.columns.values[8 + col] = i + "max"
            col = col + 8

        ############## Handle shoto detected as AKUMA etc.
        # Sorting rows by the characters that appear the most
        stats = stats.sort_values('area_count', ascending=False).reset_index()
        char1_r = stats['character'][0]  # Character with most detections

        # Setting up area and confidence outliers for roster character 1
        c1_area_avg = stats[stats['character'] == char1_r]['area_mean'].values[0]
        c1_area_std = stats[stats['character'] == char1_r]['area_std'].values[0]
        c1_area_out = [c1_area_avg-(3*c1_area_std), c1_area_avg+(3*c1_area_std)]
        c1_conf_avg = stats[stats['character'] == char1_r]['confidence_mean'].values[0]
        c1_conf_std = stats[stats['character'] == char1_r]['confidence_std'].values[0]
        c1_conf_out = [c1_conf_avg-(3*c1_conf_std), c1_conf_avg+(3*c1_conf_std)]

        if roster:  # Setting up outliers for two characters
            char2_r = stats['character'][1]  # Character with 2nd most detections

            # Setting up area and confidence outliers for roster character 2
            c2_area_avg = stats[stats['character'] == char2_r]['area_mean'].values[0]
            c2_area_std = stats[stats['character'] == char2_r]['area_std'].values[0]
            c2_area_out = [c2_area_avg - (3 * c2_area_std), c2_area_avg + (3 * c2_area_std)]
            c2_conf_avg = stats[stats['character'] == char2_r]['confidence_mean'].values[0]
            c2_conf_std = stats[stats['character'] == char2_r]['confidence_std'].values[0]
            c2_conf_out = [c2_conf_avg - (3 * c2_conf_std), c2_conf_avg + (3 * c2_conf_std)]

            cr_list = [char1_r, char2_r]

            return c1_area_out, c1_conf_out, c2_area_out, c2_conf_out, cr_list, c1_conf_std, c2_conf_std  # For roster model
        return c1_area_out, c1_conf_out, c1_conf_std  # For character models

    r_d_stats = d_stats(r, roster=True)
    char_r_list = r_d_stats[4]  # Characters found in roster model
    c1_r = char_r_list[0]  # Most detected character in roster model
    c2_r = char_r_list[1]  # 2nd most detected character in roster model

    c1_d_stats = d_stats(c1)
    c2_d_stats = d_stats(c2)

    char_c1 = c1['character'][0]  # Character from model 1 output
    char_c2 = c2['character'][0]  # Character from model 2 output
    expected_chars = [char_c1, char_c2]

    ############ Handle shotos being detected as other shotos (Yellow Ken = Sean)
    # Comparing the characters in c1 and c2 (user selected models for the video) against the characters found in the roster model (automatically detected)
    if char_c1 not in char_r_list or char_c2 not in char_r_list:
        if char_c1 in char_r_list:
            c1i = char_r_list.index(char_c1)  # Index where character 1 can be found in roster character list
            c2i = char_r_list[c1i - 1]  # The remaining index is character 2, either 0 or -1
        elif char_c2 in char_r_list:
            c2i = char_r_list.index(char_c2)  # Index where character 2 can be found in roster character list
            c1i = char_r_list[c2i - 1]  # The remaining index is character 1, either 0 or -1
        else:  ##### Neither are in the list for some reason
            if debug:
                print()
                print("ERROR: Characters found in models don't match selected character models.")
                print()
                exit()

    # Removing all rows with outlier confidences (#### Might be better to keep)
    # r = r[r['confidence'] > np.mean([r_d_stats[1][0], r_d_stats[3][0]])]
    # c1 = c1[c1['confidence'] > c1_d_stats[1][0]]
    # c2 = c2[c2['confidence'] > c2_d_stats[1][0]]
    # r.to_csv('r_conf_filter_test.csv', index=False)
    # c1.to_csv('c1_conf_filter_test.csv', index=False)
    # c2.to_csv('c2_conf_filter_test.csv', index=False)

    # Setting up empty dataframes to fill with winning detections
    c1_val = pd.DataFrame(columns=["video", "frame", "time", "character", "action", "description", "x", "y", "w", "h", "area", "confidence"])
    c2_val = pd.DataFrame(columns=["video", "frame", "time", "character", "action", "description", "x", "y", "w", "h", "area", "confidence"])
    f = 0  # Frame of video the loop is currently looking at
    t1 = 0  # Character 1 tracking trigger counter
    t2 = 0  # Character 2 tracking trigger counter
    tracking_c1 = False  # Character 1 tracking trigger
    tracking_c2 = False  # Character 2 tracking trigger
    # Lists used during pre-tracking for getting the average area of the past 5 detections
    t1_prev_5_conf = []
    t2_prev_5_conf = []

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- MAIN LOOP -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------
    while f < last_frame:
        if debug:
            print("----------------------------",str(f) + "f || ", str(round(f/60,2)) + "s ----------------------------")
            print("TRACKING C1 ->", tracking_c1, t1)
            print("TRACKING C2 ->", tracking_c2, t2)

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

        if len(r_f) > 0:
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
        if tracking_c1 and len(c1_f) > 0:
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
            if len(c1_f[(c1_f['distance_average'] < 0.2) & (c1_f['confidence'] > c1_d_stats[1][0])]) > 0:
                c1_f_d = c1_f[(c1_f['distance_average'] < 0.2) & (c1_f['confidence'] > c1_d_stats[1][0])]
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
                    print(c1_f_d.to_string())
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
                if tracking_c2:  # Store character 1 values to return to in case tracking for c1 is successful and we -5f
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
            if len(c2_f[(c2_f['distance_average'] < 0.2) & (c2_f['confidence'] > c2_d_stats[1][0])]) > 0:
                c2_f_d = c2_f[(c2_f['distance_average'] < 0.2) & (c2_f['confidence'] > c2_d_stats[1][0])]
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
                    print(c2_f_d.to_string())
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

# ------------------------------------------ ACTION MERGING ---------------------------------------------------------------
# Takes character frame by frame data and converts it into full "actions".
# Ex.] A sprite appearing on screen for 10 frames consecutively turns into one "action"

######### loop over first action DF again and merge any repeated actions
def actions(df):
    new_df = pd.DataFrame(columns=["startup_frame", "ending_frame", "total_frames", "time", "character", "action",
                                      "description", "avg_confidence", 'starting_position', 'ending_position',
                                      'distance_moved'])
    # Initialize variables on first row
    prev_act = df["action"].iloc[0]
    prev_desc = df["description"].iloc[0]
    firstframe = df["frame"].iloc[0]
    lastframe = 0; duration = 1
    conf = df["confidence"].iloc[0]
    xy1 = [df["x"].iloc[0], df["y"].iloc[0]]
    i = 1
    # Iterating over each row/frame and merging consecutive appearances of the same sprite into one row/aciton
    while i < len(df):
        # If previous action is different from current, stop tracking current move
        if [prev_act, prev_desc] != [df["action"].iloc[i], df["description"].iloc[i]]:
            # print(i, prev_act, prev_desc, firstframe, lastframe, duration, conf)
            xy2 = [df["x"].iloc[i], df["y"].iloc[i]]
            dxy = float(math.dist(xy1, xy2))
            if duration <= 1:  # Only appears for one frame
                lastframe = firstframe
            else:  # Appears for multiple frames
                conf = conf / duration
            new_row = [firstframe, lastframe, duration, round(firstframe/60,2), df["character"].iloc[i], prev_act,
                       prev_desc, conf, xy1, xy2, dxy]
            new_df.loc[len(new_df)] = new_row

            # Reset for next move
            prev_act = df["action"].iloc[i]
            prev_desc = df["description"].iloc[i]
            firstframe = df["frame"].iloc[i]
            xy1 = [df["x"].iloc[i], df["y"].iloc[i]]
            lastframe = 0
            conf = df["confidence"].iloc[i]
            duration = 1
        else:  # If previous action is the same, keep going
            # print(i, prev_act, prev_desc, firstframe, lastframe, duration, conf)
            # print('============================== continue')
            lastframe = df["frame"].iloc[i]
            conf = conf + float(df["confidence"].iloc[i])
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
                new_row = [firstframe, lastframe, lastframe-firstframe, round(int(firstframe)/60,2), new_df1["character"].iloc[j], prev_act,
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
    return new_df2.drop_duplicates(subset=['startup_frame', 'time', 'action', 'description'], keep='last')

# ------------------------------------------ RESULTS MERGING ---------------------------------------------------------------
# Adding "results" to attacks in action dataframe
# *Can find the results of an attack by looking at the opposing character's animation during the attack
# Ex.] If P1 is doing 5HP and P2 is currently being hit, the 5HP's result is "hit"
def results(df):
    # Making sure the df is sorted by frame
    df = df.sort_values(by=["startup_frame"])
    df = df.reset_index(drop=True)
    # Turning frame -> seconds
    df["time"] = df["startup_frame"] / 60

    # List of non-normals / attacks to filter out of hit/block checks
    states = ["standing", "walking", "crouching", "jump", "hit", "block",
              "forwarddash", "backdash", "knockdown", "rise", "walk", "block", "hit", "", "parry", np.nan]

    # Default result values for attacks
    df["P1_result"] = np.where(~df.action_x.isin(states), "whiff", "")
    df["P2_result"] = np.where(~df.action_y.isin(states), "whiff", "")

    # Iterating over actions/sprites and looking for the characters to be in "hit", "block", or "parry" state, then looking at the other character to see what caused that state.
    for index, row in df.iterrows():
        # Converting time doulbe into an actual datetime value
        df.loc[index, "time"] = datetime.timedelta(seconds= int(df.loc[index, "time"]))
        # df.loc[index, "time"] = pd.to_datetime(int(df.loc[index, "time"]), unit='s')

        ####  have to do something here to handle projectiles?
        if df.loc[index, "action_x"] == "hit" or df.loc[index, "action_x"] == "block" or df.loc[index, "action_x"] == "parry":
        # Look at P1 action, if HIT or BLOCK, look for previous P2 action that caused that state
            i = 0
            while True: # Found the move that caused the hit/block
                if df.loc[index - i, "action_y"] not in states and pd.isna(df.loc[index - i, "action_y"]) == False:
                    df.loc[index - i, "P2_result"] = df.loc[index, "action_x"]
                    break
                else: # Keep searching for the move that caused the hit/block
                    i = i + 1
        elif df.loc[index, "action_y"] == "hit" or df.loc[index, "action_y"] == "block" or df.loc[index, "action_y"] == "parry":
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
