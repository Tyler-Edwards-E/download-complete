

# Tyler Edwards
# 4-11-2022
# yolov5 .txt > .csv

import os
import pandas as pd
import datetime
import time

start = time.time()
print("Starting...")

P1_filepath = r"C:\Users\Ty\3D Objects\~RD\detection_output\chunCC\labels" # Path to yolov5 output folder
P1_keypath = r"C:\Users\Ty\3D Objects\~RD\RD-SF3S\SF3rdStrike-YARDS\chunlabels.txt" # Path to detection labels/keys

P2_filepath = r"C:\Users\Ty\3D Objects\~RD\detection_output\kenCC\labels" # Path to yolov5 output folder
P2_keypath = r"C:\Users\Ty\3D Objects\~RD\RD-SF3S\SF3rdStrike-YARDS\kenlabels.txt" # Path to detection labels/keys


def frames(filepath, keypath): # Create dataset of frame by frame sprites on screen

    df = pd.DataFrame(columns = ["video", "frame", "character", "action", "description", "confidence"])

    with open(keypath) as f:
        lines = f.readlines()

        i_list = list(range(0,len(lines)))
        lines2 = []
        for i in lines:
            lines2.append(i.strip())

        keys = dict(zip(i_list, lines2))

    for file in os.listdir(filepath):
        with open(str(filepath + "/" + file)) as f:
            lines = f.readlines()

        filesplit = file.split("_")
        linesplit = lines[0].split()

        #character = lines2[0].split("-")[0]
        video = filesplit[0]

        frame = int(filesplit[-1].replace(".txt", ""))

        clas = linesplit[0] # Class

        actiondesc = keys[int(clas)].split("-")
        character = actiondesc[0]
        # Default values
        action = actiondesc[1]
        desc = ""

        # Values for jumps, blocking description, etc.
        if len(actiondesc) > 2:
            desc = actiondesc[2]
            if len(actiondesc) == 4:
                action = desc + actiondesc[3]
                desc = ""

        # Label cleanup
        if desc == "stand": # Label cleanup
            desc = "standing"
        elif desc == "crouch":
            desc = "crouching"
        elif desc == "8":
            desc = "neutral"
        elif desc == "79":
            desc = "forward/backward"

        conf = linesplit[5]

        # print("VIDEO:", video)
        # print("FRAME:", frame)
        # print("CHARACTER:", character)
        # # print(actiondesc)
        # print("ACTION:", action)
        # print("DESCRIPTION:", desc)
        # print("CONFIDENCE:", conf)
        # print("--------------------------------------------------------------------")

        row = [video, frame, character, action, desc, conf]
        df.loc[len(df)] = row

    return df

print("(1/5) Converting P1 sprite detection output to dataframe...")
A = frames(P1_filepath, P1_keypath)
print("(2/5) Converting P2 sprite detection output to dataframe...")
B = frames(P2_filepath, P2_keypath)
A = A.sort_values(by=['frame'])
B = B.sort_values(by=['frame'])
# Merge both characters to one dataframe
C = pd.merge(A, B, on=['video',  'frame'])
C = C.sort_values(by=['frame'])
C.to_csv('RD1(Frames).csv', index=False)

# ------------------------------------------ ACTION MERGING ---------------------------------------------------------------
def actions(df): # Takes character frame by frame dataframe and converts it into full "actions"
    new_df = pd.DataFrame(columns = ["startup_frame", "ending_frame", "total_frames", "character", "action", "description", "avg_confidence"])
    prev_act = df["action"].iloc[0]
    desc2 = df["description"].iloc[0]
    firstframe = df["frame"].iloc[0]
    lastframe = ""
    duration = 0
    count = 0
    conf = 0

    for index, row in df.iterrows():
        if(prev_act != row["action"] or count == len(df) - 1): # If previous action is different from current, stop tracking current move
            if duration <= 1: # Only appears for one frame
                firstframe = lastframe
                totalframe = 1
            else:
                totalframe = lastframe - firstframe
                conf = conf / totalframe
            new_row = {"startup_frame": firstframe, "ending_frame": lastframe, "total_frames": totalframe,  "character": row["character"],  "action": prev_act, "description": desc2, "avg_confidence": conf}
            new_df.loc[len(new_df)] = new_row

            # print(firstframe, [firstframe, lastframe], prev_act, duration, conf)
            # print("----------------------------------")

            # Reset for next move
            prev_act = row["action"]
            desc2 = row["description"]
            firstframe = row["frame"]
            duration = 0
            conf = 0
        elif prev_act == row["action"]: # If previous action is the same
            desc2 = row["description"]
            lastframe = row["frame"]
            conf = conf + float(row["confidence"])
            duration = duration + 1
        count = count + 1
    return new_df

print("(3/5) Converting P1 frames to character actions...")
A2 = actions(A)
A2 = A2[A2["total_frames"] >= 4]
# A2.to_csv('P1_SOLO.csv', index=False)
print("(4/5) Converting P2 frames to character actions...")
B2 = actions(B)
B2 = B2[B2["total_frames"] >= 4]
# B2.to_csv('P2_SOLO.csv', index=False)

C2 = pd.merge(A2, B2, on = "startup_frame", how = "outer")
C2 = C2.sort_values(by = ["startup_frame"])
C2.to_csv('RD2(Actions).csv', index=False)

def results(df): # Adding "results" to attacks in action dataframe and adding a proper time column
    df = df.sort_values(by=["startup_frame"])
    df = df.reset_index(drop=True)
    df["time"] = df["startup_frame"] / 60
    df["result1"] = ""
    df["result2"] = ""

    # List of non-normals / attacks to filter out of hit/block checks
    states = ["standing", "walking", "crouching", "jump", "hit", "block", "forwarddash", "backdash", "knockdown", "rise"]

    for index, row in df.iterrows():
        df.loc[index, "time"] = str(datetime.timedelta(seconds= df.loc[index, "time"]))

        if df.loc[index, "action_x"] == "hit" or df.loc[index, "action_x"] == "block" or df.loc[index, "action_x"] == "parry":
        # Look at P1 action, if HIT or BLOCK, look for previous P2 action that caused that state
            i = 0
            while True:
                if df.loc[index - i, "action_y"] not in states and pd.isna(df.loc[index - i, "action_y"]) == False:
                    df.loc[index - i, "result2"] = df.loc[index, "action_x"]
                    break
                else:
                    i = i + 1
        elif df.loc[index, "action_y"] == "hit" or df.loc[index, "action_y"] == "block" or df.loc[index, "action_y"] == "parry":
        # Look at P2 action, if HIT or BLOCK, look for previous P1 action that caused that state
            i = 0
            while True:
                if df.loc[index - i, "action_x"] not in states and pd.isna(df.loc[index - i, "action_x"]) == False:
                    df.loc[index - i, "result1"] = df.loc[index, "action_y"]
                    break
                else:
                    i = i + 1

        # Reorder and rename columns
    df = df[['time', 'startup_frame', 'ending_frame_x', 'total_frames_x', "character_x", "action_x", "description_x", "result1", "avg_confidence_x",
            'ending_frame_y', 'total_frames_y', "character_y", "action_y", "description_y", "result2", "avg_confidence_y",]]
    df.rename(columns={'time': 'timestamp', 'ending_frame_x': 'P1_ending_frame', 'total_frames_x': 'P1_total_frames',
                        'character_x': 'P1_character', 'action_x': 'P1_action', 'description_x': 'P1_description',
                        'result1': 'P1_result', 'avg_confidence_x': 'P1_avg_confidence',
                        'ending_frame_y': 'P2_ending_frame', 'total_frames_y': 'P2_total_frames',
                        'character_y': 'P2_character', 'action_y': 'P2_action', 'description_y': 'P2_description',
                        'result2': 'P2_result', 'avg_confidence_y': 'P2_avg_confidence'}, inplace=True)
    return df


print("(5/5) Generating results for both players' actions...")
C3 = results(C2)
C3.to_csv('RD3(Results).csv', index=False)
print("Done! =]")

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print()