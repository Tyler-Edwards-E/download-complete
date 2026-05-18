# Main user workspace for postprocessing

import time
import datetime
import pandas as pd
from ultralytics import YOLO
from rd_aggregate import character_frames, actions, results, roster_frames, validate_characters

start = time.time()
print(datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'), "||", "Starting...")

video_filepath = r'..\[01]-detect\~sample videos\sf3\ibuki_vs_ken.mp4'  # Edit if necessary
video = video_filepath.split("\\")[4].split('.')[0]

roster_filepath = r'..\[01]-detect\~weights\sf3\roster-2024-11-07.pt'  # Edit if necessary
roster_model_name = roster_filepath.split("\\")[4].replace(".pt", "")

character1 = 'KEN'  # Edit if necessary
model1_filepath = r'..\[01]-detect\~weights\sf3\ken-2024-11-04.pt'  # Edit if necessary
model1_name = model1_filepath.split("\\")[4].replace(".pt", "")

character2 = 'IBUKI'  # Edit if necessary
model2_filepath = r"..\[01]-detect\~weights\sf3\ibuki-2024-11-16.pt"  # Edit if necessary
model2_name = model2_filepath.split("\\")[4].replace(".pt", "")

# Folder Names  ######## make sure this can handle any video filetype
r_output_name = f'{video}({roster_model_name})'
c1_output_name = f'{video}({model1_name})'
c2_output_name = f'{video}({model2_name})'


####### Classes parameter, filter for only selected character (or all shotos if there is one)
# roster_m = YOLO(roster_filepath)
# roster_m.predict(source=video_filepath,
#                  save=True,
#                  max_det=4,
#                  save_txt=True,
#                  save_conf=True,
#                  project=r'..\[01]-detect\prediction_output',
#                  name=r_output_name,
#                  verbose=False)

# model1 = YOLO(model1_filepath)
# model1.predict(source=video_filepath,
#                save=True,
#                max_det=4,
#                save_txt=True,
#                save_conf=True,
#                project=r'..\[01]-detect\prediction_output',
#                name=c1_output_name,
#                verbose=False)
#
# model2 = YOLO(model2_filepath)
# model2.predict(source=video_filepath,
#                save=True,
#                max_det=4,
#                save_txt=True,
#                save_conf=True,
#                project=r'..\[01]-detect\prediction_output',
#                name=c2_output_name,
#                verbose=False)

# iou and half parameters ### Experiment with these
# https://docs.ultralytics.com/modes/predict/#inference-arguments

################################################

debug = True

if debug:
      # Shortcut filepaths for debugging/testing
      roster = pd.read_csv(r"[02]-output\00-frames(ROSTER).csv")
      c1_frames = pd.read_csv(r"[02]-output\01-frames(KEN).csv")
      c2_frames = pd.read_csv(r"[02]-output\02-frames(IBUKI).csv")
      c1_val = pd.read_csv(r"[02]-output\10-validate_frames(KEN).csv")
      c2_val = pd.read_csv(r"[02]-output\11-validate_frames(IBUKI).csv")

else:
      # ------------------------------------------ FRAME READING & MERGING -------------------------------------------------
      # Create dataframe of frame by frame sprites on screen. (Raw output of .txt files into one dataframe)
      # Path to yolo output folder
      # Path to detection labels/keys

      ts = time.time()
      print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
            f"(1/7) Reading {r_output_name} labels...")
      r_output_path = f"../[01]-detect/prediction_output/{r_output_name}/labels"
      r_keys_path = "[01]-labelmaps/~roster-labels.txt"  # Should only be one universal roster labelmap/model
      roster = roster_frames(r_output_path, r_keys_path)
      roster = roster.sort_values(by=['frame'])
      roster.to_csv('[02]-output/00-frames(ROSTER).csv', index=False)

      ts = time.time()
      print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
            f"(2/7) Reading {c1_output_name} labels...")
      c1_output_path = f"../[01]-detect/prediction_output/{c1_output_name}/labels"
      c1_keys_path = f"[01]-labelmaps/{character1}-labels.txt"
      c1_frames = character_frames(c1_output_path, c1_keys_path)
      c1_frames = c1_frames.sort_values(by=['frame'])
      c1_frames.to_csv(f'[02]-output/01-frames({character1}).csv', index=False)

      ts = time.time()
      print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
            f"(3/7) Reading {c2_output_name} labels...")
      c2_output_path = f"../[01]-detect/prediction_output/{c2_output_name}/labels"
      c2_keys_path = f"[01]-labelmaps/{character2}-labels.txt"
      c2_frames = character_frames(c2_output_path, c2_keys_path)
      c2_frames = c2_frames.sort_values(by=['frame'])
      c2_frames.to_csv(f'[02]-output/02-frames({character2}).csv', index=False)

      ts = time.time()
      print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
            f"(4/7) Validating frames...")
      c1_val, c2_val = validate_characters(roster, c1_frames, c2_frames)
      c1_val.to_csv(f'[02]-output/10-validate_frames({character1}).csv', index=False)
      c2_val.to_csv(f'[02]-output/11-validate_frames({character2}).csv', index=False)

# ------------------------------------------ ACTION MERGING -----------------------------------------------------------
# Takes character frame by frame data and converts it into full "actions".
# Ex.] A sprite appearing on screen for 10 frames consecutively turns into one "action"

ts = time.time()
print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
      f"(5/7) Converting {character1} frames to character actions...")
A2 = actions(c1_val)
A2.to_csv(f'[02]-output/20-actions({character1}).csv', index=False)

ts = time.time()
print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
      f"(6/7) Converting {character2} frames to character actions...")
B2 = actions(c2_val)
B2.to_csv(f'[02]-output/21-actions({character2}).csv', index=False)

# Sorting and merging both characters to one dataframe
C2 = pd.merge(A2, B2, on="startup_frame", how="outer")
C2 = C2.sort_values(by=["startup_frame"])
C2.to_csv(f'[02]-output/22-actions({character1}&{character2}).csv', index=False)

ts = time.time()
print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||",
      "(7/7) Generating results for both players' actions...")
C3 = results(C2)
C3.to_csv('[02]-output/30-results.csv', index=False)

ts = time.time()
print(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), "||", "Done! =]")

end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
print()
