"""
Tyler Edwards

xxx
"""

import cv2
import time
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import dc_aggregate as dc
start = time.time()
dc.timestamp(f"Starting...")
# ------------------------------------------ EDIT -------------------------------------------------

# Filepaths and variables that need manual attention
video_filepath = r"D:\~dc\rd_videos\sf3\input_videos\matches\urien(ag42)_vs_dudley(halfire).mp4"
detection_output_folder = r'D:\~dc\rd_videos\sf3\output_videos'  # Where you want to detection output to go

# Weights
roster_filepath = r"weights\sf3\roster-20260527.pt"
vfx_filepath = r"weights\sf3\vfx-20260602.pt"
model1_filepath = r"weights\sf3\urien-20260506.pt"
model2_filepath = r"weights\sf3\dudley-20260527.pt"

# *** Only allowing two characters for now, can consider adjusting script to handle sets with multiple characters
character1 = 'URIEN'
character2 = 'DUDLEY'

# Print out extra log .csv's etc.
debug = True
skip_detection = ['ROSTER', 'C1', 'C2', 'VFX']
shortcut = 3

# -------------------------------------------------------------------------------------------------

# --- Don't edit anything below ---
if character1 in ['AKUMA', 'KEN', 'RYU', 'SEAN'] or character2 in ['AKUMA', 'KEN', 'RYU', 'SEAN']:
    shoto = True
else:  # Might need to do this for twins too
    shoto = False

# Grabs name of video and models for convenient file naming later
video = video_filepath.split("\\")[-1] # .split('.')[0]
video_name = video.split('.')[0]
roster_model_name = roster_filepath.split("\\")[-1].replace(".pt", "")
vfx_model_name = vfx_filepath.split("\\")[-1].replace(".pt", "")
model1_name = model1_filepath.split("\\")[-1].replace(".pt", "")
model2_name = model2_filepath.split("\\")[-1].replace(".pt", "")

# Folder Names for output
r_output_name = f'{video_name}({roster_model_name})'
v_output_name = f'{video_name}({vfx_model_name})'
c1_output_name = f'{video_name}({model1_name})'
c2_output_name = f'{video_name}({model2_name})'

# ------------------------------------------ DETECTION -----------------------------------------------------------------
# Run YOLO detection for roster tracking and two characters

dc.timestamp(f"Running YOLO detection on {video}...")
print("-----------------------------------------------------------------------------------------------------------------")

cap = cv2.VideoCapture(video_filepath)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# # --- ROSTER DETECTION ---
roster_m = YOLO(roster_filepath)
roster_m_classes = list(roster_m.names.values())
if debug: # Create .txt file with classes
    with open(f'labelmaps/{roster_model_name}-labels.txt', 'w') as f:
        for i in range(len(roster_m_classes)):
            f.write(f"{roster_m_classes[i]}\n")
roster_whitelist = [roster_m_classes.index(character1),  roster_m_classes.index(character2)]
if shoto:
    for c in ['AKUMA', 'KEN', 'RYU', 'SEAN']:
        if c not in roster_whitelist:
            roster_whitelist.append(roster_m_classes.index(c))
if 'ROSTER' not in skip_detection:
    roster_results = roster_m.predict(source=video_filepath, project=detection_output_folder, name=r_output_name,
                     save=True, save_txt=True, save_conf=True, line_width=2, stream=True,
                     verbose=False, max_det=4, iou=0.1, agnostic_nms=True, classes=roster_whitelist)
    for roster_results in tqdm(roster_results, total=total_frames, desc=f"Detecting {character1} and {character2} in '{video}'", unit="frame"):
        pass
print("-----------------------------------------------------------------------------------------------------------------")

# # --- CHARACTER 1 DETECTION ---
model1 = YOLO(model1_filepath)
model1_classes = list(model1.names.values())
if debug: # Create .txt file with classes
    with open(f'labelmaps/{model1_name}-labels.txt', 'w') as f:
        for i in range(len(model1_classes)):
            f.write(f"{model1_classes[i]}\n")
if 'C1' not in skip_detection:
    m1_results = model1.predict(source=video_filepath, project=detection_output_folder, name=c1_output_name,
                     save=True, save_txt=True, save_conf=True, line_width=2, stream=True,
                     verbose=False, max_det=4, iou=0.1, agnostic_nms=True)
    for m1_results in tqdm(m1_results, total=total_frames, desc=f"Detecting {character1} ACTIONS in '{video}'", unit="frame"):
        pass
print("-----------------------------------------------------------------------------------------------------------------")

# --- CHARACTER 2 DETECTION ---
model2 = YOLO(model2_filepath)
model2_classes = list(model2.names.values())
if debug: # Create .txt file with classes
    with open(f'labelmaps/{model2_name}-labels.txt', 'w') as f:
        for i in range(len(model2_classes)):
            f.write(f"{model2_classes[i]}\n")
if 'C2' not in skip_detection:
    m2_results = model2.predict(source=video_filepath, project=detection_output_folder, name=c2_output_name,
                     save=True, save_txt=True, save_conf=True, line_width=2, stream=True,
                     verbose=False, max_det=4, iou=0.1, agnostic_nms=True)
    for m2_results in tqdm(m2_results, total=total_frames, desc=f"Detecting {character2} ACTIONS in '{video}'", unit="frame"):
        pass
print("-----------------------------------------------------------------------------------------------------------------")

# --- VFX DETECTION ---
vfx_m = YOLO(vfx_filepath)
vfx_m_classes = list(vfx_m.names.values())
if debug: # Create .txt file with classes
    with open(f'labelmaps/{vfx_model_name}-labels.txt', 'w') as f:
        for i in range(len(vfx_m_classes)):
            f.write(f"{vfx_m_classes[i]}\n")
if shoto:
    vfx_whitelist = [i for i, vfx in enumerate(vfx_m_classes) if vfx.startswith((character1 + "-", character2+ "-", 'ALL-', 'SHOTO-'))]
else:
    vfx_whitelist = [i for i, vfx in enumerate(vfx_m_classes) if vfx.startswith((character1 + "-", character2+ "-", 'ALL-'))]
if 'VFX' not in skip_detection:
    vfx_results = vfx_m.predict(source=video_filepath, project=detection_output_folder, name=v_output_name,
                     save=True, save_txt=True, save_conf=True, line_width=2, stream=True,
                     verbose=False,  max_det=4, iou=0.1, classes=vfx_whitelist)
    for vfx_results in tqdm(vfx_results, total=total_frames, desc=f"Detecting VFX in '{video}'", unit="frame"):
        pass
print("-----------------------------------------------------------------------------------------------------------------")
# ------------------------------------------ FRAME READING -------------------------------------------------------------
# Create dataframe of frame by frame sprites on screen. (Raw output of .txt files into one dataframe)

if shortcut < 1:
    # -- ROSTER FRAMES --
    dc.timestamp(f"(1/9) Reading {r_output_name} labels...")
    r_output_path = f"{detection_output_folder}/{r_output_name}/labels"
    roster = dc.roster_frames(r_output_path, roster_m.names, [character1,character2],shoto=shoto, debug=debug)
    roster = roster.sort_values(by=['frame'])
    roster.to_excel(f'{video_name}.xlsx', sheet_name='00-roster_frames', index=False)


    # -- CHARACTER 1 FRAMES --
    dc.timestamp(f"(2/9) Reading {c1_output_name} labels...")
    c1_output_path = f"{detection_output_folder}/{c1_output_name}/labels"
    c1_frames = dc.character_frames(c1_output_path, model1.names, debug=debug)
    c1_frames = c1_frames[~(c1_frames['character'].isin(['NPC', 'VFX']))]
    c1_frames = c1_frames.sort_values(by=['frame'])
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        c1_frames.to_excel(writer, sheet_name=f'01-{character1}_frames', index=False)


    # -- CHARACTER 2 FRAMES --
    dc.timestamp(f"(3/9) Reading {c2_output_name} labels...")
    c2_output_path = f"{detection_output_folder}/{c2_output_name}/labels"
    c2_frames = dc.character_frames(c2_output_path, model2.names, debug=debug)
    c2_frames = c2_frames[~(c2_frames['character'].isin(['NPC', 'VFX']))]
    c2_frames = c2_frames.sort_values(by=['frame'])
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        c2_frames.to_excel(writer, sheet_name=f'02-{character2}_frames', index=False)


    # -- VFX FRAMES --
    dc.timestamp(f"(4/9) Reading {v_output_name} labels...")
    v_output_path = f"{detection_output_folder}/{v_output_name}/labels"
    vfx_frames = dc.character_frames(v_output_path, vfx_m.names, vfx=True, debug=debug)
    vfx_frames = vfx_frames.sort_values(by=['frame'])
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        vfx_frames.to_excel(writer, sheet_name=f'03-vfx_frames', index=False)

# -- SHORTCUT 1: Skip frame detection --
else:
    dc.timestamp('SHORTCUT 1: reading excel sheets')  # shortcut for testing
    roster = pd.read_excel(f'{video_name}.xlsx', sheet_name='00-roster_frames')
    c1_frames = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'01-{character1}_frames')
    c2_frames = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'02-{character2}_frames')
    vfx_frames = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'03-vfx_frames')

# ------------------------------------------ FRAME VALIDATION ----------------------------------------------------------

if shortcut < 2:
    # -- VALIDATE CHARACTER FRAMES --
    dc.timestamp(f"(5/9) Validating character detections...")
    c1_val = dc.validate_characters(roster, c1_frames, iou_threshold=0.1)
    c2_val = dc.validate_characters(roster, c2_frames, iou_threshold=0.1)
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        c1_val.to_excel(writer, sheet_name=f'10-validate_frames({character1})', index=False)
        c2_val.to_excel(writer, sheet_name=f'11-validate_frames({character2})', index=False)
else:
    dc.timestamp('SHORTCUT 2: reading excel sheets')  # shortcut for testing
    c1_val = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'10-validate_frames({character1})')
    c2_val = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'11-validate_frames({character2})')

# ------------------------------------------ FRAME MERGING -------------------------------------------------------------
# Takes character frame by frame data and converts it into full "actions".
# Ex.] A sprite appearing on screen for 10 frames consecutively turns into one "action"

if shortcut < 3:
    dc.timestamp(f"(6/9) Merging {character1} frames into character actions...")
    c1_actions = dc.merge_frames(c1_val)
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        c1_actions.to_excel(writer, sheet_name=f'20-merge_frames({character1})', index=False)

    dc.timestamp(f"(7/9) Merging {character2} frames into character actions...")
    c2_actions = dc.merge_frames(c2_val)
    with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
        c2_actions.to_excel(writer, sheet_name=f'21-merge_frames({character2})', index=False)
else:
    dc.timestamp('SHORTCUT 3: reading excel sheets')  # shortcut for testing
    c1_actions = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'20-merge_frames({character1})')
    c2_actions = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'21-merge_frames({character2})')

dc.timestamp(f"Loading sf3_context.yaml")
context_yaml = dc.load_context_rules('sf3_context.yaml')
dc.timestamp(f"apply_sequence_context")
c1_context = dc.apply_advanced_sequence_context(c1_actions, context_yaml)
c2_context = dc.apply_advanced_sequence_context(c2_actions, context_yaml)

with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
    c1_context.to_excel(writer, sheet_name=f'30-context_({character1})', index=False)
    c2_context.to_excel(writer, sheet_name=f'31-context_({character2})', index=False)

exit()
########## Add average conf filter to actions

# timestamp(f"(8/9) Merging VFX frames...")
# vfx_actions = merge_frames(vfx_frames, debug=debug)
# with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
#     vfx_actions.to_excel(writer, sheet_name=f'22-actions(VFX)', index=False)

# Sorting and merging both characters to one dataframe
merged_actions = c1_actions.merge(c2_actions, on="startup_frame", how='outer')
merged_actions = merged_actions.sort_values(by=["startup_frame"])
with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
    merged_actions.to_excel(writer, sheet_name=f'23-merged-actions)', index=False)

# ### SHORTCUT 2
# timestamp('SHORTCUT: reading excel sheets')  # shortcut for testing
# C2 = pd.read_excel(f'{video_name}.xlsx', sheet_name=f'22-actions({character1}&{character2})')

timestamp("(9/9) Generating results for both players' actions...")
final_results = results(merged_actions)
with pd.ExcelWriter(f'{video_name}.xlsx', engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
    final_results.to_excel(writer, sheet_name=f'30-results', index=False)


timestamp("Download Complete! =]")
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
print()
